from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

ProposalDecision = Literal["approved", "rejected"]
ProposalState = Literal["pending", "approved", "rejected"]

_GENESIS_HASH = "0" * 64
_FORBIDDEN_CANDIDATE_FIELDS = {
    "observed_at",
    "status",
    "supersedes",
    "version",
}


@dataclass(frozen=True)
class ChangeProposal:
    """Candidate content that has no authority version until promotion."""

    proposal_id: str
    record_id: str
    candidate: dict[str, Any]
    proposed_by: str
    proposed_at: datetime
    base_reference: str | None = None

    def validate(self) -> None:
        if not self.proposal_id.strip() or not self.record_id.strip():
            raise ValueError("proposal_id and record_id cannot be empty")
        if not self.proposed_by.strip():
            raise ValueError("proposed_by cannot be empty")
        if self.proposed_at.utcoffset() is None:
            raise ValueError("proposed_at must include a timezone")
        forbidden = _FORBIDDEN_CANDIDATE_FIELDS.intersection(self.candidate)
        if forbidden:
            fields = ", ".join(sorted(forbidden))
            raise ValueError(f"Proposal candidate contains authoritative fields: {fields}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "record_id": self.record_id,
            "base_reference": self.base_reference,
            "candidate": self.candidate,
            "proposed_by": self.proposed_by,
            "proposed_at": self.proposed_at.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ChangeProposal:
        proposed_at = datetime.fromisoformat(str(value["proposed_at"]).replace("Z", "+00:00"))
        if proposed_at.utcoffset() is None:
            raise ValueError("proposed_at must include a timezone")
        proposal = cls(
            proposal_id=str(value["proposal_id"]),
            record_id=str(value["record_id"]),
            base_reference=(
                str(value["base_reference"]) if value.get("base_reference") is not None else None
            ),
            candidate=dict(value["candidate"]),
            proposed_by=str(value["proposed_by"]),
            proposed_at=proposed_at.astimezone(UTC),
        )
        proposal.validate()
        return proposal


class ProposalStore:
    """Append-only local workflow audit, deliberately separate from authority."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self._connection = sqlite3.connect(self.path, timeout=30)
        self._connection.row_factory = sqlite3.Row
        self._initialize()

    def __enter__(self) -> ProposalStore:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        self._connection.close()

    def submit(self, proposal: ChangeProposal) -> str:
        proposal.validate()
        payload = {
            "event": "submitted",
            "proposal": proposal.to_dict(),
        }
        return self._append_event(
            proposal.proposal_id,
            "submitted",
            payload,
            expected_state=None,
        )

    def decide(
        self,
        proposal_id: str,
        *,
        decision: ProposalDecision,
        reviewed_by: str,
        reviewed_at: datetime,
        comment: str = "",
    ) -> str:
        if decision not in {"approved", "rejected"}:
            raise ValueError(f"Unsupported proposal decision: {decision}")
        if not reviewed_by.strip():
            raise ValueError("reviewed_by cannot be empty")
        if reviewed_at.utcoffset() is None:
            raise ValueError("reviewed_at must include a timezone")
        payload = {
            "event": decision,
            "proposal_id": proposal_id,
            "reviewed_by": reviewed_by,
            "reviewed_at": reviewed_at.astimezone(UTC).isoformat().replace("+00:00", "Z"),
            "comment": comment,
        }
        return self._append_event(
            proposal_id,
            decision,
            payload,
            expected_state="pending",
        )

    def state(self, proposal_id: str) -> ProposalState | None:
        return self._state(proposal_id)

    def get(self, proposal_id: str) -> ChangeProposal | None:
        row = self._connection.execute(
            """
            SELECT payload
            FROM proposal_events
            WHERE proposal_id = ? AND event_type = 'submitted'
            """,
            (proposal_id,),
        ).fetchone()
        if row is None:
            return None
        payload = json.loads(row["payload"])
        return ChangeProposal.from_dict(payload["proposal"])

    def verify_chain(self) -> bool:
        previous_hash = _GENESIS_HASH
        rows = self._connection.execute(
            """
            SELECT payload, previous_event_hash, event_hash
            FROM proposal_events
            ORDER BY sequence
            """
        ).fetchall()
        for row in rows:
            if row["previous_event_hash"] != previous_hash:
                return False
            expected = _event_digest(previous_hash, row["payload"])
            if row["event_hash"] != expected:
                return False
            previous_hash = row["event_hash"]
        return True

    def _state(self, proposal_id: str) -> ProposalState | None:
        row = self._connection.execute(
            """
            SELECT event_type
            FROM proposal_events
            WHERE proposal_id = ?
            ORDER BY sequence DESC
            LIMIT 1
            """,
            (proposal_id,),
        ).fetchone()
        if row is None:
            return None
        return "pending" if row["event_type"] == "submitted" else row["event_type"]

    def _append_event(
        self,
        proposal_id: str,
        event_type: str,
        value: dict[str, Any],
        *,
        expected_state: ProposalState | None,
    ) -> str:
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        try:
            with self._connection:
                self._connection.execute("BEGIN IMMEDIATE")
                current_state = self._state(proposal_id)
                if current_state != expected_state:
                    if expected_state is None:
                        raise ValueError(f"Proposal already exists: {proposal_id}")
                    raise ValueError(f"Proposal is not {expected_state}: {proposal_id}")
                last_event = self._connection.execute(
                    "SELECT event_hash FROM proposal_events ORDER BY sequence DESC LIMIT 1"
                ).fetchone()
                previous_hash = last_event["event_hash"] if last_event else _GENESIS_HASH
                event_hash = _event_digest(previous_hash, payload)
                self._connection.execute(
                    """
                    INSERT INTO proposal_events (
                        proposal_id,
                        event_type,
                        payload,
                        previous_event_hash,
                        event_hash
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (proposal_id, event_type, payload, previous_hash, event_hash),
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"Cannot append proposal event for {proposal_id}: {exc}") from exc
        return event_hash

    def _initialize(self) -> None:
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS proposal_events (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                proposal_id TEXT NOT NULL,
                event_type TEXT NOT NULL CHECK (
                    event_type IN ('submitted', 'approved', 'rejected')
                ),
                payload TEXT NOT NULL,
                previous_event_hash TEXT NOT NULL,
                event_hash TEXT NOT NULL UNIQUE
            );

            CREATE INDEX IF NOT EXISTS proposal_event_lookup
                ON proposal_events (proposal_id, sequence);

            CREATE UNIQUE INDEX IF NOT EXISTS proposal_single_chain_successor
                ON proposal_events (previous_event_hash);

            CREATE UNIQUE INDEX IF NOT EXISTS proposal_single_submission
                ON proposal_events (proposal_id)
                WHERE event_type = 'submitted';

            CREATE TRIGGER IF NOT EXISTS proposal_events_no_update
            BEFORE UPDATE ON proposal_events
            BEGIN
                SELECT RAISE(ABORT, 'proposal_events is append-only');
            END;

            CREATE TRIGGER IF NOT EXISTS proposal_events_no_delete
            BEFORE DELETE ON proposal_events
            BEGIN
                SELECT RAISE(ABORT, 'proposal_events is append-only');
            END;
            """
        )
        self._connection.commit()


def _event_digest(previous_hash: str, payload: str) -> str:
    return hashlib.sha256(f"{previous_hash}\n{payload}".encode()).hexdigest()
