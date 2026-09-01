from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from gkr.ai import Generation
from gkr.context import EvidenceBundle
from gkr.decision import DecisionParseResult
from gkr.retrieval import RetrievalPlan
from gkr.schemas import Actor
from gkr.verification import CitationVerification, SemanticVerification

_GENESIS_HASH = "0" * 64


@dataclass(frozen=True)
class ExecutionTrace:
    trace_id: str
    created_at: datetime
    question: str
    request_hash: str
    resolved_principals: tuple[str, ...]
    scope_hash: str
    as_of: date
    known_at: datetime
    authority_snapshot_id: str
    evidence_bundle_id: str
    evidence_references: tuple[str, ...]
    retriever_id: str
    retriever_configuration: tuple[tuple[str, str], ...]
    decision_parse: dict[str, object]
    model: str | None
    model_metadata: dict[str, Any] | None
    prompt_hash: str
    candidate_answer_hash: str | None
    citation_verification: dict[str, Any] | None
    semantic_verification: dict[str, Any] | None
    duration_ms: float
    peak_process_rss_bytes: int | None
    publication_status: str

    @classmethod
    def create(
        cls,
        *,
        question: str,
        actor: Actor,
        as_of: date,
        evidence: EvidenceBundle,
        retrieval: RetrievalPlan,
        decision_parse: DecisionParseResult,
        generation: Generation | None,
        citation_verification: CitationVerification | None,
        semantic_verification: SemanticVerification | None,
        duration_ms: float,
        peak_process_rss_bytes: int | None,
        publication_status: str,
        created_at: datetime | None = None,
    ) -> ExecutionTrace:
        timestamp = (created_at or datetime.now(UTC)).astimezone(UTC)
        principals = tuple(sorted(actor.principals))
        request = {
            "question": question.strip(),
            "as_of": as_of.isoformat(),
            "known_at": evidence.known_at.isoformat().replace("+00:00", "Z"),
        }
        values: dict[str, Any] = {
            "created_at": timestamp.isoformat().replace("+00:00", "Z"),
            "question": question.strip(),
            "request_hash": _digest(request),
            "resolved_principals": list(principals),
            "scope_hash": _digest({"principals": principals}),
            "as_of": as_of.isoformat(),
            "known_at": evidence.known_at.isoformat().replace("+00:00", "Z"),
            "authority_snapshot_id": evidence.authority_snapshot_id,
            "evidence_bundle_id": evidence.evidence_bundle_id,
            "evidence_references": list(evidence.record_references),
            "retriever_id": retrieval.retriever_id,
            "retriever_configuration": [list(item) for item in retrieval.configuration],
            "decision_parse": decision_parse.to_dict(),
            "model": generation.model if generation else None,
            "model_metadata": generation.metadata if generation else None,
            "prompt_hash": _text_digest(evidence.prompt),
            "candidate_answer_hash": _text_digest(generation.text) if generation else None,
            "citation_verification": (
                citation_verification.to_dict() if citation_verification else None
            ),
            "semantic_verification": (
                semantic_verification.to_dict() if semantic_verification else None
            ),
            "duration_ms": round(duration_ms, 3),
            "peak_process_rss_bytes": peak_process_rss_bytes,
            "publication_status": publication_status,
        }
        return cls.from_dict({"trace_id": _digest(values), **values})

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "created_at": self.created_at.isoformat().replace("+00:00", "Z"),
            "question": self.question,
            "request_hash": self.request_hash,
            "resolved_principals": list(self.resolved_principals),
            "scope_hash": self.scope_hash,
            "as_of": self.as_of.isoformat(),
            "known_at": self.known_at.isoformat().replace("+00:00", "Z"),
            "authority_snapshot_id": self.authority_snapshot_id,
            "evidence_bundle_id": self.evidence_bundle_id,
            "evidence_references": list(self.evidence_references),
            "retriever_id": self.retriever_id,
            "retriever_configuration": [list(item) for item in self.retriever_configuration],
            "decision_parse": self.decision_parse,
            "model": self.model,
            "model_metadata": self.model_metadata,
            "prompt_hash": self.prompt_hash,
            "candidate_answer_hash": self.candidate_answer_hash,
            "citation_verification": self.citation_verification,
            "semantic_verification": self.semantic_verification,
            "duration_ms": self.duration_ms,
            "peak_process_rss_bytes": self.peak_process_rss_bytes,
            "publication_status": self.publication_status,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ExecutionTrace:
        return cls(
            trace_id=str(value["trace_id"]),
            created_at=_datetime(value["created_at"]),
            question=str(value["question"]),
            request_hash=str(value["request_hash"]),
            resolved_principals=tuple(value["resolved_principals"]),
            scope_hash=str(value["scope_hash"]),
            as_of=date.fromisoformat(str(value["as_of"])),
            known_at=_datetime(value["known_at"]),
            authority_snapshot_id=str(value["authority_snapshot_id"]),
            evidence_bundle_id=str(value["evidence_bundle_id"]),
            evidence_references=tuple(value["evidence_references"]),
            retriever_id=str(value["retriever_id"]),
            retriever_configuration=tuple(
                (str(item[0]), str(item[1])) for item in value["retriever_configuration"]
            ),
            decision_parse=dict(value["decision_parse"]),
            model=str(value["model"]) if value.get("model") is not None else None,
            model_metadata=(
                dict(value["model_metadata"]) if value.get("model_metadata") is not None else None
            ),
            prompt_hash=str(value["prompt_hash"]),
            candidate_answer_hash=(
                str(value["candidate_answer_hash"])
                if value.get("candidate_answer_hash") is not None
                else None
            ),
            citation_verification=(
                dict(value["citation_verification"])
                if value.get("citation_verification") is not None
                else None
            ),
            semantic_verification=(
                dict(value["semantic_verification"])
                if value.get("semantic_verification") is not None
                else None
            ),
            duration_ms=float(value["duration_ms"]),
            peak_process_rss_bytes=(
                int(value["peak_process_rss_bytes"])
                if value.get("peak_process_rss_bytes") is not None
                else None
            ),
            publication_status=str(value["publication_status"]),
        )


class TraceStore:
    """Append-only durable query traces, separate from authority data."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self._connection = sqlite3.connect(self.path, timeout=30)
        self._connection.row_factory = sqlite3.Row
        self._initialize()

    def __enter__(self) -> TraceStore:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        self._connection.close()

    def append(self, trace: ExecutionTrace) -> str:
        value = trace.to_dict()
        if trace.trace_id != _trace_id(value):
            raise ValueError("Execution trace ID does not match its canonical payload")
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        try:
            with self._connection:
                self._connection.execute("BEGIN IMMEDIATE")
                last_event = self._connection.execute(
                    "SELECT event_hash FROM execution_traces ORDER BY sequence DESC LIMIT 1"
                ).fetchone()
                previous_hash = last_event["event_hash"] if last_event else _GENESIS_HASH
                event_hash = hashlib.sha256(f"{previous_hash}\n{payload}".encode()).hexdigest()
                self._connection.execute(
                    """
                    INSERT INTO execution_traces (
                        trace_id,
                        payload,
                        previous_event_hash,
                        event_hash
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (trace.trace_id, payload, previous_hash, event_hash),
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"Cannot append execution trace {trace.trace_id}: {exc}") from exc
        return event_hash

    def get(self, trace_id: str) -> ExecutionTrace | None:
        row = self._connection.execute(
            "SELECT payload FROM execution_traces WHERE trace_id = ?",
            (trace_id,),
        ).fetchone()
        return ExecutionTrace.from_dict(json.loads(row["payload"])) if row else None

    def count(self) -> int:
        row = self._connection.execute(
            "SELECT COUNT(*) AS count FROM execution_traces"
        ).fetchone()
        return int(row["count"])

    def verify_chain(self) -> bool:
        previous_hash = _GENESIS_HASH
        rows = self._connection.execute(
            """
            SELECT trace_id, payload, previous_event_hash, event_hash
            FROM execution_traces
            ORDER BY sequence
            """
        ).fetchall()
        for row in rows:
            if row["previous_event_hash"] != previous_hash:
                return False
            expected = hashlib.sha256(f"{previous_hash}\n{row['payload']}".encode()).hexdigest()
            if row["event_hash"] != expected:
                return False
            try:
                value = json.loads(row["payload"])
            except (TypeError, json.JSONDecodeError):
                return False
            if row["trace_id"] != value.get("trace_id") or row["trace_id"] != _trace_id(value):
                return False
            previous_hash = row["event_hash"]
        return True

    def _initialize(self) -> None:
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS execution_traces (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                trace_id TEXT NOT NULL UNIQUE,
                payload TEXT NOT NULL,
                previous_event_hash TEXT NOT NULL,
                event_hash TEXT NOT NULL UNIQUE
            );

            CREATE UNIQUE INDEX IF NOT EXISTS trace_single_chain_successor
                ON execution_traces (previous_event_hash);

            CREATE TRIGGER IF NOT EXISTS execution_traces_no_update
            BEFORE UPDATE ON execution_traces
            BEGIN
                SELECT RAISE(ABORT, 'execution_traces is append-only');
            END;

            CREATE TRIGGER IF NOT EXISTS execution_traces_no_delete
            BEFORE DELETE ON execution_traces
            BEGIN
                SELECT RAISE(ABORT, 'execution_traces is append-only');
            END;
            """
        )
        self._connection.commit()


def _digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _text_digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _trace_id(value: dict[str, Any]) -> str:
    trace_value = dict(value)
    trace_value.pop("trace_id", None)
    return _digest(trace_value)


def _datetime(value: object) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.utcoffset() is None:
        raise ValueError("Trace datetime must include a timezone")
    return parsed.astimezone(UTC)
