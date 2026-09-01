from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

from gkr.schemas import Actor, KnowledgeRecord
from gkr.source import SourceArtifact, verify_record_source

_GENESIS_HASH = "0" * 64


@dataclass(frozen=True)
class AuthorizedCorpus:
    actor: Actor
    as_of: date
    known_at: datetime
    records: tuple[KnowledgeRecord, ...]
    authority_snapshot_id: str


class AuthorityStore:
    """Append-only SQLite authority ledger with temporal and ACL-aware reads."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self._connection = sqlite3.connect(self.path, timeout=30)
        self._connection.row_factory = sqlite3.Row
        self._initialize()

    def __enter__(self) -> AuthorityStore:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        self._connection.close()

    def append(
        self,
        record: KnowledgeRecord,
        *,
        source_artifact: SourceArtifact | None = None,
    ) -> str:
        with self._connection:
            self._connection.execute("BEGIN IMMEDIATE")
            return self._append(record, source_artifact)

    def append_many(self, records: Iterable[KnowledgeRecord]) -> list[str]:
        event_hashes: list[str] = []
        with self._connection:
            self._connection.execute("BEGIN IMMEDIATE")
            for record in records:
                event_hashes.append(self._append(record, None))
        return event_hashes

    def import_jsonl(self, path: str | Path) -> list[str]:
        source_path = Path(path)
        records: list[tuple[KnowledgeRecord, SourceArtifact | None]] = []
        with source_path.open(encoding="utf-8") as source:
            for line_number, raw_line in enumerate(source, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    value = json.loads(line)
                    if "record" in value:
                        record = KnowledgeRecord.from_dict(value["record"])
                        artifact = SourceArtifact.from_descriptor(
                            value["source_artifact"],
                            base_directory=source_path.parent,
                        )
                    else:
                        record = KnowledgeRecord.from_dict(value)
                        artifact = None
                    records.append((record, artifact))
                except (KeyError, json.JSONDecodeError, TypeError, ValueError) as exc:
                    raise ValueError(f"{path}:{line_number}: {exc}") from exc
        event_hashes: list[str] = []
        with self._connection:
            self._connection.execute("BEGIN IMMEDIATE")
            for record, artifact in records:
                event_hashes.append(self._append(record, artifact))
        return event_hashes

    def current_records(
        self,
        *,
        actor: Actor,
        as_of: date,
        known_at: datetime | None = None,
    ) -> AuthorizedCorpus:
        """Return one latest applicable version per record, then enforce authorization.

        Filtering by status and ACL deliberately happens after temporal version resolution.
        This prevents a caller from falling back to an older version after a later version
        retires a record or revokes access.
        """

        effective_known_at = known_at or datetime.now(UTC)
        if effective_known_at.utcoffset() is None:
            raise ValueError("known_at must include a timezone")
        rows = self._connection.execute(
            """
            SELECT payload
            FROM knowledge_ledger
            WHERE valid_from <= ?
              AND (valid_to IS NULL OR valid_to > ?)
            ORDER BY record_id, version
            """,
            (
                as_of.isoformat(),
                as_of.isoformat(),
            ),
        ).fetchall()

        latest: dict[str, KnowledgeRecord] = {}
        for row in rows:
            record = KnowledgeRecord.from_dict(json.loads(row["payload"]))
            if not record.is_known_at(effective_known_at):
                continue
            latest[record.record_id] = record

        records = tuple(
            record
            for record in latest.values()
            if record.status == "approved" and record.is_permitted(actor)
        )
        return AuthorizedCorpus(
            actor=actor,
            as_of=as_of,
            known_at=effective_known_at.astimezone(UTC),
            records=records,
            authority_snapshot_id=_authority_snapshot_id(as_of, records),
        )

    def get(self, reference: str) -> KnowledgeRecord | None:
        try:
            record_id, raw_version = reference.rsplit(":v", maxsplit=1)
            version = int(raw_version)
        except (ValueError, TypeError):
            return None
        row = self._connection.execute(
            "SELECT payload FROM knowledge_ledger WHERE record_id = ? AND version = ?",
            (record_id, version),
        ).fetchone()
        if row is None:
            return None
        return KnowledgeRecord.from_dict(json.loads(row["payload"]))

    def count(self) -> int:
        row = self._connection.execute("SELECT COUNT(*) AS count FROM knowledge_ledger").fetchone()
        return int(row["count"])

    def verify_chain(self) -> bool:
        previous_hash = _GENESIS_HASH
        rows = self._connection.execute(
            """
            SELECT
                record_id,
                version,
                valid_from,
                valid_to,
                observed_at,
                payload,
                previous_event_hash,
                event_hash
            FROM knowledge_ledger
            ORDER BY sequence
            """
        ).fetchall()
        for row in rows:
            if row["previous_event_hash"] != previous_hash:
                return False
            expected = _event_digest(previous_hash, row["payload"])
            if row["event_hash"] != expected:
                return False
            try:
                record = KnowledgeRecord.from_dict(json.loads(row["payload"]))
            except (json.JSONDecodeError, TypeError, ValueError):
                return False
            indexed_values = (
                row["record_id"],
                int(row["version"]),
                row["valid_from"],
                row["valid_to"],
                row["observed_at"],
            )
            payload_values = (
                record.record_id,
                record.version,
                record.valid_from.isoformat(),
                record.valid_to.isoformat() if record.valid_to else None,
                record.observed_at.isoformat().replace("+00:00", "Z"),
            )
            if indexed_values != payload_values:
                return False
            previous_hash = row["event_hash"]
        return True

    def _append(
        self,
        record: KnowledgeRecord,
        source_artifact: SourceArtifact | None,
    ) -> str:
        record.validate()
        if record.status not in {"approved", "retired"}:
            raise ValueError(f"{record.reference}: workflow states do not belong in authority")
        verify_record_source(record, source_artifact)
        previous_version = self._connection.execute(
            """
            SELECT version, observed_at
            FROM knowledge_ledger
            WHERE record_id = ?
            ORDER BY version DESC
            LIMIT 1
            """,
            (record.record_id,),
        ).fetchone()

        if previous_version is None:
            if record.version != 1:
                raise ValueError(f"{record.reference}: the first stored version must be v1")
        else:
            expected_version = int(previous_version["version"]) + 1
            if record.version != expected_version:
                raise ValueError(
                    f"{record.reference}: expected contiguous version v{expected_version}"
                )
            expected_reference = f"{record.record_id}:v{expected_version - 1}"
            if record.supersedes != expected_reference:
                raise ValueError(f"{record.reference}: must supersede {expected_reference}")
            previous_observed = _database_datetime(previous_version["observed_at"])
            if record.observed_at < previous_observed:
                raise ValueError(f"{record.reference}: observed_at precedes its prior version")

        last_event = self._connection.execute(
            "SELECT event_hash FROM knowledge_ledger ORDER BY sequence DESC LIMIT 1"
        ).fetchone()
        previous_event_hash = last_event["event_hash"] if last_event else _GENESIS_HASH
        payload = record.canonical_json()
        event_hash = _event_digest(previous_event_hash, payload)
        try:
            self._connection.execute(
                """
                INSERT INTO knowledge_ledger (
                    record_id,
                    version,
                    valid_from,
                    valid_to,
                    observed_at,
                    payload,
                    previous_event_hash,
                    event_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.record_id,
                    record.version,
                    record.valid_from.isoformat(),
                    record.valid_to.isoformat() if record.valid_to else None,
                    record.observed_at.isoformat().replace("+00:00", "Z"),
                    payload,
                    previous_event_hash,
                    event_hash,
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"Cannot append {record.reference}: {exc}") from exc
        return event_hash

    def _initialize(self) -> None:
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS knowledge_ledger (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                record_id TEXT NOT NULL,
                version INTEGER NOT NULL CHECK (version > 0),
                valid_from TEXT NOT NULL,
                valid_to TEXT,
                observed_at TEXT NOT NULL,
                payload TEXT NOT NULL,
                previous_event_hash TEXT NOT NULL,
                event_hash TEXT NOT NULL UNIQUE,
                UNIQUE (record_id, version)
            );

            CREATE INDEX IF NOT EXISTS knowledge_temporal_lookup
                ON knowledge_ledger (valid_from, valid_to, observed_at);

            CREATE UNIQUE INDEX IF NOT EXISTS knowledge_single_chain_successor
                ON knowledge_ledger (previous_event_hash);

            CREATE TRIGGER IF NOT EXISTS knowledge_ledger_no_update
            BEFORE UPDATE ON knowledge_ledger
            BEGIN
                SELECT RAISE(ABORT, 'knowledge_ledger is append-only');
            END;

            CREATE TRIGGER IF NOT EXISTS knowledge_ledger_no_delete
            BEFORE DELETE ON knowledge_ledger
            BEGIN
                SELECT RAISE(ABORT, 'knowledge_ledger is append-only');
            END;
            """
        )
        self._connection.commit()


def _event_digest(previous_hash: str, payload: str) -> str:
    return hashlib.sha256(f"{previous_hash}\n{payload}".encode()).hexdigest()


def _authority_snapshot_id(as_of: date, records: tuple[KnowledgeRecord, ...]) -> str:
    value = {
        "as_of": as_of.isoformat(),
        "records": [
            {"reference": record.reference, "source_hash": record.source_hash}
            for record in records
        ],
    }
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _database_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)
