from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from conftest import make_record

from gkr.authority import AuthorityStore
from gkr.schemas import KnowledgeRecord
from gkr.source import SourceArtifact


def test_inline_statement_hash_is_recomputed_by_importer(
    store: AuthorityStore,
    tmp_path: Path,
) -> None:
    value = make_record().to_dict()
    value["source_hash"] = "0" * 64
    source = tmp_path / "records.jsonl"
    source.write_text(json.dumps(value) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="does not match the canonical statement"):
        store.import_jsonl(source)

    assert store.count() == 0


def test_raw_source_bytes_are_computed_and_verified_locally(
    store: AuthorityStore,
    tmp_path: Path,
) -> None:
    source_file = tmp_path / "policy.txt"
    source_file.write_text("Approved source bytes\n", encoding="utf-8")
    artifact = SourceArtifact.from_file(
        source_file,
        source_uri="file://policy.txt",
        source_version="git:abc123",
        content_type="text/plain",
    )
    value = make_record().to_dict()
    value.update(
        {
            "source_uri": artifact.source_uri,
            "source_hash": artifact.raw_sha256,
            "metadata": {
                "hash_scope": "raw_bytes",
                "source_version": artifact.source_version,
                "content_type": artifact.content_type,
                "raw_byte_length": artifact.raw_byte_length,
                "extracted_text_sha256": None,
            },
        }
    )
    record = KnowledgeRecord.from_dict(value)

    with pytest.raises(ValueError, match="requires a SourceArtifact"):
        store.append(record)

    store.append(record, source_artifact=artifact)
    assert store.count() == 1


def test_raw_source_hash_mismatch_rejects_transaction(
    store: AuthorityStore,
    tmp_path: Path,
) -> None:
    source_file = tmp_path / "policy.txt"
    source_file.write_text("Actual source\n", encoding="utf-8")
    artifact = SourceArtifact.from_file(
        source_file,
        source_uri="file://policy.txt",
        source_version="v1",
        content_type="text/plain",
    )
    value = make_record().to_dict()
    value.update(
        {
            "source_uri": artifact.source_uri,
            "source_hash": "f" * 64,
            "metadata": {
                "hash_scope": "raw_bytes",
                "source_version": artifact.source_version,
                "content_type": artifact.content_type,
                "raw_byte_length": artifact.raw_byte_length,
                "extracted_text_sha256": None,
            },
        }
    )

    with pytest.raises(ValueError, match="does not match source bytes"):
        store.append(KnowledgeRecord.from_dict(value), source_artifact=artifact)

    assert store.count() == 0


def test_raw_source_envelope_is_verified_during_jsonl_import(
    store: AuthorityStore,
    tmp_path: Path,
) -> None:
    source_file = tmp_path / "policy.txt"
    source_file.write_text("Authority source\n", encoding="utf-8")
    artifact = SourceArtifact.from_file(
        source_file,
        source_uri="file://policy.txt",
        source_version="v7",
        content_type="text/plain",
    )
    record = make_record().to_dict()
    record.update(
        {
            "source_uri": artifact.source_uri,
            "source_hash": artifact.raw_sha256,
            "metadata": {
                "hash_scope": "raw_bytes",
                "source_version": artifact.source_version,
                "content_type": artifact.content_type,
                "raw_byte_length": artifact.raw_byte_length,
                "extracted_text_sha256": None,
            },
        }
    )
    envelope = {
        "record": record,
        "source_artifact": {
            "path": source_file.name,
            "source_uri": artifact.source_uri,
            "source_version": artifact.source_version,
            "content_type": artifact.content_type,
        },
    }
    manifest = tmp_path / "ingest.jsonl"
    manifest.write_text(json.dumps(envelope) + "\n", encoding="utf-8")

    store.import_jsonl(manifest)

    assert store.count() == 1
    assert store.verify_chain()


def test_jsonl_source_artifact_cannot_escape_import_directory(
    store: AuthorityStore,
    tmp_path: Path,
) -> None:
    import_dir = tmp_path / "import"
    import_dir.mkdir()
    outside = tmp_path / "outside-source.txt"
    outside.write_text("should not be readable via import\n", encoding="utf-8")
    record = make_record().to_dict()
    record.update(
        {
            "source_uri": "file://outside-source.txt",
            "source_hash": "a" * 64,
            "metadata": {
                "hash_scope": "raw_bytes",
                "source_version": "v1",
                "content_type": "text/plain",
                "raw_byte_length": 1,
                "extracted_text_sha256": None,
            },
        }
    )
    envelope = {
        "record": record,
        "source_artifact": {
            "path": "../outside-source.txt",
            "source_uri": "file://outside-source.txt",
            "source_version": "v1",
            "content_type": "text/plain",
        },
    }
    manifest = import_dir / "ingest.jsonl"
    manifest.write_text(json.dumps(envelope) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="stay inside the import directory"):
        store.import_jsonl(manifest)

    envelope["source_artifact"]["path"] = str(outside)
    manifest.write_text(json.dumps(envelope) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be relative"):
        store.import_jsonl(manifest)

    assert store.count() == 0


def test_source_changed_after_digest_is_rejected(
    store: AuthorityStore,
    tmp_path: Path,
) -> None:
    source_file = tmp_path / "policy.txt"
    source_file.write_text("Original authority source\n", encoding="utf-8")
    artifact = SourceArtifact.from_file(
        source_file,
        source_uri="file://policy.txt",
        source_version="v1",
        content_type="text/plain",
    )
    record = _record_for_artifact(artifact)
    source_file.write_text("Changed authority source\n", encoding="utf-8")

    with pytest.raises(ValueError, match="changed after digest computation"):
        store.append(record, source_artifact=artifact)

    assert store.count() == 0


def test_invalid_source_version_is_rejected(tmp_path: Path) -> None:
    source_file = tmp_path / "policy.txt"
    source_file.write_text("Authority source\n", encoding="utf-8")

    with pytest.raises(ValueError, match="source_version contains invalid"):
        SourceArtifact.from_file(
            source_file,
            source_uri="file://policy.txt",
            source_version="v1\nforged",
            content_type="text/plain",
        )


def test_chain_verification_compares_index_columns_to_hashed_payload(
    store: AuthorityStore,
) -> None:
    store.append(make_record())
    store._connection.execute(  # noqa: SLF001 - simulates a broken migration below SQL guards
        "DROP TRIGGER knowledge_ledger_no_update"
    )
    store._connection.execute(  # noqa: SLF001
        "UPDATE knowledge_ledger SET valid_from = '2030-01-01'"
    )

    assert not store.verify_chain()


def test_chain_verification_detects_hashed_payload_tampering(
    store: AuthorityStore,
) -> None:
    store.append(make_record())
    store._connection.execute(  # noqa: SLF001
        "DROP TRIGGER knowledge_ledger_no_update"
    )
    row = store._connection.execute(  # noqa: SLF001
        "SELECT payload FROM knowledge_ledger WHERE sequence = 1"
    ).fetchone()
    value = json.loads(row["payload"])
    value["statement"] = "Tampered authority statement."
    tampered_payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    store._connection.execute(  # noqa: SLF001
        "UPDATE knowledge_ledger SET payload = ? WHERE sequence = 1",
        (tampered_payload,),
    )

    assert not store.verify_chain()


def test_chain_verification_detects_replaced_earlier_event(
    store: AuthorityStore,
) -> None:
    store.append_many(
        (
            make_record(record_id="POL-001"),
            make_record(record_id="POL-002"),
        )
    )
    store._connection.execute(  # noqa: SLF001
        "DROP TRIGGER knowledge_ledger_no_update"
    )
    row = store._connection.execute(  # noqa: SLF001
        "SELECT payload FROM knowledge_ledger WHERE sequence = 1"
    ).fetchone()
    value = json.loads(row["payload"])
    value["title"] = "Replacement event"
    replacement = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    replacement_hash = hashlib.sha256(f"{'0' * 64}\n{replacement}".encode()).hexdigest()
    store._connection.execute(  # noqa: SLF001
        "UPDATE knowledge_ledger SET payload = ?, event_hash = ? WHERE sequence = 1",
        (replacement, replacement_hash),
    )

    assert not store.verify_chain()


def test_duplicate_authority_event_is_rejected(store: AuthorityStore) -> None:
    record = make_record()
    store.append(record)

    with pytest.raises(ValueError, match="expected contiguous version"):
        store.append(record)

    assert store.count() == 1
    assert store.verify_chain()


def _record_for_artifact(artifact: SourceArtifact) -> KnowledgeRecord:
    value = make_record().to_dict()
    value.update(
        {
            "source_uri": artifact.source_uri,
            "source_hash": artifact.raw_sha256,
            "metadata": {
                "hash_scope": "raw_bytes",
                "source_version": artifact.source_version,
                "content_type": artifact.content_type,
                "raw_byte_length": artifact.raw_byte_length,
                "extracted_text_sha256": artifact.extracted_text_sha256,
            },
        }
    )
    return KnowledgeRecord.from_dict(value)
