from __future__ import annotations

import json
import os
import shutil
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from gkr.authority import AuthorityStore, AuthorizedCorpus
from gkr.context import ContextCompiler
from gkr.m1_corpus import (
    AS_OF_ANCHOR,
    AUTHORITY_FILENAME,
    DEFAULT_CORPUS_DIR,
    MANIFEST_FILENAME,
    SOURCE_URI_PREFIX,
    build_authority_records,
    load_authority_records,
    publish_corpus_files,
    serialize_authority_jsonl,
    serialize_manifest,
    validate_m1_corpus,
    write_corpus,
)
from gkr.retrieval import RetrievalHit, RetrievalPlan
from gkr.retrieval.router import _record_text
from gkr.schemas import Actor, KnowledgeRecord, source_digest

_MARKER_TOKENS = (
    "fixture_kind",
    "conflict_group",
    "adversarial",
    "future_effective",
    "temporal_correction",
    "acl_transition",
    "adversarial_evidence",
    "retirement",
    "weekend-change-freeze",
    "overtime-premium",
    "vendor-insurance",
    "retention-period",
)
_CONFLICT_PAIRS = {
    "retention-period": ("LEG-RETENTION", "OPS-ARCHIVE"),
    "vendor-insurance": ("LEG-RISK-INS", "PRC-VENDOR-INS"),
    "overtime-premium": ("HR-OVERTIME", "OPS-OVERTIME"),
    "weekend-change-freeze": ("ENG-REL-WINDOW", "ITS-CHANGE-CAB"),
}


def test_committed_corpus_validates_and_ingests() -> None:
    report = validate_m1_corpus()

    assert 40 <= report["stable_record_count"] <= 50
    assert report["authority_event_count"] >= 60
    assert report["ledger_chain_valid"] is True
    assert report["manifest_valid"] is True
    assert report["status"] == "reviewed"
    assert report["status"] != "frozen"
    assert report["relation_count"] >= 12
    assert report["rule_count"] >= 8
    assert report["coverage"]["versioned_records"] >= 8
    assert report["coverage"]["future_effective_records"] >= 3
    assert report["coverage"]["retired_records"] >= 3
    assert report["coverage"]["acl_changed_records"] >= 4
    assert report["coverage"]["conflict_records"] >= 4
    assert report["coverage"]["adversarial_evidence_records"] >= 6
    assert set(report["current_retired_ids"]) >= {
        "FIN-PETTY-CASH",
        "HR-FAX-LEAVE",
        "PRC-PAPER-PO",
    }
    assert set(report["current_restricted_ids"]) >= {
        "FIN-PAYROLL-CAL",
        "SEC-VPN-ACCESS",
        "SEC-INCIDENT-SEV",
        "PRC-VENDOR-SCORE",
        "ITS-ADMIN-RUNBOOK",
        "SEC-BREAK-GLASS",
        "SEC-RADIO-KEYING",
    }


def test_reviewed_manifest_is_not_frozen_and_does_not_pass_gate1() -> None:
    manifest = json.loads((DEFAULT_CORPUS_DIR / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    programme = json.loads(Path("evaluation/m1/programme.json").read_text(encoding="utf-8"))

    assert manifest["status"] == "reviewed"
    assert manifest["status"] != "frozen"
    assert manifest["generation"]["producer"] == "grok"
    assert manifest["generation"]["independent_review"] == "completed"
    assert programme["current_pass_status"]["corpus"] == "not_run"
    assert programme["current_pass_status"]["development"] == "not_run"
    assert programme["current_pass_status"]["validation"] == "not_run"
    assert programme["current_pass_status"]["test"] == "not_run"


def test_builder_matches_committed_jsonl() -> None:
    committed = (DEFAULT_CORPUS_DIR / "authority.jsonl").read_text(encoding="utf-8")
    assert committed == serialize_authority_jsonl(build_authority_records())


def test_every_record_is_statement_scoped_and_synthetic() -> None:
    records = build_authority_records()
    statuses = {record.status for record in records}
    sensitivities = {record.sensitivity for record in records}

    assert statuses == {"approved", "retired"}
    assert {"public", "internal", "restricted", "secret"} <= sensitivities
    for record in records:
        assert record.source_uri.startswith(SOURCE_URI_PREFIX)
        assert record.source_hash == source_digest(record.statement)
        assert record.metadata["hash_scope"] == "statement"
        assert "question" not in record.metadata
        assert "expected_answer" not in record.metadata
        assert "oracle" not in record.metadata


def test_restriction_and_retirement_do_not_fall_back(tmp_path: Path) -> None:
    store = AuthorityStore(tmp_path / "authority.sqlite")
    store.import_jsonl(DEFAULT_CORPUS_DIR / "authority.jsonl")
    employee = Actor("alice", ("employees",))
    payroll = Actor("sam", ("payroll-officers",))
    known_at = datetime(2026, 9, 2, 12, 0, tzinfo=UTC)

    employee_now = store.current_records(actor=employee, as_of=AS_OF_ANCHOR, known_at=known_at)
    payroll_now = store.current_records(actor=payroll, as_of=AS_OF_ANCHOR, known_at=known_at)
    employee_ids = {record.record_id for record in employee_now.records}
    payroll_ids = {record.record_id for record in payroll_now.records}

    assert "FIN-PAYROLL-CAL" not in employee_ids
    assert "FIN-PETTY-CASH" not in employee_ids
    assert "FIN-PAYROLL-CAL" in payroll_ids
    assert all(record.reference != "FIN-PAYROLL-CAL:v1" for record in employee_now.records)
    assert store.verify_chain()
    store.close()


def test_temporal_correction_depends_on_known_at(tmp_path: Path) -> None:
    store = AuthorityStore(tmp_path / "authority.sqlite")
    store.import_jsonl(DEFAULT_CORPUS_DIR / "authority.jsonl")
    actor = Actor("alice", ("employees",))

    before_known = store.current_records(
        actor=actor,
        as_of=date(2026, 4, 1),
        known_at=datetime(2026, 6, 1, tzinfo=UTC),
    )
    after_known = store.current_records(
        actor=actor,
        as_of=date(2026, 4, 1),
        known_at=datetime(2026, 8, 1, tzinfo=UTC),
    )
    before_effect = store.current_records(
        actor=actor,
        as_of=date(2026, 2, 1),
        known_at=datetime(2026, 8, 1, tzinfo=UTC),
    )

    assert _reference(before_known.records, "FIN-MILEAGE") == "FIN-MILEAGE:v1"
    assert _reference(after_known.records, "FIN-MILEAGE") == "FIN-MILEAGE:v2"
    assert _reference(before_effect.records, "FIN-MILEAGE") == "FIN-MILEAGE:v1"
    store.close()


def test_future_effective_version_is_hidden_at_anchor(tmp_path: Path) -> None:
    store = AuthorityStore(tmp_path / "authority.sqlite")
    store.import_jsonl(DEFAULT_CORPUS_DIR / "authority.jsonl")
    finance = Actor("renee", ("finance",))

    current = store.current_records(
        actor=finance,
        as_of=AS_OF_ANCHOR,
        known_at=datetime(2026, 9, 2, 12, 0, tzinfo=UTC),
    )
    later = store.current_records(
        actor=finance,
        as_of=date(2027, 1, 2),
        known_at=datetime(2027, 1, 2, tzinfo=UTC),
    )

    assert _reference(current.records, "FIN-CARD-LIMIT") == "FIN-CARD-LIMIT:v1"
    assert _reference(later.records, "FIN-CARD-LIMIT") == "FIN-CARD-LIMIT:v2"
    store.close()


def test_winter_roster_predecessor_is_current_at_anchor(tmp_path: Path) -> None:
    store = AuthorityStore(tmp_path / "authority.sqlite")
    store.import_jsonl(DEFAULT_CORPUS_DIR / "authority.jsonl")
    actor = Actor("alice", ("employees",))
    known_at = datetime(2026, 9, 2, 12, 0, tzinfo=UTC)

    at_anchor = store.current_records(actor=actor, as_of=AS_OF_ANCHOR, known_at=known_at)
    after_winter = store.current_records(
        actor=actor,
        as_of=date(2026, 11, 16),
        known_at=datetime(2026, 11, 16, tzinfo=UTC),
    )

    assert _reference(at_anchor.records, "OPS-WINTER-ROSTER") == "OPS-WINTER-ROSTER:v1"
    assert _reference(after_winter.records, "OPS-WINTER-ROSTER") == "OPS-WINTER-ROSTER:v2"
    store.close()


def test_retired_statements_contain_no_replacement_policy() -> None:
    records = {
        record.reference: record
        for record in build_authority_records()
        if record.reference
        in {"FIN-PETTY-CASH:v2", "HR-FAX-LEAVE:v2", "PRC-PAPER-PO:v2"}
    }
    assert "purchasing card" not in records["FIN-PETTY-CASH:v2"].statement.lower()
    assert "portal" not in records["HR-FAX-LEAVE:v2"].statement.lower()
    assert "digital" not in records["PRC-PAPER-PO:v2"].statement.lower()
    for record in records.values():
        assert record.status == "retired"
        assert "withdrawn" in record.statement.lower() or "retired" in record.statement.lower()


def test_four_conflict_pairs_are_exact_and_unlabelled() -> None:
    records = {record.reference: record for record in build_authority_records()}
    groups = {
        record.metadata["conflict_group"]
        for record in records.values()
        if record.metadata.get("conflict_group")
    }
    assert groups == set(_CONFLICT_PAIRS)
    assert records["ITS-CHANGE-CAB:v1"].metadata.get("conflict_group") is None
    assert records["ITS-CHANGE-CAB:v2"].metadata["conflict_group"] == "weekend-change-freeze"

    for record in records.values():
        lowered = f"{record.statement} {' '.join(record.aliases)}".lower()
        assert "approved conflict" not in lowered
        assert "not aligned" not in lowered

    legal = records["LEG-RETENTION:v2"].statement.lower()
    ops = records["OPS-ARCHIVE:v1"].statement.lower()
    assert "closed quayside operational files" in legal
    assert "seven years" in legal
    assert "closed quayside operational files" in ops
    assert "three years" in ops

    counsel = records["LEG-RISK-INS:v1"].statement.lower()
    procurement = records["PRC-VENDOR-INS:v1"].statement.lower()
    assert "airside quay" in counsel and "2,000,000" in counsel
    assert "airside quay" in procurement and "5,000,000" in procurement
    assert "gate pass" in counsel and "gate pass" in procurement

    people = records["HR-OVERTIME:v1"].statement.lower()
    harbor = records["OPS-OVERTIME:v1"].statement.lower()
    assert "quayside rostered" in people and "1.5" in people
    assert "quayside rostered" in harbor and "1.25" in harbor
    assert "unless" not in people

    freeze = records["ENG-REL-WINDOW:v2"].statement.lower()
    sunday = records["ITS-CHANGE-CAB:v2"].statement.lower()
    assert "including emergency slots" in freeze
    assert "does not authorize a production change" in freeze
    assert "sunday emergency production changes" in sunday
    assert "security-ops acknowledgement" in sunday
    assert "group:security-ops" in records["ITS-CHANGE-CAB:v2"].acl


def test_structured_rules_match_prose() -> None:
    records = {record.reference: record for record in build_authority_records()}
    exception_free = (
        "FIN-CARD-LIMIT:v1",
        "FIN-CARD-LIMIT:v2",
        "FIN-USD-WIRE:v1",
        "HR-OVERTIME:v1",
    )
    for reference in exception_free:
        for rule in records[reference].rules:
            assert rule.exceptions == ()

    assert records["FIN-MILEAGE:v1"].rules[0].measure == "reimbursement-rate-per-mile"
    assert records["FIN-MILEAGE:v2"].rules[0].measure == "reimbursement-rate-per-mile"
    assert "reaches 30 minutes" in records["SEC-VPN-ACCESS:v1"].statement
    assert "reaches 30 minutes" in records["SEC-VPN-ACCESS:v2"].statement
    assert "after generated lockfiles are excluded" in records["ENG-CHANGE-SIZE:v1"].statement
    assert "after generated lockfiles are excluded" in records["ENG-CHANGE-SIZE:v2"].statement
    assert records["ENG-CHANGE-SIZE:v1"].rules[0].measure == "touched-lines"


def test_fixture_markers_are_absent_from_search_text_and_prompts() -> None:
    records = build_authority_records()
    searchable = "\n".join(_record_text(record) for record in records).casefold()
    for token in _MARKER_TOKENS:
        assert token not in searchable

    actor = Actor("reviewer", ("employees", "finance", "security-ops", "payroll-officers"))
    corpus = AuthorizedCorpus(
        actor=actor,
        as_of=AS_OF_ANCHOR,
        known_at=datetime(2026, 9, 2, 12, 0, tzinfo=UTC),
        records=records,
        authority_snapshot_id="review-snapshot",
    )
    plan = RetrievalPlan(
        mode="full_context",
        retriever_id="test-all-records",
        configuration=(),
        hits=tuple(RetrievalHit(record=record, score=None) for record in records),
        available_records=len(records),
        estimated_evidence_tokens=0,
        reason="compile every candidate event",
    )
    prompt = ContextCompiler().compile(
        question="What rules apply?",
        corpus=corpus,
        plan=plan,
    ).prompt.casefold()
    for token in _MARKER_TOKENS:
        assert token not in prompt


def test_load_authority_records_includes_file_and_line(tmp_path: Path) -> None:
    template = json.loads((DEFAULT_CORPUS_DIR / AUTHORITY_FILENAME).read_text().splitlines()[0])
    cases = (
        ({"status": "proposed"}, "Unsupported status"),
        ({"acl": ["employees"]}, "typed principals"),
        ({"record_id": "ENG-CHANGE-SIZE", "version": 2, "supersedes": "ENG-CHANGE-SIZE:v99"},
         "earlier version"),
    )
    for index, (patch, fragment) in enumerate(cases, start=1):
        path = tmp_path / f"bad-{index}.jsonl"
        value = dict(template)
        value.update(patch)
        path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match=rf"{path}:1: .*{fragment}") as captured:
            load_authority_records(path)
        assert captured.value.__cause__ is not None


def test_missing_corpus_files_raise_value_error(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="missing required file") as missing_both:
        validate_m1_corpus(empty)
    assert AUTHORITY_FILENAME in str(missing_both.value)
    assert MANIFEST_FILENAME in str(missing_both.value)

    partial = tmp_path / "partial"
    partial.mkdir()
    shutil.copy(DEFAULT_CORPUS_DIR / AUTHORITY_FILENAME, partial / AUTHORITY_FILENAME)
    with pytest.raises(ValueError, match="missing required file") as missing_manifest:
        validate_m1_corpus(partial)
    assert MANIFEST_FILENAME in str(missing_manifest.value)
    assert isinstance(missing_manifest.value, ValueError)


def test_write_corpus_replaces_complete_files(tmp_path: Path) -> None:
    manifest = write_corpus(tmp_path)
    leftovers = list(tmp_path.glob(".*tmp*")) + list(tmp_path.glob("*.tmp"))
    assert leftovers == []
    assert (tmp_path / AUTHORITY_FILENAME).is_file()
    assert (tmp_path / MANIFEST_FILENAME).is_file()
    assert manifest["status"] == "reviewed"
    assert manifest["status"] != "frozen"
    assert manifest["generation"]["independent_review"] == "completed"
    report = validate_m1_corpus(tmp_path)
    assert report["authority_event_count"] == manifest["authority_event_count"]


def test_publish_cleans_up_when_first_replace_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_corpus(tmp_path)
    original_authority = (tmp_path / AUTHORITY_FILENAME).read_bytes()
    original_manifest = (tmp_path / MANIFEST_FILENAME).read_bytes()

    def boom(_source: str | os.PathLike[str], _dest: str | os.PathLike[str]) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError, match="simulated replace failure"):
        write_corpus(tmp_path)

    assert (tmp_path / AUTHORITY_FILENAME).read_bytes() == original_authority
    assert (tmp_path / MANIFEST_FILENAME).read_bytes() == original_manifest
    leftovers = [
        path for path in tmp_path.iterdir() if path.suffix == ".tmp" or ".tmp" in path.name
    ]
    assert leftovers == []


def test_publish_rejects_pair_after_second_replace_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publish_corpus_files(
        tmp_path,
        authority_text='{"record_id":"OLD"}\n',
        manifest_text='{"stale": true}\n',
    )
    calls = {"count": 0}
    real_replace = os.replace

    def flaky(source: str | os.PathLike[str], dest: str | os.PathLike[str]) -> None:
        calls["count"] += 1
        if calls["count"] == 1:
            real_replace(source, dest)
            return
        raise OSError("simulated second replace failure")

    monkeypatch.setattr(os, "replace", flaky)
    with pytest.raises(OSError, match="simulated second replace failure"):
        publish_corpus_files(
            tmp_path,
            authority_text=serialize_authority_jsonl(build_authority_records()),
            manifest_text=serialize_manifest({"fresh": True}),
        )

    assert (tmp_path / AUTHORITY_FILENAME).read_text(encoding="utf-8") != '{"record_id":"OLD"}\n'
    assert (tmp_path / MANIFEST_FILENAME).read_text(encoding="utf-8") == '{"stale": true}\n'
    leftovers = [
        path for path in tmp_path.iterdir() if path.suffix == ".tmp" or ".tmp" in path.name
    ]
    assert leftovers == []
    with pytest.raises(ValueError):
        validate_m1_corpus(tmp_path)


def _copy_lines(corpus: Path) -> list[str]:
    return (corpus / AUTHORITY_FILENAME).read_text(encoding="utf-8").splitlines()


def _write_lines(corpus: Path, lines: list[str]) -> None:
    (corpus / AUTHORITY_FILENAME).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _mutate_statement_keep_hash(corpus: Path) -> None:
    lines = _copy_lines(corpus)
    value = json.loads(lines[0])
    value["statement"] = value["statement"] + " MUTATED"
    lines[0] = json.dumps(value, ensure_ascii=False)
    _write_lines(corpus, lines)


def _mutate_manifest_count(corpus: Path) -> None:
    manifest = json.loads((corpus / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    manifest["authority_event_count"] = int(manifest["authority_event_count"]) - 1
    (corpus / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _mutate_file_digest(corpus: Path) -> None:
    manifest = json.loads((corpus / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    manifest["authority_jsonl_sha256"] = "0" * 64
    (corpus / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _mutate_order(corpus: Path) -> None:
    lines = _copy_lines(corpus)
    lines[0], lines[1] = lines[1], lines[0]
    _write_lines(corpus, lines)


def _mutate_skip_version(corpus: Path) -> None:
    kept = []
    for line in _copy_lines(corpus):
        value = json.loads(line)
        if value["record_id"] == "ENG-CHANGE-SIZE" and value["version"] == 1:
            continue
        kept.append(line)
    _write_lines(corpus, kept)


def _mutate_supersedes(corpus: Path) -> None:
    lines = _copy_lines(corpus)
    for index, line in enumerate(lines):
        value = json.loads(line)
        if value["record_id"] == "ENG-CHANGE-SIZE" and value["version"] == 2:
            value["supersedes"] = "ENG-CHANGE-SIZE:v99"
            lines[index] = json.dumps(value, ensure_ascii=False)
    _write_lines(corpus, lines)


def _mutate_workflow_status(corpus: Path) -> None:
    lines = _copy_lines(corpus)
    value = json.loads(lines[0])
    value["status"] = "proposed"
    lines[0] = json.dumps(value, ensure_ascii=False)
    _write_lines(corpus, lines)


def _mutate_source_uri(corpus: Path) -> None:
    lines = _copy_lines(corpus)
    value = json.loads(lines[0])
    value["source_uri"] = "https://example.com/real-policy"
    lines[0] = json.dumps(value, ensure_ascii=False)
    _write_lines(corpus, lines)


def _reference(records: tuple[KnowledgeRecord, ...], record_id: str) -> str | None:
    for record in records:
        if record.record_id == record_id:
            return record.reference
    return None


@pytest.mark.parametrize(
    ("mutator", "fragment", "expect_line"),
    [
        (_mutate_statement_keep_hash, "source_hash does not match", False),
        (_mutate_manifest_count, "does not match recomputed hashes/counts", False),
        (_mutate_file_digest, "does not match recomputed hashes/counts", False),
        (_mutate_order, "ordered by record_id", False),
        (_mutate_skip_version, "first event must be v1", False),
        (_mutate_supersedes, "earlier version", True),
        (_mutate_workflow_status, "Unsupported status", True),
        (_mutate_source_uri, "synthetic://m1/", False),
    ],
)
def test_copied_corpus_mutations_fail_closed(
    tmp_path: Path,
    mutator: object,
    fragment: str,
    expect_line: bool,
) -> None:
    corpus = tmp_path / "corpus"
    shutil.copytree(DEFAULT_CORPUS_DIR, corpus)
    mutator(corpus)
    with pytest.raises(ValueError, match=fragment) as captured:
        validate_m1_corpus(corpus)
    if expect_line:
        message = str(captured.value)
        assert f"{AUTHORITY_FILENAME}:" in message
        line_token = message.split(f"{AUTHORITY_FILENAME}:", 1)[1].split(":", 1)[0]
        assert line_token.isdigit()
