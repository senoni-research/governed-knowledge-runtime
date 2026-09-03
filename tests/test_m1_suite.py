from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest
from test_m1_freeze import (
    FIXTURE,
    _complete_scoring_suite,
    _encrypted_descriptor,
    _write_cases,
)

from gkr.m1_authoring import (
    DEDUP_ATTESTATION_BOUNDARY,
    lexical_cross_split_candidates,
    question_set_digest,
)
from gkr.m1_freeze import freeze_m1_suite, repository_root
from gkr.m1_hash import canonical_jsonl_digest, recipient_fingerprint_sha256
from gkr.m1_suite import (
    finalize_m1_scoring_suite,
    load_age_recipient,
    prepare_m1_scoring_suite,
    recipient_fingerprint_with_newline,
)
from gkr.m1_validation import question_digest

FAKE_RECIPIENT = "age1qyqszqgpqyqszqgpqyqszqgpqyqszqgpqyqszqgpqyqszqgpqyqs329t38"


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _dedup_report_for(cases_path: Path, dest: Path) -> Path:
    cases = [
        json.loads(line)
        for line in cases_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    candidates = lexical_cross_split_candidates(cases)
    report = {
        "schema_version": "gkr-m1-semantic-dedup-report-v1",
        "hash_profile_id": "gkr-m1-hash-profile-v1",
        "question_set_sha256": question_set_digest(cases),
        "exact_cross_split_duplicates": 0,
        "semantic_cross_split_candidates_reviewed": True,
        "all_scenarios_reviewed_for_semantic_cross_split_duplication": True,
        "unresolved_semantic_duplicates": 0,
        "candidates": [
            {
                "left_case_id": item["left_case_id"],
                "right_case_id": item["right_case_id"],
                "left_split": item["left_split"],
                "right_split": item["right_split"],
                "disposition": "distinct",
                "finding_codes": [],
            }
            for item in candidates
        ],
        "reviewer_kind": "model",
        "reviewer_identity": "chatgpt",
        "reviewer_session_id": "chatgpt-dedup-session",
        "reviewer_model_family_id": "openai-gpt",
        "reviewer_model_id": "gpt-4.1",
        "reviewer_model_revision": "chatgpt-015",
        "reviewer_prompt_sha256": "cc" * 32,
        "independent_from_retriever_tuning": True,
        "attestation_boundary": DEDUP_ATTESTATION_BOUNDARY,
    }
    return _write_json(dest, report)


def _age_files(tmp_path: Path) -> tuple[Path, Path]:
    recipient = tmp_path / "recipient.txt"
    recipient.write_text(FAKE_RECIPIENT + "\n", encoding="utf-8")
    artifact = tmp_path / "test-sealed.age"
    artifact.write_bytes(b"age-encryption.org/v1\n-> X25519 test\n--- test\n" + b"\x00\x01")
    return recipient, artifact


def _prepare(tmp_path: Path, cases: Path, **kwargs: object) -> dict[str, object]:
    return prepare_m1_scoring_suite(
        cases,
        plaintext_staging_path=tmp_path / "staging.jsonl",
        public_test_candidate_path=tmp_path / "public.jsonl",
        preparation_report_path=tmp_path / "preparation.json",
        **kwargs,
    )


def test_prepare_refuses_incomplete_conformance_and_pending_review(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="conformance|non-scoring"):
        _prepare(tmp_path, FIXTURE)
    assert (tmp_path / "staging.jsonl").exists() is False
    assert (tmp_path / "public.jsonl").exists() is False
    assert (tmp_path / "preparation.json").exists() is False

    incomplete = _complete_scoring_suite(tmp_path / "almost.jsonl")
    rows = [
        json.loads(line)
        for line in incomplete.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    _write_cases(incomplete, rows[:-1])
    with pytest.raises(ValueError, match="expected 120|expected 15"):
        _prepare(tmp_path, incomplete)
    assert (tmp_path / "staging.jsonl").exists() is False


def test_prepare_refuses_pending_review_on_scoring_file(tmp_path: Path) -> None:
    path = _complete_scoring_suite(tmp_path / "pending-review.jsonl")
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    rows[0]["oracle_review"] = {
        "status": "pending",
        "semantically_reviewed": False,
        "reviewer_kind": None,
        "reviewer_identity": None,
        "reviewer_session_id": None,
        "reviewer_model_family_id": None,
        "reviewer_model_id": None,
        "reviewer_model_revision": None,
        "reviewer_prompt_sha256": None,
        "review_sha256": None,
        "independent_from_retriever_tuning": None,
    }
    _write_cases(path, rows)
    with pytest.raises(ValueError, match="pending or incomplete semantic review"):
        _prepare(tmp_path, path)
    assert (tmp_path / "staging.jsonl").exists() is False


def test_prepare_refuses_staging_inside_repository(tmp_path: Path) -> None:
    path = _complete_scoring_suite(tmp_path / "complete.jsonl")
    inside = repository_root() / "evaluation" / "m1" / "must-not-write-prepare-staging.jsonl"
    with pytest.raises(ValueError, match="outside the repository"):
        prepare_m1_scoring_suite(
            path,
            plaintext_staging_path=inside,
            public_test_candidate_path=tmp_path / "public.jsonl",
            preparation_report_path=tmp_path / "preparation.json",
        )
    assert inside.exists() is False
    assert (repository_root() / "evaluation" / "m1" / "suites").exists() is False


def test_prepare_sets_no_suite_status(tmp_path: Path) -> None:
    path = _complete_scoring_suite(tmp_path / "complete.jsonl")
    report = _prepare(tmp_path, path)
    assert "status" not in report
    assert "gate_1_status" not in report
    assert "suite_manifest" not in report
    dumped = json.dumps(report)
    assert "frozen" not in dumped
    assert report["publication_warning"].startswith("This preparation report binds")
    verify = json.loads((tmp_path / "preparation.json").read_text(encoding="utf-8"))
    assert verify["complete_cases_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert verify["plaintext_staging_sha256"] == hashlib.sha256(
        (tmp_path / "staging.jsonl").read_bytes()
    ).hexdigest()
    assert verify["public_test_candidate_sha256"] == hashlib.sha256(
        (tmp_path / "public.jsonl").read_bytes()
    ).hexdigest()


def _finalize_kwargs(tmp_path: Path, cases: Path) -> dict[str, object]:
    recipient, artifact = _age_files(tmp_path)
    return {
        "preparation_report_path": tmp_path / "preparation.json",
        "plaintext_staging_path": tmp_path / "staging.jsonl",
        "public_test_candidate_path": tmp_path / "public.jsonl",
        "semantic_dedup_report_path": _dedup_report_for(cases, tmp_path / "dedup.json"),
        "age_recipient_file": recipient,
        "age_tool_version": "1.2.0",
        "encrypted_artifact_path": artifact,
        "output_dir": tmp_path / "suite",
    }


def test_changed_preparation_bytes_and_commitment_tampering(tmp_path: Path) -> None:
    cases = _complete_scoring_suite(tmp_path / "complete.jsonl")
    _prepare(tmp_path, cases)
    kwargs = _finalize_kwargs(tmp_path, cases)
    staging = tmp_path / "staging.jsonl"
    staging.write_bytes(staging.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="plaintext staging digest"):
        finalize_m1_scoring_suite(cases, **kwargs)
    assert (tmp_path / "suite").exists() is False

    _prepare(tmp_path, cases)
    public = tmp_path / "public.jsonl"
    rows = [json.loads(line) for line in public.read_text(encoding="utf-8").splitlines()]
    rows[0]["question_commitment_sha256"] = "0" * 64
    public.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    report = json.loads((tmp_path / "preparation.json").read_text(encoding="utf-8"))
    report["public_test_candidate_sha256"] = hashlib.sha256(public.read_bytes()).hexdigest()
    (tmp_path / "preparation.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="commitment mismatch"):
        finalize_m1_scoring_suite(cases, **_finalize_kwargs(tmp_path, cases))
    assert (tmp_path / "suite").exists() is False


def test_staging_test_mismatch_is_rejected(tmp_path: Path) -> None:
    cases = _complete_scoring_suite(tmp_path / "complete.jsonl")
    _prepare(tmp_path, cases)
    staging = tmp_path / "staging.jsonl"
    rows = [json.loads(line) for line in staging.read_text(encoding="utf-8").splitlines()]
    rows[0]["case"]["question"] = rows[0]["case"]["question"] + " tampered"
    rows[0]["case"]["question_sha256"] = question_digest(rows[0]["case"]["question"])
    staging.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    report = json.loads((tmp_path / "preparation.json").read_text(encoding="utf-8"))
    report["plaintext_staging_sha256"] = hashlib.sha256(staging.read_bytes()).hexdigest()
    (tmp_path / "preparation.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not match the input"):
        finalize_m1_scoring_suite(cases, **_finalize_kwargs(tmp_path, cases))
    assert (tmp_path / "suite").exists() is False


def test_finalizer_refuses_arbitrary_digest_without_loaded_report(tmp_path: Path) -> None:
    cases = _complete_scoring_suite(tmp_path / "complete.jsonl")
    _prepare(tmp_path, cases)
    kwargs = _finalize_kwargs(tmp_path, cases)
    kwargs["semantic_dedup_report_path"] = _write_json(
        tmp_path / "digest-only.json",
        {"report_sha256": "c" * 64},
    )
    with pytest.raises(ValueError, match="schema_version|semantic dedup report"):
        finalize_m1_scoring_suite(cases, **kwargs)
    assert (tmp_path / "suite").exists() is False


def test_wrong_recipient_preimage_and_ciphertext_refusals(tmp_path: Path) -> None:
    cases = _complete_scoring_suite(tmp_path / "complete.jsonl")
    _prepare(tmp_path, cases)
    kwargs = _finalize_kwargs(tmp_path, cases)
    recipient = load_age_recipient(kwargs["age_recipient_file"])
    assert recipient == FAKE_RECIPIENT
    correct = recipient_fingerprint_sha256(recipient)
    wrong = recipient_fingerprint_with_newline(recipient)
    assert correct != wrong

    missing = deepcopy(kwargs)
    missing["encrypted_artifact_path"] = tmp_path / "missing.age"
    with pytest.raises(ValueError, match="does not exist"):
        finalize_m1_scoring_suite(cases, **missing)

    inside = repository_root() / "evaluation" / "m1" / "must-not-bind.age"
    inside_kwargs = deepcopy(kwargs)
    inside_kwargs["encrypted_artifact_path"] = inside
    with pytest.raises(ValueError, match="outside the repository"):
        finalize_m1_scoring_suite(cases, **inside_kwargs)
    assert inside.exists() is False

    mismatched = tmp_path / "not-age.bin"
    mismatched.write_bytes(b"not-ciphertext")
    bad = deepcopy(kwargs)
    bad["encrypted_artifact_path"] = mismatched
    with pytest.raises(ValueError, match="external .age file|age ciphertext"):
        finalize_m1_scoring_suite(cases, **bad)
    assert (tmp_path / "suite").exists() is False


def test_no_suite_output_on_validation_or_late_publication_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cases = _complete_scoring_suite(tmp_path / "complete.jsonl")
    _prepare(tmp_path, cases)
    kwargs = _finalize_kwargs(tmp_path, cases)

    def boom(manifest: dict[str, object]) -> None:
        raise ValueError("forced late suite validation failure")

    monkeypatch.setattr("gkr.m1_suite._validate_suite_manifest", boom)
    with pytest.raises(ValueError, match="forced late suite validation failure"):
        finalize_m1_scoring_suite(cases, **kwargs)
    assert (tmp_path / "suite").exists() is False
    leftovers = [
        path
        for path in tmp_path.iterdir()
        if path.name.startswith(".suite") or path.suffix == ".tmp"
    ]
    assert leftovers == []


def test_successful_four_file_release_uses_prepared_bytes(tmp_path: Path) -> None:
    cases = _complete_scoring_suite(tmp_path / "complete.jsonl")
    prepared = _prepare(tmp_path, cases)
    kwargs = _finalize_kwargs(tmp_path, cases)
    report = finalize_m1_scoring_suite(cases, **kwargs)
    dest = tmp_path / "suite"
    assert sorted(path.name for path in dest.iterdir()) == [
        "development.jsonl",
        "suite-manifest.json",
        "test-public.jsonl",
        "validation.jsonl",
    ]
    public_bytes = (tmp_path / "public.jsonl").read_bytes()
    assert (dest / "test-public.jsonl").read_bytes() == public_bytes
    assert prepared["public_test_candidate_sha256"] == hashlib.sha256(public_bytes).hexdigest()
    staging_bytes = (tmp_path / "staging.jsonl").read_bytes()
    assert prepared["plaintext_staging_sha256"] == hashlib.sha256(staging_bytes).hexdigest()

    manifest = json.loads((dest / "suite-manifest.json").read_text(encoding="utf-8"))
    assert manifest == report["suite_manifest"]
    assert manifest["status"] == "frozen"
    assert manifest["gate_1_status"] == "passed"
    assert manifest["encryption_boundary"].startswith("The encrypted artifact is produced")
    public_rows = [
        json.loads(line)
        for line in (dest / "test-public.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert manifest["splits"]["test"]["public_cases_sha256"] == canonical_jsonl_digest(public_rows)
    assert report["file_sha256"]["suite-manifest.json"] == hashlib.sha256(
        (dest / "suite-manifest.json").read_bytes()
    ).hexdigest()
    assert report["file_sha256"]["test-public.jsonl"] == hashlib.sha256(public_bytes).hexdigest()

    dumped = (dest / "test-public.jsonl").read_text(encoding="utf-8") + (
        dest / "suite-manifest.json"
    ).read_text(encoding="utf-8")
    for token in (
        "question",
        "oracle",
        "salt_hex",
        "plaintext_staging_sha256",
        str(tmp_path),
        FAKE_RECIPIENT,
        "gkr-sealed",
    ):
        if token == "question":
            assert "question_commitment_sha256" in dumped
            assert '"question"' not in dumped
            continue
        if token == "oracle":
            assert '"oracle"' not in dumped
            continue
        assert token not in dumped

    with pytest.raises(ValueError, match="refuse to overwrite"):
        finalize_m1_scoring_suite(cases, **kwargs)


def test_public_release_excludes_test_plaintext_and_private_paths(tmp_path: Path) -> None:
    cases = _complete_scoring_suite(tmp_path / "complete.jsonl")
    source_cases = [
        json.loads(line) for line in cases.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    test_questions = {case["question"] for case in source_cases if case["split"] == "test"}
    _prepare(tmp_path, cases)
    finalize_m1_scoring_suite(cases, **_finalize_kwargs(tmp_path, cases))
    public_text = (tmp_path / "suite" / "test-public.jsonl").read_text(encoding="utf-8")
    manifest_text = (tmp_path / "suite" / "suite-manifest.json").read_text(encoding="utf-8")
    for question in test_questions:
        assert question not in public_text
        assert question not in manifest_text
    assert "salt_hex" not in public_text
    assert "plaintext_staging_sha256" not in manifest_text
    assert "complete_cases_sha256" not in manifest_text


def test_conformance_freeze_behavior_is_unchanged(tmp_path: Path) -> None:
    report = freeze_m1_suite(
        FIXTURE,
        mode="conformance",
        plaintext_staging_path=tmp_path / "staging.jsonl",
        public_test_path=tmp_path / "public.jsonl",
    )
    assert report["scoring_suite"] is False
    assert report["suite_manifest"] is None
    with pytest.raises(TypeError):
        freeze_m1_suite(
            FIXTURE,
            mode="conformance",
            plaintext_staging_path=tmp_path / "staging.jsonl",
            public_test_path=tmp_path / "public.jsonl",
            retrieval_configuration_sha256="d" * 64,
        )


def test_existing_scoring_freeze_api_still_requires_ciphertext(tmp_path: Path) -> None:
    path = _complete_scoring_suite(tmp_path / "complete.jsonl")
    with pytest.raises(ValueError, match="encrypted artifact descriptor"):
        freeze_m1_suite(
            path,
            mode="scoring",
            plaintext_staging_path=tmp_path / "staging.jsonl",
            public_test_path=tmp_path / "public.jsonl",
            dedup_report_sha256="c" * 64,
        )
    encrypted = _encrypted_descriptor(tmp_path)
    report = freeze_m1_suite(
        path,
        mode="scoring",
        plaintext_staging_path=tmp_path / "staging.jsonl",
        public_test_path=tmp_path / "public.jsonl",
        encrypted_artifact=encrypted,
        dedup_report_sha256="c" * 64,
    )
    assert report["suite_manifest"]["gate_1_status"] == "passed"
