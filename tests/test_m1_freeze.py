from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from gkr import m1_freeze
from gkr.m1_freeze import (
    STAGING_RECORD_VERSION,
    freeze_m1_suite,
    question_commitment,
    repository_root,
    verify_question_commitments,
    verify_staging_bundle_digest,
)
from gkr.m1_oracle_validation import validate_m1_oracles
from gkr.m1_validation import question_digest

FIXTURE = Path("tests/fixtures/m1/conformance-cases.jsonl")
QUERY_CLASSES = (
    "exact_factual",
    "semantic_paraphrase",
    "numeric_conditional",
    "temporal",
    "authorization",
    "unknown_oos",
    "multi_record",
    "adversarial_conflicting",
)


def _cases() -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in FIXTURE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_cases(path: Path, cases: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases),
        encoding="utf-8",
    )


def _freeze(tmp_path: Path, case_path: Path = FIXTURE, **kwargs: object) -> dict[str, object]:
    return freeze_m1_suite(
        case_path,
        mode="conformance",
        plaintext_staging_path=tmp_path / "staging.jsonl",
        public_test_path=tmp_path / "public.jsonl",
        **kwargs,
    )


def _scoring_roles() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    question = {
        "method": "model",
        "session_id": "grok-question-session",
        "identity": "grok",
        "model_family_id": "xai-grok",
        "model_id": "grok-4",
        "model_revision": "grok-high-fast-014",
        "prompt_retained": True,
        "prompt_sha256": "5c75c3fe8f948e6404086214965f305cbe176398380e8f32b5e4cbf0c067e21d",
        "independent_from_retriever_tuning": True,
    }
    oracle = {**question, "session_id": "grok-oracle-session"}
    review = {
        "status": "completed",
        "semantically_reviewed": True,
        "reviewer_kind": "model",
        "reviewer_identity": "chatgpt",
        "reviewer_session_id": "chatgpt-review-session",
        "reviewer_model_family_id": "openai-gpt",
        "reviewer_model_id": "gpt-4.1",
        "reviewer_model_revision": "chatgpt-015",
        "reviewer_prompt_sha256": "aa" * 32,
        "review_sha256": "bb" * 32,
        "independent_from_retriever_tuning": True,
    }
    return question, oracle, review


def _templates() -> dict[str, dict[str, object]]:
    chosen: dict[str, dict[str, object]] = {}
    for case in _cases():
        query_class = str(case["query_class"])
        if query_class not in chosen:
            chosen[query_class] = case
        if query_class == "authorization" and case["oracle"]["publication"] == "published":
            chosen[query_class] = case
        if query_class == "multi_record" and case["case_id"] == "m1-cf-release-planning-01":
            chosen[query_class] = case
    return chosen


def _authorization_denied() -> dict[str, object]:
    return next(
        case
        for case in _cases()
        if case["query_class"] == "authorization" and case["oracle"]["publication"] == "refused"
    )


def _complete_scoring_suite(
    path: Path,
    *,
    same_sessions: bool = False,
    reviewer_family: str = "openai-gpt",
    independent: bool = True,
    authorization_denied: bool = True,
) -> Path:
    templates = _templates()
    denied = _authorization_denied()
    question, oracle, review = _scoring_roles()
    if same_sessions:
        oracle["session_id"] = question["session_id"]
        review["reviewer_session_id"] = question["session_id"]
    review["reviewer_model_family_id"] = reviewer_family
    if reviewer_family == "xai-grok":
        review["reviewer_model_id"] = "grok-4"
    elif reviewer_family == "openai-gpt":
        review["reviewer_model_id"] = "gpt-4.1"
    question["independent_from_retriever_tuning"] = independent
    oracle["independent_from_retriever_tuning"] = independent
    review["independent_from_retriever_tuning"] = independent
    rows: list[dict[str, object]] = []
    for split in ("development", "validation", "test"):
        for query_class in QUERY_CLASSES:
            for index in range(1, 16):
                if query_class == "authorization" and authorization_denied and index > 8:
                    template = denied
                else:
                    template = templates[query_class]
                case = deepcopy(template)
                case_id = f"m1-sc-{split}-{query_class.replace('_', '-')}-{index:02d}"
                question_text = f"{template['question']} ({split} {query_class} {index:02d})"
                case["case_id"] = case_id
                case["scenario_id"] = case_id
                case["split"] = split
                case["case_kind"] = "scoring"
                case["question"] = question_text
                case["question_sha256"] = question_digest(question_text)
                case["question_authorship"] = deepcopy(question)
                case["oracle_authorship"] = deepcopy(oracle)
                case["oracle_review"] = deepcopy(review)
                case["tags"] = ["conformance-fixture"]
                rows.append(case)
    _write_cases(path, rows)
    return path


def _encrypted_descriptor(tmp_path: Path) -> dict[str, str]:
    artifact = tmp_path / "encrypted.bin"
    artifact.write_bytes(b"external-ciphertext-placeholder")
    return {
        "encryption_profile_id": "age-x25519-v1",
        "tool_family": "age",
        "tool_version": "1.2.0",
        "recipient_key_fingerprint_sha256": "a" * 64,
        "encrypted_artifact_path": str(artifact),
        "encrypted_artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
    }


def test_conformance_fixture_freeze_round_trip(tmp_path: Path) -> None:
    report = _freeze(tmp_path)

    assert report["conformance_fixture"] is True
    assert report["scoring_suite"] is False
    assert report["externally_encrypted_artifact_bound"] is False
    assert report["suite_manifest"] is None
    assert report["semantic_support_established"] is False
    assert report["splits"]["development"]["independent_scenario_count"] == 8
    assert report["splits"]["validation"]["independent_scenario_count"] == 4
    assert report["splits"]["test"]["independent_scenario_count"] == 4
    assert report["splits"]["test"]["externally_encrypted_artifact_bound"] is False
    assert report["splits"]["test"]["case_count"] == 4

    staging = tmp_path / "staging.jsonl"
    public = tmp_path / "public.jsonl"
    verify_staging_bundle_digest(staging, str(report["plaintext_staging_sha256"]))
    round_trip = verify_question_commitments(
        public_test_path=public,
        plaintext_staging_path=staging,
    )
    assert round_trip["commitments_match"] is True
    assert round_trip["externally_encrypted_artifact_bound"] is False
    assert round_trip["cases"] == 4

    public_rows = [json.loads(line) for line in public.read_text(encoding="utf-8").splitlines()]
    assert all(row["schema_version"] == "gkr-m1-test-public-case-v3" for row in public_rows)
    assert all(row["hash_profile_id"] == "gkr-m1-hash-profile-v1" for row in public_rows)
    assert all("question" not in row for row in public_rows)
    assert all("oracle" not in row for row in public_rows)
    assert all(row["split"] == "test" for row in public_rows)

    bundle_rows = [json.loads(line) for line in staging.read_text(encoding="utf-8").splitlines()]
    for row in bundle_rows:
        assert row["schema_version"] == STAGING_RECORD_VERSION
        salt = bytes.fromhex(row["salt_hex"])
        expected = question_commitment(salt, row["case"]["question"])
        public_row = next(item for item in public_rows if item["case_id"] == row["case"]["case_id"])
        assert public_row["question_commitment_sha256"] == expected


def test_plaintext_staging_refused_inside_repository(tmp_path: Path) -> None:
    inside = repository_root() / "evaluation" / "m1" / "must-not-write-staging.jsonl"
    with pytest.raises(ValueError, match="outside the repository"):
        freeze_m1_suite(
            FIXTURE,
            mode="conformance",
            plaintext_staging_path=inside,
            public_test_path=tmp_path / "public.jsonl",
        )
    assert inside.exists() is False


def test_conformance_tag_cannot_select_mode(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="mode must be"):
        freeze_m1_suite(
            FIXTURE,
            mode="not-a-mode",
            plaintext_staging_path=tmp_path / "staging.jsonl",
            public_test_path=tmp_path / "public.jsonl",
        )


def test_commitment_mismatch_on_temp_public_file(tmp_path: Path) -> None:
    _freeze(tmp_path)
    public = tmp_path / "public.jsonl"
    rows = [json.loads(line) for line in public.read_text(encoding="utf-8").splitlines()]
    rows[0]["question_commitment_sha256"] = "0" * 64
    public.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="commitment mismatch"):
        verify_question_commitments(
            public_test_path=public,
            plaintext_staging_path=tmp_path / "staging.jsonl",
        )


def test_tampered_public_file_fails_round_trip(tmp_path: Path) -> None:
    _freeze(tmp_path)
    public = tmp_path / "public.jsonl"
    rows = [json.loads(line) for line in public.read_text(encoding="utf-8").splitlines()]
    rows[0]["case_id"] = "tampered-case"
    public.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match the plaintext staging bundle"):
        verify_question_commitments(
            public_test_path=public,
            plaintext_staging_path=tmp_path / "staging.jsonl",
        )


def test_tampered_staging_bundle_digest(tmp_path: Path) -> None:
    report = _freeze(tmp_path)
    staging = tmp_path / "staging.jsonl"
    staging.write_bytes(staging.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="plaintext staging digest"):
        verify_staging_bundle_digest(staging, str(report["plaintext_staging_sha256"]))


def test_freeze_refuses_validation_failure(tmp_path: Path) -> None:
    cases = _cases()
    cases[0]["oracle"]["sufficient_reference_sets"] = [["NO-SUCH-RECORD:v1"]]
    path = tmp_path / "bad.jsonl"
    _write_cases(path, cases)
    with pytest.raises(ValueError, match="does not exist in the corpus"):
        _freeze(tmp_path, path)
    assert (tmp_path / "staging.jsonl").exists() is False
    assert (tmp_path / "public.jsonl").exists() is False


def test_late_public_validation_failure_leaves_no_new_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = m1_freeze._validate_public_cases

    def wrapped(cases: list[dict[str, object]], *, source: str | Path) -> None:
        if source != m1_freeze._GENERATED_PUBLIC_SOURCE:
            raise ValueError("forced late public validation failure")
        original(cases, source=source)

    monkeypatch.setattr(m1_freeze, "_validate_public_cases", wrapped)
    with pytest.raises(ValueError, match="forced late public validation failure"):
        _freeze(tmp_path)
    assert (tmp_path / "staging.jsonl").exists() is False
    assert (tmp_path / "public.jsonl").exists() is False
    leftovers = [
        path for path in tmp_path.iterdir() if path.suffix == ".tmp" or path.name.startswith(".")
    ]
    assert leftovers == []


def test_preexisting_replace_is_not_a_pair_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging = tmp_path / "staging.jsonl"
    public = tmp_path / "public.jsonl"
    staging.write_text("old-staging\n", encoding="utf-8")
    public.write_text("old-public\n", encoding="utf-8")
    real_replace = m1_freeze.os.replace
    calls = {"n": 0}

    def boom(src: str | Path, dst: str | Path) -> None:
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("forced second replace failure")
        real_replace(src, dst)

    monkeypatch.setattr(m1_freeze.os, "replace", boom)
    with pytest.raises(OSError, match="forced second replace failure"):
        freeze_m1_suite(
            FIXTURE,
            mode="conformance",
            plaintext_staging_path=staging,
            public_test_path=public,
        )
    assert staging.read_text(encoding="utf-8") != "old-staging\n"
    assert public.read_text(encoding="utf-8") == "old-public\n"


def test_fixed_salt_freeze_is_byte_identical(tmp_path: Path) -> None:
    def factory() -> bytes:
        factory.n += 1
        return bytes([factory.n]) * 32

    first = tmp_path / "a"
    second = tmp_path / "b"
    first.mkdir()
    second.mkdir()
    factory.n = 0
    report_a = _freeze(first, salt_factory=factory)
    factory.n = 0
    report_b = _freeze(second, salt_factory=factory)
    assert (first / "staging.jsonl").read_bytes() == (second / "staging.jsonl").read_bytes()
    assert (first / "public.jsonl").read_bytes() == (second / "public.jsonl").read_bytes()
    assert report_a["plaintext_staging_sha256"] == report_b["plaintext_staging_sha256"]
    assert report_a["public_test_sha256"] == report_b["public_test_sha256"]
    assert (
        report_a["splits"]["test"]["question_commitments_sha256"]
        == report_b["splits"]["test"]["question_commitments_sha256"]
    )


def test_complete_suite_counts_are_enforced_on_fixture(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="expected 120"):
        validate_m1_oracles(FIXTURE, allow_incomplete=False)
    with pytest.raises(ValueError, match="scoring mode rejects"):
        freeze_m1_suite(
            FIXTURE,
            mode="scoring",
            plaintext_staging_path=tmp_path / "staging.jsonl",
            public_test_path=tmp_path / "public.jsonl",
            encrypted_artifact=_encrypted_descriptor(tmp_path),
            dedup_report_sha256="c" * 64,
        )


def test_conformance_tag_bypass_on_360_scoring_file(tmp_path: Path) -> None:
    path = _complete_scoring_suite(tmp_path / "bypass.jsonl", same_sessions=True)
    with pytest.raises(ValueError, match="case_kind=conformance"):
        freeze_m1_suite(
            path,
            mode="conformance",
            plaintext_staging_path=tmp_path / "staging.jsonl",
            public_test_path=tmp_path / "public.jsonl",
        )
    with pytest.raises(ValueError, match="pairwise-distinct"):
        freeze_m1_suite(
            path,
            mode="scoring",
            plaintext_staging_path=tmp_path / "staging.jsonl",
            public_test_path=tmp_path / "public.jsonl",
            encrypted_artifact=_encrypted_descriptor(tmp_path),
            dedup_report_sha256="c" * 64,
        )
    assert (tmp_path / "staging.jsonl").exists() is False
    assert (tmp_path / "public.jsonl").exists() is False


def test_scoring_freeze_rejects_pending_review(tmp_path: Path) -> None:
    path = tmp_path / "pending.jsonl"
    _write_cases(path, _cases())
    with pytest.raises(ValueError, match="scoring mode rejects"):
        freeze_m1_suite(
            path,
            mode="scoring",
            plaintext_staging_path=tmp_path / "staging.jsonl",
            public_test_path=tmp_path / "public.jsonl",
            encrypted_artifact=_encrypted_descriptor(tmp_path),
            dedup_report_sha256="c" * 64,
        )


def test_absent_encrypted_artifact_blocks_scoring_freeze(tmp_path: Path) -> None:
    path = _complete_scoring_suite(tmp_path / "complete.jsonl")
    with pytest.raises(ValueError, match="encrypted artifact descriptor"):
        freeze_m1_suite(
            path,
            mode="scoring",
            plaintext_staging_path=tmp_path / "staging.jsonl",
            public_test_path=tmp_path / "public.jsonl",
            dedup_report_sha256="c" * 64,
        )
    assert (tmp_path / "staging.jsonl").exists() is False


def test_tampered_encrypted_artifact_hash_blocks_scoring_freeze(tmp_path: Path) -> None:
    path = _complete_scoring_suite(tmp_path / "complete.jsonl")
    encrypted = _encrypted_descriptor(tmp_path)
    encrypted["encrypted_artifact_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="encrypted artifact digest"):
        freeze_m1_suite(
            path,
            mode="scoring",
            plaintext_staging_path=tmp_path / "staging.jsonl",
            public_test_path=tmp_path / "public.jsonl",
            encrypted_artifact=encrypted,
            dedup_report_sha256="c" * 64,
        )


def test_scoring_freeze_emits_suite_manifest(tmp_path: Path) -> None:
    path = _complete_scoring_suite(tmp_path / "complete.jsonl")
    encrypted = _encrypted_descriptor(tmp_path)
    report = freeze_m1_suite(
        path,
        mode="scoring",
        plaintext_staging_path=tmp_path / "staging.jsonl",
        public_test_path=tmp_path / "public.jsonl",
        encrypted_artifact=encrypted,
        dedup_report_sha256="c" * 64,
    )
    assert report["scoring_suite"] is True
    assert report["externally_encrypted_artifact_bound"] is True
    assert report["splits"]["development"]["externally_encrypted_artifact_bound"] is False
    assert report["splits"]["validation"]["externally_encrypted_artifact_bound"] is False
    assert report["splits"]["test"]["externally_encrypted_artifact_bound"] is True
    manifest = report["suite_manifest"]
    assert isinstance(manifest, dict)
    assert manifest["case_kind"] == "scoring"
    assert manifest["status"] == "frozen"
    assert manifest["gate_1_status"] == "passed"
    assert manifest["suite_id"] == "gkr-m1-scoring-suite-v1"
    assert manifest["hash_profile_id"] == "gkr-m1-hash-profile-v1"
    assert "retrieval_configuration" not in manifest
    assert "retrieval_configuration_sha256" not in json.dumps(manifest)
    assert "cases_sha256" not in manifest["splits"]["test"]
    assert "plaintext_staging_sha256" not in manifest["splits"]["test"]
    assert "public_cases_sha256" in manifest["splits"]["test"]
    assert "question_commitments_sha256" in manifest["splits"]["test"]
    assert "encrypted_artifact_path" not in json.dumps(manifest)
    assert "sealed" not in json.dumps(manifest)
    assert manifest["encrypted_test_artifact"]["encryption_profile_id"] == "age-x25519-v1"
    assert manifest["encrypted_test_artifact"]["encrypted_artifact_sha256"] == encrypted[
        "encrypted_artifact_sha256"
    ]
    assert manifest["corpus"]["authority_jsonl_sha256"].startswith("3325db84")
    exact = manifest["evidence_applicability"]["development"]["exact_factual"]
    assert exact["total_scenario_count"] == 15
    assert (
        exact["evidence_bearing_scenario_count"] + exact["zero_set_scenario_count"] == 15
    )
    authorization = manifest["evidence_applicability"]["development"]["authorization"]
    assert authorization["total_scenario_count"] == 15
    assert (
        authorization["evidence_bearing_authorized_scenario_count"]
        + authorization["zero_set_denied_scenario_count"]
        == 15
    )
    assert (tmp_path / "staging.jsonl").exists() is True


def test_test_role_tuning_independence_required(tmp_path: Path) -> None:
    path = _complete_scoring_suite(tmp_path / "tuned.jsonl", independent=False)
    with pytest.raises(ValueError, match="not independent from retriever tuning"):
        freeze_m1_suite(
            path,
            mode="scoring",
            plaintext_staging_path=tmp_path / "staging.jsonl",
            public_test_path=tmp_path / "public.jsonl",
            encrypted_artifact=_encrypted_descriptor(tmp_path),
            dedup_report_sha256="c" * 64,
        )


def test_reviewer_family_match_blocks_scoring_freeze(tmp_path: Path) -> None:
    path = _complete_scoring_suite(tmp_path / "family.jsonl", reviewer_family="xai-grok")
    with pytest.raises(ValueError, match="reviewer model family"):
        freeze_m1_suite(
            path,
            mode="scoring",
            plaintext_staging_path=tmp_path / "staging.jsonl",
            public_test_path=tmp_path / "public.jsonl",
            encrypted_artifact=_encrypted_descriptor(tmp_path),
            dedup_report_sha256="c" * 64,
        )


def test_plaintext_cat_none_descriptor_rejected_before_output(tmp_path: Path) -> None:
    path = _complete_scoring_suite(tmp_path / "complete.jsonl")
    artifact = tmp_path / "encrypted.bin"
    artifact.write_bytes(b"external-ciphertext-placeholder")
    with pytest.raises(ValueError, match="plaintext|legacy encryption|age-x25519-v1"):
        freeze_m1_suite(
            path,
            mode="scoring",
            plaintext_staging_path=tmp_path / "staging.jsonl",
            public_test_path=tmp_path / "public.jsonl",
            encrypted_artifact={
                "scheme": "plaintext",
                "tool": "cat",
                "recipient_key_fingerprint": "none",
                "encrypted_artifact_path": str(artifact),
                "encrypted_artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            },
            dedup_report_sha256="c" * 64,
        )
    assert (tmp_path / "staging.jsonl").exists() is False
    assert (tmp_path / "public.jsonl").exists() is False


def test_suite_manifest_validation_failure_leaves_no_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _complete_scoring_suite(tmp_path / "complete.jsonl")

    def boom(manifest: dict[str, object]) -> None:
        raise ValueError("forced suite schema failure")

    monkeypatch.setattr(m1_freeze, "_validate_suite_manifest", boom)
    with pytest.raises(ValueError, match="forced suite schema failure"):
        freeze_m1_suite(
            path,
            mode="scoring",
            plaintext_staging_path=tmp_path / "staging.jsonl",
            public_test_path=tmp_path / "public.jsonl",
            encrypted_artifact=_encrypted_descriptor(tmp_path),
            dedup_report_sha256="c" * 64,
        )
    assert (tmp_path / "staging.jsonl").exists() is False
    assert (tmp_path / "public.jsonl").exists() is False
    leftovers = [
        item for item in tmp_path.iterdir() if item.suffix == ".tmp" or item.name.startswith(".")
    ]
    assert leftovers == []


def test_complete_suite_requires_evidence_applicability(tmp_path: Path) -> None:
    path = _complete_scoring_suite(
        tmp_path / "all-auth-published.jsonl", authorization_denied=False
    )
    with pytest.raises(ValueError, match="zero-set denied scenario|under-counts"):
        validate_m1_oracles(path, allow_incomplete=False)
    with pytest.raises(ValueError, match="zero-set denied scenario|under-counts"):
        freeze_m1_suite(
            path,
            mode="scoring",
            plaintext_staging_path=tmp_path / "staging.jsonl",
            public_test_path=tmp_path / "public.jsonl",
            encrypted_artifact=_encrypted_descriptor(tmp_path),
            dedup_report_sha256="c" * 64,
        )
    assert (tmp_path / "staging.jsonl").exists() is False
    assert (tmp_path / "public.jsonl").exists() is False


def test_authorization_unclassified_scenario_under_counts_partition(tmp_path: Path) -> None:
    path = _complete_scoring_suite(tmp_path / "auth-gap.jsonl")
    cases = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    mutated = False
    for case in cases:
        if (
            case["split"] == "development"
            and case["query_class"] == "authorization"
            and not case["oracle"]["sufficient_reference_sets"]
        ):
            case["oracle"]["publication"] = "published"
            case["oracle"].pop("disposition_reason", None)
            mutated = True
            break
    assert mutated
    _write_cases(path, cases)
    with pytest.raises(ValueError, match="under-counts"):
        validate_m1_oracles(path, allow_incomplete=False)


def test_conformance_freeze_emits_no_suite_or_retrieval_digest(tmp_path: Path) -> None:
    report = _freeze(tmp_path)
    assert report["suite_manifest"] is None
    assert "retrieval_configuration" not in report
    assert "evidence_applicability" not in report
    assert "cases_sha256" not in report["splits"]["test"]
    dumped = json.dumps(report)
    assert "retrieval_configuration_sha256" not in dumped
    assert "gkr-m1-retrieval-configuration-v1" not in dumped


def test_scoring_freeze_rejects_retrieval_configuration_kwarg(tmp_path: Path) -> None:
    path = _complete_scoring_suite(tmp_path / "complete.jsonl")
    with pytest.raises(TypeError):
        freeze_m1_suite(
            path,
            mode="scoring",
            plaintext_staging_path=tmp_path / "staging.jsonl",
            public_test_path=tmp_path / "public.jsonl",
            encrypted_artifact=_encrypted_descriptor(tmp_path),
            dedup_report_sha256="c" * 64,
            retrieval_configuration_sha256="d" * 64,
        )
