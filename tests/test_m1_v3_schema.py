"""JSON Schema mutation probes for M1 v3 (not runtime-validator-only)."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker

CASE_SCHEMA = Path("evaluation/m1/benchmark-case-v3.schema.json")
SUITE_SCHEMA = Path("evaluation/m1/suite-manifest-v3.schema.json")
FIXTURE = Path("tests/fixtures/m1/conformance-cases.jsonl")
PROGRAMME = Path("evaluation/m1/programme-v3.json")
METRICS = Path("evaluation/m1/metric-contract-v3.json")
AUTHORITY_SHA256 = "3325db848fe13a2f0f3e7b2e0894e92d9b27e99c42a16b3c90f03b40a39c81a2"
CONTENT_DIGEST = "1a30668d55910c544fa00aa6927ad7dc3064f6585a6186bc543fc590dec64300"
FREEZE_DIGEST = "961e91927798ddfecf55133591481ddf6f5d181729296c21937ff036b2563a1a"


def _case_validator() -> Draft202012Validator:
    return Draft202012Validator(
        json.loads(CASE_SCHEMA.read_text(encoding="utf-8")),
        format_checker=FormatChecker(),
    )


def _suite_validator() -> Draft202012Validator:
    return Draft202012Validator(json.loads(SUITE_SCHEMA.read_text(encoding="utf-8")))


def _fixture_case(case_id: str) -> dict[str, object]:
    for line in FIXTURE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        if case["case_id"] == case_id:
            return case
    raise AssertionError(case_id)


def _errors(validator: Draft202012Validator, instance: object) -> list[str]:
    return [error.message for error in validator.iter_errors(instance)]


def test_schema_rejects_oos_with_sufficient_set() -> None:
    case = _fixture_case("m1-cf-share-options-01")
    case["oracle"]["sufficient_reference_sets"] = [["FIN-EXP-THRESHOLD:v2"]]
    assert _errors(_case_validator(), case)


def test_schema_rejects_unknown_oos_bundle_field_mutations() -> None:
    validator = _case_validator()
    for field, value in (
        ("support_label", "supported"),
        ("publication", "published"),
        ("disposition_reason", "unauthorized_actor"),
    ):
        case = _fixture_case("m1-cf-share-options-01")
        case["oracle"][field] = value
        assert _errors(validator, case), field


def test_schema_rejects_supported_with_empty_sets() -> None:
    case = _fixture_case("m1-cf-travel-approval-01")
    case["oracle"]["sufficient_reference_sets"] = []
    assert _errors(_case_validator(), case)


def test_schema_rejects_conflicting_with_empty_sets() -> None:
    case = _fixture_case("m1-cf-overtime-01")
    case["oracle"]["sufficient_reference_sets"] = []
    assert _errors(_case_validator(), case)


def test_schema_rejects_zero_set_dispositions_with_sufficient_set() -> None:
    validator = _case_validator()
    for disposition, source_id in (
        ("unauthorized_actor", "m1-cf-payroll-02"),
        ("no_authorized_evidence", "m1-cf-share-options-01"),
        ("stale_or_future_only", "m1-cf-share-options-01"),
        ("ambiguous_question", "m1-cf-share-options-01"),
    ):
        case = _fixture_case(source_id)
        if disposition != "unauthorized_actor":
            case["query_class"] = "authorization"
        case["oracle"]["disposition_reason"] = disposition
        case["oracle"]["sufficient_reference_sets"] = [["FIN-EXP-THRESHOLD:v2"]]
        assert _errors(validator, case), disposition


def test_schema_rejects_missing_hash_profile_id() -> None:
    case = _fixture_case("m1-cf-travel-approval-01")
    del case["hash_profile_id"]
    assert _errors(_case_validator(), case)


def test_schema_rejects_different_model_family_method() -> None:
    case = _fixture_case("m1-cf-travel-approval-01")
    case["question_authorship"]["method"] = "different_model_family"
    assert _errors(_case_validator(), case)


def test_schema_rejects_non_canonical_model_family() -> None:
    case = _fixture_case("m1-cf-travel-approval-01")
    case["question_authorship"]["model_family_id"] = "Grok"
    assert _errors(_case_validator(), case)


def test_schema_rejects_out_of_scope_on_non_oos_class() -> None:
    case = _fixture_case("m1-cf-payroll-02")
    case["oracle"]["disposition_reason"] = "out_of_scope"
    assert _errors(_case_validator(), case)


def test_schema_rejects_scoring_null_prompt_digest() -> None:
    case = _fixture_case("m1-cf-travel-approval-01")
    case["case_kind"] = "scoring"
    case["question_authorship"]["prompt_retained"] = False
    case["question_authorship"]["prompt_sha256"] = None
    case["oracle_review"] = {
        "status": "completed",
        "semantically_reviewed": True,
        "reviewer_kind": "human",
        "reviewer_identity": "reviewer",
        "reviewer_session_id": "review-session",
        "reviewer_model_family_id": None,
        "reviewer_model_id": None,
        "reviewer_model_revision": None,
        "reviewer_prompt_sha256": None,
        "review_sha256": "bb" * 32,
        "independent_from_retriever_tuning": True,
    }
    assert _errors(_case_validator(), case)


def _minimal_suite() -> dict[str, object]:
    digest = "a" * 64
    split = {
        "independent_scenario_count": 120,
        "case_count": 120,
        "cases_sha256": digest,
        "questions_sha256": digest,
        "externally_encrypted_artifact_bound": False,
    }
    test_split = {
        "independent_scenario_count": 120,
        "case_count": 120,
        "public_case_count": 120,
        "public_cases_sha256": digest,
        "question_commitments_sha256": digest,
        "externally_encrypted_artifact_bound": True,
    }
    class_counts = {
        "exact_factual": 15,
        "semantic_paraphrase": 15,
        "numeric_conditional": 15,
        "temporal": 15,
        "authorization": 15,
        "unknown_oos": 15,
        "multi_record": 15,
        "adversarial_conflicting": 15,
    }
    evidence_class = {
        "evidence_bearing_scenario_count": 15,
        "zero_set_scenario_count": 0,
        "total_scenario_count": 15,
    }
    applicability = {
        "exact_factual": deepcopy(evidence_class),
        "semantic_paraphrase": deepcopy(evidence_class),
        "numeric_conditional": deepcopy(evidence_class),
        "temporal": deepcopy(evidence_class),
        "authorization": {
            "evidence_bearing_authorized_scenario_count": 8,
            "zero_set_denied_scenario_count": 7,
            "total_scenario_count": 15,
        },
        "unknown_oos": {
            "evidence_bearing_scenario_count": 0,
            "zero_set_scenario_count": 15,
            "total_scenario_count": 15,
        },
        "multi_record": deepcopy(evidence_class),
        "adversarial_conflicting": deepcopy(evidence_class),
    }
    return {
        "schema_version": "gkr-m1-suite-manifest-v3",
        "suite_id": "gkr-m1-scoring-suite-v1",
        "status": "frozen",
        "case_kind": "scoring",
        "case_schema_version": "gkr-m1-case-v3",
        "metric_contract_version": "gkr-m1-retrieval-metrics-v3",
        "hash_profile_id": "gkr-m1-hash-profile-v1",
        "gate_1_status": "passed",
        "semantic_review": {"all_completed": True, "all_semantically_reviewed": True},
        "deterministic_oracle_validation": {"passed": True},
        "corpus": {
            "stable_record_count": 47,
            "authority_event_count": 63,
            "authority_jsonl_sha256": AUTHORITY_SHA256,
            "content_digest_sha256": CONTENT_DIGEST,
            "corpus_freeze_manifest_sha256": FREEZE_DIGEST,
        },
        "splits": {
            "development": split,
            "validation": deepcopy(split),
            "test": test_split,
        },
        "query_class_scenario_counts": {
            "development": class_counts,
            "validation": deepcopy(class_counts),
            "test": deepcopy(class_counts),
        },
        "evidence_applicability": {
            "development": deepcopy(applicability),
            "validation": deepcopy(applicability),
            "test": deepcopy(applicability),
        },
        "encrypted_test_artifact": {
            "encryption_profile_id": "age-x25519-v1",
            "tool_family": "age",
            "tool_version": "1.2.0",
            "recipient_key_fingerprint_sha256": digest,
            "encrypted_artifact_sha256": digest,
        },
        "encryption_boundary": (
            "The encrypted artifact is produced by an external age-x25519-v1 "
            "authenticated-encryption tool. This runtime verifies the artifact exists "
            "outside the repository, matches encryption profile age-x25519-v1, and "
            "matches the declared SHA-256 of exact ciphertext bytes. It does not "
            "implement or decrypt age and does not prove the ciphertext decrypts to "
            "the plaintext staging bytes."
        ),
        "deduplication": {
            "exact_cross_split_duplicates": 0,
            "semantic_cross_split_candidates_reviewed": True,
            "report_sha256": digest,
        },
    }


def test_valid_frozen_suite_schema_accepts_age_profile() -> None:
    assert _errors(_suite_validator(), _minimal_suite()) == []


def test_plaintext_cat_none_schema_probe_fails() -> None:
    manifest = _minimal_suite()
    manifest["encrypted_test_artifact"] = {
        "scheme": "plaintext",
        "tool": "cat",
        "recipient_key_fingerprint": "none",
        "encrypted_artifact_sha256": "a" * 64,
    }
    for split in ("development", "validation", "test"):
        manifest["splits"][split]["sealed"] = True
        manifest["splits"][split]["externally_encrypted_artifact_bound"] = True
    errors = _errors(_suite_validator(), manifest)
    assert errors


def test_sealed_true_on_development_rejected() -> None:
    manifest = _minimal_suite()
    manifest["splits"]["development"]["externally_encrypted_artifact_bound"] = True
    assert _errors(_suite_validator(), manifest)


def test_unknown_encryption_profile_rejected() -> None:
    manifest = _minimal_suite()
    manifest["encrypted_test_artifact"]["encryption_profile_id"] = "password-zip"
    assert _errors(_suite_validator(), manifest)


def test_lifecycle_combinations() -> None:
    validator = _suite_validator()
    assert _errors(validator, _minimal_suite()) == []
    for status in ("draft", "frozen", "validation_failed"):
        for gate in ("not_run", "passed", "failed"):
            manifest = _minimal_suite()
            manifest["status"] = status
            manifest["gate_1_status"] = gate
            errors = _errors(validator, manifest)
            if (status, gate) == ("frozen", "passed"):
                assert errors == [], (status, gate, errors)
            else:
                assert errors, (status, gate)


def test_unknown_v9_encryption_profile_rejected() -> None:
    manifest = _minimal_suite()
    manifest["encrypted_test_artifact"]["encryption_profile_id"] = "unknown-v9"
    assert _errors(_suite_validator(), manifest)


def test_validation_external_encryption_true_rejected() -> None:
    manifest = _minimal_suite()
    manifest["splits"]["validation"]["externally_encrypted_artifact_bound"] = True
    assert _errors(_suite_validator(), manifest)


def test_test_external_encryption_false_rejected() -> None:
    manifest = _minimal_suite()
    manifest["splits"]["test"]["externally_encrypted_artifact_bound"] = False
    assert _errors(_suite_validator(), manifest)


def test_suite_identity_mutations_rejected() -> None:
    validator = _suite_validator()
    for mutator in (
        lambda manifest: manifest.__setitem__("suite_id", "other-suite"),
        lambda manifest: manifest["corpus"].__setitem__(
            "authority_jsonl_sha256", "a" * 64
        ),
        lambda manifest: manifest["corpus"].__setitem__(
            "content_digest_sha256", "a" * 64
        ),
        lambda manifest: manifest["corpus"].__setitem__(
            "corpus_freeze_manifest_sha256", "a" * 64
        ),
    ):
        manifest = _minimal_suite()
        mutator(manifest)
        assert _errors(validator, manifest)


def test_test_split_full_case_digest_rejected() -> None:
    manifest = _minimal_suite()
    manifest["splits"]["test"]["cases_sha256"] = "a" * 64
    assert _errors(_suite_validator(), manifest)


def test_suite_schema_has_no_retrieval_configuration() -> None:
    schema = json.loads(SUITE_SCHEMA.read_text(encoding="utf-8"))
    assert "retrieval_configuration" not in schema["properties"]
    assert "retrieval_configuration" not in schema["required"]
    manifest = _minimal_suite()
    manifest["retrieval_configuration"] = {
        "schema_id": "gkr-m1-retrieval-configuration-v1",
        "sha256": "a" * 64,
        "test_opened": False,
    }
    assert _errors(_suite_validator(), manifest)


def test_gate_1_docs_select_no_retriever() -> None:
    programme = json.loads(PROGRAMME.read_text(encoding="utf-8"))
    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    assert programme["phase_boundary"]["gate_1"].startswith("Gate 1 freezes corpus")
    assert "selects no retriever" in programme["phase_boundary"]["gate_1"]
    assert programme["statistical_policy"]["selection_unit"] == "one_global_candidate_arm"
    assert "wins each query class" not in json.dumps(programme)
    assert "wins each query class" not in json.dumps(metrics)
    assert metrics["selection_policy"]["selection_unit"] == "one_global_candidate_arm"
    assert "retrieval-configuration-v1" not in json.dumps(programme)
    assert "retrieval-configuration-v1" not in json.dumps(metrics)


def test_applicability_requires_total_scenario_count() -> None:
    manifest = _minimal_suite()
    del manifest["evidence_applicability"]["development"]["exact_factual"][
        "total_scenario_count"
    ]
    assert _errors(_suite_validator(), manifest)
    manifest = _minimal_suite()
    manifest["evidence_applicability"]["development"]["exact_factual"][
        "total_scenario_count"
    ] = 14
    assert _errors(_suite_validator(), manifest)


def test_applicability_undercount_overcount_and_authorization_mismatch_rejected() -> None:
    from gkr.m1_freeze import _validate_suite_manifest

    under = _minimal_suite()
    under["evidence_applicability"]["development"]["exact_factual"] = {
        "evidence_bearing_scenario_count": 1,
        "zero_set_scenario_count": 0,
        "total_scenario_count": 15,
    }
    with pytest.raises(ValueError, match="under-counts"):
        _validate_suite_manifest(under)

    over = _minimal_suite()
    over["evidence_applicability"]["validation"]["multi_record"] = {
        "evidence_bearing_scenario_count": 10,
        "zero_set_scenario_count": 10,
        "total_scenario_count": 15,
    }
    with pytest.raises(ValueError, match="over-counts"):
        _validate_suite_manifest(over)

    mismatch = _minimal_suite()
    mismatch["evidence_applicability"]["test"]["authorization"] = {
        "evidence_bearing_authorized_scenario_count": 1,
        "zero_set_denied_scenario_count": 1,
        "total_scenario_count": 15,
    }
    with pytest.raises(ValueError, match="under-counts|partition mismatch"):
        _validate_suite_manifest(mismatch)


def test_suite_schema_rejects_prohibited_tool_version_tokens() -> None:
    for token in (
        "plaintext",
        "cat",
        "none",
        "zip",
        "password",
        "password-zip",
        "password_zip",
        "archive",
        "archive-only",
        "archive_only",
    ):
        manifest = _minimal_suite()
        manifest["encrypted_test_artifact"]["tool_version"] = token
        assert _errors(_suite_validator(), manifest), token
