from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from conftest import make_record

from gkr.authority import AuthorityStore
from gkr.evaluation import run_retrieval_suite
from gkr.m1_validation import question_digest, validate_m1_cases
from gkr.runtime import GovernedKnowledgeRuntime


def test_retrieval_suite_reports_temporal_and_acl_expectations(
    store: AuthorityStore,
    tmp_path: Path,
) -> None:
    store.append(make_record())
    case = {
        "case_id": "visible-policy",
        "question": "What approval is required?",
        "actor": "alice",
        "groups": ["employees"],
        "as_of": "2026-06-01",
        "expected_contains": ["POL-001:v1"],
        "expected_absent": ["SEC-001:v1"],
    }
    suite = tmp_path / "suite.jsonl"
    suite.write_text(json.dumps(case) + "\n", encoding="utf-8")

    report = run_retrieval_suite(GovernedKnowledgeRuntime(store), suite)

    assert report["passed"] == 1
    assert report["failed"] == 0


def test_frozen_m0_suite_passes_against_demo_authority(tmp_path: Path) -> None:
    with AuthorityStore(tmp_path / "authority.sqlite") as store:
        store.import_jsonl("knowledge/demo_records.jsonl")
        report = run_retrieval_suite(
            GovernedKnowledgeRuntime(store),
            "evaluation/m0_retrieval.jsonl",
        )

    assert report["cases"] == 5
    assert report["passed"] == 5
    assert report["failed"] == 0


def test_m1_benchmark_contract_is_frozen_and_self_consistent() -> None:
    contract_directory = Path("evaluation/m1")
    manifest = json.loads(
        (contract_directory / "contract-manifest.json").read_text(encoding="utf-8")
    )
    schema = json.loads(
        (contract_directory / "benchmark-case.schema.json").read_text(encoding="utf-8")
    )
    metrics = json.loads(
        (contract_directory / "metric-contract.json").read_text(encoding="utf-8")
    )

    assert manifest["status"] == "frozen"
    assert metrics["status"] == "frozen"
    assert schema["properties"]["query_class"]["enum"] == metrics["query_classes"]
    assert schema["properties"]["metric_targets"]["items"]["enum"] == metrics["metrics"]
    for file_name, expected_hash in manifest["files"].items():
        content = (contract_directory / file_name).read_bytes()
        assert hashlib.sha256(content).hexdigest() == expected_hash


def test_m1_v2_programme_freezes_splits_oracles_and_safety_metrics() -> None:
    contract_directory = Path("evaluation/m1")
    manifest = _json_file(contract_directory / "contract-manifest-v2.json")
    case_schema = _json_file(contract_directory / "benchmark-case-v2.schema.json")
    suite_schema = _json_file(contract_directory / "suite-manifest.schema.json")
    metrics = _json_file(contract_directory / "metric-contract-v2.json")
    programme = _json_file(contract_directory / "programme.json")

    query_classes = case_schema["properties"]["query_class"]["enum"]
    suite_classes = suite_schema["properties"]["query_class_scenario_counts"]["required"]
    assert query_classes == suite_classes
    assert programme["independent_scenarios"] == {
        "development": 120,
        "validation": 120,
        "test": 120,
    }
    assert metrics["absolute_gates"] == {
        "unauthorized_record_exposure": 0,
        "stale_version_retrieval_rate": 0,
    }
    assert "sufficient_reference_sets" in case_schema["properties"]["oracle"]["required"]
    assert manifest["cases_status"] == "pending-independent-owner-labelled-corpus"
    assert programme["current_pass_status"] == {
        "contract": "passed",
        "corpus": "not_run",
        "development": "not_run",
        "validation": "not_run",
        "test": "not_run",
    }
    for file_name, expected_hash in manifest["files"].items():
        content = (contract_directory / file_name).read_bytes()
        assert hashlib.sha256(content).hexdigest() == expected_hash


def test_m1_validator_checks_question_hash_and_allows_incomplete_authoring(
    tmp_path: Path,
) -> None:
    question = "Which policy applies?"
    case = {
        "schema_version": "gkr-m1-case-v2",
        "case_id": "dev-policy-001-a",
        "scenario_id": "dev-policy-001",
        "variant_id": "a",
        "split": "development",
        "query_class": "exact_factual",
        "question": question,
        "question_sha256": question_digest(question),
        "authorship": {
            "method": "human",
            "independent_from_retriever_tuning": True,
        },
        "scope": {
            "actor": "alice",
            "groups": ["employees"],
            "as_of": "2026-06-01",
        },
        "oracle": {
            "sufficient_reference_sets": [["POL-001:v1"]],
            "forbidden_references": [],
            "publication": "published",
            "support_label": "supported",
        },
    }
    cases = tmp_path / "cases.jsonl"
    cases.write_text(json.dumps(case) + "\n", encoding="utf-8")

    report = validate_m1_cases(cases, allow_incomplete=True)
    assert report["cases"] == 1
    assert report["independent_scenarios"] == {
        "development": 1,
        "validation": 0,
        "test": 0,
    }

    case["question_sha256"] = "0" * 64
    cases.write_text(json.dumps(case) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="question_sha256"):
        validate_m1_cases(cases, allow_incomplete=True)


def _json_file(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))
