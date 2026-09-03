from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import pytest
from test_m1_freeze import (
    _cases,
    _complete_scoring_suite,
    _scoring_roles,
    _write_cases,
)

from gkr.m1_authoring import (
    DEDUP_ATTESTATION_BOUNDARY,
    SEMANTIC_REVIEW_SCOPE,
    assemble_reviewed_cases,
    build_dedup_candidates,
    hash_dedup_file,
    hash_prompt_file,
    hash_review_file,
    lexical_cross_split_candidates,
    load_semantic_dedup_report,
    question_set_digest,
    validate_semantic_dedup_report,
)
from gkr.m1_hash import prompt_digest, question_digest
from gkr.m1_io import repository_root
from gkr.m1_validation import validate_m1_cases

PYTHON = sys.executable


def _question_draft(case: dict[str, object]) -> dict[str, object]:
    draft = {
        "schema_version": "gkr-m1-question-draft-v1",
        "hash_profile_id": "gkr-m1-hash-profile-v1",
        "case_id": case["case_id"],
        "scenario_id": case["scenario_id"],
        "variant_id": case["variant_id"],
        "split": case["split"],
        "query_class": case["query_class"],
        "question": case["question"],
        "question_sha256": case["question_sha256"],
        "question_authorship": deepcopy(case["question_authorship"]),
        "scope": deepcopy(case["scope"]),
    }
    if "tags" in case:
        draft["tags"] = deepcopy(case["tags"])
    return draft


def _oracle_draft(case: dict[str, object]) -> dict[str, object]:
    draft = _question_draft(case)
    draft["schema_version"] = "gkr-m1-oracle-draft-v1"
    draft["case_kind"] = "scoring"
    draft["oracle_authorship"] = deepcopy(case["oracle_authorship"])
    draft["oracle"] = deepcopy(case["oracle"])
    if "oracle_notes" in case:
        draft["oracle_notes"] = case["oracle_notes"]
    return draft


def _review_artifact(oracles: list[dict[str, object]], *, split: str) -> dict[str, object]:
    split_rows = [row for row in oracles if row["split"] == split]
    case_ids = sorted(str(row["case_id"]) for row in split_rows)
    return {
        "schema_version": "gkr-m1-semantic-review-artifact-v1",
        "split": split,
        "reviewed_oracle_draft_question_set_sha256": question_set_digest(split_rows),
        "case_count": len(case_ids),
        "case_ids": case_ids,
        "overall_status": "APPROVED",
        "results": [
            {"case_id": case_id, "status": "APPROVED", "finding_codes": []}
            for case_id in case_ids
        ],
        "reviewer_kind": "model",
        "reviewer_identity": "chatgpt",
        "reviewer_session_id": "chatgpt-review-session",
        "reviewer_model_family_id": "openai-gpt",
        "reviewer_model_id": "gpt-4.1",
        "reviewer_model_revision": "chatgpt-015",
        "reviewer_prompt_sha256": "aa" * 32,
        "independent_from_retriever_tuning": True,
        "semantic_review_scope": SEMANTIC_REVIEW_SCOPE,
    }


def _scoring_sample() -> dict[str, object]:
    case = deepcopy(next(item for item in _cases() if item["split"] == "development"))
    question, oracle, review = _scoring_roles()
    case["case_kind"] = "scoring"
    case["question_authorship"] = question
    case["oracle_authorship"] = oracle
    case["oracle_review"] = review
    return case


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.write_bytes(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    return path


def _assemble_sample(tmp_path: Path, **overrides: object) -> dict[str, object]:
    case = _scoring_sample()
    questions = [_question_draft(case)]
    oracles = [_oracle_draft(case)]
    artifact = _review_artifact(oracles, split="development")
    for key, value in overrides.items():
        if key == "question":
            questions[0] = value  # type: ignore[assignment]
        elif key == "oracle":
            oracles[0] = value  # type: ignore[assignment]
        elif key == "artifact":
            artifact = value  # type: ignore[assignment]
    question_path = _write_jsonl(tmp_path / "questions.jsonl", questions)
    oracle_path = _write_jsonl(tmp_path / "oracles.jsonl", oracles)
    review_path = _write_json(tmp_path / "review.json", artifact)
    return assemble_reviewed_cases(
        questions_path=question_path,
        oracle_drafts_path=oracle_path,
        review_artifact_paths=[review_path],
        output_path=tmp_path / "reviewed.jsonl",
    )


def test_question_oracle_drift_is_rejected(tmp_path: Path) -> None:
    case = _scoring_sample()
    oracle = _oracle_draft(case)
    oracle["question"] = case["question"] + " drifted"
    oracle["question_sha256"] = question_digest(str(oracle["question"]))
    with pytest.raises(ValueError, match="question-field drift"):
        _assemble_sample(tmp_path, oracle=oracle)
    assert (tmp_path / "reviewed.jsonl").exists() is False


def test_role_session_collision_is_rejected(tmp_path: Path) -> None:
    case = _scoring_sample()
    case["oracle_authorship"] = deepcopy(case["question_authorship"])
    with pytest.raises(ValueError, match="pairwise distinct"):
        _assemble_sample(tmp_path, question=_question_draft(case), oracle=_oracle_draft(case))


def test_reviewer_family_collision_is_rejected(tmp_path: Path) -> None:
    case = _scoring_sample()
    artifact = _review_artifact([_oracle_draft(case)], split="development")
    artifact["reviewer_model_family_id"] = "xai-grok"
    artifact["reviewer_model_id"] = "grok-4"
    with pytest.raises(ValueError, match="reviewer model family"):
        _assemble_sample(tmp_path, artifact=artifact)


def test_unknown_model_mapping_is_rejected(tmp_path: Path) -> None:
    case = _scoring_sample()
    case["question_authorship"]["model_id"] = "not-a-registered-model"
    with pytest.raises(ValueError, match="unknown family/model mapping"):
        _assemble_sample(tmp_path, question=_question_draft(case), oracle=_oracle_draft(case))


def test_missing_extra_and_blocked_review_results(tmp_path: Path) -> None:
    case = _scoring_sample()
    artifact = _review_artifact([_oracle_draft(case)], split="development")
    artifact["results"] = []
    artifact["case_count"] = 0
    artifact["case_ids"] = []
    with pytest.raises(ValueError, match="case_count|review case set|minItems"):
        _assemble_sample(tmp_path, artifact=artifact)

    extra = _review_artifact([_oracle_draft(case)], split="development")
    extra["case_ids"] = sorted([str(case["case_id"]), "m1-extra-case-01"])
    extra["case_count"] = 2
    extra["results"] = [
        {"case_id": str(case["case_id"]), "status": "APPROVED", "finding_codes": []},
        {"case_id": "m1-extra-case-01", "status": "APPROVED", "finding_codes": []},
    ]
    with pytest.raises(ValueError, match="review case set"):
        _assemble_sample(tmp_path, artifact=extra)

    blocked = _review_artifact([_oracle_draft(case)], split="development")
    blocked["overall_status"] = "BLOCKED"
    blocked["results"][0]["status"] = "BLOCKED"
    blocked["results"][0]["finding_codes"] = ["unsupported-claim"]
    with pytest.raises(ValueError, match="BLOCKED|blocked semantic-review"):
        _assemble_sample(tmp_path, artifact=blocked)


def test_review_digest_is_stamped_from_exact_raw_bytes(tmp_path: Path) -> None:
    report = _assemble_sample(tmp_path)
    review_bytes = (tmp_path / "review.json").read_bytes()
    expected = hashlib.sha256(review_bytes).hexdigest()
    assert report["review_sha256_by_split"]["development"] == expected
    assembled = json.loads((tmp_path / "reviewed.jsonl").read_text(encoding="utf-8"))
    assert assembled["oracle_review"]["review_sha256"] == expected
    assert assembled["oracle_review"]["status"] == "completed"
    assert assembled["question"] == _scoring_sample()["question"]
    assert assembled["oracle"] == _scoring_sample()["oracle"]


def test_prompt_review_and_dedup_hash_profiles(tmp_path: Path) -> None:
    prompt = tmp_path / "prompt.txt"
    prompt.write_bytes(b"hello\r\nworld\r!\n")
    assert hash_prompt_file(prompt) == prompt_digest("hello\nworld\n!\n")
    review = tmp_path / "review.bin"
    review.write_bytes(b"exact\r\nbytes")
    assert hash_review_file(review) == hashlib.sha256(b"exact\r\nbytes").hexdigest()
    assert hash_review_file(review) != hash_prompt_file(review)
    payload = {"z": 1, "a": {"b": 2}}
    pretty = tmp_path / "dedup-pretty.json"
    pretty.write_text(json.dumps(payload, indent=4), encoding="utf-8")
    compact = tmp_path / "dedup-compact.json"
    compact.write_text(json.dumps({"a": {"b": 2}, "z": 1}, separators=(",", ":")), encoding="utf-8")
    assert hash_dedup_file(pretty) == hash_dedup_file(compact)
    hashed = subprocess.check_output(
        [PYTHON, "scripts/hash_m1_artifact.py", "prompt", str(prompt)],
        text=True,
    ).strip()
    assert hashed == hash_prompt_file(prompt)


def test_exact_cross_split_duplicate_fails_candidate_build(tmp_path: Path) -> None:
    path = _complete_scoring_suite(tmp_path / "complete.jsonl")
    cases = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    donor = next(case for case in cases if case["split"] == "development")
    for case in cases:
        if case["split"] == "test":
            case["question"] = donor["question"]
            case["question_sha256"] = donor["question_sha256"]
            break
    _write_cases(path, cases)
    with pytest.raises(ValueError, match="exact normalized cross-split duplicates"):
        build_dedup_candidates(path, output_path=tmp_path / "outside" / "candidates.json")


def test_candidate_ordering_is_deterministic(tmp_path: Path) -> None:
    path = _complete_scoring_suite(tmp_path / "complete.jsonl")
    cases = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    first = lexical_cross_split_candidates(cases)
    second = lexical_cross_split_candidates(list(reversed(cases)))
    assert first == second
    ordered = sorted(
        first,
        key=lambda item: (-item["score"], item["left_case_id"], item["right_case_id"]),
    )
    assert first == ordered
    outside = tmp_path / "dedup-candidates.json"
    report = build_dedup_candidates(path, output_path=outside)
    assert report["semantic_cross_split_candidates_reviewed"] is False
    assert report["exact_cross_split_duplicates"] == 0
    assert len(report["question_inventory"]) == 360
    assert "reviewer_session_id" not in report
    inside = repository_root() / "evaluation" / "m1" / "must-not-write-dedup.json"
    with pytest.raises(ValueError, match="outside the repository"):
        build_dedup_candidates(path, output_path=inside)
    assert inside.exists() is False


def _dedup_report(
    cases: list[dict[str, object]],
    candidates: list[dict[str, object]],
) -> dict[str, object]:
    return {
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


def test_dedup_report_mismatch_unreviewed_and_unresolved(tmp_path: Path) -> None:
    path = _complete_scoring_suite(tmp_path / "complete.jsonl")
    cases = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    candidates = lexical_cross_split_candidates(cases)
    report = _dedup_report(cases, candidates)
    valid_path = tmp_path / "dedup.json"
    valid_path.write_text(json.dumps(report), encoding="utf-8")
    loaded, digest = load_semantic_dedup_report(valid_path, cases=cases)
    assert loaded["semantic_cross_split_candidates_reviewed"] is True
    assert digest == hash_dedup_file(valid_path)

    mismatched = deepcopy(report)
    mismatched["question_set_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="question_set_sha256"):
        validate_semantic_dedup_report(
            mismatched, cases=cases, generated_candidates=candidates
        )

    unreviewed = deepcopy(report)
    unreviewed["candidates"] = report["candidates"][1:]
    with pytest.raises(ValueError, match="missing a disposition"):
        validate_semantic_dedup_report(
            unreviewed, cases=cases, generated_candidates=candidates
        )

    unresolved = deepcopy(report)
    unresolved["unresolved_semantic_duplicates"] = 1
    with pytest.raises(ValueError, match="unresolved_semantic_duplicates"):
        validate_semantic_dedup_report(
            unresolved, cases=cases, generated_candidates=candidates
        )


def test_validate_m1_cases_refuses_silent_v3_against_v2(tmp_path: Path) -> None:
    case = _scoring_sample()
    path = _write_jsonl(tmp_path / "v3.jsonl", [case])
    with pytest.raises(ValueError, match="v3 cases must not be silently validated"):
        validate_m1_cases(path)
    report = validate_m1_cases(
        path,
        schema_path="evaluation/m1/benchmark-case-v3.schema.json",
        allow_incomplete=True,
    )
    assert report["cases"] == 1
    detected = subprocess.check_output(
        [PYTHON, "scripts/validate_m1_cases.py", str(path), "--allow-incomplete"],
        text=True,
    )
    assert '"schema_version": "v3"' in detected
    rejected = subprocess.run(
        [
            PYTHON,
            "scripts/validate_m1_cases.py",
            str(path),
            "--schema-version",
            "v2",
            "--allow-incomplete",
        ],
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0
    assert "v3 cases must not be silently validated" in rejected.stderr + rejected.stdout


DEDUP_CLI_STDOUT_KEYS = frozenset(
    {
        "output",
        "case_count",
        "scenario_count",
        "question_set_sha256",
        "candidate_count",
        "exact_cross_split_duplicates",
    }
)


@contextmanager
def _ephemeral_repo_output(name: str):
    path = repository_root() / "evaluation" / "m1" / name
    path.unlink(missing_ok=True)
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)


def _scoring_case_for_split(split: str, *, case_id: str) -> dict[str, object]:
    case = _scoring_sample()
    case["split"] = split
    case["case_id"] = case_id
    case["scenario_id"] = case_id
    return case


def _write_role_packets(
    tmp_path: Path,
    cases: list[dict[str, object]],
    *,
    prefix: str = "",
) -> tuple[Path, Path, list[Path]]:
    questions = [_question_draft(case) for case in cases]
    oracles = [_oracle_draft(case) for case in cases]
    question_path = _write_jsonl(tmp_path / f"{prefix}questions.jsonl", questions)
    oracle_path = _write_jsonl(tmp_path / f"{prefix}oracles.jsonl", oracles)
    review_paths = [
        _write_json(
            tmp_path / f"{prefix}review-{split}.json",
            _review_artifact(oracles, split=split),
        )
        for split in sorted({str(case["split"]) for case in cases})
    ]
    return question_path, oracle_path, review_paths


def test_test_only_assembly_is_rejected_inside_repository(tmp_path: Path) -> None:
    questions, oracles, reviews = _write_role_packets(
        tmp_path,
        [_scoring_case_for_split("test", case_id="m1-sc-test-exact-factual-01")],
    )
    with _ephemeral_repo_output("must-not-exist-028-test-assembled.jsonl") as inside:
        with pytest.raises(ValueError, match="test-split assembled cases"):
            assemble_reviewed_cases(
                questions_path=questions,
                oracle_drafts_path=oracles,
                review_artifact_paths=reviews,
                output_path=inside,
            )
        assert inside.exists() is False


def test_mixed_assembly_containing_test_is_rejected_inside_repository(
    tmp_path: Path,
) -> None:
    questions, oracles, reviews = _write_role_packets(
        tmp_path,
        [
            _scoring_case_for_split("development", case_id="m1-sc-development-exact-factual-01"),
            _scoring_case_for_split("test", case_id="m1-sc-test-exact-factual-01"),
        ],
        prefix="mixed-",
    )
    with _ephemeral_repo_output("must-not-exist-028-mixed-assembled.jsonl") as inside:
        with pytest.raises(ValueError, match="test-split assembled cases"):
            assemble_reviewed_cases(
                questions_path=questions,
                oracle_drafts_path=oracles,
                review_artifact_paths=reviews,
                output_path=inside,
            )
        assert inside.exists() is False


def test_test_containing_assembly_succeeds_outside_repository(tmp_path: Path) -> None:
    questions, oracles, reviews = _write_role_packets(
        tmp_path,
        [_scoring_case_for_split("test", case_id="m1-sc-test-exact-factual-01")],
        prefix="outside-",
    )
    output = tmp_path / "outside" / "test-assembled.jsonl"
    report = assemble_reviewed_cases(
        questions_path=questions,
        oracle_drafts_path=oracles,
        review_artifact_paths=reviews,
        output_path=output,
    )
    assert output.exists()
    assert report["cases"] == 1
    assembled = json.loads(output.read_text(encoding="utf-8"))
    assert assembled["split"] == "test"


def test_development_only_assembly_may_write_inside_repository(tmp_path: Path) -> None:
    questions, oracles, reviews = _write_role_packets(
        tmp_path,
        [_scoring_case_for_split("development", case_id="m1-sc-development-exact-factual-01")],
        prefix="dev-",
    )
    with _ephemeral_repo_output("028-development-only-assembled.jsonl") as inside:
        report = assemble_reviewed_cases(
            questions_path=questions,
            oracle_drafts_path=oracles,
            review_artifact_paths=reviews,
            output_path=inside,
        )
        assert inside.exists()
        assert report["cases"] == 1
        assert report["relative_to_repository"] is True
        assembled = json.loads(inside.read_text(encoding="utf-8"))
        assert assembled["split"] == "development"


def test_dedup_candidate_cli_stdout_omits_question_inventory(tmp_path: Path) -> None:
    cases_path = _complete_scoring_suite(tmp_path / "complete.jsonl")
    cases = [
        json.loads(line)
        for line in cases_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    questions = [str(case["question"]) for case in cases]
    output = tmp_path / "private-candidates.json"
    completed = subprocess.run(
        [
            PYTHON,
            "scripts/build_m1_dedup_candidates.py",
            str(cases_path),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    summary = json.loads(completed.stdout)
    assert set(summary) == DEDUP_CLI_STDOUT_KEYS
    assert summary["output"] == str(output)
    assert summary["case_count"] == 360
    assert summary["exact_cross_split_duplicates"] == 0
    for question in questions:
        assert question not in completed.stdout
        assert question not in completed.stderr
    private = json.loads(output.read_text(encoding="utf-8"))
    assert len(private["question_inventory"]) == 360
    assert private["candidates"]
    assert summary["candidate_count"] == len(private["candidates"])
    assert summary["question_set_sha256"] == private["question_set_sha256"]
    assert summary["scenario_count"] == private["scenario_count"]
