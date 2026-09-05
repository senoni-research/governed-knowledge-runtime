"""Role-separated M1 authoring packets, review assembly, and lexical dedup support.

Question, oracle, and semantic-review packets are distinct inputs. Assembly
copies question fields from the question draft and oracle fields from the
oracle draft after a canonical drift check; it stamps ``oracle_review`` only
from a review artifact that binds the canonical full oracle-draft row.
Content-addressed approvals from multiple review batches may compose; stale
bindings do not approve changed content, and competing current-content reviews
fail closed. No role can silently overwrite another.

Lexical candidate generation is stdlib-only. A later semantic dedup report is
an attestation: this module validates schema, question-set digest, and
candidate completeness. It cannot prove that a model actually performed the
review.

These support schemas are not part of the frozen v3 contract.
"""

from __future__ import annotations

import json
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from gkr.m1_hash import (
    HASH_PROFILE_ID,
    canonical_json_bytes,
    canonical_json_digest,
    canonical_jsonl_bytes,
    dedup_report_digest,
    load_model_family_registry,
    normalize_question,
    prompt_digest,
    question_digest,
    resolve_model_family_id,
    review_artifact_digest,
)
from gkr.m1_io import (
    assert_outside_repository,
    load_jsonl,
    publish_text_files,
    repository_root,
)
from gkr.m1_oracle_validation import validate_m1_oracles

AUTHORING_SUPPORT_DIR = Path("evaluation/m1/authoring-support")
QUESTION_DRAFT_SCHEMA = AUTHORING_SUPPORT_DIR / "question-draft-v1.schema.json"
ORACLE_DRAFT_SCHEMA = AUTHORING_SUPPORT_DIR / "oracle-draft-v1.schema.json"
REVIEW_ARTIFACT_SCHEMA = AUTHORING_SUPPORT_DIR / "semantic-review-artifact-v2.schema.json"
DEDUP_REPORT_SCHEMA = AUTHORING_SUPPORT_DIR / "semantic-dedup-report-v1.schema.json"
QUESTION_DRAFT_VERSION = "gkr-m1-question-draft-v1"
ORACLE_DRAFT_VERSION = "gkr-m1-oracle-draft-v1"
REVIEW_ARTIFACT_VERSION = "gkr-m1-semantic-review-artifact-v2"
REVIEW_BINDING_MANIFEST_VERSION = "gkr-m1-review-binding-manifest-v1"
DEDUP_REPORT_VERSION = "gkr-m1-semantic-dedup-report-v1"
CANDIDATE_REPORT_VERSION = "gkr-m1-dedup-candidate-report-v1"
CASE_VERSION = "gkr-m1-case-v3"
LEXICAL_METHOD_ID = "gkr-m1-lexical-ngram-overlap-v1"
TOKEN_RE = re.compile(r"[a-z0-9]+")
CHAR_NGRAM = 3
NEAREST_PER_FOREIGN_SPLIT = 1
CANDIDATE_SCORE_FLOOR = 0.25
SEMANTIC_REVIEW_SCOPE = (
    "Every listed case's question, scope, evidence sets, claims, citations, "
    "publication decision and disposition were semantically reviewed."
)
DEDUP_ATTESTATION_BOUNDARY = (
    "This artifact attests that a reviewer recorded these dispositions. "
    "Support tooling validates schema, question-set digest, candidate "
    "completeness, and recorded counts. It cannot prove the model actually "
    "performed the review."
)
QUESTION_OWNED_FIELDS = (
    "case_id",
    "scenario_id",
    "variant_id",
    "split",
    "query_class",
    "question",
    "question_sha256",
    "question_authorship",
    "scope",
    "tags",
)
APPROVED_DEDUP_DISPOSITIONS = frozenset(
    {"distinct", "different_intent", "not_semantic_duplicate"}
)


def question_set_digest(cases: Sequence[Mapping[str, Any]]) -> str:
    """SHA-256 of sorted case_id/split/scenario_id/question_sha256 entries."""

    entries = [
        {
            "case_id": str(case["case_id"]),
            "question_sha256": str(case["question_sha256"]),
            "scenario_id": str(case["scenario_id"]),
            "split": str(case["split"]),
        }
        for case in cases
    ]
    ordered = sorted(
        entries,
        key=lambda item: (
            item["case_id"],
            item["split"],
            item["scenario_id"],
            item["question_sha256"],
        ),
    )
    return canonical_json_digest(ordered)


def oracle_draft_digest(case: Mapping[str, Any]) -> str:
    """SHA-256 of one canonical full oracle-draft row."""

    return canonical_json_digest(case)


def oracle_draft_case_set_digest(cases: Sequence[Mapping[str, Any]]) -> str:
    """SHA-256 of sorted case IDs and their canonical oracle-draft digests."""

    entries = [
        {
            "case_id": str(case["case_id"]),
            "oracle_draft_sha256": oracle_draft_digest(case),
        }
        for case in cases
    ]
    ordered = sorted(
        entries,
        key=lambda item: (item["case_id"], item["oracle_draft_sha256"]),
    )
    return canonical_json_digest(ordered)


def review_result_case_set_digest(results: Sequence[Mapping[str, Any]]) -> str:
    """SHA-256 of sorted v2 review-result case/content bindings."""

    entries = [
        {
            "case_id": str(result["case_id"]),
            "oracle_draft_sha256": str(result["reviewed_oracle_draft_sha256"]),
        }
        for result in results
    ]
    ordered = sorted(
        entries,
        key=lambda item: (item["case_id"], item["oracle_draft_sha256"]),
    )
    return canonical_json_digest(ordered)


def build_review_binding_manifest(
    oracle_drafts_path: str | Path,
    *,
    split: str,
    case_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build the content bindings a reviewer copies into one v2 batch."""

    oracles = _load_schema_jsonl(
        oracle_drafts_path,
        ORACLE_DRAFT_SCHEMA,
        ORACLE_DRAFT_VERSION,
    )
    split_rows = [row for row in oracles if str(row["split"]) == split]
    by_case_id = _unique_by_case_id(split_rows, oracle_drafts_path, "oracle draft")
    if case_ids is None:
        selected_ids = sorted(by_case_id)
    else:
        requested = [str(case_id) for case_id in case_ids]
        if len(requested) != len(set(requested)):
            raise ValueError("review binding request contains duplicate case IDs")
        unknown = sorted(set(requested) - set(by_case_id))
        if unknown:
            raise ValueError(
                f"{oracle_drafts_path}: cases are absent from split {split}: "
                + ", ".join(unknown)
            )
        selected_ids = sorted(requested)
    if not selected_ids:
        raise ValueError(f"{oracle_drafts_path}: no oracle drafts selected for split {split}")
    selected = [by_case_id[case_id] for case_id in selected_ids]
    return {
        "schema_version": REVIEW_BINDING_MANIFEST_VERSION,
        "hash_profile_id": HASH_PROFILE_ID,
        "split": split,
        "reviewed_oracle_draft_case_set_sha256": oracle_draft_case_set_digest(selected),
        "case_count": len(selected_ids),
        "case_ids": selected_ids,
        "results": [
            {
                "case_id": case_id,
                "reviewed_oracle_draft_sha256": oracle_draft_digest(by_case_id[case_id]),
            }
            for case_id in selected_ids
        ],
    }


def hash_prompt_file(path: str | Path) -> str:
    text = Path(path).read_text(encoding="utf-8")
    return prompt_digest(text)


def hash_review_file(path: str | Path) -> str:
    return review_artifact_digest(Path(path).read_bytes())


def hash_dedup_file(path: str | Path) -> str:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: dedup hash profile requires one JSON object")
    return dedup_report_digest(payload)


def assemble_reviewed_cases(
    *,
    questions_path: str | Path,
    oracle_drafts_path: str | Path,
    review_artifact_paths: Sequence[str | Path],
    output_path: str | Path,
    corpus_dir: str | Path = "evaluation/m1/corpus",
    require_complete: bool = False,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    questions = _load_schema_jsonl(questions_path, QUESTION_DRAFT_SCHEMA, QUESTION_DRAFT_VERSION)
    oracles = _load_schema_jsonl(oracle_drafts_path, ORACLE_DRAFT_SCHEMA, ORACLE_DRAFT_VERSION)
    artifacts = [_load_review_artifact(path) for path in review_artifact_paths]
    if not artifacts:
        raise ValueError("assembly requires at least one semantic-review artifact")

    question_map = _unique_by_case_id(questions, questions_path, "question draft")
    oracle_map = _unique_by_case_id(oracles, oracle_drafts_path, "oracle draft")
    if set(question_map) != set(oracle_map):
        missing = sorted(set(question_map).symmetric_difference(oracle_map))
        raise ValueError(
            "question and oracle packets must cover the same case_id set; "
            f"drift: {', '.join(missing)}"
        )

    registry = load_model_family_registry()
    for case_id in sorted(question_map):
        _assert_question_field_match(question_map[case_id], oracle_map[case_id], case_id)
    matching_reviews: dict[
        str,
        list[
            tuple[
                dict[str, Any],
                Path,
                str,
                Mapping[str, Any],
            ]
        ],
    ] = {}
    for artifact, raw_path, raw_bytes in artifacts:
        split = str(artifact["split"])
        digest = review_artifact_digest(raw_bytes)
        _assert_review_artifact_integrity(artifact, raw_path)
        artifact_ids = [str(case_id) for case_id in artifact["case_ids"]]
        unknown = sorted(set(artifact_ids) - set(oracle_map))
        if unknown:
            raise ValueError(
                f"{raw_path}: review batch contains unknown case IDs: "
                + ", ".join(unknown)
            )
        wrong_split = sorted(
            case_id
            for case_id in artifact_ids
            if str(oracle_map[case_id]["split"]) != split
        )
        if wrong_split:
            raise ValueError(
                f"{raw_path}: review batch split {split} does not match cases: "
                + ", ".join(wrong_split)
            )
        for result in artifact["results"]:
            case_id = str(result["case_id"])
            if result.get("reviewed_oracle_draft_sha256") != oracle_draft_digest(
                oracle_map[case_id]
            ):
                continue
            matching_reviews.setdefault(case_id, []).append(
                (artifact, raw_path, digest, result)
            )

    assembled: list[dict[str, Any]] = []
    review_digests: dict[str, set[str]] = {}
    scenario_review_digests: dict[str, str] = {}
    missing_review: list[str] = []
    for case_id in sorted(oracle_map):
        matches = matching_reviews.get(case_id, [])
        blockers = [
            (source, result)
            for _artifact, source, _digest, result in matches
            if result.get("status") == "BLOCKED"
        ]
        if blockers:
            sources = ", ".join(str(source) for source, _result in blockers)
            raise ValueError(
                f"{case_id}: current oracle content has a blocked semantic-review "
                f"in {sources}"
            )
        approvals = [
            match for match in matches if match[3].get("status") == "APPROVED"
        ]
        if not approvals:
            missing_review.append(case_id)
            continue
        if len(approvals) != 1:
            sources = ", ".join(str(match[1]) for match in approvals)
            raise ValueError(
                f"{case_id}: current oracle content has multiple matching semantic "
                f"approvals: {sources}"
            )
        artifact, raw_path, digest, result = approvals[0]
        question = question_map[case_id]
        oracle = oracle_map[case_id]
        scenario_id = str(oracle["scenario_id"])
        prior_review_digest = scenario_review_digests.setdefault(scenario_id, digest)
        if prior_review_digest != digest:
            raise ValueError(
                f"{raw_path}: all variants of scenario {scenario_id} must be "
                "covered by the same review artifact"
            )
        review_digests.setdefault(str(oracle["split"]), set()).add(digest)
        assembled.append(
            _stamp_scoring_case(
                question,
                oracle,
                artifact,
                review_sha256=digest,
                registry=registry,
            )
        )

    if missing_review:
        raise ValueError(
            "oracle drafts include cases with no matching approved semantic review: "
            + ", ".join(missing_review)
        )

    assembled.sort(key=lambda case: str(case["case_id"]))
    output = Path(output_path)
    root = Path(repo_root) if repo_root is not None else repository_root()
    if any(str(case["split"]) == "test" for case in assembled):
        assert_outside_repository(output, root, "test-split assembled cases")
    text = canonical_jsonl_bytes(assembled).decode("utf-8")
    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp) / "reviewed.jsonl"
        staged.write_text(text, encoding="utf-8")
        oracle_report = validate_m1_oracles(
            staged,
            corpus_dir=corpus_dir,
            allow_incomplete=not require_complete,
        )
    oracle_report["case_file"] = str(output)
    publish_text_files([(output, text)])
    return {
        "case_file": str(output),
        "cases": len(assembled),
        "review_sha256s_by_split": {
            split: sorted(digests) for split, digests in sorted(review_digests.items())
        },
        "relative_to_repository": _is_inside(output, root),
        "oracle_validation": oracle_report,
        "complete": require_complete,
    }


def build_dedup_candidates(
    cases_path: str | Path,
    *,
    output_path: str | Path,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    cases = [case for _line, case in load_jsonl(cases_path)]
    _assert_complete_three_splits(cases, cases_path)
    exact = _exact_cross_split_duplicates(cases)
    if exact:
        raise ValueError(
            f"{cases_path}: exact normalized cross-split duplicates are forbidden: "
            + "; ".join(exact)
        )
    digest = question_set_digest(cases)
    inventory = _question_inventory(cases)
    candidates = lexical_cross_split_candidates(cases)
    report = {
        "schema_version": CANDIDATE_REPORT_VERSION,
        "hash_profile_id": HASH_PROFILE_ID,
        "question_set_sha256": digest,
        "case_count": len(cases),
        "scenario_count": len({str(case["scenario_id"]) for case in cases}),
        "exact_cross_split_duplicates": 0,
        "method": {
            "id": LEXICAL_METHOD_ID,
            "description": (
                "stdlib-only lexical overlap: Unicode-normalized question tokens "
                f"and character {CHAR_NGRAM}-grams. Score is the mean of token "
                f"Jaccard and character-{CHAR_NGRAM} Jaccard. Each case contributes "
                f"its nearest neighbor in every other split, plus every pair at or "
                f"above score {CANDIDATE_SCORE_FLOOR}."
            ),
            "nearest_per_foreign_split": NEAREST_PER_FOREIGN_SPLIT,
            "score_floor": CANDIDATE_SCORE_FLOOR,
        },
        "question_inventory": inventory,
        "candidates": candidates,
        "semantic_cross_split_candidates_reviewed": False,
        "note": (
            "This report lists deterministic lexical candidates and the complete "
            "sorted question inventory. It does not claim semantic review and does "
            "not set final dedup pass fields."
        ),
        "attestation_boundary": DEDUP_ATTESTATION_BOUNDARY,
    }
    dest = Path(output_path)
    root = Path(repo_root) if repo_root is not None else repository_root()
    assert_outside_repository(dest, root, "dedup candidate report")
    text = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    publish_text_files([(dest, text)])
    return report


def lexical_cross_split_candidates(cases: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Deterministic cross-split nearest-neighbor candidates."""

    indexed = [
        {
            "case_id": str(case["case_id"]),
            "scenario_id": str(case["scenario_id"]),
            "split": str(case["split"]),
            "question": str(case["question"]),
            "normalized": normalize_question(str(case["question"])),
            "tokens": _token_set(normalize_question(str(case["question"]))),
            "ngrams": _char_ngrams(normalize_question(str(case["question"]))),
        }
        for case in cases
    ]
    by_split: dict[str, list[dict[str, Any]]] = {
        "development": [],
        "validation": [],
        "test": [],
    }
    for item in indexed:
        if item["split"] in by_split:
            by_split[item["split"]].append(item)

    pairs: dict[tuple[str, str], dict[str, Any]] = {}
    splits = ("development", "validation", "test")
    for left_split_index, left_split in enumerate(splits):
        for right_split in splits[left_split_index + 1 :]:
            for left in by_split[left_split]:
                scored = [
                    (lexical_overlap_score(left, right), right) for right in by_split[right_split]
                ]
                scored.sort(key=lambda item: (-item[0], item[1]["case_id"]))
                nearest = scored[:NEAREST_PER_FOREIGN_SPLIT]
                extras = [item for item in scored if item[0] >= CANDIDATE_SCORE_FLOOR]
                for score, right in nearest + extras:
                    _record_candidate_pair(pairs, left, right, score)
            for right in by_split[right_split]:
                scored = [
                    (lexical_overlap_score(right, left), left) for left in by_split[left_split]
                ]
                scored.sort(key=lambda item: (-item[0], item[1]["case_id"]))
                for score, left in scored[:NEAREST_PER_FOREIGN_SPLIT]:
                    _record_candidate_pair(pairs, left, right, score)
    return [
        pairs[key]
        for key in sorted(
            pairs,
            key=lambda item: (-pairs[item]["score"], item[0], item[1]),
        )
    ]


def lexical_overlap_score(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    token_score = _jaccard(left["tokens"], right["tokens"])
    ngram_score = _jaccard(left["ngrams"], right["ngrams"])
    return (token_score + ngram_score) / 2.0


def load_semantic_dedup_report(
    path: str | Path,
    *,
    cases: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], str]:
    """Load, schema-validate, and canonical-hash a semantic dedup report."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: semantic dedup report must be one JSON object")
    _validate_schema(payload, DEDUP_REPORT_SCHEMA, source=path)
    generated = lexical_cross_split_candidates(cases)
    validate_semantic_dedup_report(payload, cases=cases, generated_candidates=generated)
    return payload, dedup_report_digest(payload)


def validate_semantic_dedup_report(
    report: Mapping[str, Any],
    *,
    cases: Sequence[Mapping[str, Any]],
    generated_candidates: Sequence[Mapping[str, Any]],
) -> None:
    errors: list[str] = []
    expected_digest = question_set_digest(cases)
    if report.get("question_set_sha256") != expected_digest:
        errors.append(
            "semantic dedup report question_set_sha256 does not match the "
            "assembled scoring cases"
        )
    if report.get("exact_cross_split_duplicates") != 0:
        errors.append("semantic dedup report exact_cross_split_duplicates must be 0")
    if report.get("semantic_cross_split_candidates_reviewed") is not True:
        errors.append("semantic dedup report must record candidates reviewed")
    if report.get("all_scenarios_reviewed_for_semantic_cross_split_duplication") is not True:
        errors.append("semantic dedup report must record review of all 360 scenarios")
    if report.get("unresolved_semantic_duplicates") != 0:
        errors.append("semantic dedup report unresolved_semantic_duplicates must be 0")
    if len({str(case["scenario_id"]) for case in cases}) != 360:
        errors.append("semantic dedup report requires 360 independent scenarios")

    recorded = {
        _pair_key(item["left_case_id"], item["right_case_id"]): item
        for item in report.get("candidates", [])
        if isinstance(item, Mapping)
    }
    for candidate in generated_candidates:
        key = _pair_key(candidate["left_case_id"], candidate["right_case_id"])
        item = recorded.get(key)
        if item is None:
            errors.append(
                "semantic dedup report is missing a disposition for generated "
                f"candidate {key[0]} / {key[1]}"
            )
            continue
        disposition = item.get("disposition")
        if disposition not in APPROVED_DEDUP_DISPOSITIONS:
            errors.append(
                f"semantic dedup candidate {key[0]} / {key[1]} is not an approved "
                "distinct disposition"
            )
    if errors:
        raise ValueError("\n".join(errors))


def _load_schema_jsonl(
    path: str | Path,
    schema_path: Path,
    schema_version: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, row in load_jsonl(path):
        if row.get("schema_version") != schema_version:
            raise ValueError(f"{path}:{line_number}: expected schema_version {schema_version}")
        _validate_schema(row, schema_path, source=f"{path}:{line_number}")
        if row.get("hash_profile_id") != HASH_PROFILE_ID:
            raise ValueError(f"{path}:{line_number}: hash_profile_id must be {HASH_PROFILE_ID}")
        question = row.get("question")
        if isinstance(question, str) and row.get("question_sha256") != question_digest(question):
            raise ValueError(f"{path}:{line_number}: question_sha256 does not match question")
        rows.append(row)
    return rows


def _load_review_artifact(path: str | Path) -> tuple[dict[str, Any], Path, bytes]:
    raw_path = Path(path)
    raw_bytes = raw_path.read_bytes()
    try:
        payload = json.loads(raw_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: review artifact must be UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: review artifact must be one JSON object")
    if "review_sha256" in payload:
        raise ValueError(f"{path}: review artifact must not contain its own digest")
    if payload.get("schema_version") != REVIEW_ARTIFACT_VERSION:
        raise ValueError(
            f"{path}: expected {REVIEW_ARTIFACT_VERSION}; legacy v1 review "
            "artifacts do not bind oracle content"
        )
    _validate_schema(payload, REVIEW_ARTIFACT_SCHEMA, source=path)
    if payload.get("semantic_review_scope") != SEMANTIC_REVIEW_SCOPE:
        raise ValueError(f"{path}: review artifact is missing the required semantic-review scope")
    case_ids = [str(item) for item in payload.get("case_ids", [])]
    if case_ids != sorted(case_ids):
        raise ValueError(f"{path}: review artifact case_ids must be sorted")
    if payload.get("case_count") != len(case_ids):
        raise ValueError(f"{path}: review artifact case_count must equal sorted case_ids")
    result_ids = [str(item.get("case_id", "")) for item in payload.get("results", [])]
    if len(result_ids) != len(set(result_ids)):
        raise ValueError(f"{path}: review artifact has duplicate case results")
    if result_ids != case_ids:
        raise ValueError(
            f"{path}: review results must contain exactly the sorted case_ids in order"
        )
    has_blocked_result = any(
        item.get("status") == "BLOCKED" for item in payload.get("results", [])
    )
    expected_status = "BLOCKED" if has_blocked_result else "APPROVED"
    if payload.get("overall_status") != expected_status:
        raise ValueError(
            f"{path}: overall_status must be {expected_status} for its result statuses"
        )
    return payload, raw_path, raw_bytes


def _unique_by_case_id(
    rows: Sequence[Mapping[str, Any]],
    source: str | Path,
    label: str,
) -> dict[str, dict[str, Any]]:
    mapped: dict[str, dict[str, Any]] = {}
    for row in rows:
        case_id = str(row["case_id"])
        if case_id in mapped:
            raise ValueError(f"{source}: duplicate {label} case_id {case_id}")
        mapped[case_id] = dict(row)
    return mapped


def _assert_question_field_match(
    question: Mapping[str, Any],
    oracle: Mapping[str, Any],
    case_id: str,
) -> None:
    for field in QUESTION_OWNED_FIELDS:
        left = question.get(field, _MISSING)
        right = oracle.get(field, _MISSING)
        if left is _MISSING and right is _MISSING:
            continue
        if canonical_json_bytes(left) != canonical_json_bytes(right):
            raise ValueError(
                f"case {case_id}: question-field drift on {field} between "
                "question and oracle packets"
            )


def _assert_review_artifact_integrity(
    artifact: Mapping[str, Any],
    source: str | Path,
) -> None:
    results = [
        item for item in artifact["results"] if isinstance(item, Mapping)
    ]
    expected_digest = review_result_case_set_digest(results)
    if artifact.get("reviewed_oracle_draft_case_set_sha256") != expected_digest:
        raise ValueError(
            f"{source}: reviewed oracle-draft case-set digest does not match its "
            "result bindings"
        )


def _stamp_scoring_case(
    question: Mapping[str, Any],
    oracle: Mapping[str, Any],
    artifact: Mapping[str, Any],
    *,
    review_sha256: str,
    registry: Mapping[str, Any],
) -> dict[str, Any]:
    question_role = dict(question["question_authorship"])
    oracle_role = dict(oracle["oracle_authorship"])
    sessions = (
        str(question_role.get("session_id") or ""),
        str(oracle_role.get("session_id") or ""),
        str(artifact.get("reviewer_session_id") or ""),
    )
    if any(not session for session in sessions) or len(set(sessions)) != 3:
        raise ValueError(
            f"{question['case_id']}: question, oracle, and reviewer session IDs "
            "must be pairwise distinct"
        )
    author_families = {
        _resolve_role_family(question_role, question["case_id"], "question_authorship", registry),
        _resolve_role_family(oracle_role, question["case_id"], "oracle_authorship", registry),
    }
    review = {
        "status": "completed",
        "semantically_reviewed": True,
        "reviewer_kind": artifact["reviewer_kind"],
        "reviewer_identity": artifact["reviewer_identity"],
        "reviewer_session_id": artifact["reviewer_session_id"],
        "reviewer_model_family_id": artifact["reviewer_model_family_id"],
        "reviewer_model_id": artifact["reviewer_model_id"],
        "reviewer_model_revision": artifact["reviewer_model_revision"],
        "reviewer_prompt_sha256": artifact["reviewer_prompt_sha256"],
        "review_sha256": review_sha256,
        "independent_from_retriever_tuning": True,
    }
    if review["reviewer_kind"] == "model":
        reviewer_family = resolve_model_family_id(
            review["reviewer_model_family_id"],
            review["reviewer_model_id"],
            registry=registry,
        )
        if reviewer_family in author_families:
            raise ValueError(
                f"{question['case_id']}: reviewer model family must differ from "
                "both author families"
            )
    case = {
        "schema_version": CASE_VERSION,
        "hash_profile_id": HASH_PROFILE_ID,
        "case_id": question["case_id"],
        "scenario_id": question["scenario_id"],
        "variant_id": question["variant_id"],
        "split": question["split"],
        "query_class": question["query_class"],
        "case_kind": "scoring",
        "question": question["question"],
        "question_sha256": question["question_sha256"],
        "question_authorship": question_role,
        "oracle_authorship": oracle_role,
        "oracle_review": review,
        "scope": oracle["scope"],
        "oracle": oracle["oracle"],
    }
    if "tags" in question:
        case["tags"] = question["tags"]
    if "oracle_notes" in oracle:
        case["oracle_notes"] = oracle["oracle_notes"]
    return case


def _resolve_role_family(
    role: Mapping[str, Any],
    case_id: str,
    role_name: str,
    registry: Mapping[str, Any],
) -> str | None:
    family = role.get("model_family_id")
    if family is None:
        return None
    try:
        return resolve_model_family_id(family, role.get("model_id"), registry=registry)
    except ValueError as exc:
        raise ValueError(f"{case_id}: {role_name} {exc}") from exc


def _assert_complete_three_splits(cases: Sequence[Mapping[str, Any]], source: str | Path) -> None:
    from collections import defaultdict

    scenarios: dict[str, set[str]] = defaultdict(set)
    classes: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for case in cases:
        split = str(case.get("split", ""))
        scenarios[split].add(str(case.get("scenario_id", "")))
        classes[split][str(case.get("query_class", ""))].add(str(case.get("scenario_id", "")))
    errors: list[str] = []
    for split in ("development", "validation", "test"):
        count = len(scenarios.get(split, set()))
        if count != 120:
            errors.append(f"{source}: split {split} has {count} scenarios; expected 120")
        for query_class in (
            "exact_factual",
            "semantic_paraphrase",
            "numeric_conditional",
            "temporal",
            "authorization",
            "unknown_oos",
            "multi_record",
            "adversarial_conflicting",
        ):
            class_count = len(classes.get(split, {}).get(query_class, set()))
            if class_count != 15:
                errors.append(
                    f"{source}: split {split} class {query_class} has "
                    f"{class_count} scenarios; expected 15"
                )
    if errors:
        raise ValueError("\n".join(errors))


def _exact_cross_split_duplicates(cases: Sequence[Mapping[str, Any]]) -> list[str]:
    seen: dict[str, tuple[str, str]] = {}
    errors: list[str] = []
    for case in cases:
        digest = str(case.get("question_sha256") or question_digest(str(case.get("question", ""))))
        split = str(case.get("split", ""))
        case_id = str(case.get("case_id", ""))
        previous = seen.get(digest)
        if previous is None:
            seen[digest] = (split, case_id)
            continue
        previous_split, previous_id = previous
        if previous_split != split:
            errors.append(f"{previous_id} ({previous_split}) == {case_id} ({split})")
    return errors


def _question_inventory(cases: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    inventory = [
        {
            "case_id": str(case["case_id"]),
            "query_class": str(case["query_class"]),
            "question": str(case["question"]),
            "question_sha256": str(case["question_sha256"]),
            "scenario_id": str(case["scenario_id"]),
            "split": str(case["split"]),
        }
        for case in cases
    ]
    inventory.sort(key=lambda item: item["case_id"])
    return inventory


def _record_candidate_pair(
    pairs: dict[tuple[str, str], dict[str, Any]],
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    score: float,
) -> None:
    key = _pair_key(left["case_id"], right["case_id"])
    first, second = (left, right) if left["case_id"] == key[0] else (right, left)
    current = pairs.get(key)
    rounded = round(float(score), 12)
    if current is None or rounded > current["score"]:
        pairs[key] = {
            "left_case_id": first["case_id"],
            "left_scenario_id": first["scenario_id"],
            "left_split": first["split"],
            "right_case_id": second["case_id"],
            "right_scenario_id": second["scenario_id"],
            "right_split": second["split"],
            "score": rounded,
        }


def _pair_key(left: object, right: object) -> tuple[str, str]:
    first, second = sorted((str(left), str(right)))
    return first, second


def _token_set(normalized: str) -> frozenset[str]:
    return frozenset(TOKEN_RE.findall(normalized))


def _char_ngrams(normalized: str) -> frozenset[str]:
    compact = normalized.replace(" ", "")
    if len(compact) < CHAR_NGRAM:
        return frozenset([compact] if compact else [])
    return frozenset(compact[index : index + CHAR_NGRAM] for index in range(len(compact) - 2))


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _validate_schema(value: Mapping[str, Any], schema_path: Path, *, source: str | Path) -> None:
    try:
        from jsonschema import Draft202012Validator, FormatChecker
    except ImportError as exc:
        raise RuntimeError("M1 authoring requires the development dependencies") from exc

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    errors = [
        f"{source}:{'.'.join(str(part) for part in error.absolute_path) or '<root>'}: "
        f"{error.message}"
        for error in Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(
            value
        )
    ]
    if errors:
        raise ValueError("\n".join(errors))


def _is_inside(path: Path, repo_root: Path) -> bool:
    try:
        path.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return False
    return True


_MISSING = object()
