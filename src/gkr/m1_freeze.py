"""Corpus-freeze and suite-staging helpers for M1 v3 cases.

Salted question commitments follow gkr-m1-hash-profile-v1: Unicode NFKC, Unicode
whitespace runs to U+0020, trim, Unicode case-fold, then SHA-256 of
``salt_bytes || utf8(normalized_question)`` with a 32-byte salt.

Plaintext full test cases plus salts are a ``plaintext_staging_bundle``.
Development and validation remain ``externally_encrypted_artifact_bound=false``.
Scoring-suite finalization requires a caller-supplied age-x25519-v1 encrypted
artifact descriptor; this module verifies the artifact exists outside the
repository and matches the declared SHA-256 of exact ciphertext bytes. It does
not implement or decrypt age and does not prove the ciphertext decrypts to the
staged bytes.

``dedup_report_sha256`` is an attestation digest. This freeze API accepts an
externally computed 64-hex value and binds it on the suite manifest. It does
not load, parse, or rehash a dedup-report file. Authors must compute that
digest from the gkr-m1-hash-profile-v1 canonical JSON object preimage.

Gate 1 binds no retrieval-configuration digest. A later Gate 2 execution
contract must be created, frozen from validation, and bound before test
opening.

The in-memory suite manifest is built and schema-validated before any
plaintext staging or public file is published. Schema or constructor failure
before publish leaves no outputs. Replacing a pre-existing staging/public pair
is not a pair transaction.

Conformance mode may write staging under a temporary directory and must return
``scoring_suite=false``. It never emits a suite manifest. Incomplete or
validation-failure paths also emit no suite manifest.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from gkr.m1_corpus import content_digest, load_authority_records
from gkr.m1_hash import (
    HASH_PROFILE_ID,
    canonical_jsonl_digest,
    is_sha256_hex,
    load_model_family_registry,
    normalize_question,
    question_list_digest,
    resolve_model_family_id,
)
from gkr.m1_oracle_validation import (
    applicability_partition_errors,
    evidence_applicability_counts,
    load_case_jsonl,
    validate_m1_oracles,
)

SALT_SIZE = 32
STAGING_RECORD_VERSION = "gkr-m1-plaintext-staging-record-v1"
PUBLIC_CASE_VERSION = "gkr-m1-test-public-case-v3"
PUBLIC_SCHEMA_PATH = Path("evaluation/m1/test-public-case-v3.schema.json")
SUITE_SCHEMA_PATH = Path("evaluation/m1/suite-manifest-v3.schema.json")
CORPUS_FREEZE_PATH = Path("evaluation/m1/corpus-freeze-manifest.json")
SUITE_ID = "gkr-m1-scoring-suite-v1"
AUTHORITY_JSONL_SHA256 = "3325db848fe13a2f0f3e7b2e0894e92d9b27e99c42a16b3c90f03b40a39c81a2"
CONTENT_DIGEST_SHA256 = "1a30668d55910c544fa00aa6927ad7dc3064f6585a6186bc543fc590dec64300"
CORPUS_FREEZE_MANIFEST_SHA256 = (
    "961e91927798ddfecf55133591481ddf6f5d181729296c21937ff036b2563a1a"
)
MODES = {"scoring", "conformance"}
ENCRYPTION_PROFILE_ID = "age-x25519-v1"
TOOL_FAMILY = "age"
ENCRYPTED_ARTIFACT_FIELDS = (
    "encryption_profile_id",
    "tool_family",
    "tool_version",
    "recipient_key_fingerprint_sha256",
    "encrypted_artifact_path",
    "encrypted_artifact_sha256",
)
_PROHIBITED_ENCRYPTION_TOKENS = frozenset(
    {
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
    }
)
ENCRYPTION_BOUNDARY = (
    "The encrypted artifact is produced by an external age-x25519-v1 "
    "authenticated-encryption tool. This runtime verifies the artifact exists "
    "outside the repository, matches encryption profile age-x25519-v1, and "
    "matches the declared SHA-256 of exact ciphertext bytes. It does not "
    "implement or decrypt age and does not prove the ciphertext decrypts to "
    "the plaintext staging bytes."
)
_GENERATED_PUBLIC_SOURCE = "<generated>"


def question_commitment(salt: bytes, question: str) -> str:
    if len(salt) != SALT_SIZE:
        raise ValueError(f"question commitment salt must be {SALT_SIZE} bytes")
    payload = salt + normalize_question(question).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def freeze_m1_suite(
    case_path: str | Path,
    *,
    mode: str,
    plaintext_staging_path: str | Path,
    public_test_path: str | Path,
    encrypted_artifact: Mapping[str, str] | None = None,
    dedup_report_sha256: str | None = None,
    corpus_dir: str | Path = "evaluation/m1/corpus",
    salt_factory: Callable[[], bytes] | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Validate cases, write plaintext staging, and emit the public redacted form."""

    if mode not in MODES:
        raise ValueError("mode must be 'scoring' or 'conformance'")

    cases = load_case_jsonl(case_path)
    root = Path(repo_root) if repo_root is not None else repository_root()
    staging_path = Path(plaintext_staging_path)
    public_path = Path(public_test_path)
    _assert_outside_repository(staging_path, root, "plaintext staging bundle")

    _assert_mode_matches_case_kind(cases, mode, case_path)
    if mode == "scoring":
        _assert_scoring_freeze_eligibility(cases, case_path)

    encrypted = None
    if mode == "scoring":
        encrypted = _require_encrypted_artifact(encrypted_artifact, root)
        if not isinstance(dedup_report_sha256, str) or not is_sha256_hex(dedup_report_sha256):
            raise ValueError(
                "scoring freeze requires dedup_report_sha256 as a 64-character hex digest"
            )
        encrypted = {
            **encrypted,
            "dedup_report_sha256": dedup_report_sha256,
        }
        _require_corpus_identity(corpus_dir)
    elif encrypted_artifact is not None:
        raise ValueError("conformance mode must not supply an encrypted artifact descriptor")

    oracle_report = validate_m1_oracles(
        case_path,
        corpus_dir=corpus_dir,
        allow_incomplete=mode == "conformance",
    )

    factory = salt_factory or (lambda: secrets.token_bytes(SALT_SIZE))
    test_records, public_cases, salts = _stage_test_split(cases, factory)
    if not test_records:
        raise ValueError(f"{case_path}: freeze requires at least one test-split case")

    staging_text = _jsonl([record for _line, record in test_records])
    public_text = _jsonl(public_cases)
    _validate_public_cases(public_cases, source=_GENERATED_PUBLIC_SOURCE)
    split_records = _split_freeze_records(cases, public_cases, salts)
    split_records["test"]["externally_encrypted_artifact_bound"] = False

    suite_manifest = None
    bound = False
    if mode == "scoring" and encrypted is not None:
        bound = True
        split_records["development"]["externally_encrypted_artifact_bound"] = False
        split_records["validation"]["externally_encrypted_artifact_bound"] = False
        split_records["test"]["externally_encrypted_artifact_bound"] = True
        suite_manifest = _scoring_suite_manifest(
            cases,
            split_records,
            encrypted=encrypted,
            corpus_dir=corpus_dir,
        )
        _validate_suite_manifest(suite_manifest)

    _publish_text_files(
        [
            (staging_path, staging_text),
            (public_path, public_text),
        ],
        public_cases=public_cases,
        public_text=public_text,
    )

    staging_digest = hashlib.sha256(staging_text.encode("utf-8")).hexdigest()
    public_digest = hashlib.sha256(public_text.encode("utf-8")).hexdigest()
    report = {
        "case_file": str(case_path),
        "mode": mode,
        "conformance_fixture": mode == "conformance",
        "scoring_suite": mode == "scoring",
        "externally_encrypted_artifact_bound": bound,
        "oracle_validation": oracle_report,
        "splits": split_records,
        "plaintext_staging_path": str(staging_path),
        "plaintext_staging_sha256": staging_digest,
        "public_test_path": str(public_path),
        "public_test_sha256": public_digest,
        "suite_manifest": suite_manifest,
        "semantic_support_established": False,
        "note": (
            "Plaintext staging binds case bytes, salted question commitments, and split "
            "counts. It is not a sealed bundle. Semantic support and Gate 1 passage of "
            "the repository Gate 1 programme remain not established by a conformance freeze. "
            "A scoring freeze records suite status=frozen and gate_1_status=passed for "
            "construction/validation only."
        ),
    }
    if encrypted is not None:
        report["encrypted_artifact_sha256"] = encrypted["encrypted_artifact_sha256"]
        report["encryption_boundary"] = ENCRYPTION_BOUNDARY
        report["encryption_profile_id"] = ENCRYPTION_PROFILE_ID
    if report["externally_encrypted_artifact_bound"] and mode != "scoring":
        raise RuntimeError("plaintext staging cannot be marked externally encrypted")
    return report


def verify_question_commitments(
    *,
    public_test_path: str | Path,
    plaintext_staging_path: str | Path,
) -> dict[str, Any]:
    """Recompute commitments from staging salts and compare with the public file."""

    public_cases = {case["case_id"]: case for _line, case in load_case_jsonl(public_test_path)}
    bundle = load_case_jsonl(plaintext_staging_path)
    errors: list[str] = []
    if set(public_cases) != {record["case"]["case_id"] for _line, record in bundle}:
        errors.append(
            f"{public_test_path}: public case_id set does not match the plaintext staging bundle"
        )
    for line_number, record in bundle:
        prefix = f"{plaintext_staging_path}:{line_number}"
        case = record.get("case")
        salt_hex = record.get("salt_hex")
        if record.get("schema_version") != STAGING_RECORD_VERSION:
            errors.append(f"{prefix}: plaintext staging record has an unexpected schema_version")
        if not isinstance(case, dict) or not isinstance(salt_hex, str):
            errors.append(f"{prefix}: staging record must contain case and salt_hex")
            continue
        try:
            salt = bytes.fromhex(salt_hex)
        except ValueError:
            errors.append(f"{prefix}: salt_hex is not valid hex")
            continue
        expected = question_commitment(salt, str(case["question"]))
        public = public_cases.get(case["case_id"])
        if public is None:
            errors.append(f"{prefix}: case_id {case['case_id']} is missing from the public file")
            continue
        if public.get("question_commitment_sha256") != expected:
            errors.append(
                f"{public_test_path}: commitment mismatch for case_id {case['case_id']}"
            )
    if errors:
        raise ValueError("\n".join(errors))
    return {
        "cases": len(public_cases),
        "commitments_match": True,
        "plaintext_staging_sha256": hashlib.sha256(
            Path(plaintext_staging_path).read_bytes()
        ).hexdigest(),
        "externally_encrypted_artifact_bound": False,
    }


def verify_staging_bundle_digest(path: str | Path, expected_sha256: str) -> str:
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise ValueError(f"{path}: plaintext staging digest {digest} != {expected_sha256}")
    return digest


def _assert_mode_matches_case_kind(
    cases: list[tuple[int, dict[str, Any]]],
    mode: str,
    case_path: str | Path,
) -> None:
    kinds = [case.get("case_kind") for _line, case in cases]
    if not kinds:
        raise ValueError(f"{case_path}: freeze requires at least one case")
    if any(kind != mode for kind in kinds):
        if mode == "conformance":
            raise ValueError(
                f"{case_path}: conformance mode requires every case to have "
                "case_kind=conformance; tags cannot select conformance mode"
            )
        raise ValueError(
            f"{case_path}: scoring mode rejects any case with case_kind=conformance; "
            "tags cannot bypass scoring requirements"
        )


def _assert_scoring_freeze_eligibility(
    cases: list[tuple[int, dict[str, Any]]],
    case_path: str | Path,
) -> None:
    errors: list[str] = []
    for line_number, case in cases:
        prefix = f"{case_path}:{line_number}"
        if case.get("case_kind") != "scoring":
            errors.append(f"{prefix}: scoring freeze rejects conformance cases")
            continue
        review = case.get("oracle_review")
        question = case.get("question_authorship")
        oracle = case.get("oracle_authorship")
        if not isinstance(review, dict):
            errors.append(f"{prefix}: scoring freeze requires oracle_review")
            continue
        if review.get("status") != "completed" or review.get("semantically_reviewed") is not True:
            errors.append(f"{prefix}: scoring freeze rejects pending or incomplete semantic review")
        if not isinstance(question, dict) or not isinstance(oracle, dict):
            errors.append(f"{prefix}: scoring freeze requires question and oracle authorship")
            continue
        sessions = (
            str(question.get("session_id") or ""),
            str(oracle.get("session_id") or ""),
            str(review.get("reviewer_session_id") or ""),
        )
        if any(not session for session in sessions) or len(set(sessions)) != 3:
            errors.append(
                f"{prefix}: scoring freeze requires pairwise-distinct author and reviewer sessions"
            )
        if review.get("reviewer_kind") == "model":
            try:
                registry = load_model_family_registry()
                reviewer_family = resolve_model_family_id(
                    review.get("reviewer_model_family_id"),
                    review.get("reviewer_model_id"),
                    registry=registry,
                )
                author_families = {
                    resolve_model_family_id(
                        question.get("model_family_id"),
                        question.get("model_id"),
                        registry=registry,
                    ),
                    resolve_model_family_id(
                        oracle.get("model_family_id"),
                        oracle.get("model_id"),
                        registry=registry,
                    ),
                }
            except ValueError as exc:
                errors.append(f"{prefix}: scoring freeze {exc}")
            else:
                if reviewer_family in author_families:
                    errors.append(
                        f"{prefix}: scoring freeze rejects a reviewer model family "
                        "that matches either author family"
                    )
        if case.get("split") == "test":
            roles = (
                ("question_authorship", question.get("independent_from_retriever_tuning")),
                ("oracle_authorship", oracle.get("independent_from_retriever_tuning")),
                ("oracle_review", review.get("independent_from_retriever_tuning")),
            )
            for role_name, independent in roles:
                if independent is not True:
                    errors.append(
                        f"{prefix}: scoring freeze rejects a test-split {role_name} "
                        "role that is not independent from retriever tuning"
                    )
    if errors:
        raise ValueError("\n".join(errors))


def _require_encrypted_artifact(
    encrypted_artifact: Mapping[str, str] | None,
    repo_root: Path,
) -> dict[str, str]:
    if encrypted_artifact is None:
        raise ValueError(
            "scoring freeze requires an encrypted artifact descriptor "
            "(encryption_profile_id=age-x25519-v1, tool_family=age, tool_version, "
            "recipient_key_fingerprint_sha256, encrypted_artifact_path, "
            "encrypted_artifact_sha256)"
        )
    lowered = {str(value).strip().lower() for value in encrypted_artifact.values() if value}
    prohibited = sorted(lowered.intersection(_PROHIBITED_ENCRYPTION_TOKENS))
    if prohibited:
        raise ValueError(
            "scoring freeze rejects plaintext, cat, none, ZIP/password, archive, "
            f"or unknown encryption tokens: {', '.join(prohibited)}"
        )
    for legacy in ("scheme", "tool", "recipient_key_fingerprint", "sealed"):
        if legacy in encrypted_artifact:
            raise ValueError(
                "scoring freeze rejects legacy encryption fields scheme/tool/"
                "recipient_key_fingerprint/sealed; use encryption_profile_id "
                "age-x25519-v1"
            )
    missing = [field for field in ENCRYPTED_ARTIFACT_FIELDS if not encrypted_artifact.get(field)]
    if missing:
        raise ValueError(
            "scoring freeze encrypted artifact descriptor is missing " + ", ".join(missing)
        )
    profile_id = str(encrypted_artifact["encryption_profile_id"])
    tool_family = str(encrypted_artifact["tool_family"])
    tool_version = str(encrypted_artifact["tool_version"]).strip()
    fingerprint = str(encrypted_artifact["recipient_key_fingerprint_sha256"])
    if profile_id != ENCRYPTION_PROFILE_ID:
        raise ValueError(
            f"scoring freeze encryption_profile_id must be {ENCRYPTION_PROFILE_ID}"
        )
    if tool_family != TOOL_FAMILY:
        raise ValueError(f"scoring freeze tool_family must be {TOOL_FAMILY}")
    if not tool_version:
        raise ValueError("scoring freeze tool_version must be nonempty")
    if not is_sha256_hex(fingerprint):
        raise ValueError(
            "scoring freeze recipient_key_fingerprint_sha256 must be 64 lowercase hex"
        )
    artifact_path = Path(str(encrypted_artifact["encrypted_artifact_path"]))
    _assert_outside_repository(artifact_path, repo_root, "encrypted artifact")
    if not artifact_path.is_file():
        raise ValueError(f"{artifact_path}: encrypted artifact does not exist")
    digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    expected = str(encrypted_artifact["encrypted_artifact_sha256"])
    if not is_sha256_hex(expected) or digest != expected:
        raise ValueError(
            f"{artifact_path}: encrypted artifact digest {digest} != {expected}"
        )
    return {
        "encryption_profile_id": ENCRYPTION_PROFILE_ID,
        "tool_family": TOOL_FAMILY,
        "tool_version": tool_version,
        "recipient_key_fingerprint_sha256": fingerprint,
        "encrypted_artifact_path": str(artifact_path),
        "encrypted_artifact_sha256": expected,
    }


def _stage_test_split(
    cases: list[tuple[int, dict[str, Any]]],
    salt_factory: Callable[[], bytes],
) -> tuple[list[tuple[int, dict[str, Any]]], list[dict[str, Any]], dict[str, bytes]]:
    staged: list[tuple[int, dict[str, Any]]] = []
    public_cases: list[dict[str, Any]] = []
    salts: dict[str, bytes] = {}
    for line_number, case in cases:
        if case.get("split") != "test":
            continue
        salt = salt_factory()
        if len(salt) != SALT_SIZE:
            raise ValueError("salt_factory must return 32 bytes")
        salts[str(case["case_id"])] = salt
        staged.append(
            (
                line_number,
                {
                    "schema_version": STAGING_RECORD_VERSION,
                    "salt_hex": salt.hex(),
                    "case": case,
                },
            )
        )
        public_cases.append(_redact_test_case(case, salt))
    return staged, public_cases, salts


def _redact_test_case(case: dict[str, Any], salt: bytes) -> dict[str, Any]:
    return {
        "schema_version": PUBLIC_CASE_VERSION,
        "hash_profile_id": HASH_PROFILE_ID,
        "case_id": case["case_id"],
        "scenario_id": case["scenario_id"],
        "variant_id": case["variant_id"],
        "split": "test",
        "query_class": case["query_class"],
        "scope": case["scope"],
        "question_commitment_sha256": question_commitment(salt, str(case["question"])),
    }


def _validate_public_cases(cases: list[dict[str, Any]], *, source: str | Path) -> None:
    try:
        from jsonschema import Draft202012Validator, FormatChecker
    except ImportError as exc:
        raise RuntimeError("M1 freeze requires the development dependencies") from exc

    schema = json.loads(PUBLIC_SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors: list[str] = []
    for index, case in enumerate(cases, start=1):
        for error in validator.iter_errors(case):
            location = ".".join(str(part) for part in error.absolute_path)
            errors.append(f"{source}:{index}:{location or '<root>'}: {error.message}")
        for forbidden in (
            "question",
            "oracle",
            "oracle_notes",
            "oracle_review",
            "question_authorship",
            "oracle_authorship",
            "authorship",
            "salt_hex",
            "encrypted_artifact_path",
        ):
            if forbidden in case:
                errors.append(f"{source}:{index}: public test case must exclude {forbidden}")
    if errors:
        raise ValueError("\n".join(errors))


def _split_freeze_records(
    cases: list[tuple[int, dict[str, Any]]],
    public_cases: list[dict[str, Any]],
    _salts: dict[str, bytes],
) -> dict[str, dict[str, Any]]:
    """Build per-split digests using gkr-m1-hash-profile-v1 case_id order.

    Development and validation ``cases_sha256`` are canonical full-case JSONL.
    Test binds only redacted public-case JSONL and ordered salted commitments.
    """

    by_split: dict[str, list[dict[str, Any]]] = {
        "development": [],
        "validation": [],
        "test": [],
    }
    scenarios: dict[str, set[str]] = {split: set() for split in by_split}
    for _line, case in cases:
        split = str(case.get("split", ""))
        if split not in by_split:
            continue
        by_split[split].append(case)
        scenarios[split].add(str(case.get("scenario_id", "")))

    commitments = {
        str(case["case_id"]): str(case["question_commitment_sha256"]) for case in public_cases
    }
    records: dict[str, dict[str, Any]] = {}
    for split, split_cases in by_split.items():
        if split == "test":
            records[split] = {
                "independent_scenario_count": len(scenarios[split]),
                "case_count": len(split_cases),
                "public_case_count": len(public_cases),
                "public_cases_sha256": canonical_jsonl_digest(public_cases),
                "question_commitments_sha256": question_list_digest(
                    [
                        (str(case["case_id"]), commitments[str(case["case_id"])])
                        for case in split_cases
                    ]
                ),
                "externally_encrypted_artifact_bound": False,
            }
            continue
        records[split] = {
            "independent_scenario_count": len(scenarios[split]),
            "case_count": len(split_cases),
            "cases_sha256": canonical_jsonl_digest(split_cases),
            "questions_sha256": question_list_digest(
                [
                    (str(case["case_id"]), str(case.get("question_sha256", "")))
                    for case in split_cases
                ]
            ),
            "externally_encrypted_artifact_bound": False,
        }
    return records


def _scoring_suite_manifest(
    cases: list[tuple[int, dict[str, Any]]],
    split_records: dict[str, dict[str, Any]],
    *,
    encrypted: Mapping[str, str],
    corpus_dir: str | Path,
) -> dict[str, Any]:
    freeze = _require_corpus_identity(corpus_dir)
    return {
        "schema_version": "gkr-m1-suite-manifest-v3",
        "suite_id": SUITE_ID,
        "status": "frozen",
        "case_kind": "scoring",
        "case_schema_version": "gkr-m1-case-v3",
        "metric_contract_version": "gkr-m1-retrieval-metrics-v3",
        "hash_profile_id": HASH_PROFILE_ID,
        "gate_1_status": "passed",
        "semantic_review": {
            "all_completed": True,
            "all_semantically_reviewed": True,
        },
        "deterministic_oracle_validation": {"passed": True},
        "corpus": {
            "stable_record_count": freeze["stable_record_count"],
            "authority_event_count": freeze["authority_event_count"],
            "authority_jsonl_sha256": AUTHORITY_JSONL_SHA256,
            "content_digest_sha256": CONTENT_DIGEST_SHA256,
            "corpus_freeze_manifest_sha256": CORPUS_FREEZE_MANIFEST_SHA256,
        },
        "splits": {
            "development": _plaintext_suite_split(split_records["development"]),
            "validation": _plaintext_suite_split(split_records["validation"]),
            "test": _public_test_suite_split(split_records["test"]),
        },
        "query_class_scenario_counts": _query_class_scenario_counts(cases),
        "evidence_applicability": evidence_applicability_counts(cases),
        "encrypted_test_artifact": {
            "encryption_profile_id": encrypted["encryption_profile_id"],
            "tool_family": encrypted["tool_family"],
            "tool_version": encrypted["tool_version"],
            "recipient_key_fingerprint_sha256": encrypted[
                "recipient_key_fingerprint_sha256"
            ],
            "encrypted_artifact_sha256": encrypted["encrypted_artifact_sha256"],
        },
        "encryption_boundary": ENCRYPTION_BOUNDARY,
        "deduplication": {
            "exact_cross_split_duplicates": 0,
            "semantic_cross_split_candidates_reviewed": True,
            "report_sha256": encrypted["dedup_report_sha256"],
        },
    }


def _plaintext_suite_split(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "independent_scenario_count": record["independent_scenario_count"],
        "case_count": record["case_count"],
        "cases_sha256": record["cases_sha256"],
        "questions_sha256": record["questions_sha256"],
        "externally_encrypted_artifact_bound": record[
            "externally_encrypted_artifact_bound"
        ],
    }


def _public_test_suite_split(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "independent_scenario_count": record["independent_scenario_count"],
        "case_count": record["case_count"],
        "public_case_count": record["public_case_count"],
        "public_cases_sha256": record["public_cases_sha256"],
        "question_commitments_sha256": record["question_commitments_sha256"],
        "externally_encrypted_artifact_bound": record[
            "externally_encrypted_artifact_bound"
        ],
    }


def _require_corpus_identity(corpus_dir: str | Path) -> dict[str, Any]:
    freeze_bytes = CORPUS_FREEZE_PATH.read_bytes()
    freeze_digest = hashlib.sha256(freeze_bytes).hexdigest()
    freeze = json.loads(freeze_bytes.decode("utf-8"))
    authority_path = Path(corpus_dir) / "authority.jsonl"
    authority_digest = hashlib.sha256(authority_path.read_bytes()).hexdigest()
    records = load_authority_records(authority_path)
    computed_content = content_digest(records)
    errors: list[str] = []
    if freeze_digest != CORPUS_FREEZE_MANIFEST_SHA256:
        errors.append(
            "corpus-freeze-manifest SHA-256 does not match the frozen suite identity"
        )
    if authority_digest != AUTHORITY_JSONL_SHA256:
        errors.append("authority.jsonl SHA-256 does not match the frozen suite identity")
    if freeze.get("authority_jsonl_sha256") != AUTHORITY_JSONL_SHA256:
        errors.append("corpus freeze manifest authority digest does not match suite identity")
    if computed_content != CONTENT_DIGEST_SHA256:
        errors.append("recomputed content digest does not match the frozen suite identity")
    if freeze.get("content_digest_sha256") != CONTENT_DIGEST_SHA256:
        errors.append("corpus freeze manifest content digest does not match suite identity")
    if freeze.get("corpus_freeze_manifest_sha256") not in {
        None,
        CORPUS_FREEZE_MANIFEST_SHA256,
    } and freeze.get("corpus_freeze_manifest_sha256") != freeze_digest:
        errors.append("corpus freeze manifest self-digest does not match recomputed bytes")
    if errors:
        raise ValueError("; ".join(errors))
    return freeze


def _query_class_scenario_counts(
    cases: list[tuple[int, dict[str, Any]]],
) -> dict[str, dict[str, int]]:
    classes = (
        "exact_factual",
        "semantic_paraphrase",
        "numeric_conditional",
        "temporal",
        "authorization",
        "unknown_oos",
        "multi_record",
        "adversarial_conflicting",
    )
    counts: dict[str, dict[str, set[str]]] = {
        split: {query_class: set() for query_class in classes}
        for split in ("development", "validation", "test")
    }
    for _line, case in cases:
        split = str(case.get("split", ""))
        query_class = str(case.get("query_class", ""))
        if split in counts and query_class in counts[split]:
            counts[split][query_class].add(str(case.get("scenario_id", "")))
    return {
        split: {query_class: len(scenarios) for query_class, scenarios in class_map.items()}
        for split, class_map in counts.items()
    }


def _validate_suite_manifest(manifest: dict[str, Any]) -> None:
    try:
        from jsonschema import Draft202012Validator
    except ImportError as exc:
        raise RuntimeError("M1 freeze requires the development dependencies") from exc

    schema = json.loads(SUITE_SCHEMA_PATH.read_text(encoding="utf-8"))
    errors = [
        f"suite_manifest:{'.'.join(str(part) for part in error.absolute_path) or '<root>'}: "
        f"{error.message}"
        for error in Draft202012Validator(schema).iter_errors(manifest)
    ]
    if "encrypted_artifact_path" in json.dumps(manifest):
        errors.append("suite_manifest must not publish the encrypted artifact path")
    forbidden_keys = {
        "salt_hex",
        "recipient_key",
        "encrypted_artifact_path",
        "question",
        "oracle",
        "plaintext_digest",
        "plaintext_staging_sha256",
        "retrieval_configuration",
        "retrieval_configuration_sha256",
        "sealed",
        "scheme",
        "tool",
        "recipient_key_fingerprint",
    }
    leaked_keys = sorted(_collect_keys(manifest).intersection(forbidden_keys))
    if leaked_keys:
        errors.append(
            "suite_manifest must not publish path, key, salt, question, oracle, "
            f"plaintext digest, or legacy encryption fields: {', '.join(leaked_keys)}"
        )
    artifact = manifest.get("encrypted_test_artifact")
    if not isinstance(artifact, dict) or artifact.get("encryption_profile_id") != (
        ENCRYPTION_PROFILE_ID
    ):
        errors.append("suite_manifest encryption_profile_id must be age-x25519-v1")
    test_split = manifest.get("splits", {}).get("test") if isinstance(
        manifest.get("splits"), dict
    ) else None
    if isinstance(test_split, dict):
        leaked_test = {
            key
            for key in (
                "cases_sha256",
                "questions_sha256",
                "plaintext_staging_sha256",
            )
            if key in test_split
        }
        if leaked_test:
            errors.append(
                "suite_manifest test split must not bind a full-case or plaintext "
                f"staging digest: {', '.join(sorted(leaked_test))}"
            )
    if "retrieval_configuration" in manifest:
        errors.append("suite_manifest must not bind a Gate 1 retrieval-configuration digest")
    applicability = manifest.get("evidence_applicability")
    if isinstance(applicability, dict):
        typed: dict[str, dict[str, dict[str, int]]] = {}
        for split, class_map in applicability.items():
            if not isinstance(class_map, dict):
                continue
            typed[str(split)] = {}
            for query_class, bucket in class_map.items():
                if isinstance(bucket, dict):
                    typed[str(split)][str(query_class)] = {
                        str(key): int(value)
                        for key, value in bucket.items()
                        if isinstance(value, int)
                    }
        errors.extend(
            applicability_partition_errors(typed, source="suite_manifest")
        )
    if errors:
        raise ValueError("\n".join(errors))


def _publish_text_files(
    pairs: list[tuple[Path, str]],
    *,
    public_cases: list[dict[str, Any]],
    public_text: str,
) -> None:
    """Stage, fsync, validate, then ``os.replace`` each destination.

    All public and staging bytes are generated and validated in memory before
    this function touches destination paths. Each output is staged in its
    destination directory, flushed, and ``fsync``ed. Staged public bytes are
    validated again before publication. Publication uses ``os.replace``.
    Staged leftovers are unlinked in ``finally``.

    The replacements are not a pair transaction: if destination files already
    exist, a crash between replacements can leave a new first file beside an
    old second file. On any pre-publish or late validation failure, neither
    destination is created by this call.
    """

    staged: list[Path] = []
    destinations = [path for path, _text in pairs]
    try:
        for dest, text in pairs:
            dest.parent.mkdir(parents=True, exist_ok=True)
            staged.append(_stage_text_file(dest.parent, dest.name, text))
        public_tmp = staged[-1]
        actual = public_tmp.read_text(encoding="utf-8")
        if actual != public_text:
            raise ValueError(f"{public_tmp}: staged public bytes do not match generated text")
        _validate_public_cases(public_cases, source=public_tmp)
        remaining = list(staged)
        for dest, tmp in zip(destinations, remaining, strict=True):
            os.replace(tmp, dest)
            staged.remove(tmp)
    finally:
        for leftover in staged:
            leftover.unlink(missing_ok=True)


def _stage_text_file(directory: Path, filename: str, text: str) -> Path:
    handle, raw_path = tempfile.mkstemp(prefix=f".{filename}.", suffix=".tmp", dir=directory)
    path = Path(raw_path)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return path


def _assert_outside_repository(path: Path, repo_root: Path, label: str) -> None:
    resolved = path.resolve()
    root = repo_root.resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        return
    raise ValueError(f"{path}: {label} must be written outside the repository")


def _collect_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            keys.add(str(key))
            keys.update(_collect_keys(item))
    elif isinstance(value, list):
        for item in value:
            keys.update(_collect_keys(item))
    return keys


def _jsonl(rows: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
