"""Two-phase M1 scoring-suite preparation and finalization.

Preparation writes salted plaintext staging and a redacted public-test
candidate outside the repository. It emits no suite manifest and makes no
Gate 1 pass claim. The caller encrypts the exact staging bytes with external
``age``. Finalization consumes those unchanged prepared bytes plus the
external ciphertext and a loaded semantic dedup report.

This runtime does not implement or decrypt age and does not prove that the
ciphertext decrypts to the staged bytes.
"""

from __future__ import annotations

import hashlib
import json
import re
import secrets
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from gkr.m1_authoring import load_semantic_dedup_report
from gkr.m1_freeze import (
    ENCRYPTION_BOUNDARY,
    ENCRYPTION_PROFILE_ID,
    SALT_SIZE,
    STAGING_RECORD_VERSION,
    TOOL_FAMILY,
    _assert_scoring_freeze_eligibility,
    _require_corpus_identity,
    _require_encrypted_artifact,
    _scoring_suite_manifest,
    _split_freeze_records,
    _stage_test_split,
    _validate_public_cases,
    _validate_suite_manifest,
    question_commitment,
    verify_question_commitments,
)
from gkr.m1_hash import (
    HASH_PROFILE_ID,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    is_sha256_hex,
    recipient_fingerprint_sha256,
)
from gkr.m1_io import (
    assert_outside_repository,
    canonical_json_text,
    jsonl_text,
    load_jsonl,
    publish_new_directory,
    publish_text_files,
    repository_root,
)
from gkr.m1_oracle_validation import validate_m1_oracles

PREPARATION_VERSION = "gkr-m1-scoring-suite-preparation-v1"
SUITE_OUTPUT_FILES = (
    "development.jsonl",
    "validation.jsonl",
    "test-public.jsonl",
    "suite-manifest.json",
)
AGE_RECIPIENT_RE = re.compile(r"^age1[qpzry9x8gf2tvdw0s3jn54khce6mua7l]+$")
AGE_HEADER = b"age-encryption.org/v1"
PREPARATION_WARNING = (
    "This preparation report binds complete-case, plaintext-staging, and "
    "public-candidate digests. It may therefore identify private plaintext "
    "and must not be published with the scoring suite."
)


def prepare_m1_scoring_suite(
    case_path: str | Path,
    *,
    plaintext_staging_path: str | Path,
    public_test_candidate_path: str | Path,
    preparation_report_path: str | Path,
    corpus_dir: str | Path = "evaluation/m1/corpus",
    salt_factory: Callable[[], bytes] | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Validate a complete scoring suite and write unsalted-secret staging only."""

    root = Path(repo_root) if repo_root is not None else repository_root()
    staging_path = Path(plaintext_staging_path)
    public_path = Path(public_test_candidate_path)
    report_path = Path(preparation_report_path)
    assert_outside_repository(staging_path, root, "plaintext staging bundle")
    assert_outside_repository(public_path, root, "public test candidate")
    assert_outside_repository(report_path, root, "preparation report")

    cases = load_jsonl(case_path)
    _assert_scoring_cases(cases, case_path)
    oracle_report = validate_m1_oracles(
        case_path,
        corpus_dir=corpus_dir,
        allow_incomplete=False,
    )
    _require_corpus_identity(corpus_dir)

    factory = salt_factory or (lambda: secrets.token_bytes(SALT_SIZE))
    test_records, public_cases, _salts = _stage_test_split(cases, factory)
    if not test_records:
        raise ValueError(f"{case_path}: preparation requires at least one test-split case")
    staging_rows = [record for _line, record in test_records]
    staging_rows.sort(key=lambda row: str(row["case"]["case_id"]))
    public_cases.sort(key=lambda row: str(row["case_id"]))
    staging_text = jsonl_text(staging_rows)
    public_text = jsonl_text(public_cases)
    _validate_public_cases(public_cases, source=public_path)
    _assert_staging_matches_test_cases(staging_rows, cases, case_path)

    publish_text_files(
        [
            (staging_path, staging_text),
            (public_path, public_text),
        ]
    )
    verify_question_commitments(
        public_test_path=public_path,
        plaintext_staging_path=staging_path,
    )

    report = {
        "schema_version": PREPARATION_VERSION,
        "hash_profile_id": HASH_PROFILE_ID,
        "complete_cases_sha256": hashlib.sha256(Path(case_path).read_bytes()).hexdigest(),
        "plaintext_staging_sha256": hashlib.sha256(staging_path.read_bytes()).hexdigest(),
        "public_test_candidate_sha256": hashlib.sha256(public_path.read_bytes()).hexdigest(),
        "test_case_count": len(staging_rows),
        "test_case_ids": [str(row["case"]["case_id"]) for row in staging_rows],
        "salt_size": SALT_SIZE,
        "oracle_validation": oracle_report,
        "publication_warning": PREPARATION_WARNING,
        "note": (
            "Prepared salted staging bytes are ready for an external age-x25519-v1 "
            "step. This report sets no suite status and makes no Gate 1 pass claim. "
            + PREPARATION_WARNING
        ),
    }
    if "status" in report or "gate_1_status" in report or "suite_manifest" in report:
        raise RuntimeError("preparation report must not claim suite status")
    publish_text_files([(report_path, canonical_json_text(report, pretty=True))])
    return report


def finalize_m1_scoring_suite(
    case_path: str | Path,
    *,
    preparation_report_path: str | Path,
    plaintext_staging_path: str | Path,
    public_test_candidate_path: str | Path,
    semantic_dedup_report_path: str | Path,
    age_recipient_file: str | Path,
    age_tool_version: str,
    encrypted_artifact_path: str | Path,
    output_dir: str | Path,
    corpus_dir: str | Path = "evaluation/m1/corpus",
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Publish a four-file scoring suite from unchanged prepared bytes."""

    root = Path(repo_root) if repo_root is not None else repository_root()
    dest = Path(output_dir)
    if dest.exists():
        raise ValueError(f"{dest}: refuse to overwrite an existing suite directory")

    cases = load_jsonl(case_path)
    _assert_scoring_cases(cases, case_path)
    oracle_report = validate_m1_oracles(
        case_path,
        corpus_dir=corpus_dir,
        allow_incomplete=False,
    )
    _require_corpus_identity(corpus_dir)

    preparation = _load_preparation_report(preparation_report_path)
    _assert_exact_digest(case_path, preparation["complete_cases_sha256"], "complete cases")
    staging_path = Path(plaintext_staging_path)
    public_path = Path(public_test_candidate_path)
    _assert_exact_digest(
        staging_path, preparation["plaintext_staging_sha256"], "plaintext staging"
    )
    _assert_exact_digest(
        public_path, preparation["public_test_candidate_sha256"], "public test candidate"
    )

    staging_rows = [record for _line, record in load_jsonl(staging_path)]
    _assert_staging_matches_test_cases(staging_rows, cases, case_path)
    if [str(row["case"]["case_id"]) for row in staging_rows] != preparation["test_case_ids"]:
        raise ValueError("plaintext staging case_id order does not match the preparation report")
    public_cases = [case for _line, case in load_jsonl(public_path)]
    _validate_public_cases(public_cases, source=public_path)
    verify_question_commitments(
        public_test_path=public_path,
        plaintext_staging_path=staging_path,
    )

    case_objects = [case for _line, case in cases]
    _dedup_report, dedup_digest = load_semantic_dedup_report(
        semantic_dedup_report_path,
        cases=case_objects,
    )
    recipient = load_age_recipient(age_recipient_file)
    fingerprint = recipient_fingerprint_sha256(recipient)
    artifact_path = Path(encrypted_artifact_path)
    ciphertext_digest = _require_age_ciphertext(artifact_path, root)
    encrypted = _require_encrypted_artifact(
        {
            "encryption_profile_id": ENCRYPTION_PROFILE_ID,
            "tool_family": TOOL_FAMILY,
            "tool_version": str(age_tool_version).strip(),
            "recipient_key_fingerprint_sha256": fingerprint,
            "encrypted_artifact_path": str(artifact_path),
            "encrypted_artifact_sha256": ciphertext_digest,
        },
        root,
    )
    encrypted = {**encrypted, "dedup_report_sha256": dedup_digest}

    salts = {
        str(row["case"]["case_id"]): bytes.fromhex(str(row["salt_hex"]))
        for row in staging_rows
    }
    split_records = _split_freeze_records(cases, public_cases, salts)
    split_records["development"]["externally_encrypted_artifact_bound"] = False
    split_records["validation"]["externally_encrypted_artifact_bound"] = False
    split_records["test"]["externally_encrypted_artifact_bound"] = True
    manifest = _scoring_suite_manifest(
        cases,
        split_records,
        encrypted=encrypted,
        corpus_dir=corpus_dir,
    )
    _validate_suite_manifest(manifest)
    _assert_public_release_is_redacted(
        {
            "development.jsonl": _split_jsonl(case_objects, "development"),
            "validation.jsonl": _split_jsonl(case_objects, "validation"),
            "test-public.jsonl": public_path.read_text(encoding="utf-8"),
            "suite-manifest.json": canonical_json_text(manifest, pretty=True),
        }
    )

    files = {
        "development.jsonl": _split_jsonl(case_objects, "development"),
        "validation.jsonl": _split_jsonl(case_objects, "validation"),
        "test-public.jsonl": public_path.read_text(encoding="utf-8"),
        "suite-manifest.json": canonical_json_text(manifest, pretty=True),
    }
    publish_new_directory(dest, files)
    for name in SUITE_OUTPUT_FILES:
        if not (dest / name).is_file():
            raise RuntimeError(f"{dest / name}: suite file missing after publication")

    return {
        "output_dir": str(dest),
        "scoring_suite": True,
        "suite_manifest": manifest,
        "oracle_validation": oracle_report,
        "dedup_report_sha256": dedup_digest,
        "encrypted_artifact_sha256": ciphertext_digest,
        "recipient_key_fingerprint_sha256": fingerprint,
        "encryption_boundary": ENCRYPTION_BOUNDARY,
        "file_sha256": {
            name: hashlib.sha256((dest / name).read_bytes()).hexdigest()
            for name in SUITE_OUTPUT_FILES
        },
    }


def load_age_recipient(path: str | Path) -> str:
    """Return the exact no-newline age recipient from a one-line file."""

    raw = Path(path).read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path}: recipient file must be UTF-8") from exc
    if text.endswith("\r\n"):
        body = text[:-2]
    elif text.endswith("\n") or text.endswith("\r"):
        body = text[:-1]
    else:
        body = text
    if "\n" in body or "\r" in body or not body:
        raise ValueError(
            f"{path}: recipient file must contain exactly one canonical age recipient line"
        )
    if body != body.strip() or not AGE_RECIPIENT_RE.fullmatch(body):
        raise ValueError(f"{path}: recipient line is not a canonical age X25519 recipient")
    return body


def recipient_fingerprint_with_newline(canonical_recipient: str) -> str:
    """Incorrect preimage used only to document the frozen no-newline rule."""

    return hashlib.sha256((canonical_recipient + "\n").encode("utf-8")).hexdigest()


def _assert_scoring_cases(
    cases: Sequence[tuple[int, dict[str, Any]]],
    case_path: str | Path,
) -> None:
    kinds = [case.get("case_kind") for _line, case in cases]
    if not kinds:
        raise ValueError(f"{case_path}: scoring suite requires at least one case")
    if any(kind != "scoring" for kind in kinds):
        raise ValueError(
            f"{case_path}: scoring preparation/finalization rejects conformance or "
            "non-scoring cases"
        )
    _assert_scoring_freeze_eligibility(list(cases), case_path)


def _load_preparation_report(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: preparation report must be one JSON object")
    if payload.get("schema_version") != PREPARATION_VERSION:
        raise ValueError(f"{path}: unexpected preparation report schema_version")
    required = (
        "complete_cases_sha256",
        "plaintext_staging_sha256",
        "public_test_candidate_sha256",
        "test_case_ids",
    )
    missing = [field for field in required if field not in payload]
    if missing:
        raise ValueError(f"{path}: preparation report missing {', '.join(missing)}")
    for field in (
        "complete_cases_sha256",
        "plaintext_staging_sha256",
        "public_test_candidate_sha256",
    ):
        if not isinstance(payload[field], str) or not is_sha256_hex(payload[field]):
            raise ValueError(f"{path}: {field} must be 64 lowercase hex")
    if payload.get("status") == "frozen" or payload.get("gate_1_status") == "passed":
        raise ValueError(f"{path}: preparation report must not claim Gate 1 passage")
    return payload


def _assert_exact_digest(path: str | Path, expected: str, label: str) -> None:
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest != expected:
        raise ValueError(f"{path}: {label} digest {digest} != prepared {expected}")


def _assert_staging_matches_test_cases(
    staging_rows: Sequence[Mapping[str, Any]],
    cases: Sequence[tuple[int, dict[str, Any]]],
    case_path: str | Path,
) -> None:
    test_cases = {
        str(case["case_id"]): case for _line, case in cases if case.get("split") == "test"
    }
    staged_ids = [str(row.get("case", {}).get("case_id", "")) for row in staging_rows]
    if sorted(staged_ids) != sorted(test_cases):
        raise ValueError(
            f"{case_path}: plaintext staging case set does not match the input test split"
        )
    for row in staging_rows:
        if row.get("schema_version") != STAGING_RECORD_VERSION:
            raise ValueError("plaintext staging record has an unexpected schema_version")
        case = row.get("case")
        salt_hex = row.get("salt_hex")
        if not isinstance(case, dict) or not isinstance(salt_hex, str):
            raise ValueError("plaintext staging record must contain case and salt_hex")
        try:
            salt = bytes.fromhex(salt_hex)
        except ValueError as exc:
            raise ValueError("plaintext staging salt_hex is not valid hex") from exc
        if len(salt) != SALT_SIZE:
            raise ValueError("plaintext staging salt must be 32 bytes")
        expected = test_cases[str(case["case_id"])]
        if canonical_json_bytes(case) != canonical_json_bytes(expected):
            raise ValueError(
                f"{case_path}: staging case {case['case_id']} does not match the input"
            )
        expected_commitment = question_commitment(salt, str(case["question"]))
        if expected_commitment != question_commitment(salt, str(expected["question"])):
            raise ValueError(f"{case_path}: staging commitment mismatch for {case['case_id']}")


def _require_age_ciphertext(path: Path, repo_root: Path) -> str:
    assert_outside_repository(path, repo_root, "encrypted artifact")
    if path.suffix != ".age":
        raise ValueError(f"{path}: encrypted artifact must be an external .age file")
    if not path.is_file():
        raise ValueError(f"{path}: encrypted artifact does not exist")
    payload = path.read_bytes()
    if not payload.startswith(AGE_HEADER):
        raise ValueError(
            f"{path}: encrypted artifact is not an age ciphertext "
            "(missing age-encryption.org/v1 header)"
        )
    return hashlib.sha256(payload).hexdigest()


def _split_jsonl(cases: Sequence[Mapping[str, Any]], split: str) -> str:
    selected = [case for case in cases if case.get("split") == split]
    return canonical_jsonl_bytes(selected).decode("utf-8")


def _assert_public_release_is_redacted(files: Mapping[str, str]) -> None:
    public = files["test-public.jsonl"] + files["suite-manifest.json"]
    forbidden = (
        "salt_hex",
        "plaintext_staging_sha256",
        '"recipient_key"',
        "encrypted_artifact_path",
        "/Users/",
        "gkr-sealed",
    )
    leaked = [token for token in forbidden if token in public]
    if leaked:
        raise ValueError(
            "public suite artifacts must not contain salts, private paths, "
            f"or plaintext staging fields: {', '.join(leaked)}"
        )
    for line in files["test-public.jsonl"].splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if any(key in row for key in ("question", "oracle", "salt_hex", "oracle_review")):
            raise ValueError("public test file must not contain questions, oracles, or salts")
