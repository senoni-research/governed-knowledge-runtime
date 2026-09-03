"""Hash helpers for profile ID gkr-m1-hash-profile-v1.

Preimages follow ``evaluation/m1/hash-profile-v1.json``. These functions do not
encrypt, decrypt, or establish semantic support. They also do not load or
verify an externally attested dedup-report file. Gate 1 binds no
retrieval-configuration digest.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

HASH_PROFILE_ID = "gkr-m1-hash-profile-v1"
MODEL_FAMILY_REGISTRY_PATH = Path("evaluation/m1/model-family-registry-v1.json")
CANONICAL_FAMILY_IDS = ("anthropic-claude", "google-gemini", "openai-gpt", "xai-grok")
_WHITESPACE_RUN = re.compile(r"\s+", flags=re.UNICODE)
_MODEL_FAMILY_ID = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_HEX64 = re.compile(r"^[a-f0-9]{64}$")


def is_canonical_model_family_id(value: str) -> bool:
    return bool(_MODEL_FAMILY_ID.fullmatch(value))


def load_model_family_registry(
    path: str | Path = MODEL_FAMILY_REGISTRY_PATH,
) -> dict[str, Any]:
    """Load the bound registry or fail closed if it is missing or malformed."""

    registry_path = Path(path)
    try:
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"{registry_path}: model-family registry is unavailable; fail closed"
        ) from exc
    families = payload.get("families")
    if not isinstance(payload, dict) or not isinstance(families, list) or not families:
        raise ValueError(f"{registry_path}: model-family registry is malformed; fail closed")
    by_id: dict[str, dict[str, Any]] = {}
    for entry in families:
        if not isinstance(entry, dict) or not isinstance(entry.get("family_id"), str):
            raise ValueError(f"{registry_path}: model-family registry is malformed; fail closed")
        family_id = str(entry["family_id"])
        if family_id in by_id:
            raise ValueError(f"{registry_path}: duplicate family_id {family_id}")
        by_id[family_id] = entry
    missing = [family_id for family_id in CANONICAL_FAMILY_IDS if family_id not in by_id]
    if missing:
        raise ValueError(
            f"{registry_path}: missing required canonical family IDs: {', '.join(missing)}"
        )
    return payload


def resolve_model_family_id(
    family_id: object,
    model_id: object,
    *,
    registry: Mapping[str, Any] | None = None,
) -> str:
    """Return the canonical family ID or raise on an unknown family/model mapping.

    This checks declared provenance against the controlled registry. It cannot
    prove which external model actually ran.
    """

    bound = registry if registry is not None else load_model_family_registry()
    if not isinstance(family_id, str) or not is_canonical_model_family_id(family_id):
        raise ValueError(
            "model_family_id must be a canonical lowercase slug from the bound "
            "registry; aliases such as grok, grok-4.6, or chatgpt are not accepted"
        )
    families = {
        str(entry["family_id"]): entry
        for entry in bound.get("families", [])
        if isinstance(entry, dict) and isinstance(entry.get("family_id"), str)
    }
    entry = families.get(family_id)
    if entry is None:
        raise ValueError(
            f"unknown model family {family_id!r}; fail closed against the bound registry"
        )
    if not isinstance(model_id, str) or not model_id:
        raise ValueError(
            f"family {family_id} requires an exact model_id that matches an allowed "
            "prefix or pattern"
        )
    prefixes = [
        str(prefix) for prefix in entry.get("allowed_model_id_prefixes", []) if prefix
    ]
    patterns = [
        str(pattern) for pattern in entry.get("allowed_model_id_patterns", []) if pattern
    ]
    prefix_hit = any(model_id.startswith(prefix) for prefix in prefixes)
    pattern_hit = False
    for pattern in patterns:
        try:
            pattern_hit = bool(re.fullmatch(pattern, model_id))
        except re.error as exc:
            raise ValueError(
                f"model-family registry pattern {pattern!r} is invalid; fail closed"
            ) from exc
        if pattern_hit:
            break
    if not prefix_hit and not pattern_hit:
        raise ValueError(
            f"unknown family/model mapping {family_id!r}/{model_id!r}; fail closed"
        )
    return family_id


def is_sha256_hex(value: str) -> bool:
    return bool(_HEX64.fullmatch(value))


def normalize_question(question: str) -> str:
    nfkc = unicodedata.normalize("NFKC", question)
    collapsed = _WHITESPACE_RUN.sub("\u0020", nfkc)
    return collapsed.strip().casefold()


def question_digest(question: str) -> str:
    return hashlib.sha256(normalize_question(question).encode("utf-8")).hexdigest()


def prompt_digest(prompt: str) -> str:
    normalized = prompt.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def raw_bytes_digest(payload: bytes) -> str:
    """SHA-256 of exact bytes. Preimage for review_sha256 and encrypted_artifact_sha256."""

    return hashlib.sha256(payload).hexdigest()


def review_artifact_digest(payload: bytes) -> str:
    return raw_bytes_digest(payload)


def encrypted_artifact_digest(payload: bytes) -> str:
    return raw_bytes_digest(payload)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def canonical_json_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def dedup_report_digest(value: Mapping[str, Any]) -> str:
    """SHA-256 of a dedup report as a gkr-m1-hash-profile-v1 canonical JSON object."""

    return canonical_json_digest(value)


def canonical_jsonl_bytes(records: Sequence[Mapping[str, Any]]) -> bytes:
    ordered = sorted(records, key=lambda item: str(item.get("case_id", "")))
    return b"".join(canonical_json_bytes(record) + b"\n" for record in ordered)


def canonical_jsonl_digest(records: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(canonical_jsonl_bytes(records)).hexdigest()


def question_list_digest(
    pairs: Iterable[tuple[str, str]],
) -> str:
    """Hash question or commitment digests ordered by case_id."""

    ordered = [digest for _case_id, digest in sorted(pairs, key=lambda item: item[0])]
    payload = json.dumps(ordered, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def recipient_fingerprint_sha256(canonical_age_recipient: str) -> str:
    return hashlib.sha256(canonical_age_recipient.encode("utf-8")).hexdigest()
