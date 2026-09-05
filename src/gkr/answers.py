from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal

from gkr.context import EvidenceBundle

AnswerOutcome = Literal["answer", "abstain"]

ABSTENTION_TEXT = "I cannot establish that from the evidence available to me."

_EXHAUSTIVE_PATTERN = re.compile(
    r"\b(?:no other|no additional|nothing else|only|sole|all requirements|"
    r"complete(?:ly)?|exhaustive|exclusive(?:ly)?)\b",
    re.IGNORECASE,
)
_REFERENCE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*:v[1-9]\d*$")
_EVIDENCE_LABEL_PATTERN = re.compile(
    r"^EVIDENCE-\d+\s+\[([A-Za-z0-9][A-Za-z0-9._-]*:v[1-9]\d*)\]$",
    re.IGNORECASE,
)
_TERM_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "be",
    "before",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "may",
    "must",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
}
_TERM_NORMAL_FORMS = {
    "gates": "gate",
    "owned": "owner",
    "owns": "owner",
    "policies": "policy",
    "required": "require",
    "requirement": "require",
    "requirements": "require",
    "requires": "require",
}
_IDENTIFIER_EXCLUSIONS = {
    "a",
    "an",
    "each",
    "every",
    "if",
    "model",
    "no",
    "only",
    "record",
    "the",
    "this",
    "when",
}
_WORD_TOKEN_PATTERN = r"[A-Za-z](?:[A-Za-z0-9._-]*[A-Za-z0-9])?"


@dataclass(frozen=True)
class EvidenceClaim:
    claim: str
    record_reference: str
    supporting_passage: str

    def to_dict(self) -> dict[str, str]:
        return {
            "claim": self.claim,
            "record_reference": self.record_reference,
            "supporting_passage": self.supporting_passage,
        }


@dataclass(frozen=True)
class StructuredCandidate:
    outcome: AnswerOutcome
    claims: tuple[EvidenceClaim, ...]


def parse_structured_candidate(raw_response: str) -> StructuredCandidate:
    value = _first_json_object(raw_response)
    if set(value) != {"outcome", "claims"}:
        raise ValueError("candidate must contain exactly outcome and claims")
    outcome = value["outcome"]
    if outcome not in {"answer", "abstain"}:
        raise ValueError("candidate outcome must be answer or abstain")
    raw_claims = value["claims"]
    if not isinstance(raw_claims, list):
        raise ValueError("candidate claims must be a list")
    if len(raw_claims) > 12:
        raise ValueError("candidate contains more than 12 claims")

    claims: list[EvidenceClaim] = []
    for index, raw_claim in enumerate(raw_claims):
        if not isinstance(raw_claim, dict):
            raise ValueError(f"claim {index} must be an object")
        if set(raw_claim) != {"claim", "record_reference", "supporting_passage"}:
            raise ValueError(
                f"claim {index} must contain exactly claim, record_reference, "
                "and supporting_passage"
            )
        claim = _bounded_text(raw_claim["claim"], label=f"claim {index}", limit=800)
        reference = _canonical_reference(
            _bounded_text(
                raw_claim["record_reference"],
                label=f"claim {index} record_reference",
                limit=200,
            )
        )
        passage = _bounded_text(
            raw_claim["supporting_passage"],
            label=f"claim {index} supporting_passage",
            limit=2_000,
        )
        if "[" in claim or "]" in claim:
            raise ValueError(f"claim {index} must not contain citation markup")
        claims.append(
            EvidenceClaim(
                claim=claim,
                record_reference=reference,
                supporting_passage=passage,
            )
        )

    if outcome == "answer" and not claims:
        raise ValueError("answer outcome requires at least one claim")
    if outcome == "abstain" and claims:
        raise ValueError("abstain outcome must not contain claims")
    return StructuredCandidate(outcome=outcome, claims=tuple(claims))


def validate_claim_bindings(
    claims: tuple[EvidenceClaim, ...],
    *,
    evidence: EvidenceBundle,
) -> tuple[str, ...]:
    content_by_reference = dict(evidence.verifiable_content)
    issues: list[str] = []
    for index, claim in enumerate(claims):
        content = content_by_reference.get(claim.record_reference)
        if content is None:
            issues.append(
                f"claim {index} references evidence outside the authorized bundle: "
                f"{claim.record_reference}"
            )
            continue
        if claim.supporting_passage not in content:
            issues.append(
                f"claim {index} supporting_passage is not an exact passage from "
                f"{claim.record_reference}"
            )
        if _EXHAUSTIVE_PATTERN.search(claim.claim) and not _EXHAUSTIVE_PATTERN.search(
            claim.supporting_passage
        ):
            issues.append(
                f"claim {index} makes an exhaustive assertion not present in its passage"
            )
        claim_terms = _material_terms(claim.claim)
        passage_terms = _material_terms(claim.supporting_passage)
        missing_terms = claim_terms - passage_terms
        if missing_terms:
            issues.append(
                f"claim {index} contains material terms absent from its passage: "
                f"{', '.join(sorted(missing_terms))}"
            )
        missing_identifiers = _material_identifiers(claim.claim) - _all_tokens(
            claim.supporting_passage
        )
        if missing_identifiers:
            issues.append(
                f"claim {index} contains identifiers absent from its passage: "
                f"{', '.join(sorted(missing_identifiers))}"
            )
    return tuple(issues)


def render_claims(claims: tuple[EvidenceClaim, ...]) -> str:
    rendered = [
        f"{claim.claim.rstrip()} [{claim.record_reference}]"
        for claim in claims
    ]
    return rendered[0] if len(rendered) == 1 else "\n".join(f"- {line}" for line in rendered)


def _bounded_text(value: object, *, label: str, limit: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    normalized = value.strip()
    if len(normalized) > limit:
        raise ValueError(f"{label} exceeds {limit} characters")
    return normalized


def _canonical_reference(value: str) -> str:
    if _REFERENCE_PATTERN.fullmatch(value):
        return value
    match = _EVIDENCE_LABEL_PATTERN.fullmatch(value)
    return match.group(1) if match else value


def _material_terms(value: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", value.casefold())
    return {
        _TERM_NORMAL_FORMS.get(token, token)
        for token in tokens
        if token not in _TERM_STOPWORDS
    }


def _material_identifiers(value: str) -> set[str]:
    identifiers = {
        token.casefold()
        for token in re.findall(r"[£$€]?\d+(?:[.,]\d+)?", value)
    }
    for token in re.findall(_WORD_TOKEN_PATTERN, value):
        normalized = token.casefold()
        if token[0].isupper() and normalized not in _IDENTIFIER_EXCLUSIONS:
            identifiers.add(normalized)
        if "_" in token or (len(token) > 1 and token.isupper()):
            identifiers.add(normalized)
    return identifiers


def _all_tokens(value: str) -> set[str]:
    return {
        token.casefold()
        for token in re.findall(
            rf"[£$€]?\d+(?:[.,]\d+)?|{_WORD_TOKEN_PATTERN}",
            value,
        )
    }


def _first_json_object(value: str) -> dict[str, object]:
    decoder = json.JSONDecoder()
    for index, character in enumerate(value):
        if character != "{":
            continue
        try:
            parsed, _end = decoder.raw_decode(value[index:])
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            raise ValueError("candidate JSON must be an object")
        return parsed
    raise ValueError("candidate response did not contain a JSON object")
