from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

CitationIntegrity = Literal["pass", "fail", "not_applicable"]

_CITATION_PATTERN = re.compile(
    r"\[(?:record_id:\s*)?([A-Za-z0-9][A-Za-z0-9._-]*:v[1-9]\d*)\]",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CitationVerification:
    integrity: CitationIntegrity
    cited_references: tuple[str, ...]
    unknown_references: tuple[str, ...]
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "citation_integrity": self.integrity,
            "cited_references": list(self.cited_references),
            "unknown_references": list(self.unknown_references),
            "reason": self.reason,
        }


def verify_citations(
    answer: str,
    *,
    evidence_references: tuple[str, ...],
) -> CitationVerification:
    evidence_by_normalized: dict[str, list[str]] = {}
    for reference in evidence_references:
        evidence_by_normalized.setdefault(reference.casefold(), []).append(reference)

    cited_values: list[str] = []
    unknown_values: list[str] = []
    seen: set[str] = set()
    for extracted in _CITATION_PATTERN.findall(answer):
        matches = evidence_by_normalized.get(extracted.casefold(), [])
        normalized = matches[0] if len(matches) == 1 else extracted
        if normalized not in seen:
            cited_values.append(normalized)
            seen.add(normalized)
        if len(matches) != 1 and extracted not in unknown_values:
            unknown_values.append(extracted)
    cited = tuple(cited_values)
    unknown = tuple(unknown_values)

    if not evidence_references:
        if cited:
            return CitationVerification(
                integrity="fail",
                cited_references=cited,
                unknown_references=cited,
                reason="The answer cited records even though no evidence was supplied.",
            )
        return CitationVerification(
            integrity="not_applicable",
            cited_references=(),
            unknown_references=(),
            reason="No evidence was supplied; citation integrity is not applicable.",
        )
    if not cited:
        return CitationVerification(
            integrity="fail",
            cited_references=(),
            unknown_references=(),
            reason="The answer did not cite any supplied evidence.",
        )
    if unknown:
        return CitationVerification(
            integrity="fail",
            cited_references=cited,
            unknown_references=unknown,
            reason="The answer contains citation references outside the supplied evidence.",
        )
    return CitationVerification(
        integrity="pass",
        cited_references=cited,
        unknown_references=(),
        reason="Every citation resolves to a record in the supplied evidence.",
    )
