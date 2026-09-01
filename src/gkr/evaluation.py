from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from gkr.runtime import GovernedKnowledgeRuntime
from gkr.schemas import Actor


@dataclass(frozen=True)
class RetrievalEvaluationCase:
    case_id: str
    question: str
    actor: Actor
    as_of: date
    known_at: datetime | None
    expected_contains: tuple[str, ...]
    expected_absent: tuple[str, ...]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RetrievalEvaluationCase:
        known_at_value = value.get("known_at")
        known_at = (
            datetime.fromisoformat(str(known_at_value).replace("Z", "+00:00")).astimezone(UTC)
            if known_at_value
            else None
        )
        return cls(
            case_id=str(value["case_id"]),
            question=str(value["question"]),
            actor=Actor(
                actor_id=str(value["actor"]),
                groups=tuple(str(group) for group in value.get("groups", [])),
            ),
            as_of=date.fromisoformat(str(value["as_of"])),
            known_at=known_at,
            expected_contains=tuple(
                str(reference) for reference in value.get("expected_contains", [])
            ),
            expected_absent=tuple(
                str(reference) for reference in value.get("expected_absent", [])
            ),
        )


@dataclass(frozen=True)
class RetrievalEvaluationResult:
    case_id: str
    passed: bool
    selected_references: tuple[str, ...]
    missing_expected: tuple[str, ...]
    present_forbidden: tuple[str, ...]
    retrieval_mode: str

    def to_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "passed": self.passed,
            "selected_references": list(self.selected_references),
            "missing_expected": list(self.missing_expected),
            "present_forbidden": list(self.present_forbidden),
            "retrieval_mode": self.retrieval_mode,
        }


def run_retrieval_suite(
    runtime: GovernedKnowledgeRuntime,
    suite_path: str | Path,
) -> dict[str, object]:
    cases = _load_cases(suite_path)
    results: list[RetrievalEvaluationResult] = []
    for case in cases:
        evidence, _plan = runtime.prepare(
            case.question,
            actor=case.actor,
            as_of=case.as_of,
            known_at=case.known_at,
        )
        selected = evidence.record_references
        selected_set = set(selected)
        missing = tuple(
            reference for reference in case.expected_contains if reference not in selected_set
        )
        forbidden = tuple(
            reference for reference in case.expected_absent if reference in selected_set
        )
        results.append(
            RetrievalEvaluationResult(
                case_id=case.case_id,
                passed=not missing and not forbidden,
                selected_references=selected,
                missing_expected=missing,
                present_forbidden=forbidden,
                retrieval_mode=evidence.retrieval_mode,
            )
        )

    passed = sum(result.passed for result in results)
    return {
        "suite": str(suite_path),
        "cases": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "results": [result.to_dict() for result in results],
    }


def _load_cases(path: str | Path) -> tuple[RetrievalEvaluationCase, ...]:
    cases: list[RetrievalEvaluationCase] = []
    with Path(path).open(encoding="utf-8") as source:
        for line_number, raw_line in enumerate(source, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                cases.append(RetrievalEvaluationCase.from_dict(json.loads(line)))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    if not cases:
        raise ValueError(f"{path}: evaluation suite is empty")
    return tuple(cases)
