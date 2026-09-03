from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from gkr.m1_hash import normalize_question, question_digest

__all__ = ["normalize_question", "question_digest", "validate_m1_cases"]

_SPLITS = ("development", "validation", "test")
_QUERY_CLASSES = {
    "exact_factual",
    "semantic_paraphrase",
    "numeric_conditional",
    "temporal",
    "authorization",
    "unknown_oos",
    "multi_record",
    "adversarial_conflicting",
}


def validate_m1_cases(
    case_path: str | Path,
    *,
    schema_path: str | Path = "evaluation/m1/benchmark-case-v2.schema.json",
    allow_incomplete: bool = False,
) -> dict[str, object]:
    """Validate M1 JSONL structure, hashes, independence, and split isolation."""

    try:
        from jsonschema import Draft202012Validator
    except ImportError as exc:
        raise RuntimeError("M1 validation requires the development dependencies") from exc

    schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    cases = _load_jsonl(case_path)
    errors: list[str] = []
    case_ids: set[str] = set()
    scenario_splits: dict[str, str] = {}
    question_digests: dict[str, str] = {}
    scenario_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    scenarios_by_split: dict[str, set[str]] = {split: set() for split in _SPLITS}

    for line_number, case in cases:
        prefix = f"{case_path}:{line_number}"
        for error in validator.iter_errors(case):
            location = ".".join(str(part) for part in error.absolute_path)
            errors.append(f"{prefix}:{location or '<root>'}: {error.message}")

        case_id = str(case.get("case_id", ""))
        if case_id in case_ids:
            errors.append(f"{prefix}: duplicate case_id {case_id}")
        case_ids.add(case_id)

        split = str(case.get("split", ""))
        scenario_id = str(case.get("scenario_id", ""))
        previous_split = scenario_splits.setdefault(scenario_id, split)
        if previous_split != split:
            errors.append(f"{prefix}: scenario {scenario_id} crosses benchmark splits")
        if split in scenarios_by_split:
            scenarios_by_split[split].add(scenario_id)

        query_class = str(case.get("query_class", ""))
        if query_class in _QUERY_CLASSES:
            class_counts[query_class] += 1

        question = case.get("question")
        if isinstance(question, str):
            digest = question_digest(question)
            if case.get("question_sha256") != digest:
                errors.append(f"{prefix}: question_sha256 does not match normalized question")
            duplicate = question_digests.setdefault(digest, case_id)
            if duplicate != case_id:
                errors.append(f"{prefix}: exact question duplicate of {duplicate}")

        oracle = case.get("oracle")
        if isinstance(oracle, dict):
            sufficient: set[str] = set()
            raw_sets = oracle.get("sufficient_reference_sets", [])
            if isinstance(raw_sets, list):
                for reference_set in raw_sets:
                    if isinstance(reference_set, list):
                        sufficient.update(str(reference) for reference in reference_set)
            raw_forbidden = oracle.get("forbidden_references", [])
            forbidden = (
                {str(reference) for reference in raw_forbidden}
                if isinstance(raw_forbidden, list)
                else set()
            )
            overlap = sorted(sufficient.intersection(forbidden))
            if overlap:
                errors.append(f"{prefix}: oracle references are both sufficient and forbidden")

    for split, scenarios in scenarios_by_split.items():
        scenario_counts[split] = len(scenarios)
        if not allow_incomplete and len(scenarios) != 120:
            errors.append(
                f"{case_path}: split {split} has {len(scenarios)} scenarios; expected 120"
            )
    if not allow_incomplete:
        missing_classes = sorted(_QUERY_CLASSES.difference(class_counts))
        if missing_classes:
            errors.append(f"{case_path}: missing query classes: {', '.join(missing_classes)}")
    if errors:
        raise ValueError("\n".join(errors))

    return {
        "case_file": str(case_path),
        "cases": len(cases),
        "independent_scenarios": dict(scenario_counts),
        "query_classes": dict(sorted(class_counts.items())),
        "complete": not allow_incomplete,
    }


def _load_jsonl(path: str | Path) -> list[tuple[int, dict[str, Any]]]:
    cases: list[tuple[int, dict[str, Any]]] = []
    with Path(path).open(encoding="utf-8") as source:
        for line_number, raw_line in enumerate(source, start=1):
            if not raw_line.strip():
                continue
            try:
                value = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: benchmark case must be an object")
            cases.append((line_number, value))
    if not cases:
        raise ValueError(f"{path}: benchmark case file is empty")
    return cases
