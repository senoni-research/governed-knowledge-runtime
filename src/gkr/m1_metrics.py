"""Deterministic helpers for the M1 v3 metric contract.

These functions implement ranking-list, aggregation, and winner-selection
definitions from ``evaluation/m1/metric-contract-v3.json``. They do not
retrieve, score a live suite, or establish semantic support.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import log2
from typing import Any

NOT_APPLICABLE = "not_applicable"
COMPLEXITY_TIERS: tuple[tuple[str, ...], ...] = (
    ("bm25",),
    ("dense_candidate_a", "dense_candidate_b"),
    ("bm25_dense_rank_fusion",),
    ("hybrid_local_reranker",),
)
CONTROLS = frozenset({"full_authorized_corpus", "oracle_evidence"})
EVIDENCE_SET_METRIC_IDS = frozenset(
    {
        "set_success_at_1",
        "set_success_at_5",
        "best_sufficient_set_recall_at_1",
        "best_sufficient_set_recall_at_5",
        "mrr",
        "ndcg",
    }
)
OOS_OPERATIONAL_METRIC_IDS = frozenset(
    {
        "oos_false_load_rate",
        "oos_correct_rejection_rate",
    }
)


def dedupe_returned_list(references: Sequence[str]) -> list[str]:
    """Retain first occurrence of each reference."""

    seen: set[str] = set()
    unique: list[str] = []
    for reference in references:
        if reference in seen:
            continue
        seen.add(reference)
        unique.append(reference)
    return unique


def select_ndcg_target_set(
    sufficient_sets: Sequence[Sequence[str]],
    returned: Sequence[str],
) -> list[str] | None:
    """Return S*, or None when nDCG is not_applicable (zero sufficient sets)."""

    sets = [tuple(sorted(group)) for group in sufficient_sets]
    if not sets:
        return None
    ranking = dedupe_returned_list(returned)
    rank_index = {reference: index for index, reference in enumerate(ranking)}

    def complete_rank(group: tuple[str, ...]) -> int | None:
        if any(reference not in rank_index for reference in group):
            return None
        return max(rank_index[reference] for reference in group)

    completed: list[tuple[int, int, tuple[str, ...]]] = []
    incomplete: list[tuple[float, int, tuple[str, ...]]] = []
    for group in sets:
        rank = complete_rank(group)
        if rank is None:
            if not ranking:
                fraction = 0.0
            else:
                fraction = sum(1 for reference in group if reference in rank_index) / len(group)
            incomplete.append((-fraction, len(group), group))
        else:
            completed.append((rank, len(group), group))

    if completed:
        _rank, _size, chosen = min(completed)
        return list(chosen)

    _neg_fraction, _size, chosen = min(incomplete)
    return list(chosen)


def ndcg_status(
    sufficient_sets: Sequence[Sequence[str]],
    returned: Sequence[str],
) -> str | float:
    """Return ``not_applicable`` or a deterministic nDCG value in [0, 1]."""

    target = select_ndcg_target_set(sufficient_sets, returned)
    if target is None:
        return NOT_APPLICABLE
    ranking = dedupe_returned_list(returned)
    if not ranking:
        return 0.0
    relevant = set(target)
    dcg = _dcg(ranking, relevant)
    idcg = _dcg(target, relevant)
    if idcg == 0:
        return 0.0
    return dcg / idcg


def set_success_at_k(
    sufficient_sets: Sequence[Sequence[str]],
    returned: Sequence[str],
    k: int,
) -> str | float:
    if not sufficient_sets:
        return NOT_APPLICABLE
    ranking = set(dedupe_returned_list(returned)[:k])
    return 1.0 if any(set(group).issubset(ranking) for group in sufficient_sets) else 0.0


def best_sufficient_set_recall_at_k(
    sufficient_sets: Sequence[Sequence[str]],
    returned: Sequence[str],
    k: int,
) -> str | float:
    if not sufficient_sets:
        return NOT_APPLICABLE
    ranking = set(dedupe_returned_list(returned)[:k])
    return max(len(set(group) & ranking) / len(group) for group in sufficient_sets)


def mrr_status(
    sufficient_sets: Sequence[Sequence[str]],
    returned: Sequence[str],
) -> str | float:
    if not sufficient_sets:
        return NOT_APPLICABLE
    ranking = dedupe_returned_list(returned)
    for index in range(len(ranking)):
        prefix = set(ranking[: index + 1])
        if any(set(group).issubset(prefix) for group in sufficient_sets):
            return 1.0 / (index + 1)
    return 0.0


def _dcg(ranking: Sequence[str], relevant: set[str]) -> float:
    total = 0.0
    for index, reference in enumerate(ranking, start=1):
        if reference in relevant:
            total += 1.0 / log2(index + 1)
    return total


@dataclass(frozen=True)
class VariantObservation:
    """One case-arm variant after list production or execution failure."""

    scenario_id: str
    variant_id: str
    error: bool
    query_class: str = ""
    value: float | None = None
    unsafe: bool = False
    metric_id: str = ""
    has_sufficient_sets: bool = True


def evidence_metric_applicable(query_class: str, has_sufficient_sets: bool) -> bool:
    """Evidence-set metrics apply only to non-OOS cases that have sufficient sets."""

    return query_class != "unknown_oos" and has_sufficient_sets


def metric_applicable(
    metric_id: str,
    query_class: str,
    *,
    has_sufficient_sets: bool = True,
) -> bool:
    """Return whether ``metric_id`` applies to this query class / set shape.

    Evidence-set metrics exclude ``unknown_oos`` and zero-set cases even if a
    numeric value was supplied. OOS operational rates apply only to
    ``unknown_oos``. Other operational rates apply to every class.
    """

    if metric_id in EVIDENCE_SET_METRIC_IDS:
        return evidence_metric_applicable(query_class, has_sufficient_sets)
    if metric_id in OOS_OPERATIONAL_METRIC_IDS:
        return query_class == "unknown_oos"
    return True


def scenario_mean(values: Sequence[float | None]) -> float | None:
    applicable = [value for value in values if value is not None]
    if not applicable:
        return None
    return sum(applicable) / len(applicable)


def _resolved_metric_id(item: VariantObservation, metric_id: str | None) -> str:
    return metric_id or item.metric_id


def observation_metric_applicable(
    item: VariantObservation,
    metric_id: str | None = None,
) -> bool:
    """True when this non-error observation is in-scope for the named metric."""

    resolved = _resolved_metric_id(item, metric_id)
    if not resolved:
        raise ValueError(
            "aggregate_non_error_metric requires metric_id on the call or each observation"
        )
    if item.error:
        return False
    return metric_applicable(
        resolved,
        item.query_class,
        has_sufficient_sets=item.has_sufficient_sets,
    )


def aggregate_non_error_metric(
    observations: Sequence[VariantObservation],
    metric_id: str | None = None,
) -> dict[str, Any]:
    """Scenario-then-split mean over applicable non-error variants.

    Applicability is enforced from ``metric_id`` (argument or observation field)
    plus query class / sufficient-set shape. A numeric ``value`` on an
    out-of-scope class does not make that variant applicable. Execution-error
    variants are excluded. A scenario with zero applicable variants is
    excluded from the split mean. ``all_variants_error_scenario_count``
    increments only when execution errors are why an otherwise
    metric-applicable scenario has zero applicable variants. A scenario
    that is intrinsically not applicable (for example all-error
    ``unknown_oos`` under an evidence-set metric) does not increment it.
    """

    by_scenario: dict[str, list[VariantObservation]] = defaultdict(list)
    for item in observations:
        by_scenario[item.scenario_id].append(item)

    scenario_values: dict[str, float] = {}
    all_error_count = 0
    for scenario_id, variants in by_scenario.items():
        applicable = [
            item.value
            for item in variants
            if observation_metric_applicable(item, metric_id) and item.value is not None
        ]
        if applicable:
            scenario_values[scenario_id] = sum(applicable) / len(applicable)
            continue
        in_scope = [
            item
            for item in variants
            if metric_applicable(
                _resolved_metric_id(item, metric_id),
                item.query_class,
                has_sufficient_sets=item.has_sufficient_sets,
            )
        ]
        if in_scope and all(item.error for item in in_scope):
            all_error_count += 1

    split_value = (
        sum(scenario_values.values()) / len(scenario_values) if scenario_values else None
    )
    return {
        "scenario_values": scenario_values,
        "split_mean": split_value,
        "all_variants_error_scenario_count": all_error_count,
        "scenario_denominator": len(scenario_values),
        "metric_id": metric_id
        or next((item.metric_id for item in observations if item.metric_id), ""),
    }


def retrieval_error_rate(observations: Sequence[VariantObservation]) -> dict[str, Any]:
    """Variant error indicator averaged within scenario, then equally across scenarios."""

    by_scenario: dict[str, list[VariantObservation]] = defaultdict(list)
    for item in observations:
        by_scenario[item.scenario_id].append(item)
    scenario_values = {
        scenario_id: sum(1.0 if item.error else 0.0 for item in variants) / len(variants)
        for scenario_id, variants in by_scenario.items()
        if variants
    }
    split_value = (
        sum(scenario_values.values()) / len(scenario_values) if scenario_values else None
    )
    return {
        "scenario_values": scenario_values,
        "split_mean": split_value,
        "all_variants_error_scenario_count": sum(
            1
            for variants in by_scenario.values()
            if variants and all(item.error for item in variants)
        ),
    }


def safety_gate_count(observations: Sequence[VariantObservation]) -> int:
    """Count non-error variants whose valid returned list is unsafe."""

    return sum(1 for item in observations if not item.error and item.unsafe)


def select_winner(passing_by_id: Mapping[str, bool]) -> str | None:
    """Select one global arm: lowest complexity tier, then lexicographic ID."""

    eligible = {
        candidate_id
        for candidate_id, passing in passing_by_id.items()
        if passing and candidate_id not in CONTROLS
    }
    for tier in COMPLEXITY_TIERS:
        hits = [candidate_id for candidate_id in tier if candidate_id in eligible]
        if hits:
            return sorted(hits)[0]
    return None
