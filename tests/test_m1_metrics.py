from __future__ import annotations

import hashlib
import json
from math import log2
from pathlib import Path

from gkr.m1_hash import (
    canonical_json_digest,
    canonical_jsonl_bytes,
    canonical_jsonl_digest,
    dedup_report_digest,
    encrypted_artifact_digest,
    normalize_question,
    prompt_digest,
    question_digest,
    question_list_digest,
    raw_bytes_digest,
    recipient_fingerprint_sha256,
    review_artifact_digest,
)
from gkr.m1_metrics import (
    NOT_APPLICABLE,
    OOS_OPERATIONAL_METRIC_IDS,
    VariantObservation,
    aggregate_non_error_metric,
    best_sufficient_set_recall_at_k,
    dedupe_returned_list,
    evidence_metric_applicable,
    metric_applicable,
    mrr_status,
    ndcg_status,
    retrieval_error_rate,
    safety_gate_count,
    select_ndcg_target_set,
    select_winner,
    set_success_at_k,
)


def _examples() -> dict[str, dict[str, object]]:
    contract = json.loads(Path("evaluation/m1/metric-contract-v3.json").read_text(encoding="utf-8"))
    return contract["worked_examples"]


def test_simultaneous_ndcg_set_completion() -> None:
    example = _examples()["simultaneous_equal_size_lexicographic"]
    assert (
        select_ndcg_target_set(example["sufficient_sets"], example["returned"]) == example["S_star"]
    )

    different_ranks = _examples()["alternative_sets_different_ranks"]
    assert (
        select_ndcg_target_set(different_ranks["sufficient_sets"], different_ranks["returned"])
        == different_ranks["S_star"]
    )

    earliest = _examples()["simultaneous_completion"]
    assert (
        select_ndcg_target_set(earliest["sufficient_sets"], earliest["returned"])
        == earliest["S_star"]
    )


def test_oos_ndcg_is_not_applicable() -> None:
    example = _examples()["oos_not_applicable"]
    assert ndcg_status(example["sufficient_sets"], example["returned"]) == NOT_APPLICABLE
    assert set_success_at_k(example["sufficient_sets"], example["returned"], 1) == NOT_APPLICABLE
    assert mrr_status(example["sufficient_sets"], example["returned"]) == NOT_APPLICABLE
    assert select_ndcg_target_set([], ["A:v1"]) is None


def test_empty_ranking_and_no_completion() -> None:
    empty = _examples()["empty_ranking"]
    assert select_ndcg_target_set(empty["sufficient_sets"], empty["returned"]) == empty["S_star"]
    assert ndcg_status(empty["sufficient_sets"], empty["returned"]) == 0.0

    incomplete = _examples()["no_set_completes"]
    assert (
        select_ndcg_target_set(incomplete["sufficient_sets"], incomplete["returned"])
        == incomplete["S_star"]
    )


def test_duplicate_returned_references_keep_first() -> None:
    example = _examples()["duplicate_returned_references"]
    assert dedupe_returned_list(example["returned_raw"]) == example["returned_unique"]
    assert set_success_at_k(example["sufficient_sets"], example["returned_raw"], 1) == 0.0
    assert (
        best_sufficient_set_recall_at_k(example["sufficient_sets"], example["returned_raw"], 1)
        == 0.5
    )
    assert set_success_at_k(example["sufficient_sets"], example["returned_raw"], 5) == 1.0
    assert mrr_status(example["sufficient_sets"], example["returned_raw"]) == 0.5
    assert ndcg_status(example["sufficient_sets"], example["returned_raw"]) == 1.0


def test_worked_example_numeric_metrics() -> None:
    alt = _examples()["alternative_sets_different_ranks"]
    assert set_success_at_k(alt["sufficient_sets"], alt["returned"], 1) == 0.0
    assert best_sufficient_set_recall_at_k(alt["sufficient_sets"], alt["returned"], 1) == 0.0
    assert set_success_at_k(alt["sufficient_sets"], alt["returned"], 5) == 1.0
    assert mrr_status(alt["sufficient_sets"], alt["returned"]) == 0.5
    assert ndcg_status(alt["sufficient_sets"], alt["returned"]) == 1 / log2(3)

    simultaneous = _examples()["simultaneous_equal_size_lexicographic"]
    assert set_success_at_k(simultaneous["sufficient_sets"], simultaneous["returned"], 1) == 0.0
    assert (
        best_sufficient_set_recall_at_k(
            simultaneous["sufficient_sets"], simultaneous["returned"], 1
        )
        == 0.5
    )
    assert mrr_status(simultaneous["sufficient_sets"], simultaneous["returned"]) == 1 / 3
    expected_ndcg = 1.5 / (1 + 1 / log2(3))
    assert ndcg_status(simultaneous["sufficient_sets"], simultaneous["returned"]) == expected_ndcg


def test_mixed_and_all_error_aggregation() -> None:
    mixed = _examples()["mixed_error_scenario"]
    mixed_obs = [
        VariantObservation(
            "s1",
            "a",
            error=False,
            value=1.0,
            query_class="exact_factual",
            metric_id="set_success_at_5",
        ),
        VariantObservation(
            "s1",
            "b",
            error=True,
            query_class="exact_factual",
            metric_id="set_success_at_5",
        ),
        VariantObservation(
            "s2",
            "a",
            error=False,
            value=0.0,
            query_class="exact_factual",
            metric_id="set_success_at_5",
        ),
    ]
    mixed_result = aggregate_non_error_metric(mixed_obs, metric_id="set_success_at_5")
    assert mixed_result["scenario_values"]["s1"] == mixed["s1_applicable_mean"]
    assert mixed_result["scenario_values"]["s2"] == mixed["s2_applicable_mean"]
    assert mixed_result["all_variants_error_scenario_count"] == 0
    assert mixed_result["split_mean"] == mixed["split_mean"]
    assert retrieval_error_rate(mixed_obs)["split_mean"] == mixed["retrieval_error_rate"]

    all_error = _examples()["all_error_scenario"]
    all_obs = [
        VariantObservation(
            "s1",
            "a",
            error=True,
            query_class="exact_factual",
            metric_id="set_success_at_5",
        ),
        VariantObservation(
            "s1",
            "b",
            error=True,
            query_class="exact_factual",
            metric_id="set_success_at_5",
        ),
        VariantObservation(
            "s2",
            "a",
            error=False,
            value=1.0,
            query_class="exact_factual",
            metric_id="set_success_at_5",
        ),
    ]
    all_result = aggregate_non_error_metric(all_obs, metric_id="set_success_at_5")
    assert "s1" not in all_result["scenario_values"]
    assert all_result["scenario_values"]["s2"] == 1.0
    assert all_result["all_variants_error_scenario_count"] == 1
    assert all_result["split_mean"] == 1.0
    assert retrieval_error_rate(all_obs)["split_mean"] == all_error["retrieval_error_rate"]

    oos_example = _examples()["oos_error_is_not_correct_rejection"]
    oos = [
        VariantObservation(
            "oos1",
            "a",
            error=True,
            query_class="unknown_oos",
            metric_id="oos_correct_rejection_rate",
            has_sufficient_sets=False,
        ),
        VariantObservation(
            "oos2",
            "a",
            error=False,
            query_class="unknown_oos",
            value=1.0,
            metric_id="oos_correct_rejection_rate",
            has_sufficient_sets=False,
        ),
    ]
    oos_result = aggregate_non_error_metric(oos, metric_id="oos_correct_rejection_rate")
    assert oos_result["all_variants_error_scenario_count"] == 1
    assert oos_result["split_mean"] == oos_example["split_mean"]
    assert retrieval_error_rate(oos)["split_mean"] == oos_example["retrieval_error_rate"]


def test_evidence_metric_excludes_oos_even_if_numeric_value_supplied() -> None:
    status = set_success_at_k([["A:v1"]], ["A:v1"], 5)
    assert status == 1.0
    oos_status = set_success_at_k([], ["A:v1"], 5)
    assert oos_status == NOT_APPLICABLE
    assert evidence_metric_applicable("unknown_oos", False) is False
    assert metric_applicable("set_success_at_5", "unknown_oos", has_sufficient_sets=False) is False
    assert metric_applicable("oos_correct_rejection_rate", "unknown_oos") is True
    assert metric_applicable("oos_correct_rejection_rate", "exact_factual") is False
    assert metric_applicable("oos_false_load_rate", "exact_factual") is False
    assert "oos_correct_rejection_rate" in OOS_OPERATIONAL_METRIC_IDS

    observations = [
        VariantObservation(
            "exact1",
            "a",
            error=False,
            query_class="exact_factual",
            value=float(status),
            metric_id="set_success_at_5",
            has_sufficient_sets=True,
        ),
        VariantObservation(
            "oos1",
            "a",
            error=False,
            query_class="unknown_oos",
            value=1.0,
            metric_id="set_success_at_5",
            has_sufficient_sets=False,
        ),
    ]
    result = aggregate_non_error_metric(observations, metric_id="set_success_at_5")
    assert result["scenario_values"] == {"exact1": 1.0}
    assert result["split_mean"] == 1.0
    assert "oos1" not in result["scenario_values"]


def test_oos_rate_excludes_exact_factual_even_if_numeric_value_supplied() -> None:
    observations = [
        VariantObservation(
            "exact1",
            "a",
            error=False,
            query_class="exact_factual",
            value=1.0,
            metric_id="oos_correct_rejection_rate",
        ),
        VariantObservation(
            "oos1",
            "a",
            error=False,
            query_class="unknown_oos",
            value=0.0,
            metric_id="oos_correct_rejection_rate",
            has_sufficient_sets=False,
        ),
    ]
    result = aggregate_non_error_metric(observations, metric_id="oos_correct_rejection_rate")
    assert result["scenario_values"] == {"oos1": 0.0}
    assert result["split_mean"] == 0.0
    assert "exact1" not in result["scenario_values"]
    assert result["all_variants_error_scenario_count"] == 0


def test_all_error_oos_does_not_increment_evidence_metric_error_count() -> None:
    observations = [
        VariantObservation(
            "oos1",
            "a",
            error=True,
            query_class="unknown_oos",
            metric_id="set_success_at_5",
            has_sufficient_sets=False,
        ),
        VariantObservation(
            "exact1",
            "a",
            error=False,
            query_class="exact_factual",
            value=1.0,
            metric_id="set_success_at_5",
        ),
    ]
    result = aggregate_non_error_metric(observations, metric_id="set_success_at_5")
    assert result["scenario_values"] == {"exact1": 1.0}
    assert result["split_mean"] == 1.0
    assert result["all_variants_error_scenario_count"] == 0
    assert "oos1" not in result["scenario_values"]


def test_variant_then_scenario_split_mean() -> None:
    example = _examples()["variant_weighting"]
    observations = [
        VariantObservation(
            "s1",
            "a",
            error=False,
            value=1.0,
            query_class="exact_factual",
            metric_id="set_success_at_5",
        ),
        VariantObservation(
            "s1",
            "b",
            error=False,
            value=0.0,
            query_class="exact_factual",
            metric_id="set_success_at_5",
        ),
        VariantObservation(
            "s2",
            "a",
            error=False,
            value=1.0,
            query_class="exact_factual",
            metric_id="set_success_at_5",
        ),
    ]
    result = aggregate_non_error_metric(observations, metric_id="set_success_at_5")
    assert result["scenario_values"]["s1"] == example["scenario_value"]
    assert result["split_mean"] == example["split_mean"]


def test_safety_error_is_not_exposure() -> None:
    error_only = [
        VariantObservation("s1", "a", error=True, unsafe=False),
        VariantObservation("s1", "b", error=False, unsafe=False),
    ]
    assert safety_gate_count(error_only) == 0
    unsafe = [
        VariantObservation("s1", "a", error=False, unsafe=True),
        VariantObservation("s1", "b", error=False, unsafe=False),
    ]
    assert safety_gate_count(unsafe) == 1


def test_lowest_complexity_tier_wins() -> None:
    assert (
        select_winner(
            {
                "full_authorized_corpus": True,
                "oracle_evidence": True,
                "bm25": False,
                "dense_candidate_b": True,
                "dense_candidate_a": True,
                "bm25_dense_rank_fusion": True,
            }
        )
        == "dense_candidate_a"
    )
    assert select_winner({"bm25": True, "hybrid_local_reranker": True}) == "bm25"
    assert select_winner({"full_authorized_corpus": True, "oracle_evidence": True}) is None


def test_hash_profile_preimages() -> None:
    assert normalize_question("  Foo\t\nBAR  ") == "foo bar"
    assert normalize_question("Ａ") == "a"
    assert normalize_question("foo\u00a0bar") == "foo bar"
    assert normalize_question("Straße") == "strasse"
    assert question_digest("Foo") == question_digest("FOO")
    assert prompt_digest("a\r\nb\rc") == prompt_digest("a\nb\nc")
    assert prompt_digest("a\r\nb") != prompt_digest("a\nb ")
    assert prompt_digest("a\r\nb") == hashlib.sha256(b"a\nb").hexdigest()
    assert prompt_digest("a\rb") == hashlib.sha256(b"a\nb").hexdigest()
    review = b"review-artifact-bytes"
    ciphertext = b"age-ciphertext-bytes"
    assert raw_bytes_digest(review) == hashlib.sha256(review).hexdigest()
    assert review_artifact_digest(review) == raw_bytes_digest(review)
    assert encrypted_artifact_digest(ciphertext) == raw_bytes_digest(ciphertext)
    left = canonical_json_digest({"b": 1, "a": {"z": 2, "y": 3}})
    right = canonical_json_digest({"a": {"y": 3, "z": 2}, "b": 1})
    assert left == right
    dedup = {"pairs": [], "z": 1, "a": 2}
    assert dedup_report_digest(dedup) == canonical_json_digest({"a": 2, "pairs": [], "z": 1})
    rows = [{"case_id": "b", "n": 1}, {"case_id": "a", "n": 2}]
    payload = canonical_jsonl_bytes(rows)
    assert payload == b'{"case_id":"a","n":2}\n{"case_id":"b","n":1}\n'
    assert payload.count(b"\n") == 2
    assert b"\r" not in payload
    assert canonical_jsonl_digest(rows) == canonical_jsonl_digest(list(reversed(rows)))
    assert question_list_digest([("b", "11"), ("a", "00")]) == hashlib.sha256(
        b'["00","11"]'
    ).hexdigest()
    assert question_list_digest([("b", "1"), ("a", "0")]) == question_list_digest(
        [("a", "0"), ("b", "1")]
    )
    assert len(recipient_fingerprint_sha256("age1example")) == 64
