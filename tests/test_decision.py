from __future__ import annotations

from datetime import date

from conftest import make_record

from gkr.ai import Generation, GenerationRequest
from gkr.authority import AuthorityStore
from gkr.decision import TypedPolicyDecisionEngine
from gkr.retrieval import LocalRetrievalRouter
from gkr.runtime import GovernedKnowledgeRuntime
from gkr.schemas import Actor


class MustNotRunGenerator:
    model_id = "must-not-run"

    def generate(self, request: GenerationRequest) -> Generation:
        raise AssertionError(f"LLM should not run for a typed rule: {request.prompt}")


def test_numeric_relation_is_decided_without_llm(store: AuthorityStore) -> None:
    store.append(
        make_record(
            statement="Travel spend above £750 requires written approval.",
            rules=(_policy_rule(),),
        )
    )
    runtime = GovernedKnowledgeRuntime(store, generator=MustNotRunGenerator())

    below = runtime.ask(
        "Does a £700 travel booking require approval?",
        actor=Actor("alice", ("employees",)),
        as_of=date(2026, 6, 1),
    )
    above = runtime.ask(
        "Does a £800 travel booking require approval?",
        actor=Actor("alice", ("employees",)),
        as_of=date(2026, 6, 1),
    )

    assert below.answer is not None
    assert below.answer.startswith("No.")
    assert above.answer is not None
    assert above.answer.startswith("Yes.")
    assert below.answer_status == "published_deterministic_policy_rule"
    assert below.decision_parse.request is not None
    assert below.decision_parse.request.polarity == "positive"
    assert below.semantic_verification is None


def test_numeric_engine_does_not_guess_a_different_action(
    store: AuthorityStore,
) -> None:
    store.append(
        make_record(
            rules=(_policy_rule(),),
        )
    )
    corpus = store.current_records(
        actor=Actor("alice", ("employees",)),
        as_of=date(2026, 6, 1),
    )
    decision = TypedPolicyDecisionEngine().decide(
        "Is approval waived for £700 travel spend?",
        corpus,
        as_of=date(2026, 6, 1),
    )

    assert decision.generation is None
    assert decision.parse.status == "ambiguous"


def test_numeric_parser_fails_closed_on_negation_and_multiple_amounts(
    store: AuthorityStore,
) -> None:
    store.append(make_record(rules=(_policy_rule(),)))
    corpus = store.current_records(
        actor=Actor("alice", ("employees",)),
        as_of=date(2026, 6, 1),
    )
    engine = TypedPolicyDecisionEngine()

    negated = engine.decide(
        "Does £800 travel spend not require approval?",
        corpus,
        as_of=date(2026, 6, 1),
    )
    multiple = engine.decide(
        "Do £700 or £800 travel bookings require approval?",
        corpus,
        as_of=date(2026, 6, 1),
    )
    comparison = engine.decide(
        "Compare £700 and £800 travel spend.",
        corpus,
        as_of=date(2026, 6, 1),
    )

    assert negated.parse.status == "ambiguous"
    assert negated.generation is None
    assert multiple.parse.status == "ambiguous"
    assert multiple.generation is None
    assert comparison.parse.status == "ambiguous"
    assert comparison.generation is None


def test_numeric_parser_fails_closed_on_false_negation_and_conditionals(
    store: AuthorityStore,
) -> None:
    store.append(make_record(rules=(_policy_rule(),)))
    corpus = store.current_records(
        actor=Actor("alice", ("employees",)),
        as_of=date(2026, 6, 1),
    )
    engine = TypedPolicyDecisionEngine()

    false_negation = engine.decide(
        "Is it false that £800 travel spend requires approval?",
        corpus,
        as_of=date(2026, 6, 1),
    )
    conditional = engine.decide(
        "If £800 travel spend includes VAT, does it require approval?",
        corpus,
        as_of=date(2026, 6, 1),
    )

    assert false_negation.parse.status == "ambiguous"
    assert false_negation.generation is None
    assert conditional.parse.status == "ambiguous"
    assert conditional.generation is None


def test_numeric_parser_requires_clear_subject(store: AuthorityStore) -> None:
    store.append(make_record(rules=(_policy_rule(),)))
    corpus = store.current_records(
        actor=Actor("alice", ("employees",)),
        as_of=date(2026, 6, 1),
    )
    outcome = TypedPolicyDecisionEngine().decide(
        "Does £800 require approval?",
        corpus,
        as_of=date(2026, 6, 1),
    )

    assert outcome.parse.status == "ambiguous"
    assert outcome.generation is None


def test_policy_rule_is_pinned_even_when_lexical_budget_would_miss_it(
    store: AuthorityStore,
) -> None:
    store.append(
        make_record(
            record_id="RULE-001",
            statement="A governed threshold exists. " + "background " * 80,
            rules=(_policy_rule(rule_id="RULE-001.approval-threshold"),),
        )
    )
    store.append_many(
        make_record(
            record_id=f"OTHER-{index:03d}",
            statement="Travel approval guidance is described here. " + "noise " * 80,
        )
        for index in range(6)
    )
    runtime = GovernedKnowledgeRuntime(
        store,
        generator=MustNotRunGenerator(),
        router=LocalRetrievalRouter(max_evidence_tokens=256, top_k=1),
    )

    result = runtime.ask(
        "Does £800 travel spend require approval?",
        actor=Actor("alice", ("employees",)),
        as_of=date(2026, 6, 1),
    )

    assert result.answer_status == "published_deterministic_policy_rule"
    assert result.retrieval.mode == "policy_rule"
    assert result.evidence.record_references == ("RULE-001:v1",)
    assert "[RULE-001:v1]" in result.answer


def test_policy_rule_with_unevaluated_condition_fails_closed(
    store: AuthorityStore,
) -> None:
    store.append(
        make_record(
            rules=(
                _policy_rule(
                    conditions=["before-booking"],
                ),
            )
        )
    )
    corpus = store.current_records(
        actor=Actor("alice", ("employees",)),
        as_of=date(2026, 6, 1),
    )

    outcome = TypedPolicyDecisionEngine().decide(
        "Does £800 travel spend require approval?",
        corpus,
        as_of=date(2026, 6, 1),
    )

    assert outcome.generation is None
    assert outcome.parse.status == "ambiguous"
    assert "conditions" in outcome.parse.reason


def test_policy_rule_with_unevaluated_exception_fails_closed(
    store: AuthorityStore,
) -> None:
    store.append(
        make_record(
            rules=(
                _policy_rule(
                    exceptions=["emergency-travel"],
                ),
            )
        )
    )
    corpus = store.current_records(
        actor=Actor("alice", ("employees",)),
        as_of=date(2026, 6, 1),
    )

    outcome = TypedPolicyDecisionEngine().decide(
        "Does £800 travel spend require approval?",
        corpus,
        as_of=date(2026, 6, 1),
    )

    assert outcome.generation is None
    assert outcome.parse.status == "ambiguous"
    assert "exceptions" in outcome.parse.reason


def test_competing_policy_rules_fail_closed(store: AuthorityStore) -> None:
    store.append_many(
        (
            make_record(
                record_id="RULE-A",
                rules=(_policy_rule(rule_id="RULE-A.threshold"),),
            ),
            make_record(
                record_id="RULE-B",
                rules=(_policy_rule(rule_id="RULE-B.threshold", threshold="900"),),
            ),
        )
    )
    corpus = store.current_records(
        actor=Actor("alice", ("employees",)),
        as_of=date(2026, 6, 1),
    )

    outcome = TypedPolicyDecisionEngine().decide(
        "Does £800 travel spend require approval?",
        corpus,
        as_of=date(2026, 6, 1),
    )

    assert outcome.generation is None
    assert outcome.parse.status == "ambiguous"
    assert "Multiple approved rules" in outcome.parse.reason


def test_below_comparator_is_evaluated_deterministically(
    store: AuthorityStore,
) -> None:
    store.append(
        make_record(
            rules=(
                _policy_rule(
                    comparator="<",
                    threshold="750",
                ),
            )
        )
    )
    runtime = GovernedKnowledgeRuntime(store, generator=MustNotRunGenerator())
    actor = Actor("alice", ("employees",))

    below = runtime.ask(
        "Does £700 travel spend require approval?",
        actor=actor,
        as_of=date(2026, 6, 1),
    )
    above = runtime.ask(
        "Does £800 travel spend require approval?",
        actor=actor,
        as_of=date(2026, 6, 1),
    )

    assert below.answer is not None and below.answer.startswith("Yes.")
    assert above.answer is not None and above.answer.startswith("No.")


def test_at_or_below_comparator_includes_threshold(
    store: AuthorityStore,
) -> None:
    store.append(
        make_record(
            rules=(
                _policy_rule(
                    comparator="<=",
                    threshold="750",
                ),
            )
        )
    )
    result = GovernedKnowledgeRuntime(store, generator=MustNotRunGenerator()).ask(
        "Does £750 travel spend require approval?",
        actor=Actor("alice", ("employees",)),
        as_of=date(2026, 6, 1),
    )

    assert result.answer is not None and result.answer.startswith("Yes.")


def _policy_rule(
    *,
    rule_id: str = "POL-001.approval-threshold",
    conditions: list[str] | None = None,
    exceptions: list[str] | None = None,
    comparator: str = ">",
    threshold: str = "750",
) -> dict[str, object]:
    return {
        "rule_id": rule_id,
        "subject": "travel-spend",
        "measure": "gross-amount",
        "unit": "GBP",
        "comparator": comparator,
        "threshold": threshold,
        "effect": "written-approval-required",
        "conditions": conditions or [],
        "exceptions": exceptions or [],
    }
