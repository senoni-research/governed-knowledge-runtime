from __future__ import annotations

from datetime import date

from conftest import make_record

from gkr.ai import Generation, GenerationRequest
from gkr.answers import EvidenceClaim
from gkr.authority import AuthorityStore
from gkr.context import ContextCompiler
from gkr.retrieval import LocalRetrievalRouter
from gkr.schemas import Actor
from gkr.verification import ModelSemanticVerifier


class VerifierGenerator:
    model_id = "fake-verifier"

    def __init__(self, response: str | list[str]) -> None:
        self.responses = [response] if isinstance(response, str) else response
        self.calls = 0

    def generate(self, request: GenerationRequest) -> Generation:
        assert "Classify textual entailment" in request.prompt
        response = self.responses[self.calls]
        self.calls += 1
        return Generation(text=response, model=self.model_id)


def test_model_semantic_verifier_parses_strict_result(store: AuthorityStore) -> None:
    evidence = _evidence(store)
    verifier = ModelSemanticVerifier(
        VerifierGenerator(
            '{"verdict":"unsupported","issues":["contradictory threshold"]}'
        )
    )

    result = verifier.verify(
        candidate_answer="No approval is needed [POL-001:v1].",
        evidence=evidence,
        claims=(_claim("No approval is needed."),),
    )

    assert result.verdict == "unsupported"
    assert result.issues == ("contradictory threshold",)
    assert result.claim_results[0].claim_index == 0


def test_model_semantic_verifier_fails_closed_on_invalid_output(
    store: AuthorityStore,
) -> None:
    evidence = _evidence(store)
    verifier = ModelSemanticVerifier(VerifierGenerator("Looks fine to me."))

    result = verifier.verify(
        candidate_answer="Approval is required [POL-001:v1].",
        evidence=evidence,
        claims=(_claim(),),
    )

    assert result.verdict == "error"


def test_model_semantic_verifier_maps_each_claim_and_fails_on_one_unsupported(
    store: AuthorityStore,
) -> None:
    evidence = _evidence(store)
    verifier = ModelSemanticVerifier(
        VerifierGenerator(
            [
                '{"verdict":"supported"}',
                '{"verdict":"unsupported"}',
            ]
        )
    )
    claims = (
        _claim("Approval is required."),
        _claim("Receipts are optional."),
    )

    result = verifier.verify(
        candidate_answer=(
            "Approval is required [POL-001:v1]. "
            "Receipts are optional [POL-001:v1]."
        ),
        evidence=evidence,
        claims=claims,
    )

    assert result.verdict == "unsupported"
    assert [item.verdict for item in result.claim_results] == [
        "supported",
        "unsupported",
    ]


def test_model_semantic_verifier_fails_closed_on_one_malformed_claim_result(
    store: AuthorityStore,
) -> None:
    evidence = _evidence(store)
    verifier = ModelSemanticVerifier(
        VerifierGenerator(
            [
                '{"verdict":"supported"}',
                "not JSON",
            ]
        )
    )

    result = verifier.verify(
        candidate_answer="Two claims.",
        evidence=evidence,
        claims=(_claim(), _claim("A second claim.")),
    )

    assert result.verdict == "error"
    assert "could not be parsed" in result.issues[0]


def test_deterministic_gate_rejects_opposing_polarity_before_model(
    store: AuthorityStore,
) -> None:
    evidence = _evidence(store)
    generator = VerifierGenerator('{"verdict":"supported"}')
    verifier = ModelSemanticVerifier(generator)

    result = verifier.verify(
        candidate_answer=(
            "No, a £700 travel booking does not require special approval. "
            "Therefore, a £700 travel booking requires written approval."
        ),
        evidence=evidence,
        claims=(_claim(),),
    )

    assert result.verdict == "unsupported"
    assert result.verifier_model == "deterministic-contradiction-check"
    assert result.claim_results[0].verdict == "unsupported"
    assert generator.calls == 0


def test_deterministic_gate_ignores_unrelated_negative_scope(
    store: AuthorityStore,
) -> None:
    evidence = _evidence(store)
    generator = VerifierGenerator('{"verdict":"supported"}')
    verifier = ModelSemanticVerifier(generator)

    result = verifier.verify(
        candidate_answer=(
            "A production release requires automated tests and a rollback plan. "
            "No other evidence is relevant to production deployment on this date."
        ),
        evidence=evidence,
        claims=(_claim(),),
    )

    assert result.verdict == "supported"
    assert generator.calls == 1


def test_deterministic_gate_accepts_equivalent_avoidance_prohibitions(
    store: AuthorityStore,
) -> None:
    evidence = _evidence(store)
    generator = VerifierGenerator('{"verdict":"supported"}')
    verifier = ModelSemanticVerifier(generator)

    result = verifier.verify(
        candidate_answer=(
            "No, you may not split a purchase into smaller bookings to avoid approval. "
            "Splitting a purchase into smaller bookings to avoid approval is not permitted."
        ),
        evidence=evidence,
        claims=(_claim(),),
    )

    assert result.verdict == "supported"
    assert generator.calls == 1


def test_deterministic_gate_rejects_opposing_avoidance_permissions(
    store: AuthorityStore,
) -> None:
    evidence = _evidence(store)
    generator = VerifierGenerator('{"verdict":"supported"}')
    verifier = ModelSemanticVerifier(generator)

    result = verifier.verify(
        candidate_answer=(
            "Splitting a purchase to avoid approval is allowed. "
            "Splitting a purchase to avoid approval is not allowed."
        ),
        evidence=evidence,
        claims=(_claim(),),
    )

    assert result.verdict == "unsupported"
    assert result.verifier_model == "deterministic-contradiction-check"
    assert result.claim_results[0].verdict == "unsupported"
    assert generator.calls == 0


def _claim(text: str = "Approval is required.") -> EvidenceClaim:
    return EvidenceClaim(
        claim=text,
        record_reference="POL-001:v1",
        supporting_passage="Spend above £500 requires written approval.",
    )


def _evidence(store: AuthorityStore):
    store.append(make_record())
    corpus = store.current_records(
        actor=Actor("alice", ("employees",)),
        as_of=date(2026, 6, 1),
    )
    plan = LocalRetrievalRouter().plan("What is required?", corpus)
    return ContextCompiler().compile(
        question="What is required?",
        corpus=corpus,
        plan=plan,
    )
