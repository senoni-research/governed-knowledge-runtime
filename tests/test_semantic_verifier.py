from __future__ import annotations

from datetime import date

from conftest import make_record

from gkr.ai import Generation, GenerationRequest
from gkr.authority import AuthorityStore
from gkr.context import ContextCompiler
from gkr.retrieval import LocalRetrievalRouter
from gkr.schemas import Actor
from gkr.verification import ModelSemanticVerifier


class VerifierGenerator:
    model_id = "fake-verifier"

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = 0

    def generate(self, request: GenerationRequest) -> Generation:
        self.calls += 1
        assert "closed-domain evidence verifier" in request.prompt
        return Generation(text=self.response, model=self.model_id)


def test_model_semantic_verifier_parses_strict_result(store: AuthorityStore) -> None:
    evidence = _evidence(store)
    verifier = ModelSemanticVerifier(
        VerifierGenerator('{"verdict":"unsupported","issues":["contradictory threshold"]}')
    )

    result = verifier.verify(
        candidate_answer="No approval is needed [POL-001:v1].",
        evidence=evidence,
    )

    assert result.verdict == "unsupported"
    assert result.issues == ("contradictory threshold",)


def test_model_semantic_verifier_fails_closed_on_invalid_output(
    store: AuthorityStore,
) -> None:
    evidence = _evidence(store)
    verifier = ModelSemanticVerifier(VerifierGenerator("Looks fine to me."))

    result = verifier.verify(
        candidate_answer="Approval is required [POL-001:v1].",
        evidence=evidence,
    )

    assert result.verdict == "error"


def test_deterministic_gate_rejects_opposing_polarity_before_model(
    store: AuthorityStore,
) -> None:
    evidence = _evidence(store)
    generator = VerifierGenerator('{"verdict":"supported","issues":[]}')
    verifier = ModelSemanticVerifier(generator)

    result = verifier.verify(
        candidate_answer=(
            "No, a £700 travel booking does not require special approval. "
            "Therefore, a £700 travel booking requires written approval."
        ),
        evidence=evidence,
    )

    assert result.verdict == "unsupported"
    assert result.verifier_model == "deterministic-contradiction-check"
    assert generator.calls == 0


def test_deterministic_gate_ignores_unrelated_negative_scope(
    store: AuthorityStore,
) -> None:
    evidence = _evidence(store)
    generator = VerifierGenerator('{"verdict":"supported","issues":[]}')
    verifier = ModelSemanticVerifier(generator)

    result = verifier.verify(
        candidate_answer=(
            "A production release requires automated tests and a rollback plan. "
            "No other evidence is relevant to production deployment on this date."
        ),
        evidence=evidence,
    )

    assert result.verdict == "supported"
    assert generator.calls == 1


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
