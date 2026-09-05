from __future__ import annotations

import json
from datetime import date

from conftest import make_record

from gkr.ai import Generation, GenerationRequest
from gkr.answers import ABSTENTION_TEXT, EvidenceClaim
from gkr.authority import AuthorityStore
from gkr.context import EvidenceBundle
from gkr.runtime import GovernedKnowledgeRuntime
from gkr.schemas import Actor
from gkr.verification import SemanticVerification


class FakeGenerator:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.calls = 0

    @property
    def model_id(self) -> str:
        return "fake-local-model"

    def generate(self, request: GenerationRequest) -> Generation:
        self.calls += 1
        assert "NON-NEGOTIABLE RULES" in request.prompt
        assert "supporting_passage" in request.prompt
        return Generation(text=self.answer, model=self.model_id)


class FakeSemanticVerifier:
    def __init__(self, verdict: str, *, verifier_model: str = "fake-local-verifier") -> None:
        self.verdict = verdict
        self.verifier_model = verifier_model
        self.calls = 0

    def verify(
        self,
        *,
        candidate_answer: str,
        evidence: EvidenceBundle,
        claims: tuple[EvidenceClaim, ...],
    ) -> SemanticVerification:
        self.calls += 1
        assert candidate_answer
        assert evidence.record_references
        assert claims
        return SemanticVerification(
            verdict=self.verdict,  # type: ignore[arg-type]
            issues=() if self.verdict == "supported" else ("candidate is not supported",),
            verifier_model=self.verifier_model,
            raw_response="{}",
        )


def test_runtime_generates_locally_and_checks_citation_integrity(
    store: AuthorityStore,
) -> None:
    store.append(make_record())
    generator = FakeGenerator(
        _candidate("Written approval is required above £500.")
    )
    semantic_verifier = FakeSemanticVerifier("supported")
    runtime = GovernedKnowledgeRuntime(
        store,
        generator=generator,
        semantic_verifier=semantic_verifier,
    )

    result = runtime.ask(
        "What is the approval threshold?",
        actor=Actor("alice", ("employees",)),
        as_of=date(2026, 6, 1),
    )

    assert generator.calls == 1
    assert result.verification is not None
    assert result.verification.integrity == "pass"
    assert semantic_verifier.calls == 1
    assert result.answer is not None
    assert result.answer_status == "published_local_verifier_supported"
    assert result.to_dict()["model"] == "fake-local-model"


def test_runtime_flags_fabricated_citation(store: AuthorityStore) -> None:
    store.append(make_record())
    generator = FakeGenerator(
        _candidate("Approval is not needed.", reference="MADE-UP:v9")
    )
    runtime = GovernedKnowledgeRuntime(store, generator=generator)

    result = runtime.ask(
        "Is approval needed?",
        actor=Actor("alice", ("employees",)),
        as_of=date(2026, 6, 1),
    )

    assert result.verification is None
    assert result.answer is None
    assert result.answer_status == "withheld_claim_contract_invalid"
    assert "outside the authorized bundle" in result.claim_contract_issues[0]


def test_runtime_withholds_semantically_unsupported_candidate(
    store: AuthorityStore,
) -> None:
    store.append(make_record())
    generator = FakeGenerator(
        _candidate("Spend above £500 requires written approval.")
    )
    runtime = GovernedKnowledgeRuntime(
        store,
        generator=generator,
        semantic_verifier=FakeSemanticVerifier("unsupported"),
    )

    result = runtime.ask(
        "Is approval required?",
        actor=Actor("alice", ("employees",)),
        as_of=date(2026, 6, 1),
    )

    assert result.answer is None
    assert result.answer_status == "withheld_semantic_unsupported"


def test_runtime_does_not_publish_same_model_self_approval(
    store: AuthorityStore,
) -> None:
    store.append(make_record())
    generator = FakeGenerator(_candidate())
    runtime = GovernedKnowledgeRuntime(
        store,
        generator=generator,
        semantic_verifier=FakeSemanticVerifier(
            "supported",
            verifier_model=generator.model_id,
        ),
    )

    result = runtime.ask(
        "Is approval required?",
        actor=Actor("alice", ("employees",)),
        as_of=date(2026, 6, 1),
    )

    assert result.answer is None
    assert result.answer_status == "withheld_non_independent_verifier"


def test_runtime_refuses_without_invoking_model_when_evidence_is_unavailable(
    store: AuthorityStore,
) -> None:
    store.append(
        make_record(
            sensitivity="restricted",
            acl=("group:finance",),
        )
    )
    generator = FakeGenerator("This must never be returned.")
    runtime = GovernedKnowledgeRuntime(store, generator=generator)

    result = runtime.ask(
        "What is the rule?",
        actor=Actor("alice", ("employees",)),
        as_of=date(2026, 6, 1),
    )

    assert generator.calls == 0
    assert result.generation is not None
    assert result.generation.model == "deterministic-abstention"
    assert result.verification is not None
    assert result.verification.integrity == "not_applicable"
    assert result.answer == ABSTENTION_TEXT
    assert result.answer_status == "abstained_missing_authorized_evidence"


def test_runtime_renders_structured_abstention_without_model_prose(
    store: AuthorityStore,
) -> None:
    store.append(make_record())
    generator = FakeGenerator(
        'Model preamble {"outcome":"abstain","claims":[]} ignored epilogue'
    )
    runtime = GovernedKnowledgeRuntime(store, generator=generator)

    result = runtime.ask(
        "What is the Mars meal allowance?",
        actor=Actor("alice", ("employees",)),
        as_of=date(2026, 6, 1),
    )

    assert result.answer == ABSTENTION_TEXT
    assert result.answer_status == "abstained_insufficient_evidence"
    assert result.verification is not None
    assert result.verification.integrity == "not_applicable"
    assert "withheld_candidate" not in result.to_dict()


def test_runtime_rejects_non_exact_supporting_passage(
    store: AuthorityStore,
) -> None:
    store.append(make_record())
    generator = FakeGenerator(
        _candidate(
            "Approval is required.",
            passage="This passage was not in the authority record.",
        )
    )
    runtime = GovernedKnowledgeRuntime(store, generator=generator)

    result = runtime.ask(
        "Is approval required?",
        actor=Actor("alice", ("employees",)),
        as_of=date(2026, 6, 1),
    )

    assert result.answer is None
    assert result.answer_status == "withheld_claim_contract_invalid"
    assert "not an exact passage" in result.claim_contract_issues[0]


def test_runtime_renders_each_claim_with_its_own_reference(
    store: AuthorityStore,
) -> None:
    first = make_record()
    second = make_record(
        record_id="POL-002",
        statement="The handbook does not override approved policies.",
    )
    store.append_many((first, second))
    generator = FakeGenerator(
        json.dumps(
            {
                "outcome": "answer",
                "claims": [
                    {
                        "claim": "Spend above £500 requires written approval.",
                        "record_reference": "POL-001:v1",
                        "supporting_passage": first.statement,
                    },
                    {
                        "claim": "The handbook does not override approved policies.",
                        "record_reference": "POL-002:v1",
                        "supporting_passage": second.statement,
                    },
                ],
            }
        )
    )
    runtime = GovernedKnowledgeRuntime(
        store,
        generator=generator,
        semantic_verifier=FakeSemanticVerifier("supported"),
    )

    result = runtime.ask(
        "What does each policy say?",
        actor=Actor("alice", ("employees",)),
        as_of=date(2026, 6, 1),
    )

    assert result.answer is not None
    assert "[POL-001:v1]" in result.answer
    assert "[POL-002:v1]" in result.answer
    assert len(result.claims) == 2


def test_runtime_rejects_unquoted_exhaustive_claim(store: AuthorityStore) -> None:
    store.append(make_record())
    generator = FakeGenerator(
        _candidate("No other evidence is required.")
    )
    runtime = GovernedKnowledgeRuntime(store, generator=generator)

    result = runtime.ask(
        "What is required?",
        actor=Actor("alice", ("employees",)),
        as_of=date(2026, 6, 1),
    )

    assert result.answer_status == "withheld_claim_contract_invalid"
    assert "exhaustive assertion" in result.claim_contract_issues[0]


def _candidate(
    claim: str = "Approval is required.",
    *,
    reference: str = "POL-001:v1",
    passage: str = "Spend above £500 requires written approval.",
) -> str:
    return json.dumps(
        {
            "outcome": "answer",
            "claims": [
                {
                    "claim": claim,
                    "record_reference": reference,
                    "supporting_passage": passage,
                }
            ],
        }
    )
