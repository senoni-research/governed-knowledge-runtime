from __future__ import annotations

from datetime import date

from conftest import make_record

from gkr.ai import Generation, GenerationRequest
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
    ) -> SemanticVerification:
        self.calls += 1
        assert candidate_answer
        assert evidence.record_references
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
    generator = FakeGenerator("Written approval is required above £500 [POL-001:v1].")
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
    generator = FakeGenerator("Approval is not needed [MADE-UP:v9].")
    runtime = GovernedKnowledgeRuntime(store, generator=generator)

    result = runtime.ask(
        "Is approval needed?",
        actor=Actor("alice", ("employees",)),
        as_of=date(2026, 6, 1),
    )

    assert result.verification is not None
    assert result.verification.integrity == "fail"
    assert result.verification.unknown_references == ("MADE-UP:v9",)
    assert result.answer is None
    assert result.to_dict()["withheld_candidate"] == "Approval is not needed [MADE-UP:v9]."


def test_runtime_withholds_semantically_unsupported_candidate(
    store: AuthorityStore,
) -> None:
    store.append(make_record())
    generator = FakeGenerator("Approval is not required [POL-001:v1].")
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
    generator = FakeGenerator("Approval is required [POL-001:v1].")
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
    assert result.generation.model == "deterministic-refusal"
    assert result.verification is not None
    assert result.verification.integrity == "not_applicable"
