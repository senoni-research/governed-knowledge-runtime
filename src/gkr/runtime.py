from __future__ import annotations

import resource
import sys
import time
from dataclasses import dataclass, replace
from datetime import date, datetime
from typing import Any

from gkr.ai import Generation, GenerationRequest, LocalGenerator
from gkr.authority import AuthorityStore
from gkr.context import ContextCompiler, EvidenceBundle
from gkr.decision import (
    DecisionEngine,
    DecisionParseResult,
    TypedPolicyDecisionEngine,
)
from gkr.retrieval import LocalRetrievalRouter, RetrievalPlan
from gkr.schemas import Actor
from gkr.trace import ExecutionTrace, TraceStore
from gkr.verification import (
    CitationVerification,
    SemanticVerification,
    SemanticVerifier,
    verify_citations,
)


@dataclass(frozen=True)
class RuntimeResult:
    evidence: EvidenceBundle
    retrieval: RetrievalPlan
    generation: Generation | None
    verification: CitationVerification | None
    semantic_verification: SemanticVerification | None
    decision_parse: DecisionParseResult
    trace: ExecutionTrace | None = None

    @property
    def answer(self) -> str | None:
        if self.generation is None:
            return None
        if self.verification and self.verification.integrity == "fail":
            return None
        if self.generation.model.startswith("deterministic-"):
            return self.generation.text
        if self.semantic_verification is None:
            return None
        if self.semantic_verification.verdict != "supported":
            return None
        if self.semantic_verification.verifier_model == self.generation.model:
            return None
        return self.generation.text

    @property
    def answer_status(self) -> str:
        if self.generation is None:
            return "not_generated"
        if self.generation.model == "deterministic-refusal":
            return "refused_missing_authorized_evidence"
        if self.verification and self.verification.integrity == "fail":
            return "withheld_citation_integrity_failure"
        if self.generation.model == TypedPolicyDecisionEngine.model_id:
            return "published_deterministic_policy_rule"
        if self.semantic_verification is None:
            return "withheld_semantic_support_not_checked"
        if self.semantic_verification.verdict != "supported":
            return f"withheld_semantic_{self.semantic_verification.verdict}"
        if self.semantic_verification.verifier_model == self.generation.model:
            return "withheld_non_independent_verifier"
        return "published_local_verifier_supported"

    def to_dict(self, *, include_prompt: bool = False) -> dict[str, Any]:
        evidence = self.evidence.to_dict()
        if not include_prompt:
            evidence.pop("prompt")
        result = {
            "answer": self.answer,
            "answer_status": self.answer_status,
            "model": self.generation.model if self.generation else None,
            "decision_parse": self.decision_parse.to_dict(),
            "evidence": evidence,
            "retrieval": {
                "mode": self.retrieval.mode,
                "retriever_id": self.retrieval.retriever_id,
                "configuration": dict(self.retrieval.configuration),
                "reason": self.retrieval.reason,
                "available_records": self.retrieval.available_records,
                "selected_records": len(self.retrieval.hits),
                "estimated_evidence_tokens": self.retrieval.estimated_evidence_tokens,
            },
            "verification": self.verification.to_dict() if self.verification else None,
            "semantic_verification": (
                self.semantic_verification.to_dict() if self.semantic_verification else None
            ),
            "trace": self.trace.to_dict() if self.trace else None,
        }
        if self.generation and self.answer is None:
            result["withheld_candidate"] = self.generation.text
        return result


class GovernedKnowledgeRuntime:
    def __init__(
        self,
        store: AuthorityStore,
        *,
        generator: LocalGenerator | None = None,
        semantic_verifier: SemanticVerifier | None = None,
        decision_engine: DecisionEngine | None = None,
        router: LocalRetrievalRouter | None = None,
        compiler: ContextCompiler | None = None,
        trace_store: TraceStore | None = None,
    ) -> None:
        self.store = store
        self.generator = generator
        self.semantic_verifier = semantic_verifier
        self.decision_engine = decision_engine or TypedPolicyDecisionEngine()
        self.router = router or LocalRetrievalRouter()
        self.compiler = compiler or ContextCompiler()
        self.trace_store = trace_store

    def prepare(
        self,
        question: str,
        *,
        actor: Actor,
        as_of: date,
        known_at: datetime | None = None,
    ) -> tuple[EvidenceBundle, RetrievalPlan]:
        corpus = self.store.current_records(actor=actor, as_of=as_of, known_at=known_at)
        plan = self.router.plan(question, corpus)
        evidence = self.compiler.compile(question=question, corpus=corpus, plan=plan)
        return evidence, plan

    def ask(
        self,
        question: str,
        *,
        actor: Actor,
        as_of: date,
        known_at: datetime | None = None,
        max_tokens: int = 512,
    ) -> RuntimeResult:
        started_at = time.perf_counter()
        corpus = self.store.current_records(actor=actor, as_of=as_of, known_at=known_at)
        decision_outcome = self.decision_engine.decide(question, corpus, as_of=as_of)
        if decision_outcome.generation is not None:
            request = decision_outcome.parse.request
            if request is None:
                raise RuntimeError("A deterministic decision must identify its authority record")
            plan = self.router.plan_policy_rule(
                authority_reference=request.authority_reference,
                corpus=corpus,
            )
        else:
            plan = self.router.plan(question, corpus)
        evidence = self.compiler.compile(question=question, corpus=corpus, plan=plan)
        if evidence.missing_evidence:
            generation = Generation(
                text=(
                    "I cannot answer from the governed knowledge available to this caller "
                    "for the requested date."
                ),
                model="deterministic-refusal",
                metadata={"execution": "local", "reason": "missing_authorized_evidence"},
            )
        elif decision_outcome.generation is not None:
            generation = decision_outcome.generation
        elif self.generator is None:
            generation = None
        else:
            generation = self.generator.generate(
                GenerationRequest(prompt=evidence.prompt, max_tokens=max_tokens)
            )

        verification = (
            verify_citations(
                generation.text,
                evidence_references=evidence.record_references,
            )
            if generation
            else None
        )
        semantic_verification = (
            self.semantic_verifier.verify(
                candidate_answer=generation.text,
                evidence=evidence,
            )
            if (
                generation
                and not generation.model.startswith("deterministic-")
                and verification
                and verification.integrity == "pass"
                and self.semantic_verifier
            )
            else None
        )
        result = RuntimeResult(
            evidence=evidence,
            retrieval=plan,
            generation=generation,
            verification=verification,
            semantic_verification=semantic_verification,
            decision_parse=decision_outcome.parse,
        )
        trace = ExecutionTrace.create(
            question=question,
            actor=actor,
            as_of=as_of,
            evidence=evidence,
            retrieval=plan,
            decision_parse=decision_outcome.parse,
            generation=generation,
            citation_verification=verification,
            semantic_verification=semantic_verification,
            duration_ms=(time.perf_counter() - started_at) * 1000,
            peak_process_rss_bytes=_peak_process_rss_bytes(),
            publication_status=result.answer_status,
        )
        if self.trace_store is not None:
            self.trace_store.append(trace)
        return replace(result, trace=trace)


def _peak_process_rss_bytes() -> int | None:
    try:
        peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (OSError, ValueError):
        return None
    return peak_rss if sys.platform == "darwin" else peak_rss * 1024
