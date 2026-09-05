from __future__ import annotations

import hashlib
import resource
import sys
import time
from dataclasses import dataclass, replace
from datetime import date, datetime
from typing import Any

from gkr.ai import Generation, GenerationRequest, LocalGenerator
from gkr.answers import (
    ABSTENTION_TEXT,
    AnswerOutcome,
    EvidenceClaim,
    parse_structured_candidate,
    render_claims,
    validate_claim_bindings,
)
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
    candidate_outcome: AnswerOutcome | None
    claims: tuple[EvidenceClaim, ...]
    claim_contract_issues: tuple[str, ...]
    trace: ExecutionTrace | None = None

    @property
    def answer(self) -> str | None:
        if self.generation is None:
            return None
        if self.claim_contract_issues:
            return None
        if self.candidate_outcome == "abstain":
            return self.generation.text
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
        if self.generation.model == "deterministic-abstention":
            return "abstained_missing_authorized_evidence"
        if self.candidate_outcome == "abstain":
            return "abstained_insufficient_evidence"
        if self.claim_contract_issues:
            return "withheld_claim_contract_invalid"
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
            "generation_metadata": (
                self.generation.metadata if self.generation else None
            ),
            "claims": [claim.to_dict() for claim in self.claims],
            "claim_contract_issues": list(self.claim_contract_issues),
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
        candidate_outcome: AnswerOutcome | None = None
        claims: tuple[EvidenceClaim, ...] = ()
        claim_contract_issues: tuple[str, ...] = ()
        if evidence.missing_evidence:
            generation = Generation(
                text=ABSTENTION_TEXT,
                model="deterministic-abstention",
                metadata={
                    "execution": "local",
                    "response_kind": "abstention",
                    "reason": "missing_authorized_evidence",
                },
            )
            candidate_outcome = "abstain"
        elif decision_outcome.generation is not None:
            generation = decision_outcome.generation
        elif self.generator is None:
            generation = None
        else:
            raw_generation = self.generator.generate(
                GenerationRequest(prompt=evidence.prompt, max_tokens=max_tokens)
            )
            generation, candidate_outcome, claims, claim_contract_issues = (
                _prepare_model_candidate(raw_generation, evidence=evidence)
            )

        if generation is None or claim_contract_issues:
            verification = None
        elif candidate_outcome == "abstain":
            verification = CitationVerification(
                integrity="not_applicable",
                cited_references=(),
                unknown_references=(),
                reason="A structured abstention contains no factual answer or citations.",
            )
        else:
            verification = verify_citations(
                generation.text,
                evidence_references=evidence.record_references,
            )
        semantic_verification = (
            self.semantic_verifier.verify(
                candidate_answer=generation.text,
                evidence=evidence,
                claims=claims,
            )
            if (
                generation
                and candidate_outcome == "answer"
                and not claim_contract_issues
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
            candidate_outcome=candidate_outcome,
            claims=claims,
            claim_contract_issues=claim_contract_issues,
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


def _prepare_model_candidate(
    raw_generation: Generation,
    *,
    evidence: EvidenceBundle,
) -> tuple[
    Generation,
    AnswerOutcome | None,
    tuple[EvidenceClaim, ...],
    tuple[str, ...],
]:
    metadata = {
        **raw_generation.metadata,
        "structured_response_sha256": hashlib.sha256(
            raw_generation.text.encode()
        ).hexdigest(),
    }
    try:
        candidate = parse_structured_candidate(raw_generation.text)
    except ValueError as exc:
        return (
            Generation(
                text=raw_generation.text,
                model=raw_generation.model,
                metadata={**metadata, "response_kind": "invalid"},
            ),
            None,
            (),
            (f"Structured candidate could not be parsed: {exc}",),
        )

    if candidate.outcome == "abstain":
        return (
            Generation(
                text=ABSTENTION_TEXT,
                model=raw_generation.model,
                metadata={
                    **metadata,
                    "response_kind": "abstention",
                    "claim_count": 0,
                },
            ),
            "abstain",
            (),
            (),
        )

    issues = validate_claim_bindings(candidate.claims, evidence=evidence)
    if issues:
        return (
            Generation(
                text=raw_generation.text,
                model=raw_generation.model,
                metadata={
                    **metadata,
                    "response_kind": "invalid",
                    "claim_count": len(candidate.claims),
                },
            ),
            "answer",
            candidate.claims,
            issues,
        )
    return (
        Generation(
            text=render_claims(candidate.claims),
            model=raw_generation.model,
            metadata={
                **metadata,
                "response_kind": "claim_answer",
                "claim_count": len(candidate.claims),
            },
        ),
        "answer",
        candidate.claims,
        (),
    )


def _peak_process_rss_bytes() -> int | None:
    try:
        peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (OSError, ValueError):
        return None
    return peak_rss if sys.platform == "darwin" else peak_rss * 1024
