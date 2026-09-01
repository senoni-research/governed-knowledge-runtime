from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Literal

from gkr.authority import AuthorizedCorpus
from gkr.schemas import KnowledgeRecord

RetrievalMode = Literal["policy_rule", "full_context", "lexical", "no_evidence"]


@dataclass(frozen=True)
class RetrievalHit:
    record: KnowledgeRecord
    score: float | None


@dataclass(frozen=True)
class RetrievalPlan:
    mode: RetrievalMode
    retriever_id: str
    configuration: tuple[tuple[str, str], ...]
    hits: tuple[RetrievalHit, ...]
    available_records: int
    estimated_evidence_tokens: int
    reason: str


class LocalRetrievalRouter:
    """Choose full context when it fits; otherwise use deterministic local BM25."""

    def __init__(self, *, max_evidence_tokens: int = 12_000, top_k: int = 12) -> None:
        if max_evidence_tokens < 256:
            raise ValueError("max_evidence_tokens must be at least 256")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        self.max_evidence_tokens = max_evidence_tokens
        self.top_k = top_k

    def plan(self, question: str, corpus: AuthorizedCorpus) -> RetrievalPlan:
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty")
        configuration = (
            ("max_evidence_tokens", str(self.max_evidence_tokens)),
            ("top_k", str(self.top_k)),
            ("token_estimator", "canonical-json-chars-div-3.5"),
        )
        if not corpus.records:
            return RetrievalPlan(
                mode="no_evidence",
                retriever_id="authorization-empty-v1",
                configuration=configuration,
                hits=(),
                available_records=0,
                estimated_evidence_tokens=0,
                reason="No approved, temporally valid records are authorized for this actor.",
            )

        full_context_tokens = sum(_estimate_record_tokens(record) for record in corpus.records)
        if full_context_tokens <= self.max_evidence_tokens:
            return RetrievalPlan(
                mode="full_context",
                retriever_id="full-authorized-context-v1",
                configuration=configuration,
                hits=tuple(RetrievalHit(record=record, score=None) for record in corpus.records),
                available_records=len(corpus.records),
                estimated_evidence_tokens=full_context_tokens,
                reason="The complete authorized corpus fits within the evidence budget.",
            )

        ranked = _rank_bm25(question, corpus.records)
        selected: list[RetrievalHit] = []
        selected_tokens = 0
        for record, score in ranked:
            record_tokens = _estimate_record_tokens(record)
            if selected_tokens + record_tokens > self.max_evidence_tokens:
                continue
            selected.append(RetrievalHit(record=record, score=score))
            selected_tokens += record_tokens
            if len(selected) == self.top_k:
                break

        if not selected:
            reason = (
                "Lexical matches existed, but every matching record exceeded the evidence budget."
                if ranked
                else "No authorized record matched the lexical query."
            )
            return RetrievalPlan(
                mode="no_evidence",
                retriever_id="local-bm25-v1",
                configuration=configuration,
                hits=(),
                available_records=len(corpus.records),
                estimated_evidence_tokens=0,
                reason=reason,
            )
        return RetrievalPlan(
            mode="lexical",
            retriever_id="local-bm25-v1",
            configuration=configuration,
            hits=tuple(selected),
            available_records=len(corpus.records),
            estimated_evidence_tokens=selected_tokens,
            reason="The authorized corpus exceeded the budget; local BM25 selected evidence.",
        )

    def plan_policy_rule(
        self,
        *,
        authority_reference: str,
        corpus: AuthorizedCorpus,
    ) -> RetrievalPlan:
        record = next(
            (
                candidate
                for candidate in corpus.records
                if candidate.reference == authority_reference
            ),
            None,
        )
        if record is None:
            raise ValueError("Policy decision references a record outside the authorized corpus")
        return RetrievalPlan(
            mode="policy_rule",
            retriever_id="authorized-policy-rule-v1",
            configuration=(("authority_reference", authority_reference),),
            hits=(RetrievalHit(record=record, score=None),),
            available_records=len(corpus.records),
            estimated_evidence_tokens=_estimate_record_tokens(record),
            reason="The typed decision pinned its approved authority record into evidence.",
        )


def _rank_bm25(
    question: str,
    records: tuple[KnowledgeRecord, ...],
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[tuple[KnowledgeRecord, float]]:
    query_terms = _tokenize(question)
    if not query_terms:
        return []

    documents = [_record_terms(record) for record in records]
    document_frequencies = Counter(
        term for document in documents for term in set(document)
    )
    average_length = sum(len(document) for document in documents) / len(documents)
    query_phrase = " ".join(query_terms)
    scored: list[tuple[KnowledgeRecord, float]] = []

    for record, document in zip(records, documents, strict=True):
        frequencies = Counter(document)
        score = 0.0
        for term in query_terms:
            frequency = frequencies[term]
            if frequency == 0:
                continue
            document_frequency = document_frequencies[term]
            inverse_document_frequency = math.log(
                1 + (len(documents) - document_frequency + 0.5) / (document_frequency + 0.5)
            )
            denominator = frequency + k1 * (
                1 - b + b * len(document) / max(average_length, 1)
            )
            score += inverse_document_frequency * frequency * (k1 + 1) / denominator

        searchable_text = _record_text(record).casefold()
        if query_phrase and query_phrase in " ".join(_tokenize(searchable_text)):
            score += 2.0
        if score > 0:
            scored.append((record, score))

    return sorted(scored, key=lambda item: (-item[1], item[0].reference))


def _record_text(record: KnowledgeRecord) -> str:
    relations = " ".join(
        f"{relation.subject} {relation.predicate} {relation.object}"
        for relation in record.relations
    )
    rules = " ".join(
        " ".join(
            (
                rule.rule_id,
                rule.subject,
                rule.measure,
                rule.unit,
                rule.comparator,
                format(rule.threshold, "f"),
                rule.effect,
                " ".join(rule.conditions),
                " ".join(rule.exceptions),
            )
        )
        for rule in record.rules
    )
    return " ".join(
        (
            record.record_id,
            record.domain,
            record.title,
            " ".join(record.aliases),
            record.statement,
            " ".join(record.entities),
            relations,
            rules,
        )
    )


def _record_terms(record: KnowledgeRecord) -> list[str]:
    return _tokenize(_record_text(record))


def _tokenize(value: str) -> list[str]:
    return re.findall(r"[\w£$€.-]+", value.casefold())


def _estimate_record_tokens(record: KnowledgeRecord) -> int:
    # Conservative local estimate that avoids requiring a model tokenizer during routing.
    return max(1, math.ceil(len(record.canonical_json()) / 3.5))
