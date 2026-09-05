from __future__ import annotations

import json
from datetime import date

import pytest
from conftest import make_record

from gkr.answers import (
    ABSTENTION_TEXT,
    EvidenceClaim,
    parse_structured_candidate,
    render_claims,
    validate_claim_bindings,
)
from gkr.authority import AuthorityStore
from gkr.context import ContextCompiler
from gkr.retrieval import LocalRetrievalRouter
from gkr.schemas import Actor


def test_parse_answer_and_render_only_claim_text_and_citation() -> None:
    candidate = parse_structured_candidate(
        json.dumps(
            {
                "outcome": "answer",
                "claims": [
                    {
                        "claim": "Approval is required.",
                        "record_reference": "POL-001:v1",
                        "supporting_passage": "Spend above £500 requires approval.",
                    }
                ],
            }
        )
    )

    assert render_claims(candidate.claims) == "Approval is required. [POL-001:v1]"
    assert "Spend above" not in render_claims(candidate.claims)


def test_parse_normalizes_evidence_label_around_exact_reference() -> None:
    candidate = parse_structured_candidate(
        json.dumps(
            {
                "outcome": "answer",
                "claims": [
                    {
                        "claim": "Approval is required.",
                        "record_reference": "EVIDENCE-2 [POL-001:v1]",
                        "supporting_passage": "Spend above £500 requires approval.",
                    }
                ],
            }
        )
    )

    assert candidate.claims[0].record_reference == "POL-001:v1"


def test_parse_abstention_requires_no_claims() -> None:
    candidate = parse_structured_candidate('{"outcome":"abstain","claims":[]}')

    assert candidate.outcome == "abstain"
    assert candidate.claims == ()
    assert ABSTENTION_TEXT

    with pytest.raises(ValueError, match="must not contain claims"):
        parse_structured_candidate(
            '{"outcome":"abstain","claims":['
            '{"claim":"No.","record_reference":"POL-001:v1",'
            '"supporting_passage":"No."}]}'
        )


def test_claim_binding_requires_authorized_reference_and_exact_passage(
    store: AuthorityStore,
) -> None:
    record = make_record()
    store.append(record)
    corpus = store.current_records(
        actor=Actor("alice", ("employees",)),
        as_of=date(2026, 6, 1),
    )
    plan = LocalRetrievalRouter().plan("What is required?", corpus)
    evidence = ContextCompiler().compile(
        question="What is required?",
        corpus=corpus,
        plan=plan,
    )

    exact = EvidenceClaim(
        claim="Approval is required.",
        record_reference=record.reference,
        supporting_passage=record.statement,
    )
    fabricated = EvidenceClaim(
        claim="Approval is required.",
        record_reference="OTHER:v1",
        supporting_passage=record.statement,
    )
    altered = EvidenceClaim(
        claim="Approval is required.",
        record_reference=record.reference,
        supporting_passage=record.statement + " Altered.",
    )

    assert validate_claim_bindings((exact,), evidence=evidence) == ()
    assert "outside the authorized bundle" in validate_claim_bindings(
        (fabricated,), evidence=evidence
    )[0]
    assert "not an exact passage" in validate_claim_bindings(
        (altered,), evidence=evidence
    )[0]


@pytest.mark.parametrize(
    "claim_text",
    [
        "No other evidence is required.",
        "No additional evidence is required.",
        "Automated tests are the sole requirements.",
    ],
)
def test_claim_binding_rejects_unquoted_exhaustive_scope(
    store: AuthorityStore,
    claim_text: str,
) -> None:
    record = make_record()
    store.append(record)
    corpus = store.current_records(
        actor=Actor("alice", ("employees",)),
        as_of=date(2026, 6, 1),
    )
    plan = LocalRetrievalRouter().plan("What is required?", corpus)
    evidence = ContextCompiler().compile(
        question="What is required?",
        corpus=corpus,
        plan=plan,
    )
    claim = EvidenceClaim(
        claim=claim_text,
        record_reference=record.reference,
        supporting_passage=record.statement,
    )

    assert "exhaustive assertion" in validate_claim_bindings(
        (claim,), evidence=evidence
    )[0]


def test_claim_binding_rejects_material_terms_absent_from_passage(
    store: AuthorityStore,
) -> None:
    record = make_record(
        statement="A production release requires automated tests and a rollback plan."
    )
    store.append(record)
    corpus = store.current_records(
        actor=Actor("alice", ("employees",)),
        as_of=date(2026, 6, 1),
    )
    plan = LocalRetrievalRouter().plan("Can the handbook override this?", corpus)
    evidence = ContextCompiler().compile(
        question="Can the handbook override this?",
        corpus=corpus,
        plan=plan,
    )
    claim = EvidenceClaim(
        claim="The public handbook index does not override production requirements.",
        record_reference=record.reference,
        supporting_passage=record.statement,
    )

    issues = validate_claim_bindings((claim,), evidence=evidence)

    assert "material terms absent" in issues[0]
    assert "handbook" in issues[0]


@pytest.mark.parametrize(
    ("claim_text", "passage"),
    [
        ("Finance owns releases.", "Security owns releases."),
        (
            "The finance team owns production releases.",
            "The security team owns production releases.",
        ),
    ],
)
def test_claim_binding_rejects_mismatched_entity(
    store: AuthorityStore,
    claim_text: str,
    passage: str,
) -> None:
    record = make_record(statement=passage)
    store.append(record)
    corpus = store.current_records(
        actor=Actor("alice", ("employees",)),
        as_of=date(2026, 6, 1),
    )
    plan = LocalRetrievalRouter().plan("Who owns releases?", corpus)
    evidence = ContextCompiler().compile(
        question="Who owns releases?",
        corpus=corpus,
        plan=plan,
    )
    claim = EvidenceClaim(
        claim=claim_text,
        record_reference=record.reference,
        supporting_passage=record.statement,
    )

    issues = validate_claim_bindings((claim,), evidence=evidence)

    assert any("material terms absent" in issue and "finance" in issue for issue in issues)
