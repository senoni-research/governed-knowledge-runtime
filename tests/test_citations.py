from gkr.verification import verify_citations


def test_citation_parser_normalizes_common_record_id_label() -> None:
    result = verify_citations(
        "The gate applies [record_id: ENG-REL-001:v1].",
        evidence_references=("ENG-REL-001:v1",),
    )

    assert result.integrity == "pass"
    assert result.cited_references == ("ENG-REL-001:v1",)


def test_citation_parser_normalizes_common_citation_label() -> None:
    result = verify_citations(
        "The gate applies [CITATION: ENG-REL-001:v1].",
        evidence_references=("ENG-REL-001:v1",),
    )

    assert result.integrity == "pass"
    assert result.cited_references == ("ENG-REL-001:v1",)


def test_citation_parser_rejects_unretrieved_reference() -> None:
    result = verify_citations(
        "The gate applies [OTHER:v1].",
        evidence_references=("ENG-REL-001:v1",),
    )

    assert result.integrity == "fail"
    assert result.unknown_references == ("OTHER:v1",)


def test_citation_parser_returns_canonical_evidence_case() -> None:
    result = verify_citations(
        "The gate applies [fin-exp-001:v2].",
        evidence_references=("FIN-EXP-001:v2",),
    )

    assert result.integrity == "pass"
    assert result.cited_references == ("FIN-EXP-001:v2",)
    assert result.unknown_references == ()
