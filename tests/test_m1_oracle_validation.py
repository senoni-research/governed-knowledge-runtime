from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from gkr.m1_oracle_validation import validate_m1_oracles

FIXTURE = Path("tests/fixtures/m1/conformance-cases.jsonl")
QUERY_CLASSES = {
    "exact_factual",
    "semantic_paraphrase",
    "numeric_conditional",
    "temporal",
    "authorization",
    "unknown_oos",
    "multi_record",
    "adversarial_conflicting",
}


def _cases() -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in FIXTURE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_cases(path: Path, cases: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases),
        encoding="utf-8",
    )


def _mutate(tmp_path: Path, case_id: str, mutator: object) -> Path:
    cases = _cases()
    matched = False
    for case in cases:
        if case["case_id"] == case_id:
            mutator(case)
            matched = True
    assert matched, case_id
    path = tmp_path / "cases.jsonl"
    _write_cases(path, cases)
    return path


def _scoring_roles() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    question = {
        "method": "model",
        "session_id": "grok-question-session",
        "identity": "grok",
        "model_family_id": "xai-grok",
        "model_id": "grok-4",
        "model_revision": "grok-high-fast-014",
        "prompt_retained": True,
        "prompt_sha256": "5c75c3fe8f948e6404086214965f305cbe176398380e8f32b5e4cbf0c067e21d",
        "independent_from_retriever_tuning": True,
    }
    oracle = {
        **question,
        "session_id": "grok-oracle-session",
    }
    review = {
        "status": "completed",
        "semantically_reviewed": True,
        "reviewer_kind": "model",
        "reviewer_identity": "chatgpt",
        "reviewer_session_id": "chatgpt-review-session",
        "reviewer_model_family_id": "openai-gpt",
        "reviewer_model_id": "gpt-4.1",
        "reviewer_model_revision": "chatgpt-015",
        "reviewer_prompt_sha256": "aa" * 32,
        "review_sha256": "bb" * 32,
        "independent_from_retriever_tuning": True,
    }
    return question, oracle, review


def test_conformance_fixture_validates_against_frozen_corpus() -> None:
    report = validate_m1_oracles(FIXTURE)

    assert report["cases"] == 16
    assert report["independent_scenarios"] == {
        "development": 8,
        "validation": 4,
        "test": 4,
    }
    assert set(report["query_classes"]) == QUERY_CLASSES
    assert all(count == 2 for count in report["query_classes"].values())
    assert report["semantic_support_established"] is False
    assert "does not establish semantic support" in report["note"]

    cases = _cases()
    assert all(case["case_kind"] == "conformance" for case in cases)
    assert all("conformance-fixture" in case["tags"] for case in cases)
    assert all("oracle_notes" not in case for case in cases)
    assert all(case["scope"]["known_at"] for case in cases)
    assert all(case["oracle_review"]["status"] == "pending" for case in cases)
    assert all(case["oracle_review"]["semantically_reviewed"] is False for case in cases)
    assert all(case["oracle_review"]["reviewer_kind"] is None for case in cases)
    assert all(case["hash_profile_id"] == "gkr-m1-hash-profile-v1" for case in cases)
    session = cases[0]["question_authorship"]["session_id"]
    assert all(case["question_authorship"]["session_id"] == session for case in cases)
    assert all(case["oracle_authorship"]["session_id"] == session for case in cases)

    known_at_case = next(case for case in cases if case["case_id"] == "m1-cf-mileage-01")
    assert known_at_case["oracle"]["sufficient_reference_sets"] == [["FIN-MILEAGE:v1"]]
    assert "FIN-MILEAGE:v2" in known_at_case["oracle"]["forbidden_references"]
    assert known_at_case["scope"]["as_of"] == "2026-04-01"
    assert known_at_case["scope"]["known_at"] == "2026-06-01T00:00:00Z"

    visible = next(case for case in cases if case["case_id"] == "m1-cf-payroll-01")
    hidden = next(case for case in cases if case["case_id"] == "m1-cf-payroll-02")
    assert visible["question"] == hidden["question"]
    assert visible["oracle"]["publication"] == "published"
    assert hidden["oracle"]["publication"] == "refused"
    assert hidden["oracle"]["disposition_reason"] == "unauthorized_actor"

    oos = next(case for case in cases if case["case_id"] == "m1-cf-share-options-01")
    assert oos["oracle"]["sufficient_reference_sets"] == []
    assert oos["oracle"]["publication"] != "published"

    conflict = next(case for case in cases if case["case_id"] == "m1-cf-overtime-01")
    assert set(conflict["oracle"]["required_citations"]) == {"HR-OVERTIME:v1", "OPS-OVERTIME:v1"}

    multi = next(case for case in cases if case["case_id"] == "m1-cf-release-planning-01")
    assert multi["oracle"]["sufficient_reference_sets"] == [
        ["ENG-REL-GATE:v1", "ENG-ROLLBACK-PLAN:v1"]
    ]

    alts = next(case for case in cases if case["case_id"] == "m1-cf-rollback-01")
    assert alts["query_class"] == "exact_factual"
    assert alts["question"].startswith("Name one approved record that describes rollback")
    assert alts["oracle"]["sufficient_reference_sets"] == [
        ["ENG-REL-GATE:v1"],
        ["ENG-ROLLBACK-PLAN:v1"],
    ]
    assert all(case["question_authorship"]["method"] == "model" for case in cases)
    assert all(case["oracle_authorship"]["method"] == "model" for case in cases)
    assert all(case["question_authorship"]["model_family_id"] == "xai-grok" for case in cases)
    assert all(case["question_authorship"]["model_id"] == "grok-4" for case in cases)
    assert all(case["question_authorship"]["prompt_retained"] is False for case in cases)
    leak_tokens = (
        "visible",
        "hidden",
        "conflict",
        "refused",
        "supported",
        "answer",
        "alts",
        "oos",
        "exact",
        "para",
        "num",
        "temp",
        "multi",
        "adv",
        "knownat",
        "asof",
        "700",
        "350",
    )
    for case in cases:
        blob = f"{case['case_id']} {case['scenario_id']}"
        assert all(token not in blob for token in leak_tokens)


def test_validator_does_not_set_semantic_review_flag() -> None:
    original = next(case for case in _cases() if case["case_id"] == "m1-cf-travel-approval-01")
    validate_m1_oracles(FIXTURE)
    reread = next(case for case in _cases() if case["case_id"] == "m1-cf-travel-approval-01")
    assert original["oracle_review"] == reread["oracle_review"]
    assert reread["oracle_review"]["semantically_reviewed"] is False


def test_sufficient_references_match_current_records() -> None:
    import tempfile
    from datetime import UTC, date, datetime

    from gkr.authority import AuthorityStore
    from gkr.m1_oracle_validation import parse_reference, temporally_selected
    from gkr.schemas import Actor

    cases = _cases()
    with tempfile.TemporaryDirectory() as tmp:
        store = AuthorityStore(Path(tmp) / "authority.sqlite")
        store.import_jsonl("evaluation/m1/corpus/authority.jsonl")
        by_id: dict[str, list] = {}
        from gkr.m1_corpus import load_authority_records

        for record in load_authority_records("evaluation/m1/corpus/authority.jsonl"):
            by_id.setdefault(record.record_id, []).append(record)
        for case in cases:
            scope = case["scope"]
            actor = Actor(scope["actor"], tuple(scope["groups"]))
            as_of = date.fromisoformat(scope["as_of"])
            known_at = datetime.fromisoformat(scope["known_at"].replace("Z", "+00:00")).astimezone(
                UTC
            )
            authorized = {
                record.reference
                for record in store.current_records(
                    actor=actor, as_of=as_of, known_at=known_at
                ).records
            }
            for group in case["oracle"]["sufficient_reference_sets"]:
                for reference in group:
                    parsed = parse_reference(reference)
                    assert parsed is not None
                    record_id, _version = parsed
                    selected = temporally_selected(
                        by_id[record_id], as_of=as_of, known_at=known_at
                    )
                    assert selected is not None
                    assert selected.reference == reference
                    assert reference in authorized
        store.close()


@pytest.mark.parametrize(
    ("case_id", "mutator", "fragment"),
    [
        (
            "m1-cf-travel-approval-01",
            lambda case: case["oracle"].__setitem__(
                "sufficient_reference_sets", [["NO-SUCH-RECORD:v1"]]
            ),
            "does not exist in the corpus",
        ),
        (
            "m1-cf-travel-approval-01",
            lambda case: (
                case["oracle"].__setitem__(
                    "sufficient_reference_sets", [["FIN-EXP-THRESHOLD:v1"]]
                ),
                case["oracle"].__setitem__("forbidden_references", ["FIN-EXP-THRESHOLD:v2"]),
            ),
            "not the temporally selected version",
        ),
        (
            "m1-cf-travel-approval-01",
            lambda case: (
                case["oracle"].__setitem__(
                    "sufficient_reference_sets", [["FIN-CARD-LIMIT:v2"]]
                ),
                case["oracle"].__setitem__("forbidden_references", []),
            ),
            "is future-only at as_of",
        ),
        (
            "m1-cf-travel-approval-01",
            lambda case: case["oracle"].__setitem__(
                "sufficient_reference_sets", [["FIN-PAYROLL-CAL:v2"]]
            ),
            "published case sufficient reference .* not visible to the declared actor",
        ),
        (
            "m1-cf-overtime-01",
            lambda case: case["oracle"].__setitem__(
                "sufficient_reference_sets", [["FIN-PAYROLL-CAL:v2"]]
            ),
            "withheld case sufficient reference .* not visible to the declared actor",
        ),
        (
            "m1-cf-share-options-01",
            lambda case: case["oracle"].__setitem__(
                "sufficient_reference_sets", [["FIN-EXP-THRESHOLD:v2"]]
            ),
            "unknown_oos cases must have zero sufficient sets",
        ),
        (
            "m1-cf-payroll-02",
            lambda case: case["oracle"].__delitem__("disposition_reason"),
            "disposition_reason",
        ),
        (
            "m1-cf-travel-approval-01",
            lambda case: case["oracle"].__setitem__(
                "required_citations", ["SEC-VISITOR-LANYARD:v1"]
            ),
            "required_citations are not a subset of any sufficient set",
        ),
        (
            "m1-cf-travel-approval-01",
            lambda case: case["oracle"].__setitem__(
                "forbidden_claims", list(case["oracle"]["required_claims"])
            ),
            "required_claims and forbidden_claims overlap",
        ),
    ],
)
def test_oracle_mutations_fail_on_temp_copies(
    tmp_path: Path,
    case_id: str,
    mutator: object,
    fragment: str,
) -> None:
    path = _mutate(tmp_path, case_id, mutator)
    with pytest.raises(ValueError, match=fragment) as captured:
        validate_m1_oracles(path)
    assert f"{path}:" in str(captured.value)
    line_token = str(captured.value).split(f"{path}:", 1)[1].split(":", 1)[0]
    assert line_token.isdigit()


def test_scoring_pending_review_is_rejected(tmp_path: Path) -> None:
    question, oracle, review = _scoring_roles()
    review["status"] = "pending"
    review["semantically_reviewed"] = False
    for key in (
        "reviewer_kind",
        "reviewer_identity",
        "reviewer_session_id",
        "reviewer_model_family_id",
        "reviewer_model_id",
        "reviewer_model_revision",
        "reviewer_prompt_sha256",
        "review_sha256",
        "independent_from_retriever_tuning",
    ):
        review[key] = None
    path = _mutate(
        tmp_path,
        "m1-cf-travel-approval-01",
        lambda case: (
            case.__setitem__("case_kind", "scoring"),
            case.__setitem__("question_authorship", question),
            case.__setitem__("oracle_authorship", oracle),
            case.__setitem__("oracle_review", review),
        ),
    )
    with pytest.raises(ValueError, match="completed"):
        validate_m1_oracles(path)


def test_equal_author_reviewer_sessions_rejected(tmp_path: Path) -> None:
    question, oracle, review = _scoring_roles()
    review["reviewer_session_id"] = question["session_id"]
    path = _mutate(
        tmp_path,
        "m1-cf-travel-approval-01",
        lambda case: (
            case.__setitem__("case_kind", "scoring"),
            case.__setitem__("question_authorship", question),
            case.__setitem__("oracle_authorship", oracle),
            case.__setitem__("oracle_review", review),
        ),
    )
    with pytest.raises(ValueError, match="pairwise distinct"):
        validate_m1_oracles(path)


def test_reviewer_model_family_must_differ(tmp_path: Path) -> None:
    question, oracle, review = _scoring_roles()
    review["reviewer_model_family_id"] = "xai-grok"
    review["reviewer_model_id"] = "grok-4"
    path = _mutate(
        tmp_path,
        "m1-cf-travel-approval-01",
        lambda case: (
            case.__setitem__("case_kind", "scoring"),
            case.__setitem__("question_authorship", deepcopy(question)),
            case.__setitem__("oracle_authorship", deepcopy(oracle)),
            case.__setitem__("oracle_review", review),
        ),
    )
    with pytest.raises(ValueError, match="reviewer family must differ"):
        validate_m1_oracles(path)


def test_missing_oracle_author_provenance_rejected(tmp_path: Path) -> None:
    path = _mutate(
        tmp_path,
        "m1-cf-travel-approval-01",
        lambda case: case.__delitem__("oracle_authorship"),
    )
    with pytest.raises(ValueError, match="oracle_authorship"):
        validate_m1_oracles(path)


def _duplicate_variant(tmp_path: Path, mutator: object) -> Path:
    cases = _cases()
    original = deepcopy(
        next(case for case in cases if case["case_id"] == "m1-cf-travel-approval-01")
    )
    clone = deepcopy(original)
    clone["case_id"] = "m1-cf-travel-approval-01-b"
    clone["variant_id"] = "b"
    clone["question"] = original["question"] + " (variant b)"
    from gkr.m1_validation import question_digest

    clone["question_sha256"] = question_digest(clone["question"])
    mutator(clone)
    cases.append(clone)
    path = tmp_path / "cases.jsonl"
    _write_cases(path, cases)
    return path


@pytest.mark.parametrize(
    ("mutator", "fragment"),
    [
        (lambda case: case.__setitem__("split", "validation"), "must share split"),
        (lambda case: case.__setitem__("query_class", "temporal"), "must share split"),
        (lambda case: case.__setitem__("case_kind", "scoring"), "must share split"),
        (
            lambda case: case["scope"].__setitem__("actor", "sam"),
            "must share split",
        ),
        (
            lambda case: case["oracle"].__setitem__("strict_answer", "changed"),
            "must share split",
        ),
        (
            lambda case: case["oracle_authorship"].__setitem__(
                "session_id", "other-oracle-session"
            ),
            "must share split",
        ),
        (
            lambda case: case["oracle_review"].__setitem__("status", "completed"),
            "must share split",
        ),
    ],
)
def test_scenario_identity_mutations_fail(
    tmp_path: Path, mutator: object, fragment: str
) -> None:
    path = _duplicate_variant(tmp_path, mutator)
    with pytest.raises(ValueError, match=fragment):
        validate_m1_oracles(path)


def test_duplicate_variant_id_within_scenario_fails(tmp_path: Path) -> None:
    path = _duplicate_variant(tmp_path, lambda case: case.__setitem__("variant_id", "a"))
    with pytest.raises(ValueError, match="duplicate variant_id"):
        validate_m1_oracles(path)


def test_scenario_identity_permits_question_hash_and_authorship(tmp_path: Path) -> None:
    path = _duplicate_variant(
        tmp_path,
        lambda case: case["question_authorship"].__setitem__(
            "session_id", "other-question-session"
        ),
    )
    report = validate_m1_oracles(path)
    assert report["cases"] == 17
    assert report["independent_scenarios"]["development"] == 8


def test_supported_empty_sets_fail_runtime(tmp_path: Path) -> None:
    path = _mutate(
        tmp_path,
        "m1-cf-travel-approval-01",
        lambda case: (
            case["oracle"].__setitem__("sufficient_reference_sets", []),
            case["oracle"].__setitem__("required_citations", []),
        ),
    )
    with pytest.raises(ValueError, match="supported requires at least one sufficient set"):
        validate_m1_oracles(path)


def test_unauthorized_actor_with_sufficient_set_fails(tmp_path: Path) -> None:
    path = _mutate(
        tmp_path,
        "m1-cf-payroll-02",
        lambda case: case["oracle"].__setitem__(
            "sufficient_reference_sets", [["FIN-EXP-THRESHOLD:v2"]]
        ),
    )
    with pytest.raises(ValueError, match="unauthorized_actor cases must have zero"):
        validate_m1_oracles(path)


def test_no_authorized_evidence_with_sufficient_set_fails(tmp_path: Path) -> None:
    path = _mutate(
        tmp_path,
        "m1-cf-share-options-01",
        lambda case: (
            case.__setitem__("query_class", "authorization"),
            case["oracle"].__setitem__("disposition_reason", "no_authorized_evidence"),
            case["oracle"].__setitem__(
                "sufficient_reference_sets", [["FIN-EXP-THRESHOLD:v2"]]
            ),
        ),
    )
    with pytest.raises(ValueError, match="no_authorized_evidence must have zero"):
        validate_m1_oracles(path)


def test_stale_or_future_only_with_sufficient_set_fails(tmp_path: Path) -> None:
    path = _mutate(
        tmp_path,
        "m1-cf-share-options-01",
        lambda case: (
            case.__setitem__("query_class", "temporal"),
            case["oracle"].__setitem__("disposition_reason", "stale_or_future_only"),
            case["oracle"].__setitem__(
                "sufficient_reference_sets", [["FIN-EXP-THRESHOLD:v2"]]
            ),
        ),
    )
    with pytest.raises(ValueError, match="stale_or_future_only must have zero"):
        validate_m1_oracles(path)


def test_ambiguous_question_with_sufficient_set_fails(tmp_path: Path) -> None:
    path = _mutate(
        tmp_path,
        "m1-cf-share-options-01",
        lambda case: (
            case.__setitem__("query_class", "exact_factual"),
            case["oracle"].__setitem__("disposition_reason", "ambiguous_question"),
            case["oracle"].__setitem__(
                "sufficient_reference_sets", [["FIN-EXP-THRESHOLD:v2"]]
            ),
        ),
    )
    with pytest.raises(ValueError, match="ambiguous_question must have zero"):
        validate_m1_oracles(path)


def test_unknown_oos_bundle_field_mutations_fail(tmp_path: Path) -> None:
    path = _mutate(
        tmp_path,
        "m1-cf-share-options-01",
        lambda case: case["oracle"].__setitem__("support_label", "supported"),
    )
    with pytest.raises(ValueError, match="unknown_oos cases must have support_label=unsupported"):
        validate_m1_oracles(path)

    path = _mutate(
        tmp_path,
        "m1-cf-share-options-01",
        lambda case: case["oracle"].__setitem__("publication", "published"),
    )
    with pytest.raises(ValueError, match="publication=refused"):
        validate_m1_oracles(path)

    path = _mutate(
        tmp_path,
        "m1-cf-share-options-01",
        lambda case: case["oracle"].__setitem__("disposition_reason", "unauthorized_actor"),
    )
    with pytest.raises(ValueError, match="disposition_reason=out_of_scope"):
        validate_m1_oracles(path)


def test_conflicting_set_without_true_conflict_pair_fails(tmp_path: Path) -> None:
    path = _mutate(
        tmp_path,
        "m1-cf-overtime-01",
        lambda case: (
            case["oracle"].__setitem__("sufficient_reference_sets", [["HR-OVERTIME:v1"]]),
            case["oracle"].__setitem__("required_citations", ["HR-OVERTIME:v1"]),
        ),
    )
    with pytest.raises(ValueError, match="containing the conflicting evidence"):
        validate_m1_oracles(path)


def test_grok_alias_reviewer_bypass_rejected_via_registry(tmp_path: Path) -> None:
    question, oracle, review = _scoring_roles()
    question["model_family_id"] = "grok"
    question["model_id"] = "grok-4"
    oracle["model_family_id"] = "grok"
    oracle["model_id"] = "grok-4"
    review["reviewer_model_family_id"] = "grok-4.6"
    review["reviewer_model_id"] = "grok-4.6"
    path = _mutate(
        tmp_path,
        "m1-cf-travel-approval-01",
        lambda case: (
            case.__setitem__("case_kind", "scoring"),
            case.__setitem__("question_authorship", question),
            case.__setitem__("oracle_authorship", oracle),
            case.__setitem__("oracle_review", review),
        ),
    )
    with pytest.raises(ValueError, match="unknown model family|fail closed"):
        validate_m1_oracles(path)


def test_unknown_family_model_mapping_rejected(tmp_path: Path) -> None:
    question, oracle, review = _scoring_roles()
    question["model_id"] = "gpt-4.1"
    path = _mutate(
        tmp_path,
        "m1-cf-travel-approval-01",
        lambda case: (
            case.__setitem__("case_kind", "scoring"),
            case.__setitem__("question_authorship", question),
            case.__setitem__("oracle_authorship", deepcopy(oracle)),
            case.__setitem__("oracle_review", review),
        ),
    )
    with pytest.raises(ValueError, match="unknown family/model mapping"):
        validate_m1_oracles(path)


def test_conformance_null_prompt_digest_accepted() -> None:
    cases = _cases()
    assert all(case["question_authorship"]["prompt_retained"] is False for case in cases)
    assert all(case["question_authorship"]["prompt_sha256"] is None for case in cases)
    assert all(case["oracle_authorship"]["prompt_sha256"] is None for case in cases)
    validate_m1_oracles(FIXTURE)


def test_scoring_null_prompt_digest_rejected(tmp_path: Path) -> None:
    question, oracle, review = _scoring_roles()
    question["prompt_retained"] = False
    question["prompt_sha256"] = None
    path = _mutate(
        tmp_path,
        "m1-cf-travel-approval-01",
        lambda case: (
            case.__setitem__("case_kind", "scoring"),
            case.__setitem__("question_authorship", question),
            case.__setitem__("oracle_authorship", oracle),
            case.__setitem__("oracle_review", review),
        ),
    )
    with pytest.raises(ValueError, match="prompt_retained=true"):
        validate_m1_oracles(path)


def test_out_of_scope_requires_unknown_oos_runtime(tmp_path: Path) -> None:
    path = _mutate(
        tmp_path,
        "m1-cf-payroll-02",
        lambda case: case["oracle"].__setitem__("disposition_reason", "out_of_scope"),
    )
    with pytest.raises(ValueError, match="out_of_scope requires query_class unknown_oos"):
        validate_m1_oracles(path)

