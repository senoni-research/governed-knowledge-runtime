"""Deterministic, authority-aware validation of M1 v3 oracle cases.

This module checks corpus-grounded invariants and scoring provenance
separation. It loads the bound model-family registry and fail-closes on an
unknown family or family/model mapping. Registry checks validate declared
provenance only and cannot prove which external model actually ran. The
module does not establish semantic support, question quality, or claim
entailment. The ``oracle_review.semantically_reviewed`` field is a review
provenance claim and is never set or inferred here.

Authorized visibility for sufficient references is taken from
``AuthorityStore.current_records``. The in-memory temporal helper is used
only to classify stale, future, retired, or unknown failures.
"""

from __future__ import annotations

import json
import tempfile
from collections import Counter, defaultdict
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from gkr.authority import AuthorityStore
from gkr.m1_corpus import (
    DEFAULT_CORPUS_DIR,
    load_authority_records,
)
from gkr.m1_hash import (
    canonical_json_bytes,
    load_model_family_registry,
    resolve_model_family_id,
)
from gkr.m1_validation import question_digest
from gkr.schemas import Actor, KnowledgeRecord

CASE_SCHEMA_PATH = Path("evaluation/m1/benchmark-case-v3.schema.json")
QUERY_CLASSES = (
    "exact_factual",
    "semantic_paraphrase",
    "numeric_conditional",
    "temporal",
    "authorization",
    "unknown_oos",
    "multi_record",
    "adversarial_conflicting",
)
_REFUSAL_PUBLICATIONS = {"refused", "withheld"}
_DISPOSITIONS = {
    "no_authorized_evidence",
    "unauthorized_actor",
    "out_of_scope",
    "conflicting_authority",
    "ambiguous_question",
    "stale_or_future_only",
}

_DETERMINISM_NOTE = (
    "This report validates deterministic invariants only and does not establish "
    "semantic support. Semantic review is a separate provenance claim."
)


def validate_m1_oracles(
    case_path: str | Path,
    *,
    corpus_dir: str | Path = DEFAULT_CORPUS_DIR,
    schema_path: str | Path = CASE_SCHEMA_PATH,
    allow_incomplete: bool = True,
) -> dict[str, Any]:
    """Validate M1 v3 cases against schema and a freshly ingested frozen corpus."""

    cases = load_case_jsonl(case_path)
    errors: list[str] = []
    errors.extend(_schema_errors(cases, case_path, schema_path))
    try:
        registry = load_model_family_registry()
    except ValueError as exc:
        errors.append(str(exc))
        registry = None

    authority_path = Path(corpus_dir) / "authority.jsonl"
    records = load_authority_records(authority_path)
    by_id = _group_by_record_id(records)

    with tempfile.TemporaryDirectory() as tmp:
        store = AuthorityStore(Path(tmp) / "authority.sqlite")
        store.import_jsonl(authority_path)
        if store.count() != len(records):
            errors.append(
                f"{authority_path}: ingested {store.count()} events; expected {len(records)}"
            )
        errors.extend(_case_errors(cases, case_path, store, by_id, registry=registry))
        store.close()

    errors.extend(_cross_split_errors(cases, case_path))
    if not allow_incomplete:
        errors.extend(_complete_suite_errors(cases, case_path))
    if errors:
        raise ValueError("\n".join(errors))

    return _report(case_path, cases)


def load_case_jsonl(path: str | Path) -> list[tuple[int, dict[str, Any]]]:
    cases: list[tuple[int, dict[str, Any]]] = []
    with Path(path).open(encoding="utf-8") as source:
        for line_number, raw_line in enumerate(source, start=1):
            if not raw_line.strip():
                continue
            try:
                value = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: benchmark case must be an object")
            cases.append((line_number, value))
    if not cases:
        raise ValueError(f"{path}: benchmark case file is empty")
    return cases


def parse_reference(reference: str) -> tuple[str, int] | None:
    try:
        record_id, raw_version = reference.rsplit(":v", maxsplit=1)
        version = int(raw_version)
    except (ValueError, TypeError):
        return None
    if version < 1 or not record_id:
        return None
    return record_id, version


def temporally_selected(
    versions: list[KnowledgeRecord],
    *,
    as_of: date,
    known_at: datetime,
) -> KnowledgeRecord | None:
    chosen: KnowledgeRecord | None = None
    for record in sorted(versions, key=lambda item: item.version):
        if record.is_valid_at(as_of) and record.is_known_at(known_at):
            chosen = record
    return chosen


def _schema_errors(
    cases: list[tuple[int, dict[str, Any]]],
    case_path: str | Path,
    schema_path: str | Path,
) -> list[str]:
    try:
        from jsonschema import Draft202012Validator, FormatChecker
    except ImportError as exc:
        raise RuntimeError("M1 oracle validation requires the development dependencies") from exc

    schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors: list[str] = []
    for line_number, case in cases:
        prefix = f"{case_path}:{line_number}"
        for error in validator.iter_errors(case):
            location = ".".join(str(part) for part in error.absolute_path)
            errors.append(f"{prefix}:{location or '<root>'}: {error.message}")
    return errors


def _case_errors(
    cases: list[tuple[int, dict[str, Any]]],
    case_path: str | Path,
    store: AuthorityStore,
    by_id: dict[str, list[KnowledgeRecord]],
    *,
    registry: dict[str, Any] | None,
) -> list[str]:
    errors: list[str] = []
    case_ids: set[str] = set()
    for line_number, case in cases:
        prefix = f"{case_path}:{line_number}"
        case_id = str(case.get("case_id", ""))
        if case_id in case_ids:
            errors.append(f"{prefix}: duplicate case_id {case_id}")
        case_ids.add(case_id)

        question = case.get("question")
        if isinstance(question, str) and case.get("question_sha256") != question_digest(question):
            errors.append(f"{prefix}: question_sha256 does not match normalized question")

        errors.extend(_provenance_errors(prefix, case, registry=registry))

        scope = _parse_scope(case, prefix, errors)
        oracle = case.get("oracle")
        if not isinstance(oracle, dict) or scope is None:
            continue
        actor, as_of, known_at = scope
        errors.extend(
            _oracle_errors(
                prefix,
                case,
                oracle,
                actor=actor,
                as_of=as_of,
                known_at=known_at,
                store=store,
                by_id=by_id,
            )
        )
    return errors


def _oracle_errors(
    prefix: str,
    case: dict[str, Any],
    oracle: dict[str, Any],
    *,
    actor: Actor,
    as_of: date,
    known_at: datetime,
    store: AuthorityStore,
    by_id: dict[str, list[KnowledgeRecord]],
) -> list[str]:
    errors: list[str] = []
    sufficient_sets = _string_sets(oracle.get("sufficient_reference_sets"))
    sufficient = {reference for group in sufficient_sets for reference in group}
    forbidden = _string_list(oracle.get("forbidden_references"))
    forbidden_set = set(forbidden)
    citations = _string_list(oracle.get("required_citations"))
    required_claims = _string_list(oracle.get("required_claims"))
    forbidden_claims = _string_list(oracle.get("forbidden_claims"))
    publication = oracle.get("publication")
    support_label = oracle.get("support_label")
    disposition = oracle.get("disposition_reason")
    query_class = case.get("query_class")

    overlap = sorted(sufficient.intersection(forbidden_set))
    if overlap:
        errors.append(f"{prefix}: oracle references are both sufficient and forbidden")

    named = sufficient | forbidden_set | set(citations)
    resolved: dict[str, KnowledgeRecord] = {}
    for reference in sorted(named):
        record = store.get(reference)
        if record is None:
            errors.append(f"{prefix}: reference {reference} does not exist in the corpus")
            continue
        resolved[reference] = record

    authorized = store.current_records(actor=actor, as_of=as_of, known_at=known_at)
    authorized_refs = {record.reference for record in authorized.records}

    for reference in sorted(sufficient):
        record = resolved.get(reference)
        if record is None:
            continue
        selected = temporally_selected(
            by_id.get(record.record_id, []), as_of=as_of, known_at=known_at
        )
        diagnostic = _temporal_diagnostic(
            reference,
            record=record,
            selected=selected,
            as_of=as_of,
            known_at=known_at,
        )
        if diagnostic is not None:
            errors.append(f"{prefix}: {diagnostic}")
            continue
        if reference not in authorized_refs:
            errors.append(
                f"{prefix}: {_visibility_diagnostic(str(publication))} sufficient "
                f"reference {reference} is not visible to the declared actor"
            )

    if query_class == "unknown_oos":
        if sufficient_sets:
            errors.append(f"{prefix}: unknown_oos cases must have zero sufficient sets")
        if support_label != "unsupported":
            errors.append(f"{prefix}: unknown_oos cases must have support_label=unsupported")
        if publication != "refused":
            errors.append(
                f"{prefix}: unknown_oos cases must have publication=refused and "
                "disposition_reason=out_of_scope"
            )
        if disposition != "out_of_scope":
            errors.append(f"{prefix}: unknown_oos cases must have disposition_reason=out_of_scope")

    if publication in _REFUSAL_PUBLICATIONS:
        if disposition not in _DISPOSITIONS:
            errors.append(f"{prefix}: refused/withheld cases require disposition_reason")
        else:
            errors.extend(
                _disposition_errors(
                    prefix,
                    disposition=str(disposition),
                    query_class=str(query_class),
                    sufficient_sets=sufficient_sets,
                    forbidden=forbidden,
                    resolved=resolved,
                    actor=actor,
                    as_of=as_of,
                    known_at=known_at,
                    by_id=by_id,
                    support_label=str(support_label),
                    citations=citations,
                )
            )

    if citations:
        citation_set = set(citations)
        if not any(citation_set.issubset(group) for group in sufficient_sets):
            errors.append(f"{prefix}: required_citations are not a subset of any sufficient set")

    claim_overlap = sorted(set(required_claims).intersection(forbidden_claims))
    if claim_overlap:
        errors.append(f"{prefix}: required_claims and forbidden_claims overlap")

    if support_label == "supported" and not sufficient_sets:
        errors.append(f"{prefix}: supported requires at least one sufficient set")
    if support_label == "conflicting" and (
        not sufficient_sets
        or not any(_conflict_pairs(group, resolved) for group in sufficient_sets)
    ):
        errors.append(
            f"{prefix}: conflicting requires at least one nonempty set "
            "containing the conflicting evidence"
        )

    if query_class == "temporal":
        for reference in sorted(sufficient):
            parsed = parse_reference(reference)
            if parsed is None:
                continue
            record_id, version = parsed
            missing = [
                item.reference
                for item in by_id.get(record_id, [])
                if item.version != version and item.reference not in forbidden
            ]
            if missing:
                errors.append(
                    f"{prefix}: temporal case that expects a superseded or non-current "
                    f"version must list wrong-version references in forbidden_references: "
                    f"{', '.join(missing)}"
                )
    return errors


def _provenance_errors(
    prefix: str,
    case: dict[str, Any],
    *,
    registry: dict[str, Any] | None,
) -> list[str]:
    errors: list[str] = []
    case_kind = case.get("case_kind")
    question = case.get("question_authorship")
    oracle = case.get("oracle_authorship")
    review = case.get("oracle_review")
    if (
        not isinstance(question, dict)
        or not isinstance(oracle, dict)
        or not isinstance(review, dict)
    ):
        return errors

    errors.extend(
        _model_family_errors(prefix, question, "question_authorship", registry=registry)
    )
    errors.extend(
        _model_family_errors(prefix, oracle, "oracle_authorship", registry=registry)
    )
    if review.get("reviewer_kind") == "model":
        errors.extend(
            _model_family_errors(
                prefix,
                review,
                "oracle_review",
                registry=registry,
                family_key="reviewer_model_family_id",
                model_key="reviewer_model_id",
            )
        )

    for role_name, role in (
        ("question_authorship", question),
        ("oracle_authorship", oracle),
    ):
        errors.extend(_prompt_provenance_errors(prefix, case_kind, role_name, role))

    if case_kind == "scoring":
        if review.get("status") != "completed" or review.get("semantically_reviewed") is not True:
            errors.append(
                f"{prefix}: scoring cases require completed semantic review "
                "(oracle_review.status=completed and semantically_reviewed=true)"
            )
        sessions = (
            str(question.get("session_id") or ""),
            str(oracle.get("session_id") or ""),
            str(review.get("reviewer_session_id") or ""),
        )
        if any(not session for session in sessions):
            errors.append(
                f"{prefix}: scoring cases require question-author, oracle-author, "
                "and reviewer session IDs"
            )
        elif len(set(sessions)) != 3:
            errors.append(
                f"{prefix}: scoring question-author, oracle-author, and reviewer "
                "session IDs must be pairwise distinct"
            )
        if review.get("reviewer_kind") == "model" and registry is not None:
            try:
                reviewer_family = resolve_model_family_id(
                    review.get("reviewer_model_family_id"),
                    review.get("reviewer_model_id"),
                    registry=registry,
                )
                author_families = {
                    resolve_model_family_id(
                        question.get("model_family_id"),
                        question.get("model_id"),
                        registry=registry,
                    ),
                    resolve_model_family_id(
                        oracle.get("model_family_id"),
                        oracle.get("model_id"),
                        registry=registry,
                    ),
                }
            except ValueError:
                author_families = set()
                reviewer_family = None
            if reviewer_family is not None and reviewer_family in author_families:
                errors.append(
                    f"{prefix}: scoring model reviewer family must differ from both "
                    "author model families by resolved canonical ID"
                )
        if not isinstance(oracle.get("session_id"), str) or not oracle.get("session_id"):
            errors.append(f"{prefix}: missing oracle-author provenance")

    split = case.get("split")
    if split == "test":
        for role_name, role in (
            ("question_authorship", question),
            ("oracle_authorship", oracle),
        ):
            if role.get("independent_from_retriever_tuning") is not True:
                errors.append(
                    f"{prefix}: test-split {role_name} must be independent from retriever tuning"
                )
        if review.get("status") == "completed" and review.get(
            "independent_from_retriever_tuning"
        ) is not True:
            errors.append(
                f"{prefix}: test-split oracle_review must be independent from retriever tuning"
            )
    return errors


def _prompt_provenance_errors(
    prefix: str,
    case_kind: object,
    role_name: str,
    role: dict[str, Any],
) -> list[str]:
    method = role.get("method")
    retained = role.get("prompt_retained")
    digest = role.get("prompt_sha256")
    if method in {"model", "human_edited_model"} and case_kind == "scoring":
        if retained is not True:
            return [
                f"{prefix}: scoring {role_name} requires prompt_retained=true and a "
                "valid prompt digest under the hash profile"
            ]
        if not isinstance(digest, str) or len(digest) != 64:
            return [
                f"{prefix}: scoring {role_name} requires prompt_retained=true and a "
                "valid prompt digest under the hash profile"
            ]
    if retained is False and digest is not None:
        return [f"{prefix}: {role_name} prompt_retained=false requires prompt_sha256=null"]
    if retained is True and (not isinstance(digest, str) or len(digest) != 64):
        return [f"{prefix}: {role_name} prompt_retained=true requires a valid prompt digest"]
    return []


def _temporal_diagnostic(
    reference: str,
    *,
    record: KnowledgeRecord,
    selected: KnowledgeRecord | None,
    as_of: date,
    known_at: datetime,
) -> str | None:
    if selected is None:
        return (
            f"sufficient reference {reference} is not valid at as_of/"
            "known_at (retired, stale, superseded, or future-only)"
        )
    if selected.reference != reference:
        if record.valid_from > as_of:
            return (
                f"sufficient reference {reference} is future-only at as_of {as_of.isoformat()}"
            )
        if not record.is_known_at(known_at):
            return f"sufficient reference {reference} is not known at known_at"
        return (
            f"sufficient reference {reference} is not the temporally "
            f"selected version {selected.reference}"
        )
    if selected.status != "approved":
        return f"sufficient reference {reference} is {selected.status}, not approved"
    return None


def _visibility_diagnostic(publication: str) -> str:
    if publication == "published":
        return "published case"
    if publication == "refused":
        return "refused case"
    if publication == "withheld":
        return "withheld case"
    return "case"


def _disposition_errors(
    prefix: str,
    *,
    disposition: str,
    query_class: str,
    sufficient_sets: list[set[str]],
    forbidden: list[str],
    resolved: dict[str, KnowledgeRecord],
    actor: Actor,
    as_of: date,
    known_at: datetime,
    by_id: dict[str, list[KnowledgeRecord]],
    support_label: str,
    citations: list[str],
) -> list[str]:
    errors: list[str] = []
    unauthorized = _unauthorized_named_evidence(
        forbidden, resolved, actor=actor, as_of=as_of, known_at=known_at, by_id=by_id
    )
    if disposition == "unauthorized_actor":
        if sufficient_sets:
            errors.append(f"{prefix}: unauthorized_actor cases must have zero sufficient sets")
        if not unauthorized:
            errors.append(
                f"{prefix}: unauthorized_actor requires a forbidden reference that is "
                "temporally selected and approved, hidden from this actor, and visible "
                "to an authorized principal"
            )
    elif disposition == "out_of_scope":
        if query_class != "unknown_oos":
            errors.append(f"{prefix}: out_of_scope requires query_class unknown_oos")
        if sufficient_sets:
            errors.append(f"{prefix}: out_of_scope cases must have zero sufficient sets")
    elif disposition == "no_authorized_evidence":
        if sufficient_sets:
            errors.append(f"{prefix}: no_authorized_evidence must have zero sufficient sets")
        if unauthorized:
            errors.append(
                f"{prefix}: no_authorized_evidence is inconsistent when unauthorized "
                "but otherwise visible evidence is named; use unauthorized_actor"
            )
    elif disposition == "conflicting_authority":
        if support_label != "conflicting":
            errors.append(f"{prefix}: conflicting_authority requires support_label conflicting")
        if not _conflict_pairs(
            {reference for group in sufficient_sets for reference in group} | set(citations),
            resolved,
        ):
            errors.append(
                f"{prefix}: conflicting_authority requires two sufficient or cited "
                "references from distinct records in the same conflict_group"
            )
    elif disposition == "stale_or_future_only":
        if sufficient_sets:
            errors.append(f"{prefix}: stale_or_future_only must have zero sufficient sets")
        if not _temporal_mismatch(forbidden, as_of=as_of, known_at=known_at, by_id=by_id):
            errors.append(
                f"{prefix}: stale_or_future_only requires a named forbidden reference "
                "that is not the temporally selected approved version"
            )
    elif disposition == "ambiguous_question":
        if sufficient_sets:
            errors.append(f"{prefix}: ambiguous_question must have zero sufficient sets")
    return errors


def _unauthorized_named_evidence(
    forbidden: list[str],
    resolved: dict[str, KnowledgeRecord],
    *,
    actor: Actor,
    as_of: date,
    known_at: datetime,
    by_id: dict[str, list[KnowledgeRecord]],
) -> list[str]:
    named: list[str] = []
    for reference in forbidden:
        record = resolved.get(reference)
        if record is None:
            continue
        selected = temporally_selected(
            by_id.get(record.record_id, []), as_of=as_of, known_at=known_at
        )
        if selected is None or selected.reference != reference:
            continue
        if selected.status != "approved":
            continue
        if selected.is_permitted(actor):
            continue
        if _visible_to_any_authorized_principal(selected):
            named.append(reference)
    return named


def _visible_to_any_authorized_principal(record: KnowledgeRecord) -> bool:
    if record.sensitivity == "public":
        return True
    for principal in record.acl:
        probe = _actor_for_principal(principal)
        if probe is not None and record.is_permitted(probe):
            return True
    return False


def _actor_for_principal(principal: str) -> Actor | None:
    kind, _, name = principal.partition(":")
    if not name:
        return None
    if kind == "group":
        return Actor("authorized-probe", (name,))
    if kind == "user":
        return Actor(name, ())
    return None


def _temporal_mismatch(
    forbidden: list[str],
    *,
    as_of: date,
    known_at: datetime,
    by_id: dict[str, list[KnowledgeRecord]],
) -> bool:
    for reference in forbidden:
        parsed = parse_reference(reference)
        if parsed is None:
            continue
        record_id, _version = parsed
        selected = temporally_selected(by_id.get(record_id, []), as_of=as_of, known_at=known_at)
        if selected is None or selected.reference != reference or selected.status != "approved":
            return True
    return False


def _conflict_pairs(references: set[str], resolved: dict[str, KnowledgeRecord]) -> bool:
    groups: dict[str, set[str]] = defaultdict(set)
    for reference in references:
        record = resolved.get(reference)
        if record is None or record.status != "approved":
            continue
        group = record.metadata.get("conflict_group")
        if group:
            groups[str(group)].add(record.record_id)
    return any(len(record_ids) >= 2 for record_ids in groups.values())


def _parse_scope(
    case: dict[str, Any],
    prefix: str,
    errors: list[str],
) -> tuple[Actor, date, datetime] | None:
    scope = case.get("scope")
    if not isinstance(scope, dict):
        return None
    try:
        as_of = date.fromisoformat(str(scope["as_of"]))
        known_at = _parse_datetime(scope["known_at"])
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"{prefix}: scope as_of/known_at is not a timezone-aware instant: {exc}")
        return None
    groups = tuple(str(group) for group in scope.get("groups", []) if str(group))
    actor_id = scope.get("actor")
    actor = Actor(None if actor_id is None else str(actor_id), groups)
    return actor, as_of, known_at


def _parse_datetime(value: object) -> datetime:
    parsed = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    if parsed.utcoffset() is None:
        raise ValueError("known_at must include a timezone")
    return parsed.astimezone(UTC)


def _cross_split_errors(
    cases: list[tuple[int, dict[str, Any]]],
    case_path: str | Path,
) -> list[str]:
    errors: list[str] = []
    identities: dict[str, bytes] = {}
    variants: dict[str, set[str]] = defaultdict(set)
    scoped_questions: dict[tuple[str, str, tuple[str, ...], str, str], str] = {}
    for line_number, case in cases:
        prefix = f"{case_path}:{line_number}"
        scenario_id = str(case.get("scenario_id", ""))
        variant_id = str(case.get("variant_id", ""))
        identity = _scenario_identity_bytes(case)
        previous = identities.get(scenario_id)
        if previous is None:
            identities[scenario_id] = identity
        elif previous != identity:
            errors.append(
                f"{prefix}: scenario {scenario_id} variants must share split, query "
                "class, case kind, scope, oracle, oracle authorship, oracle review, "
                "and metric applicability"
            )
        seen_variants = variants[scenario_id]
        if variant_id in seen_variants:
            errors.append(f"{prefix}: duplicate variant_id {variant_id} in scenario {scenario_id}")
        seen_variants.add(variant_id)
        question = case.get("question")
        scope = case.get("scope")
        if isinstance(question, str) and isinstance(scope, dict):
            key = (
                question_digest(question),
                str(scope.get("actor")),
                tuple(str(group) for group in scope.get("groups", [])),
                str(scope.get("as_of")),
                str(scope.get("known_at")),
            )
            duplicate = scoped_questions.setdefault(key, str(case.get("case_id", "")))
            if duplicate != str(case.get("case_id", "")):
                errors.append(f"{prefix}: exact question+scope duplicate of {duplicate}")
    return errors


def _scenario_identity_bytes(case: dict[str, Any]) -> bytes:
    return canonical_json_bytes(
        {
            "split": case.get("split"),
            "query_class": case.get("query_class"),
            "case_kind": case.get("case_kind"),
            "scope": case.get("scope"),
            "oracle": case.get("oracle"),
            "oracle_authorship": case.get("oracle_authorship"),
            "oracle_review": case.get("oracle_review"),
        }
    )


def _model_family_errors(
    prefix: str,
    role: dict[str, Any],
    role_name: str,
    *,
    registry: dict[str, Any] | None,
    family_key: str = "model_family_id",
    model_key: str = "model_id",
) -> list[str]:
    family = role.get(family_key)
    if family is None:
        return []
    if registry is None:
        return [f"{prefix}: model-family registry is unavailable; fail closed"]
    try:
        resolve_model_family_id(family, role.get(model_key), registry=registry)
    except ValueError as exc:
        return [f"{prefix}: {role_name} {exc}"]
    return []


def _complete_suite_errors(
    cases: list[tuple[int, dict[str, Any]]],
    case_path: str | Path,
) -> list[str]:
    scenarios: dict[str, set[str]] = defaultdict(set)
    class_scenarios: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for _line_number, case in cases:
        split = str(case.get("split", ""))
        scenario_id = str(case.get("scenario_id", ""))
        query_class = str(case.get("query_class", ""))
        scenarios[split].add(scenario_id)
        class_scenarios[split][query_class].add(scenario_id)
    errors: list[str] = []
    for split in ("development", "validation", "test"):
        count = len(scenarios.get(split, set()))
        if count != 120:
            errors.append(f"{case_path}: split {split} has {count} scenarios; expected 120")
        for query_class in QUERY_CLASSES:
            class_count = len(class_scenarios.get(split, {}).get(query_class, set()))
            if class_count != 15:
                errors.append(
                    f"{case_path}: split {split} class {query_class} has "
                    f"{class_count} scenarios; expected 15"
                )
    errors.extend(_evidence_applicability_errors(cases, case_path))
    return errors


def evidence_applicability_counts(
    cases: list[tuple[int, dict[str, Any]]],
) -> dict[str, dict[str, dict[str, int]]]:
    """Derived complete-suite applicability counts. Not used for the fixture."""

    counts: dict[str, dict[str, dict[str, int]]] = {}
    for split in ("development", "validation", "test"):
        counts[split] = {
            query_class: _empty_applicability(query_class) for query_class in QUERY_CLASSES
        }
    seen: dict[tuple[str, str, str], dict[str, Any]] = {}
    for _line_number, case in cases:
        split = str(case.get("split", ""))
        query_class = str(case.get("query_class", ""))
        scenario_id = str(case.get("scenario_id", ""))
        if split not in counts or query_class not in counts[split]:
            continue
        key = (split, query_class, scenario_id)
        if key in seen:
            continue
        seen[key] = case
        oracle = case.get("oracle") if isinstance(case.get("oracle"), dict) else {}
        sufficient = _string_sets(oracle.get("sufficient_reference_sets"))
        publication = oracle.get("publication")
        bucket = counts[split][query_class]
        if query_class == "authorization":
            if sufficient and publication == "published":
                bucket["evidence_bearing_authorized_scenario_count"] += 1
            if not sufficient and publication in _REFUSAL_PUBLICATIONS:
                bucket["zero_set_denied_scenario_count"] += 1
            continue
        if sufficient:
            bucket["evidence_bearing_scenario_count"] += 1
        else:
            bucket["zero_set_scenario_count"] += 1
    return counts


def _empty_applicability(query_class: str) -> dict[str, int]:
    if query_class == "authorization":
        return {
            "evidence_bearing_authorized_scenario_count": 0,
            "zero_set_denied_scenario_count": 0,
            "total_scenario_count": 15,
        }
    return {
        "evidence_bearing_scenario_count": 0,
        "zero_set_scenario_count": 0,
        "total_scenario_count": 15,
    }


def applicability_partition_errors(
    counts: dict[str, dict[str, dict[str, int]]],
    *,
    source: str,
) -> list[str]:
    """Reject under-count, over-count, and authorization partition mismatch."""

    errors: list[str] = []
    for split, class_map in counts.items():
        for query_class, bucket in class_map.items():
            prefix = f"{source}: split {split} class {query_class}"
            if bucket.get("total_scenario_count") != 15:
                errors.append(f"{prefix}: total_scenario_count must be 15")
            left, right, left_name, right_name = _applicability_pair(query_class, bucket)
            total = left + right
            if query_class == "unknown_oos":
                if left != 0 or right != 15:
                    errors.append(
                        f"{prefix}: unknown_oos must contain only zero-set scenarios "
                        "(evidence_bearing_scenario_count=0, zero_set_scenario_count=15)"
                    )
                continue
            if query_class == "authorization":
                if left < 1:
                    errors.append(
                        f"{prefix}: authorization requires at least one "
                        "evidence-bearing authorized scenario"
                    )
                if right < 1:
                    errors.append(
                        f"{prefix}: authorization requires at least one "
                        "zero-set denied scenario"
                    )
            elif left < 1:
                errors.append(
                    f"{prefix}: requires at least one independent scenario with "
                    "one or more sufficient evidence sets"
                )
            if total < 15:
                errors.append(
                    f"{prefix}: {left_name}+{right_name}={total} under-counts the "
                    "required 15-scenario partition"
                )
            elif total > 15:
                errors.append(
                    f"{prefix}: {left_name}+{right_name}={total} over-counts the "
                    "required 15-scenario partition"
                )
            elif query_class == "authorization" and (left < 1 or right < 1):
                errors.append(
                    f"{prefix}: authorization partition mismatch; authorized "
                    "evidence-bearing and denied zero-set must each be at least 1 "
                    "and sum to 15"
                )
    return errors


def _applicability_pair(
    query_class: str, bucket: dict[str, int]
) -> tuple[int, int, str, str]:
    if query_class == "authorization":
        return (
            int(bucket.get("evidence_bearing_authorized_scenario_count", 0)),
            int(bucket.get("zero_set_denied_scenario_count", 0)),
            "evidence_bearing_authorized_scenario_count",
            "zero_set_denied_scenario_count",
        )
    return (
        int(bucket.get("evidence_bearing_scenario_count", 0)),
        int(bucket.get("zero_set_scenario_count", 0)),
        "evidence_bearing_scenario_count",
        "zero_set_scenario_count",
    )


def _evidence_applicability_errors(
    cases: list[tuple[int, dict[str, Any]]],
    case_path: str | Path,
) -> list[str]:
    return applicability_partition_errors(
        evidence_applicability_counts(cases),
        source=str(case_path),
    )


def _report(case_path: str | Path, cases: list[tuple[int, dict[str, Any]]]) -> dict[str, Any]:
    splits: Counter[str] = Counter()
    scenarios: dict[str, set[str]] = defaultdict(set)
    classes: Counter[str] = Counter()
    for _line_number, case in cases:
        split = str(case.get("split", ""))
        splits[split] += 1
        scenarios[split].add(str(case.get("scenario_id", "")))
        classes[str(case.get("query_class", ""))] += 1
    return {
        "case_file": str(case_path),
        "cases": len(cases),
        "independent_scenarios": {
            split: len(values) for split, values in sorted(scenarios.items())
        },
        "cases_by_split": dict(splits),
        "query_classes": dict(sorted(classes.items())),
        "semantic_support_established": False,
        "note": _DETERMINISM_NOTE,
    }


def _group_by_record_id(records: tuple[KnowledgeRecord, ...]) -> dict[str, list[KnowledgeRecord]]:
    grouped: dict[str, list[KnowledgeRecord]] = defaultdict(list)
    for record in records:
        grouped[record.record_id].append(record)
    return grouped


def _string_sets(value: object) -> list[set[str]]:
    if not isinstance(value, list):
        return []
    sets: list[set[str]] = []
    for item in value:
        if isinstance(item, list):
            sets.append({str(reference) for reference in item})
    return sets


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]
