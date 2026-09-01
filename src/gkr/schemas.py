from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Literal

Sensitivity = Literal["public", "internal", "restricted", "secret"]
RecordStatus = Literal["approved", "retired"]
RuleComparator = Literal[">", ">=", "<", "<=", "="]

_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_REFERENCE_PATTERN = re.compile(
    r"^(?P<record_id>[A-Za-z0-9][A-Za-z0-9._-]*):v(?P<version>[1-9]\d*)$"
)
_SHA256_PATTERN = re.compile(r"^[a-f0-9]{64}$")


@dataclass(frozen=True)
class Actor:
    """The authenticated local caller and their asserted authority groups."""

    actor_id: str | None
    groups: tuple[str, ...] = ()

    @property
    def principals(self) -> frozenset[str]:
        values = {f"group:{group}" for group in self.groups}
        if self.actor_id:
            values.update({f"user:{self.actor_id}", "authenticated"})
        return frozenset(values)

    @classmethod
    def anonymous(cls) -> Actor:
        return cls(actor_id=None)


@dataclass(frozen=True)
class KnowledgeRelation:
    subject: str
    predicate: str
    object: str

    @classmethod
    def from_value(cls, value: Any) -> KnowledgeRelation:
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise ValueError("A relation must contain [subject, predicate, object]")
        parts = tuple(str(part).strip() for part in value)
        if not all(parts):
            raise ValueError("Relation values cannot be empty")
        return cls(*parts)

    def to_list(self) -> list[str]:
        return [self.subject, self.predicate, self.object]


@dataclass(frozen=True)
class PolicyRule:
    """A reviewed, machine-executable rule attached to an authority record."""

    rule_id: str
    subject: str
    measure: str
    unit: str
    comparator: RuleComparator
    threshold: Decimal
    effect: str
    conditions: tuple[str, ...] = ()
    exceptions: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PolicyRule:
        required = ("rule_id", "subject", "measure", "unit", "comparator", "threshold", "effect")
        missing = [key for key in required if value.get(key) in (None, "")]
        if missing:
            raise ValueError(f"Policy rule is missing required fields: {', '.join(missing)}")
        try:
            threshold = Decimal(str(value["threshold"]).replace(",", ""))
        except InvalidOperation as exc:
            raise ValueError("Policy rule threshold must be numeric") from exc
        rule = cls(
            rule_id=str(value["rule_id"]).strip(),
            subject=str(value["subject"]).strip(),
            measure=str(value["measure"]).strip(),
            unit=str(value["unit"]).strip().upper(),
            comparator=str(value["comparator"]).strip(),  # type: ignore[arg-type]
            threshold=threshold,
            effect=str(value["effect"]).strip(),
            conditions=_string_tuple(value.get("conditions", [])),
            exceptions=_string_tuple(value.get("exceptions", [])),
        )
        rule.validate()
        return rule

    def validate(self) -> None:
        if not _ID_PATTERN.fullmatch(self.rule_id):
            raise ValueError(f"Invalid rule_id: {self.rule_id}")
        if not all((self.subject, self.measure, self.unit, self.effect)):
            raise ValueError("Policy rule text fields cannot be empty")
        if self.comparator not in {">", ">=", "<", "<=", "="}:
            raise ValueError(f"Unsupported policy comparator: {self.comparator}")
        if not self.threshold.is_finite():
            raise ValueError("Policy rule threshold must be finite")

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "subject": self.subject,
            "measure": self.measure,
            "unit": self.unit,
            "comparator": self.comparator,
            "threshold": format(self.threshold, "f"),
            "effect": self.effect,
            "conditions": list(self.conditions),
            "exceptions": list(self.exceptions),
        }


@dataclass(frozen=True)
class KnowledgeRecord:
    """An immutable, source-bound version of one authoritative knowledge item."""

    record_id: str
    version: int
    domain: str
    title: str
    statement: str
    valid_from: date
    observed_at: datetime
    source_uri: str
    source_hash: str
    owner: str
    status: RecordStatus = "approved"
    sensitivity: Sensitivity = "internal"
    acl: tuple[str, ...] = ()
    valid_to: date | None = None
    supersedes: str | None = None
    source_span: str | None = None
    aliases: tuple[str, ...] = ()
    entities: tuple[str, ...] = ()
    relations: tuple[KnowledgeRelation, ...] = ()
    rules: tuple[PolicyRule, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def reference(self) -> str:
        return f"{self.record_id}:v{self.version}"

    def is_valid_at(self, when: date) -> bool:
        return self.valid_from <= when and (self.valid_to is None or when < self.valid_to)

    def is_known_at(self, when: datetime) -> bool:
        return self.observed_at <= when

    def is_permitted(self, actor: Actor) -> bool:
        if self.sensitivity == "public":
            return True
        return bool(actor.principals.intersection(self.acl))

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "version": self.version,
            "domain": self.domain,
            "title": self.title,
            "statement": self.statement,
            "valid_from": self.valid_from.isoformat(),
            "valid_to": self.valid_to.isoformat() if self.valid_to else None,
            "observed_at": self.observed_at.isoformat().replace("+00:00", "Z"),
            "supersedes": self.supersedes,
            "status": self.status,
            "owner": self.owner,
            "source_uri": self.source_uri,
            "source_span": self.source_span,
            "source_hash": self.source_hash,
            "sensitivity": self.sensitivity,
            "acl": list(self.acl),
            "aliases": list(self.aliases),
            "entities": list(self.entities),
            "relations": [relation.to_list() for relation in self.relations],
            "rules": [rule.to_dict() for rule in self.rules],
            "metadata": self.metadata,
        }

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> KnowledgeRecord:
        required = (
            "record_id",
            "version",
            "domain",
            "title",
            "statement",
            "valid_from",
            "observed_at",
            "source_uri",
            "source_hash",
            "owner",
        )
        missing = [key for key in required if value.get(key) in (None, "")]
        if missing:
            raise ValueError(f"Knowledge record is missing required fields: {', '.join(missing)}")

        observed_at = _parse_datetime(value["observed_at"])
        record = cls(
            record_id=str(value["record_id"]).strip(),
            version=_parse_version(value["version"]),
            domain=str(value["domain"]).strip().lower(),
            title=str(value["title"]).strip(),
            statement=str(value["statement"]).strip(),
            valid_from=_parse_date(value["valid_from"], "valid_from"),
            valid_to=_optional_date(value.get("valid_to"), "valid_to"),
            observed_at=observed_at,
            supersedes=_optional_string(value.get("supersedes")),
            status=str(value.get("status", "approved")).strip().lower(),  # type: ignore[arg-type]
            owner=str(value["owner"]).strip(),
            source_uri=str(value["source_uri"]).strip(),
            source_span=_optional_string(value.get("source_span")),
            source_hash=str(value["source_hash"]).strip().lower(),
            sensitivity=str(value.get("sensitivity", "internal")).strip().lower(),  # type: ignore[arg-type]
            acl=_string_tuple(value.get("acl", [])),
            aliases=_string_tuple(value.get("aliases", [])),
            entities=_string_tuple(value.get("entities", [])),
            relations=tuple(
                KnowledgeRelation.from_value(item) for item in value.get("relations", [])
            ),
            rules=tuple(PolicyRule.from_dict(item) for item in value.get("rules", [])),
            metadata=dict(value.get("metadata", {})),
        )
        record.validate()
        return record

    def validate(self) -> None:
        if not _ID_PATTERN.fullmatch(self.record_id):
            raise ValueError(f"Invalid record_id: {self.record_id}")
        if not self.domain or not self.title or not self.statement:
            raise ValueError("Domain, title, and statement cannot be empty")
        if self.status not in {"approved", "retired"}:
            raise ValueError(f"Unsupported status: {self.status}")
        if self.sensitivity not in {"public", "internal", "restricted", "secret"}:
            raise ValueError(f"Unsupported sensitivity: {self.sensitivity}")
        if self.valid_to is not None and self.valid_to <= self.valid_from:
            raise ValueError(
                "valid_to must be later than valid_from (the interval is end-exclusive)"
            )
        if not _SHA256_PATTERN.fullmatch(self.source_hash):
            raise ValueError("source_hash must be a lowercase SHA-256 hex digest")
        if self.observed_at.utcoffset() is None:
            raise ValueError("observed_at must include a timezone")
        if self.sensitivity != "public" and not self.acl:
            raise ValueError("Non-public records require at least one ACL principal")
        if any(not principal or ":" not in principal for principal in self.acl):
            raise ValueError("ACL values must be typed principals such as group:employees")
        if self.version == 1 and self.supersedes is not None:
            raise ValueError("Version 1 cannot supersede another version")
        if self.version > 1:
            if self.supersedes is None:
                raise ValueError("Versions after v1 must name the version they supersede")
            match = _REFERENCE_PATTERN.fullmatch(self.supersedes)
            if not match:
                raise ValueError(f"Invalid supersedes reference: {self.supersedes}")
            if match.group("record_id") != self.record_id:
                raise ValueError("A record version can only supersede the same record_id")
            if int(match.group("version")) >= self.version:
                raise ValueError("A record version must supersede an earlier version")
        rule_ids: set[str] = set()
        for rule in self.rules:
            rule.validate()
            if rule.rule_id in rule_ids:
                raise ValueError(f"Duplicate policy rule_id: {rule.rule_id}")
            rule_ids.add(rule.rule_id)


def source_digest(content: str) -> str:
    """Return a deterministic digest for source content used by demos and importers."""

    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def utc_now() -> datetime:
    return datetime.now(UTC)


def _parse_version(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("version must be a positive integer")
    try:
        version = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("version must be a positive integer") from exc
    if version < 1 or str(version) != str(value).strip():
        raise ValueError("version must be a positive integer")
    return version


def _parse_date(value: Any, field_name: str) -> date:
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO date") from exc


def _optional_date(value: Any, field_name: str) -> date | None:
    if value in (None, ""):
        return None
    return _parse_date(value, field_name)


def _parse_datetime(value: Any) -> datetime:
    text = str(value).strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError("observed_at must be an ISO datetime") from exc
    if parsed.utcoffset() is None:
        raise ValueError("observed_at must include a timezone")
    return parsed.astimezone(UTC)


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _string_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("Expected a list of strings")
    return tuple(str(item).strip() for item in value if str(item).strip())
