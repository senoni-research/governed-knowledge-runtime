from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Literal, Protocol

from gkr.ai import Generation
from gkr.authority import AuthorizedCorpus
from gkr.schemas import PolicyRule

_CURRENCY_PATTERNS = {
    "GBP": re.compile(r"(?:£|\bGBP\s*)\s*([\d,]+(?:\.\d+)?)", re.IGNORECASE),
    "USD": re.compile(r"(?:\$|\bUSD\s*)\s*([\d,]+(?:\.\d+)?)", re.IGNORECASE),
    "EUR": re.compile(r"(?:€|\bEUR\s*)\s*([\d,]+(?:\.\d+)?)", re.IGNORECASE),
}
_EXPLICIT_NEGATION = re.compile(
    r"\b(?:no|not|never|false|without|cannot|can't|doesn't|does not|don't|do not|"
    r"isn't|is not|aren't|are not)\b",
    re.IGNORECASE,
)
_COMPLEX_CONSTRUCTION = re.compile(
    r"\b(?:if|unless|except|excluding|versus|vs\.?|either|compared with|compared to)\b",
    re.IGNORECASE,
)
_NORMAL_FORMS = {
    "approval": "approval",
    "approved": "approval",
    "approves": "approval",
    "allow": "allow",
    "allowed": "allow",
    "allows": "allow",
    "need": "require",
    "needed": "require",
    "needs": "require",
    "permitted": "allow",
    "requires": "require",
    "required": "require",
}
_ACTION_TERMS = {"allow", "approval", "prohibit", "require"}
_COMPARISON_PHRASES = {
    ">": "above",
    ">=": "at or above",
    "<": "below",
    "<=": "at or below",
    "=": "equal to",
}
_COMPARATORS = {
    ">": lambda value, threshold: value > threshold,
    ">=": lambda value, threshold: value >= threshold,
    "<": lambda value, threshold: value < threshold,
    "<=": lambda value, threshold: value <= threshold,
    "=": lambda value, threshold: value == threshold,
}

DecisionParseStatus = Literal["supported", "ambiguous", "unsupported"]
DecisionPolarity = Literal["positive", "negative"]


class DecisionEngine(Protocol):
    def decide(
        self,
        question: str,
        corpus: AuthorizedCorpus,
        *,
        as_of: date,
    ) -> DecisionOutcome: ...


@dataclass(frozen=True)
class DecisionRequest:
    subject: str
    measure: str
    value: Decimal
    unit: str
    requested_effect: str
    polarity: DecisionPolarity
    as_of: date
    authority_reference: str
    rule_id: str
    comparator: str
    threshold: Decimal
    conditions: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "subject": self.subject,
            "measure": self.measure,
            "value": str(self.value),
            "unit": self.unit,
            "requested_effect": self.requested_effect,
            "polarity": self.polarity,
            "as_of": self.as_of.isoformat(),
            "authority_reference": self.authority_reference,
            "rule_id": self.rule_id,
            "comparator": self.comparator,
            "threshold": str(self.threshold),
            "conditions": list(self.conditions),
        }


@dataclass(frozen=True)
class DecisionParseResult:
    status: DecisionParseStatus
    reason: str
    request: DecisionRequest | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "reason": self.reason,
            "request": self.request.to_dict() if self.request else None,
        }


@dataclass(frozen=True)
class DecisionOutcome:
    parse: DecisionParseResult
    generation: Generation | None


@dataclass(frozen=True)
class _RuleCandidate:
    reference: str
    rule: PolicyRule


class TypedPolicyDecisionEngine:
    """Parse a narrow typed request and evaluate only unambiguous approved rules."""

    model_id = "deterministic-policy-rule-v2"

    def decide(
        self,
        question: str,
        corpus: AuthorizedCorpus,
        *,
        as_of: date,
    ) -> DecisionOutcome:
        parsed = self.parse(question, corpus, as_of=as_of)
        if parsed.status != "supported" or parsed.request is None:
            return DecisionOutcome(parse=parsed, generation=None)

        request = parsed.request
        comparator = _COMPARATORS[request.comparator]
        applies = comparator(request.value, request.threshold)
        comparison_phrase = _COMPARISON_PHRASES[request.comparator]
        answer = (
            f"{'Yes' if applies else 'No'}. "
            f"{request.unit} {_format_decimal(request.value)} is "
            f"{'' if applies else 'not '}{comparison_phrase} "
            f"{request.unit} {_format_decimal(request.threshold)}, so "
            f"`{request.requested_effect}` {'applies' if applies else 'does not apply'} "
            f"[{request.authority_reference}]."
        )
        generation = Generation(
            text=answer,
            model=self.model_id,
            metadata={
                "execution": "local",
                "decision": "typed_policy_rule",
                "decision_parse_status": parsed.status,
                "rule_id": request.rule_id,
                "effect": request.requested_effect,
                "amount": str(request.value),
                "threshold": str(request.threshold),
                "unit": request.unit,
                "polarity": request.polarity,
            },
        )
        return DecisionOutcome(parse=parsed, generation=generation)

    def parse(
        self,
        question: str,
        corpus: AuthorizedCorpus,
        *,
        as_of: date,
    ) -> DecisionParseResult:
        if _EXPLICIT_NEGATION.search(question):
            return DecisionParseResult(
                status="ambiguous",
                reason="Explicit negation is outside the deterministic parser boundary.",
            )
        if _COMPLEX_CONSTRUCTION.search(question):
            return DecisionParseResult(
                status="ambiguous",
                reason="Conditional or comparative wording is outside the parser boundary.",
            )
        currency_values = _currency_values(question)
        if not currency_values:
            return DecisionParseResult(
                status="unsupported",
                reason="The question contains no supported currency amount.",
            )
        if len(currency_values) != 1:
            return DecisionParseResult(
                status="ambiguous",
                reason="The question must contain exactly one currency amount.",
            )
        unit, amount = currency_values[0]
        question_terms = _terms(question)
        candidates: list[_RuleCandidate] = []

        for record in corpus.records:
            for rule in record.rules:
                if rule.unit != unit:
                    continue
                subject_terms = _terms(rule.subject)
                effect_terms = _terms(rule.effect).intersection(_ACTION_TERMS)
                if not subject_terms.intersection(question_terms):
                    continue
                if not effect_terms or not effect_terms.issubset(question_terms):
                    continue
                candidates.append(_RuleCandidate(reference=record.reference, rule=rule))

        if not candidates:
            return DecisionParseResult(
                status="ambiguous",
                reason="No unique rule matches the question subject, effect, and unit.",
            )
        if len(candidates) > 1:
            return DecisionParseResult(
                status="ambiguous",
                reason="Multiple approved rules match; deterministic execution was refused.",
            )

        candidate = candidates[0]
        rule = candidate.rule
        if rule.conditions:
            return DecisionParseResult(
                status="ambiguous",
                reason="The matching rule has conditions this parser cannot evaluate.",
            )
        if rule.exceptions:
            return DecisionParseResult(
                status="ambiguous",
                reason="The matching rule has exceptions this parser cannot evaluate.",
            )
        request = DecisionRequest(
            subject=rule.subject,
            measure=rule.measure,
            value=amount,
            unit=unit,
            requested_effect=rule.effect,
            polarity="positive",
            as_of=as_of,
            authority_reference=candidate.reference,
            rule_id=rule.rule_id,
            comparator=rule.comparator,
            threshold=rule.threshold,
            conditions=rule.conditions,
        )
        return DecisionParseResult(
            status="supported",
            reason="Exactly one approved structured rule matched the typed request.",
            request=request,
        )


def _currency_values(question: str) -> list[tuple[str, Decimal]]:
    matches: list[tuple[str, Decimal]] = []
    for unit, pattern in _CURRENCY_PATTERNS.items():
        for match in pattern.finditer(question):
            try:
                matches.append((unit, Decimal(match.group(1).replace(",", ""))))
            except InvalidOperation:
                return []
    return matches


def _terms(value: str) -> set[str]:
    return {
        _NORMAL_FORMS.get(term, term)
        for term in re.findall(r"[a-z]+", value.casefold())
        if len(term) > 2
    }


def _format_decimal(value: Decimal) -> str:
    formatted = format(value, "f")
    return formatted.rstrip("0").rstrip(".") if "." in formatted else formatted


# Compatibility alias for the M0 public name.
NumericRelationDecisionEngine = TypedPolicyDecisionEngine
