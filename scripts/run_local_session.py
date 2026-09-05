#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from gkr.ai.mlx import MLXGenerator
from gkr.authority import AuthorityStore
from gkr.retrieval import LocalRetrievalRouter
from gkr.runtime import GovernedKnowledgeRuntime, RuntimeResult
from gkr.schemas import Actor
from gkr.trace import TraceStore
from gkr.verification import ModelSemanticVerifier


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a configured, non-production interactive local GKR session."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--question",
        action="append",
        dest="questions",
        help="Run one question non-interactively; repeat as needed.",
    )
    args = parser.parse_args()
    config_path = args.config.expanduser().resolve()
    config = _load_config(config_path)

    database = _path(config, "database", relative_to=config_path.parent)
    trace_database = _path(config, "trace_database", relative_to=config_path.parent)
    generator_model = _path(config, "generator_model", relative_to=config_path.parent)
    verifier_model = _path(config, "verifier_model", relative_to=config_path.parent)
    if not generator_model.is_dir():
        parser.error(f"generator_model is not a local directory: {generator_model}")
    if not verifier_model.is_dir():
        parser.error(f"verifier_model is not a local directory: {verifier_model}")
    if generator_model == verifier_model:
        parser.error("generator_model and verifier_model must be distinct")
    if not database.is_file():
        parser.error(f"database does not exist: {database}")

    trace_database.parent.mkdir(parents=True, exist_ok=True)
    actor = Actor(
        actor_id=_required_string(config, "actor"),
        groups=_string_tuple(config.get("groups", []), label="groups"),
    )
    as_of = date.fromisoformat(_required_string(config, "as_of"))
    known_at = _optional_datetime(config.get("known_at"))
    max_tokens = _positive_int(config.get("max_tokens", 512), label="max_tokens")
    verifier_max_tokens = _positive_int(
        config.get("verifier_max_tokens", 256),
        label="verifier_max_tokens",
    )
    evidence_tokens = _positive_int(
        config.get("evidence_tokens", 12_000),
        label="evidence_tokens",
    )

    generator = MLXGenerator(generator_model)
    verifier_generator = MLXGenerator(verifier_model)
    with AuthorityStore(database) as store, TraceStore(trace_database) as trace_store:
        runtime = GovernedKnowledgeRuntime(
            store,
            generator=generator,
            semantic_verifier=ModelSemanticVerifier(
                verifier_generator,
                max_tokens=verifier_max_tokens,
            ),
            router=LocalRetrievalRouter(max_evidence_tokens=evidence_tokens),
            trace_store=trace_store,
        )
        print("GKR local development session")
        print("Actor/group values are simulated assertions, not authentication.")
        print("A local verifier-supported answer is not certified correct.")
        print(f"Database: {database}")
        print(f"Decision date: {as_of.isoformat()}")

        if args.questions:
            for question in args.questions:
                result = runtime.ask(
                    question,
                    actor=actor,
                    as_of=as_of,
                    known_at=known_at,
                    max_tokens=max_tokens,
                )
                _display(result, as_of=as_of)
            return 0

        print("Commands: /as-of YYYY-MM-DD, /known-at TIMESTAMP|now, /status, /quit")
        while True:
            try:
                value = input("\ngkr> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if not value:
                continue
            if value in {"/quit", "/exit"}:
                return 0
            if value == "/status":
                print(f"Decision date: {as_of.isoformat()}")
                print(f"Known-at: {known_at.isoformat() if known_at else 'now'}")
                continue
            if value.startswith("/as-of "):
                as_of = date.fromisoformat(value.removeprefix("/as-of ").strip())
                print(f"Decision date: {as_of.isoformat()}")
                continue
            if value.startswith("/known-at "):
                known_at = _optional_datetime(value.removeprefix("/known-at ").strip())
                print(f"Known-at: {known_at.isoformat() if known_at else 'now'}")
                continue
            result = runtime.ask(
                value,
                actor=actor,
                as_of=as_of,
                known_at=known_at,
                max_tokens=max_tokens,
            )
            _display(result, as_of=as_of)


def _display(result: RuntimeResult, *, as_of: date) -> None:
    print()
    if result.answer is not None:
        label = "Abstention" if result.answer_status.startswith("abstained_") else "Answer"
        print(f"{label}: {result.answer}")
    else:
        print("Answer: withheld")
    print(f"Decision date: {as_of.isoformat()}")
    print(f"Check status: {_status_label(result.answer_status)}")
    cited = result.verification.cited_references if result.verification else ()
    print(f"Cited sources: {', '.join(cited) if cited else 'none'}")
    print(f"Permitted evidence: {', '.join(result.evidence.record_references) or 'none'}")
    for claim in result.claims:
        print(f"Claim: {claim.claim}")
        print(f"  Source: {claim.record_reference}")
        print(f"  Passage: {claim.supporting_passage}")
    if result.answer is None and result.generation is not None:
        print("Diagnostic withheld candidate (not an answer):")
        print(result.generation.text)
    if result.claim_contract_issues:
        print("Claim-contract issues:")
        for issue in result.claim_contract_issues:
            print(f"- {issue}")
    if result.trace is not None:
        print(f"Elapsed: {result.trace.duration_ms:.1f} ms")
        print(f"Trace: {result.trace.trace_id}")


def _status_label(status: str) -> str:
    if status == "published_local_verifier_supported":
        return "local verifier supported; not certified correct"
    if status == "published_deterministic_policy_rule":
        return "deterministic policy decision"
    if status.startswith("abstained_"):
        return "runtime abstention"
    return status.replace("_", " ")


def _load_config(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot load session config {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("Session config must be a JSON object")
    return value


def _path(config: dict[str, Any], key: str, *, relative_to: Path) -> Path:
    raw = Path(_required_string(config, key)).expanduser()
    return raw.resolve() if raw.is_absolute() else (relative_to / raw).resolve()


def _required_string(config: dict[str, Any], key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Session config {key} must be a non-empty string")
    return value.strip()


def _string_tuple(value: object, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ValueError(f"Session config {label} must be an array of non-empty strings")
    return tuple(dict.fromkeys(item.strip() for item in value))


def _optional_datetime(value: object) -> datetime | None:
    if value in (None, "", "now"):
        return None
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.utcoffset() is None:
        raise ValueError("known_at must include a timezone")
    return parsed.astimezone(UTC)


def _positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a positive integer")
    parsed = int(value)
    if parsed < 1:
        raise ValueError(f"{label} must be a positive integer")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
