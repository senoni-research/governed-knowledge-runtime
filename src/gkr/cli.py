from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sqlite3
import sys
from datetime import UTC, date, datetime
from pathlib import Path

from gkr.ai.mlx import MLXGenerator
from gkr.authority import AuthorityStore
from gkr.evaluation import run_retrieval_suite
from gkr.retrieval import LocalRetrievalRouter
from gkr.runtime import GovernedKnowledgeRuntime
from gkr.schemas import Actor
from gkr.trace import TraceStore
from gkr.verification import ModelSemanticVerifier

DEFAULT_DATABASE = Path("artifacts/authority.sqlite")
DEFAULT_TRACE_DATABASE = Path("artifacts/query-traces.sqlite")
DEFAULT_DEMO = Path("knowledge/demo_records.jsonl")
DEFAULT_EVALUATION = Path("evaluation/m0_retrieval.jsonl")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gkr",
        description="Local-first governed company knowledge on Apple Silicon.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("doctor", help="Check the local Apple Silicon runtime")

    ingest = subparsers.add_parser("ingest", help="Append validated JSONL records to the ledger")
    ingest.add_argument("source", type=Path, nargs="?", default=DEFAULT_DEMO)
    ingest.add_argument("--db", type=Path, default=DEFAULT_DATABASE)

    ledger = subparsers.add_parser("ledger", help="Inspect the append-only authority ledger")
    ledger.add_argument("--db", type=Path, default=DEFAULT_DATABASE)

    evaluate = subparsers.add_parser("eval", help="Run the frozen local retrieval suite")
    evaluate.add_argument("suite", type=Path, nargs="?", default=DEFAULT_EVALUATION)
    evaluate.add_argument("--db", type=Path, default=DEFAULT_DATABASE)
    evaluate.add_argument("--evidence-tokens", type=int, default=12_000)

    context = subparsers.add_parser(
        "context",
        help="Compile authorized evidence without invoking an LLM",
    )
    _add_query_arguments(context)
    context.add_argument("--json", action="store_true", help="Emit a machine-readable bundle")

    ask = subparsers.add_parser("ask", help="Answer deterministically or with local MLX")
    _add_query_arguments(ask)
    ask.add_argument("--model", help="Local MLX model directory for non-deterministic queries")
    ask.add_argument("--adapter", type=Path, help="Optional local MLX LoRA adapter")
    ask.add_argument("--verifier-model", help="Optional separate local MLX verifier directory")
    ask.add_argument("--verifier-adapter", type=Path, help="Optional verifier LoRA adapter")
    ask.add_argument("--max-tokens", type=int, default=512)
    ask.add_argument("--verifier-max-tokens", type=int, default=256)
    ask.add_argument("--trace-db", type=Path, default=DEFAULT_TRACE_DATABASE)
    ask.add_argument(
        "--allow-model-download",
        action="store_true",
        help="Explicitly allow MLX-LM to download a model before local execution",
    )
    ask.add_argument("--json", action="store_true", help="Emit the answer and evidence metadata")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "doctor":
            return _doctor()
        if args.command == "ingest":
            return _ingest(args)
        if args.command == "ledger":
            return _ledger(args)
        if args.command == "eval":
            return _evaluate(args)
        if args.command == "context":
            return _context(args)
        if args.command == "ask":
            return _ask(args)
    except (OSError, RuntimeError, sqlite3.DatabaseError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 2


def _doctor() -> int:
    checks = {
        "operating_system": platform.system(),
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "sqlite": sqlite3.sqlite_version,
        "mlx_lm_installed": importlib.util.find_spec("mlx_lm") is not None,
        "inference_policy": "local-only",
    }
    compatible = (
        checks["operating_system"] == "Darwin"
        and checks["architecture"] == "arm64"
        and (3, 11) <= sys.version_info[:2] < (3, 14)
    )
    checks["core_compatible"] = compatible
    print(json.dumps(checks, indent=2))
    return 0 if compatible else 1


def _ingest(args: argparse.Namespace) -> int:
    _ensure_database_parent(args.db)
    with AuthorityStore(args.db) as store:
        if not store.verify_chain():
            raise ValueError("Refusing ingestion because the existing ledger chain is invalid")
        hashes = store.import_jsonl(args.source)
        result = {
            "database": str(args.db),
            "appended_records": len(hashes),
            "ledger_records": store.count(),
            "chain_valid": store.verify_chain(),
            "last_event_hash": hashes[-1] if hashes else None,
        }
    print(json.dumps(result, indent=2))
    return 0 if result["chain_valid"] else 2


def _ledger(args: argparse.Namespace) -> int:
    with AuthorityStore(args.db) as store:
        result = {
            "database": str(args.db),
            "records": store.count(),
            "chain_valid": store.verify_chain(),
        }
    print(json.dumps(result, indent=2))
    return 0 if result["chain_valid"] else 2


def _evaluate(args: argparse.Namespace) -> int:
    with AuthorityStore(args.db) as store:
        runtime = GovernedKnowledgeRuntime(
            store,
            router=LocalRetrievalRouter(max_evidence_tokens=args.evidence_tokens),
        )
        report = run_retrieval_suite(runtime, args.suite)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["failed"] == 0 else 2


def _context(args: argparse.Namespace) -> int:
    actor, as_of, known_at = _query_scope(args)
    with AuthorityStore(args.db) as store:
        runtime = GovernedKnowledgeRuntime(
            store,
            router=LocalRetrievalRouter(max_evidence_tokens=args.evidence_tokens),
        )
        evidence, plan = runtime.prepare(
            args.question,
            actor=actor,
            as_of=as_of,
            known_at=known_at,
        )
    if args.json:
        result = evidence.to_dict()
        result["retrieval_reason"] = plan.reason
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(evidence.prompt)
    return 2 if evidence.missing_evidence else 0


def _ask(args: argparse.Namespace) -> int:
    actor, as_of, known_at = _query_scope(args)
    if args.verifier_model and not args.model:
        raise ValueError("--verifier-model requires --model")
    generator = (
        MLXGenerator(
            args.model,
            adapter_path=args.adapter,
            allow_download=args.allow_model_download,
        )
        if args.model
        else None
    )
    verifier_generator = (
        MLXGenerator(
            args.verifier_model,
            adapter_path=args.verifier_adapter,
            allow_download=args.allow_model_download,
        )
        if args.verifier_model
        else generator
    )
    semantic_verifier = (
        ModelSemanticVerifier(
            verifier_generator,
            max_tokens=args.verifier_max_tokens,
        )
        if verifier_generator
        else None
    )
    _ensure_database_parent(args.trace_db)
    with AuthorityStore(args.db) as store, TraceStore(args.trace_db) as trace_store:
        runtime = GovernedKnowledgeRuntime(
            store,
            generator=generator,
            semantic_verifier=semantic_verifier,
            router=LocalRetrievalRouter(max_evidence_tokens=args.evidence_tokens),
            trace_store=trace_store,
        )
        result = runtime.ask(
            args.question,
            actor=actor,
            as_of=as_of,
            known_at=known_at,
            max_tokens=args.max_tokens,
        )
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    else:
        if result.answer is not None:
            print(result.answer)
        elif result.answer_status.startswith("withheld"):
            reason = result.answer_status.removeprefix("withheld_").replace("_", " ")
            print(f"Candidate answer withheld: {reason}.")
        else:
            print("No answer was generated.")
        if result.verification:
            print(f"\nCitation integrity: {result.verification.integrity}")
            print(f"Authority snapshot: {result.evidence.authority_snapshot_id}")
            print(f"Evidence bundle: {result.evidence.evidence_bundle_id}")
            if result.trace:
                print(f"Execution trace: {result.trace.trace_id}")
    successful_outcome = result.answer_status.startswith(("published_", "abstained_"))
    return 0 if successful_outcome else 2


def _add_query_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("question")
    parser.add_argument("--db", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument(
        "--actor",
        required=True,
        help="Claimed actor ID for the local development harness; this is not authentication",
    )
    parser.add_argument(
        "--group",
        action="append",
        default=[],
        help="Claimed authority group; repeat for multiple groups. This is not authentication.",
    )
    parser.add_argument("--as-of", type=date.fromisoformat, default=date.today())
    parser.add_argument("--known-at", type=_parse_datetime)
    parser.add_argument("--evidence-tokens", type=int, default=12_000)


def _query_scope(args: argparse.Namespace) -> tuple[Actor, date, datetime | None]:
    actor = Actor(actor_id=args.actor, groups=tuple(dict.fromkeys(args.group)))
    return actor, args.as_of, args.known_at


def _parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.utcoffset() is None:
        raise argparse.ArgumentTypeError("known-at must include a timezone")
    return parsed.astimezone(UTC)


def _ensure_database_parent(path: Path) -> None:
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
