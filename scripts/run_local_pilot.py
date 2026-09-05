#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_RECORDS = REPO_ROOT / "knowledge" / "demo_records.jsonl"


@dataclass(frozen=True)
class PilotPaths:
    output: Path
    authority_db: Path
    trace_db: Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a non-production local MLX knowledge-assistant pilot against "
            "an isolated copy of the synthetic demo ledger."
        )
    )
    parser.add_argument("--generator-model", type=Path, required=True)
    parser.add_argument("--verifier-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    generator = args.generator_model.expanduser().resolve()
    verifier = args.verifier_model.expanduser().resolve()
    for label, model in (("generator", generator), ("verifier", verifier)):
        if not model.is_dir():
            parser.error(f"{label} model is not an existing local directory: {model}")
    if generator == verifier:
        parser.error("generator and verifier models must use distinct local directories")

    output = args.output_dir.expanduser().resolve()
    if output.exists():
        parser.error(
            f"output directory already exists: {output}; choose a fresh directory"
        )
    output.mkdir(parents=True)
    paths = PilotPaths(
        output=output,
        authority_db=output / "authority.sqlite",
        trace_db=output / "query-traces.sqlite",
    )
    updates = _write_updates(output)
    report: dict[str, Any] = {
        "schema_version": "gkr-local-pilot-report-v1",
        "purpose": (
            "Synthetic single-user development evidence only; not a benchmark, "
            "production approval, or authenticated authorization test."
        ),
        "started_at": _utc_now(),
        "generator_model": str(generator),
        "verifier_model": str(verifier),
        "authority_database": str(paths.authority_db),
        "trace_database": str(paths.trace_db),
        "cases": [],
    }

    _record(
        report,
        paths,
        "00_seed_demo",
        _cli(
            "ingest",
            str(DEMO_RECORDS),
            "--db",
            str(paths.authority_db),
        ),
        expected_exit_codes={0},
    )

    model_cases = (
        (
            "01_release_requirements",
            "What evidence is required before a production deployment?",
            "2026-09-01",
            "2026-09-05T12:30:00Z",
            ("employees",),
        ),
        (
            "02_release_paraphrase",
            "Before we ship to production, what proof must the release record contain?",
            "2026-09-01",
            "2026-09-05T12:30:00Z",
            ("employees",),
        ),
        (
            "03_high_risk_release",
            "What extra approval is required for a high-risk production change?",
            "2026-09-01",
            "2026-09-05T12:30:00Z",
            ("employees",),
        ),
        (
            "04_travel_policy_before_update",
            "What approval and filing requirements apply to travel and subsistence?",
            "2026-09-30",
            "2026-09-30T12:00:00Z",
            ("employees",),
        ),
        (
            "05_travel_policy_historical",
            "What approval and filing requirements applied to travel and subsistence?",
            "2026-08-31",
            "2026-09-30T12:00:00Z",
            ("employees",),
        ),
        (
            "06_two_record_question",
            (
                "Can the public handbook index override production-release "
                "requirements, who owns those requirements, and what evidence "
                "must be linked before deployment?"
            ),
            "2026-09-01",
            "2026-09-05T12:30:00Z",
            ("employees",),
        ),
        (
            "07_unknown_question",
            "What meal allowance applies to business travel on Mars?",
            "2026-09-01",
            "2026-09-05T12:30:00Z",
            ("employees",),
        ),
        (
            "08_denied_recovery_answer",
            "Summarize the restricted cryptographic recovery procedure.",
            "2026-09-01",
            "2026-09-05T12:30:00Z",
            ("employees",),
        ),
        (
            "11_authorized_recovery_answer",
            "Who may access cryptographic recovery material and where may it be copied?",
            "2026-09-01",
            "2026-09-05T12:30:00Z",
            ("security-recovery",),
        ),
        (
            "15_split_booking_safeguard",
            "May I split one travel purchase into smaller bookings to avoid approval?",
            "2026-09-30",
            "2026-09-30T12:00:00Z",
            ("employees",),
        ),
    )
    for case_id, question, as_of, known_at, groups in model_cases:
        _record(
            report,
            paths,
            case_id,
            _ask_command(
                question,
                paths=paths,
                generator=generator,
                verifier=verifier,
                as_of=as_of,
                known_at=known_at,
                groups=groups,
            ),
            expected_exit_codes={0, 2},
        )

    _record(
        report,
        paths,
        "09_denied_recovery_context",
        _context_command(
            "Summarize the restricted cryptographic recovery procedure.",
            paths=paths,
            as_of="2026-09-01",
            known_at="2026-09-05T12:30:00Z",
            groups=("employees",),
        ),
        expected_exit_codes={0, 2},
    )
    _record(
        report,
        paths,
        "10_authorized_recovery_context",
        _context_command(
            "Summarize the restricted cryptographic recovery procedure.",
            paths=paths,
            as_of="2026-09-01",
            known_at="2026-09-05T12:30:00Z",
            groups=("security-recovery",),
        ),
        expected_exit_codes={0, 2},
    )

    deterministic_cases = (
        (
            "12_threshold_historical_v1",
            "Does a £700 travel booking require special approval?",
            "2026-08-31",
            "2026-09-30T12:00:00Z",
        ),
        (
            "13_threshold_current_v2",
            "Does a £700 travel booking require special approval?",
            "2026-09-01",
            "2026-09-30T12:00:00Z",
        ),
        (
            "14_threshold_boundary_v2",
            "Does a £750 travel booking require special approval?",
            "2026-09-01",
            "2026-09-30T12:00:00Z",
        ),
    )
    for case_id, question, as_of, known_at in deterministic_cases:
        _record(
            report,
            paths,
            case_id,
            _ask_command(
                question,
                paths=paths,
                generator=generator,
                verifier=verifier,
                as_of=as_of,
                known_at=known_at,
                groups=("employees",),
            ),
            expected_exit_codes={0, 2},
        )

    _record(
        report,
        paths,
        "16_proposed_update_rejected",
        _cli(
            "ingest",
            str(updates["proposed"]),
            "--db",
            str(paths.authority_db),
        ),
        expected_exit_codes={2},
    )
    _record(
        report,
        paths,
        "17_after_proposal_unchanged",
        _ask_command(
            "Does an £800 travel booking require special approval?",
            paths=paths,
            generator=generator,
            verifier=verifier,
            as_of="2026-09-30",
            known_at="2026-09-30T12:00:00Z",
            groups=("employees",),
        ),
        expected_exit_codes={0, 2},
    )
    _record(
        report,
        paths,
        "18_approved_update_ingested",
        _cli(
            "ingest",
            str(updates["approved"]),
            "--db",
            str(paths.authority_db),
        ),
        expected_exit_codes={0},
    )

    post_update_cases = (
        (
            "19_current_after_update_v3",
            "Does an £800 travel booking require special approval?",
            "2026-10-01",
            "2026-10-02T09:00:00Z",
        ),
        (
            "20_history_after_update_v2",
            "Does an £800 travel booking require special approval?",
            "2026-09-30",
            "2026-10-02T09:00:00Z",
        ),
        (
            "21_known_at_before_update",
            "Does an £800 travel booking require special approval?",
            "2026-10-01",
            "2026-09-04T09:00:00Z",
        ),
    )
    for case_id, question, as_of, known_at in post_update_cases:
        _record(
            report,
            paths,
            case_id,
            _ask_command(
                question,
                paths=paths,
                generator=generator,
                verifier=verifier,
                as_of=as_of,
                known_at=known_at,
                groups=("employees",),
            ),
            expected_exit_codes={0, 2},
        )

    _record(
        report,
        paths,
        "22_travel_policy_after_update",
        _ask_command(
            "What approval and filing requirements apply to travel and subsistence?",
            paths=paths,
            generator=generator,
            verifier=verifier,
            as_of="2026-10-01",
            known_at="2026-10-02T09:00:00Z",
            groups=("employees",),
        ),
        expected_exit_codes={0, 2},
    )
    _record(
        report,
        paths,
        "23_final_ledger",
        _cli("ledger", "--db", str(paths.authority_db)),
        expected_exit_codes={0},
    )

    report["completed_at"] = _utc_now()
    report["case_count"] = len(report["cases"])
    report["unexpected_exit_count"] = sum(
        not bool(case["exit_code_expected"]) for case in report["cases"]
    )
    report["published_answer_count"] = sum(
        str(case.get("answer_status", "")).startswith("published_")
        for case in report["cases"]
    )
    report["withheld_or_refused_count"] = sum(
        str(case.get("answer_status", "")).startswith(("withheld_", "refused_"))
        for case in report["cases"]
    )
    (output / "pilot-report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output / "pilot-report.md").write_text(_render_markdown(report), encoding="utf-8")
    print(json.dumps(_terminal_summary(report), indent=2, ensure_ascii=False))
    return 0 if report["unexpected_exit_count"] == 0 else 1


def _ask_command(
    question: str,
    *,
    paths: PilotPaths,
    generator: Path,
    verifier: Path,
    as_of: str,
    known_at: str,
    groups: tuple[str, ...],
) -> list[str]:
    command = [
        "ask",
        question,
        "--db",
        str(paths.authority_db),
        "--actor",
        "pilot-user",
    ]
    for group in groups:
        command.extend(("--group", group))
    command.extend(
        (
            "--as-of",
            as_of,
            "--known-at",
            known_at,
            "--model",
            str(generator),
            "--verifier-model",
            str(verifier),
            "--max-tokens",
            "256",
            "--verifier-max-tokens",
            "96",
            "--trace-db",
            str(paths.trace_db),
            "--json",
        )
    )
    return _cli(*command)


def _context_command(
    question: str,
    *,
    paths: PilotPaths,
    as_of: str,
    known_at: str,
    groups: tuple[str, ...],
) -> list[str]:
    command = [
        "context",
        question,
        "--db",
        str(paths.authority_db),
        "--actor",
        "pilot-user",
    ]
    for group in groups:
        command.extend(("--group", group))
    command.extend(("--as-of", as_of, "--known-at", known_at, "--json"))
    return _cli(*command)


def _cli(*arguments: str) -> list[str]:
    return [sys.executable, "-m", "gkr.cli", *arguments]


def _record(
    report: dict[str, Any],
    paths: PilotPaths,
    case_id: str,
    command: list[str],
    *,
    expected_exit_codes: set[int],
) -> None:
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    wall_ms = (time.perf_counter() - started) * 1000
    payload = _parse_json(completed.stdout)
    raw = {
        "case_id": case_id,
        "command": command,
        "exit_code": completed.returncode,
        "expected_exit_codes": sorted(expected_exit_codes),
        "wall_ms": round(wall_ms, 3),
        "stdout": payload if payload is not None else completed.stdout,
        "stderr": completed.stderr,
    }
    (paths.output / f"{case_id}.json").write_text(
        json.dumps(raw, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    summary = {
        "case_id": case_id,
        "operation": command[3] if len(command) > 3 else "unknown",
        "exit_code": completed.returncode,
        "exit_code_expected": completed.returncode in expected_exit_codes,
        "wall_ms": round(wall_ms, 3),
        "stderr": completed.stderr.strip() or None,
    }
    if isinstance(payload, dict):
        summary.update(_summarize_payload(payload))
    report["cases"].append(summary)
    print(
        f"{case_id}: exit={completed.returncode} "
        f"status={summary.get('answer_status', summary.get('operation', 'unknown'))} "
        f"wall_ms={summary['wall_ms']}"
    )


def _parse_json(value: str) -> Any | None:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _summarize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if "answer_status" in payload:
        trace = payload.get("trace") or {}
        verification = payload.get("verification") or {}
        semantic = payload.get("semantic_verification") or {}
        evidence = payload.get("evidence") or {}
        return {
            "operation": "ask",
            "question": trace.get("question") or evidence.get("question"),
            "answer_status": payload.get("answer_status"),
            "answer": payload.get("answer"),
            "withheld_candidate": payload.get("withheld_candidate"),
            "model": payload.get("model"),
            "verifier_model": semantic.get("verifier_model"),
            "semantic_verdict": semantic.get("verdict"),
            "semantic_issues": semantic.get("issues", []),
            "citation_integrity": verification.get("citation_integrity"),
            "cited_references": verification.get("cited_references", []),
            "evidence_references": evidence.get("record_references", []),
            "authority_snapshot_id": evidence.get("authority_snapshot_id"),
            "evidence_bundle_id": evidence.get("evidence_bundle_id"),
            "trace_id": trace.get("trace_id"),
            "runtime_duration_ms": trace.get("duration_ms"),
            "peak_process_rss_bytes": trace.get("peak_process_rss_bytes"),
        }
    if "record_references" in payload:
        return {
            "operation": "context",
            "question": payload.get("question"),
            "evidence_references": payload.get("record_references", []),
            "missing_evidence": payload.get("missing_evidence"),
            "authority_snapshot_id": payload.get("authority_snapshot_id"),
            "evidence_bundle_id": payload.get("evidence_bundle_id"),
        }
    if "appended_records" in payload:
        return {
            "operation": "ingest",
            "appended_records": payload.get("appended_records"),
            "ledger_records": payload.get("ledger_records"),
            "chain_valid": payload.get("chain_valid"),
        }
    if "records" in payload and "chain_valid" in payload:
        return {
            "operation": "ledger",
            "ledger_records": payload.get("records"),
            "chain_valid": payload.get("chain_valid"),
        }
    return {"operation": "unknown"}


def _write_updates(output: Path) -> dict[str, Path]:
    proposed_statement = (
        "Travel or subsistence spend above £1,000 requires written approval from "
        "the relevant budget owner before booking. The claimant must submit an "
        "itemised receipt and business purpose within five calendar days after the trip."
    )
    approved_statement = (
        "Travel or subsistence spend above £900 requires written approval from "
        "the relevant budget owner before booking. An item must not be split into "
        "smaller transactions to avoid the threshold. The claimant must submit an "
        "itemised receipt and business purpose within seven calendar days after the trip."
    )
    common = {
        "record_id": "FIN-EXP-001",
        "version": 3,
        "domain": "finance",
        "title": "Travel and subsistence approval",
        "valid_from": "2026-10-01",
        "valid_to": None,
        "supersedes": "FIN-EXP-001:v2",
        "owner": "Finance Policy",
        "source_span": "section-4.2",
        "sensitivity": "internal",
        "acl": ["group:employees"],
        "aliases": ["travel expenses", "expense threshold", "receipt deadline"],
        "entities": ["travel-spend", "budget-owner"],
        "relations": [["travel-spend", "requires-approval-above", "GBP-900"]],
        "metadata": {"demo": True, "pilot": True, "hash_scope": "statement"},
    }
    proposed = {
        **common,
        "statement": proposed_statement,
        "observed_at": "2026-09-05T11:00:00Z",
        "status": "proposed",
        "source_uri": "synthetic://pilot/finance/expense-policy/proposed-v3",
        "source_hash": hashlib.sha256(proposed_statement.encode()).hexdigest(),
        "rules": [],
    }
    approved = {
        **common,
        "statement": approved_statement,
        "observed_at": "2026-09-05T12:00:00Z",
        "status": "approved",
        "source_uri": "synthetic://pilot/finance/expense-policy/v3",
        "source_hash": hashlib.sha256(approved_statement.encode()).hexdigest(),
        "rules": [
            {
                "rule_id": "FIN-EXP-001.approval-threshold",
                "subject": "travel-spend",
                "measure": "gross-amount",
                "unit": "GBP",
                "comparator": ">",
                "threshold": "900",
                "effect": "written-budget-owner-approval-required-before-booking",
                "conditions": [],
                "exceptions": [],
            }
        ],
    }
    proposed_path = output / "proposed-update.jsonl"
    approved_path = output / "approved-update.jsonl"
    proposed_path.write_text(
        json.dumps(proposed, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    approved_path.write_text(
        json.dumps(approved, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return {"proposed": proposed_path, "approved": approved_path}


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Local MLX knowledge-assistant pilot",
        "",
        str(report["purpose"]),
        "",
        f"- Generator: `{report['generator_model']}`",
        f"- Verifier: `{report['verifier_model']}`",
        f"- Cases/operations: {report['case_count']}",
        f"- Unexpected exit codes: {report['unexpected_exit_count']}",
        f"- Published answers: {report['published_answer_count']}",
        f"- Withheld/refused answers: {report['withheld_or_refused_count']}",
        "",
        "## Observed cases",
        "",
    ]
    for case in report["cases"]:
        status = case.get("answer_status") or case.get("operation")
        lines.extend(
            (
                f"### {case['case_id']}",
                "",
                f"- Exit: `{case['exit_code']}` (expected: `{case['exit_code_expected']}`)",
                f"- Status: `{status}`",
                f"- Wall time: `{case['wall_ms']} ms`",
            )
        )
        if case.get("question"):
            lines.append(f"- Question: {case['question']}")
        if case.get("evidence_references") is not None:
            refs = ", ".join(case["evidence_references"]) or "none"
            lines.append(f"- Permitted evidence: `{refs}`")
        if case.get("cited_references") is not None:
            refs = ", ".join(case["cited_references"]) or "none"
            lines.append(f"- Cited evidence: `{refs}`")
        if case.get("trace_id"):
            lines.append(f"- Trace: `{case['trace_id']}`")
        if case.get("answer"):
            lines.extend(("", str(case["answer"])))
        elif case.get("withheld_candidate"):
            lines.extend(
                (
                    "",
                    "**Diagnostic withheld candidate:**",
                    "",
                    str(case["withheld_candidate"]),
                )
            )
        if case.get("stderr"):
            lines.extend(("", f"Error: `{case['stderr']}`"))
        lines.append("")
    return "\n".join(lines)


def _terminal_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "output_directory": str(Path(report["authority_database"]).parent),
        "case_count": report["case_count"],
        "unexpected_exit_count": report["unexpected_exit_count"],
        "published_answer_count": report["published_answer_count"],
        "withheld_or_refused_count": report["withheld_or_refused_count"],
        "report_json": str(Path(report["authority_database"]).parent / "pilot-report.json"),
        "report_markdown": str(Path(report["authority_database"]).parent / "pilot-report.md"),
    }


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
