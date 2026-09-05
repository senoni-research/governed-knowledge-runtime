#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from gkr.ai import GenerationRequest
from gkr.ai.mlx import MLXGenerator
from gkr.verification.semantic import _first_json_object

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_RECORDS = REPO_ROOT / "knowledge" / "demo_records.jsonl"

ExpectedVerdict = Literal["supported", "unsupported"]


@dataclass(frozen=True)
class ReplayCase:
    case_id: str
    run_directory: str
    source_case: str
    candidate_field: str
    evidence_references: tuple[str, ...]
    expected_verdict: ExpectedVerdict


CASES = (
    ReplayCase(
        "faulty_exhaustive_release",
        "local-pilot-20260905-1",
        "01_release_requirements",
        "answer",
        ("ENG-REL-001:v1",),
        "unsupported",
    ),
    ReplayCase(
        "faulty_missing_handbook_citation",
        "local-pilot-20260905-1",
        "06_two_record_question",
        "withheld_candidate",
        ("ENG-REL-001:v1", "PUB-HBK-001:v1"),
        "unsupported",
    ),
    ReplayCase(
        "faulty_broadened_copy_prohibition",
        "local-pilot-20260905-1",
        "11_authorized_recovery_answer",
        "answer",
        ("SEC-REC-001:v1",),
        "unsupported",
    ),
    ReplayCase(
        "correct_split_booking_false_rejection",
        "local-pilot-20260905-1",
        "15_split_booking_safeguard",
        "withheld_candidate",
        ("FIN-EXP-001:v2",),
        "supported",
    ),
    ReplayCase(
        "correct_claim_bound_release",
        "local-pilot-20260905-9",
        "01_release_requirements",
        "answer",
        ("ENG-REL-001:v1",),
        "supported",
    ),
    ReplayCase(
        "correct_claim_bound_two_record",
        "local-pilot-20260905-9",
        "06_two_record_question",
        "answer",
        ("ENG-REL-001:v1", "PUB-HBK-001:v1"),
        "supported",
    ),
    ReplayCase(
        "correct_conditional_copy_prohibition",
        "local-pilot-20260905-9",
        "11_authorized_recovery_answer",
        "answer",
        ("SEC-REC-001:v1",),
        "supported",
    ),
    ReplayCase(
        "correct_claim_bound_split_booking",
        "local-pilot-20260905-9",
        "15_split_booking_safeguard",
        "answer",
        ("FIN-EXP-001:v2",),
        "supported",
    ),
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Replay the previously saved faulty and correct pilot candidates through "
            "one local verifier. This is a bounded diagnostic, not a benchmark."
        )
    )
    parser.add_argument("--verifier-model", type=Path, required=True)
    parser.add_argument("--verifier-revision")
    parser.add_argument("--artifacts-dir", type=Path, default=REPO_ROOT / "artifacts")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    model_path = args.verifier_model.expanduser().resolve()
    if not model_path.is_dir():
        parser.error(f"verifier model is not an existing local directory: {model_path}")

    evidence = _load_evidence()
    generator = MLXGenerator(model_path, enable_thinking=False)
    report: dict[str, Any] = {
        "schema_version": "gkr-saved-verifier-replay-v1",
        "purpose": (
            "Diagnostic replay of eight previously saved candidates; not verifier "
            "certification or an accuracy benchmark."
        ),
        "verifier_model": str(model_path),
        "verifier_revision": args.verifier_revision,
        "max_tokens": args.max_tokens,
        "thinking_enabled": False,
        "cases": [],
    }

    for case in CASES:
        candidate, source_path = _load_candidate(args.artifacts_dir, case)
        prompt = _prompt(candidate, case.evidence_references, evidence)
        started = time.perf_counter()
        result: dict[str, Any] = {
            "case_id": case.case_id,
            "source_artifact": str(source_path),
            "candidate": candidate,
            "evidence_references": list(case.evidence_references),
            "expected_verdict": case.expected_verdict,
        }
        try:
            generation = generator.generate(
                GenerationRequest(
                    prompt=prompt,
                    max_tokens=args.max_tokens,
                    temperature=0.0,
                )
            )
            parsed = _first_json_object(generation.text)
            verdict = str(parsed.get("verdict", "")).strip().lower()
            if verdict not in {"supported", "unsupported", "inconclusive"}:
                raise ValueError(f"invalid verifier verdict: {verdict or '<empty>'}")
            issues = parsed.get("issues", [])
            if not isinstance(issues, list):
                raise ValueError("verifier issues must be a list")
            result.update(
                {
                    "actual_verdict": verdict,
                    "issues": [str(issue) for issue in issues],
                    "raw_response": generation.text,
                    "generation_metadata": generation.metadata,
                    "execution_error": None,
                    "matched_expected": verdict == case.expected_verdict,
                }
            )
        except (KeyError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
            result.update(
                {
                    "actual_verdict": "error",
                    "issues": [],
                    "raw_response": None,
                    "generation_metadata": {},
                    "execution_error": str(exc),
                    "matched_expected": False,
                }
            )
        result["wall_ms"] = round((time.perf_counter() - started) * 1000, 3)
        report["cases"].append(result)
        print(
            f"{case.case_id}: expected={case.expected_verdict} "
            f"actual={result['actual_verdict']} matched={result['matched_expected']}"
        )

    report["matched_expected_count"] = sum(
        bool(case["matched_expected"]) for case in report["cases"]
    )
    report["execution_error_count"] = sum(
        case["execution_error"] is not None for case in report["cases"]
    )
    report["peak_memory_gb"] = max(
        (
            float(case["generation_metadata"].get("peak_memory_gb", 0.0))
            for case in report["cases"]
        ),
        default=0.0,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"saved {output}")
    return 0 if report["matched_expected_count"] == len(CASES) else 1


def _load_evidence() -> dict[str, str]:
    records: dict[str, str] = {}
    for line in DEMO_RECORDS.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        records[f"{value['record_id']}:v{value['version']}"] = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
        )
    return records


def _load_candidate(artifacts_dir: Path, case: ReplayCase) -> tuple[str, Path]:
    source = (
        artifacts_dir.expanduser().resolve()
        / case.run_directory
        / f"{case.source_case}.json"
    )
    value = json.loads(source.read_text(encoding="utf-8"))
    payload = value["stdout"]
    candidate = payload.get(case.candidate_field)
    if not isinstance(candidate, str) or not candidate.strip():
        raise ValueError(
            f"{source} has no non-empty stdout.{case.candidate_field} candidate"
        )
    return candidate.strip(), source


def _prompt(
    candidate: str,
    references: tuple[str, ...],
    evidence: dict[str, str],
) -> str:
    evidence_text = "\n\n".join(
        f"[{reference}]\n{evidence[reference]}" for reference in references
    )
    return f"""Audit this saved candidate using only the supplied evidence.

A candidate is supported only if every material factual statement is entailed by
the evidence and every material statement has a citation to its supporting record.
Reject unsupported exhaustive conclusions, broadened scope, lost conditions,
incorrect prohibitions or permissions, and missing source coverage. Do not repair
the candidate and do not infer facts from missing evidence.

EVIDENCE
{evidence_text}

CANDIDATE
{candidate}

Return exactly one JSON object:
{{"verdict":"supported|unsupported|inconclusive","issues":["brief issue"]}}
"""


if __name__ == "__main__":
    raise SystemExit(main())
