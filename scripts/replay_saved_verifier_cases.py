#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from gkr.ai import GenerationRequest
from gkr.ai.mlx import MLXGenerator
from gkr.verification.semantic import _first_json_object

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_RECORDS = REPO_ROOT / "knowledge" / "demo_records.jsonl"
DEFAULT_FIXTURE = REPO_ROOT / "examples" / "verifier-replay" / "cases.jsonl"

ExpectedVerdict = Literal["supported", "unsupported"]


@dataclass(frozen=True)
class ReplayEvidence:
    record_reference: str
    supporting_passage: str


@dataclass(frozen=True)
class ReplayProvenance:
    original_run_id: str
    original_case_id: str
    original_candidate_field: str
    historical_artifact_path: str
    historical_artifact_sha256: str
    candidate_sha256: str
    source_trace_id: str


@dataclass(frozen=True)
class ReplayCase:
    case_id: str
    question: str
    candidate: str
    evidence: tuple[ReplayEvidence, ...]
    expected_verdict: ExpectedVerdict
    provenance: ReplayProvenance

    @property
    def evidence_references(self) -> tuple[str, ...]:
        return tuple(item.record_reference for item in self.evidence)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Replay the previously saved faulty and correct pilot candidates through "
            "one local verifier. This is a bounded diagnostic, not a benchmark."
        )
    )
    parser.add_argument("--verifier-model", type=Path)
    parser.add_argument("--verifier-revision")
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        help=(
            "Optional ignored artifacts root. When supplied, re-read each historical "
            "candidate and verify it against the public fixture hashes."
        ),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate fixture hashes and public evidence without loading a model",
    )
    args = parser.parse_args()

    evidence = _load_evidence()
    cases = _load_cases(args.fixture, evidence)
    if args.validate_only:
        if args.artifacts_dir is not None:
            for case in cases:
                _load_historical_candidate(args.artifacts_dir, case)
        print(
            json.dumps(
                {
                    "fixture": _display_path(args.fixture),
                    "case_count": len(cases),
                    "evidence_source": _display_path(DEMO_RECORDS),
                    "historical_artifacts_verified": args.artifacts_dir is not None,
                    "valid": True,
                },
                indent=2,
            )
        )
        return 0
    if args.verifier_model is None:
        parser.error("--verifier-model is required unless --validate-only is used")
    if args.output is None:
        parser.error("--output is required unless --validate-only is used")

    model_path = args.verifier_model.expanduser().resolve()
    if not model_path.is_dir():
        parser.error(f"verifier model is not an existing local directory: {model_path}")

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
        "fixture": _display_path(args.fixture),
        "historical_artifacts_verified": args.artifacts_dir is not None,
        "cases": [],
    }

    for case in cases:
        candidate = case.candidate
        candidate_origin = "public_fixture"
        if args.artifacts_dir is not None:
            candidate = _load_historical_candidate(args.artifacts_dir, case)
            candidate_origin = "historical_artifact_verified"
        prompt = _prompt(candidate, case.evidence_references, evidence)
        started = time.perf_counter()
        result: dict[str, Any] = {
            "case_id": case.case_id,
            "source_artifact": case.provenance.historical_artifact_path,
            "candidate_origin": candidate_origin,
            "candidate": candidate,
            "candidate_sha256": case.provenance.candidate_sha256,
            "historical_artifact_sha256": (
                case.provenance.historical_artifact_sha256
            ),
            "source_trace_id": case.provenance.source_trace_id,
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
    return 0 if report["matched_expected_count"] == len(cases) else 1


def _load_evidence() -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for line in DEMO_RECORDS.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        records[f"{value['record_id']}:v{value['version']}"] = value
    return records


def _load_cases(
    fixture: Path,
    evidence_records: dict[str, dict[str, Any]],
) -> tuple[ReplayCase, ...]:
    cases: list[ReplayCase] = []
    seen_ids: set[str] = set()
    fixture_path = fixture.expanduser().resolve()
    for line_number, line in enumerate(
        fixture_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError("each replay fixture row must be an object")
            case = _parse_case(value, evidence_records=evidence_records)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"{fixture_path}:{line_number}: {exc}") from exc
        if case.case_id in seen_ids:
            raise ValueError(f"{fixture_path}:{line_number}: duplicate case_id")
        seen_ids.add(case.case_id)
        cases.append(case)
    if len(cases) != 8:
        raise ValueError(f"{fixture_path}: expected exactly eight replay cases")
    return tuple(cases)


def _parse_case(
    value: dict[str, Any],
    *,
    evidence_records: dict[str, dict[str, Any]],
) -> ReplayCase:
    if value.get("schema_version") != "gkr-verifier-replay-case-v1":
        raise ValueError("unsupported replay fixture schema")
    case_id = _nonempty(value["case_id"], "case_id")
    question = _nonempty(value["question"], "question")
    candidate = _nonempty(value["candidate"], "candidate")
    expected = _nonempty(
        value["expected_diagnostic_verdict"],
        "expected_diagnostic_verdict",
    )
    if expected not in {"supported", "unsupported"}:
        raise ValueError("expected_diagnostic_verdict must be supported or unsupported")

    evidence_values = value["evidence"]
    if not isinstance(evidence_values, list) or not evidence_values:
        raise ValueError("evidence must be a non-empty list")
    replay_evidence: list[ReplayEvidence] = []
    for item in evidence_values:
        if not isinstance(item, dict):
            raise TypeError("each evidence item must be an object")
        reference = _nonempty(item["record_reference"], "record_reference")
        passage = _nonempty(item["supporting_passage"], "supporting_passage")
        record = evidence_records.get(reference)
        if record is None:
            raise ValueError(f"unknown public evidence reference: {reference}")
        if passage != record["statement"]:
            raise ValueError(f"supporting passage does not match {reference}")
        if _digest(passage) != item.get("supporting_passage_sha256"):
            raise ValueError(f"supporting passage hash does not match {reference}")
        if item.get("authority_source_hash") != record["source_hash"]:
            raise ValueError(f"authority source hash does not match {reference}")
        replay_evidence.append(
            ReplayEvidence(
                record_reference=reference,
                supporting_passage=passage,
            )
        )

    provenance_value = value["provenance"]
    if not isinstance(provenance_value, dict):
        raise TypeError("provenance must be an object")
    if provenance_value.get("kind") != "exported_existing_synthetic_pilot_candidate":
        raise ValueError("unsupported replay provenance kind")
    candidate_sha256 = _require_digest(
        provenance_value["candidate_sha256"],
        "candidate_sha256",
    )
    if _digest(candidate) != candidate_sha256:
        raise ValueError("candidate hash does not match exported candidate")
    historical_path = _nonempty(
        provenance_value["historical_artifact_path"],
        "historical_artifact_path",
    )
    path = Path(historical_path)
    if path.is_absolute() or not path.parts or path.parts[0] != "artifacts":
        raise ValueError("historical_artifact_path must be repository-relative under artifacts")
    provenance = ReplayProvenance(
        original_run_id=_nonempty(
            provenance_value["original_run_id"],
            "original_run_id",
        ),
        original_case_id=_nonempty(
            provenance_value["original_case_id"],
            "original_case_id",
        ),
        original_candidate_field=_nonempty(
            provenance_value["original_candidate_field"],
            "original_candidate_field",
        ),
        historical_artifact_path=historical_path,
        historical_artifact_sha256=_require_digest(
            provenance_value["historical_artifact_sha256"],
            "historical_artifact_sha256",
        ),
        candidate_sha256=candidate_sha256,
        source_trace_id=_require_digest(
            provenance_value["source_trace_id"],
            "source_trace_id",
        ),
    )
    return ReplayCase(
        case_id=case_id,
        question=question,
        candidate=candidate,
        evidence=tuple(replay_evidence),
        expected_verdict=cast(ExpectedVerdict, expected),
        provenance=provenance,
    )


def _load_historical_candidate(artifacts_dir: Path, case: ReplayCase) -> str:
    source = (
        artifacts_dir.expanduser().resolve()
        / case.provenance.original_run_id
        / f"{case.provenance.original_case_id}.json"
    )
    source_bytes = source.read_bytes()
    if _digest_bytes(source_bytes) != case.provenance.historical_artifact_sha256:
        raise ValueError(f"{source}: historical artifact hash does not match fixture")
    value = json.loads(source_bytes)
    payload = value["stdout"]
    candidate = payload.get(case.provenance.original_candidate_field)
    if not isinstance(candidate, str) or not candidate.strip():
        raise ValueError(
            f"{source} has no non-empty "
            f"stdout.{case.provenance.original_candidate_field} candidate"
        )
    candidate = candidate.strip()
    if _digest(candidate) != case.provenance.candidate_sha256:
        raise ValueError(f"{source}: historical candidate hash does not match fixture")
    return candidate


def _prompt(
    candidate: str,
    references: tuple[str, ...],
    evidence: dict[str, dict[str, Any]],
) -> str:
    evidence_text = "\n\n".join(
        f"[{reference}]\n"
        + json.dumps(evidence[reference], ensure_ascii=False, sort_keys=True)
        for reference in references
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


def _nonempty(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _require_digest(value: object, field: str) -> str:
    digest = _nonempty(value, field)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return digest


def _digest(value: str) -> str:
    return _digest_bytes(value.encode())


def _digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _display_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


if __name__ == "__main__":
    raise SystemExit(main())
