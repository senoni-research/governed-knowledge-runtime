from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "examples" / "verifier-replay" / "cases.jsonl"


def test_public_verifier_replay_fixture_is_portable_and_valid() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/replay_saved_verifier_cases.py",
            "--validate-only",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {
        "fixture": "examples/verifier-replay/cases.jsonl",
        "case_count": 8,
        "evidence_source": "knowledge/demo_records.jsonl",
        "historical_artifacts_verified": False,
        "valid": True,
    }

    fixture_text = FIXTURE.read_text(encoding="utf-8")
    assert "/Users/" not in fixture_text
    cases = [json.loads(line) for line in fixture_text.splitlines()]
    assert {case["expected_diagnostic_verdict"] for case in cases} == {
        "supported",
        "unsupported",
    }
    assert {case["provenance"]["original_run_id"] for case in cases} == {
        "local-pilot-20260905-1",
        "local-pilot-20260905-9",
    }
