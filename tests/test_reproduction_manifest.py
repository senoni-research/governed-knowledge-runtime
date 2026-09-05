from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "reproduction" / "strong-pilot-2026-09-05.json"
CONSTRAINTS = ROOT / "reproduction" / "strong-pilot-2026-09-05-constraints.txt"


def test_strong_pilot_constraints_match_tested_environment_manifest() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    constraints = dict(
        line.split("==", maxsplit=1)
        for line in CONSTRAINTS.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    )

    assert manifest["release_classification"] == "experimental"
    assert (
        manifest["measured_source_revision"]
        == "419fcd276a638048e65682146868fc2ae30d7a91"
    )
    assert constraints == manifest["runtime_packages"]
    assert manifest["generation"] == {
        "thinking_enabled": False,
        "temperature": 0.0,
        "generator_max_tokens": 512,
        "verifier_max_tokens": 256,
        "evidence_token_budget": 12000,
        "token_limit_finish_is_execution_error": True,
    }
