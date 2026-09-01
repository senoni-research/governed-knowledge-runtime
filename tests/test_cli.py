from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import make_record

from gkr.authority import AuthorityStore
from gkr.cli import main


def test_ingest_cli_returns_success_for_valid_chain(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "records.jsonl"
    source.write_text(json.dumps(make_record().to_dict()) + "\n", encoding="utf-8")
    database = tmp_path / "authority.sqlite"

    exit_code = main(["ingest", str(source), "--db", str(database)])
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output["chain_valid"] is True
    assert output["appended_records"] == 1


def test_ingest_cli_refuses_invalid_existing_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "records.jsonl"
    source.write_text(json.dumps(make_record().to_dict()) + "\n", encoding="utf-8")
    monkeypatch.setattr(AuthorityStore, "verify_chain", lambda _store: False)

    exit_code = main(["ingest", str(source), "--db", str(tmp_path / "authority.sqlite")])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "Refusing ingestion" in captured.err


def test_ask_cli_publishes_typed_policy_and_writes_trace(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = tmp_path / "authority.sqlite"
    trace_database = tmp_path / "traces.sqlite"
    with AuthorityStore(database) as store:
        store.append(
            make_record(
                statement="Travel spend above £750 requires written approval.",
                rules=(
                    {
                        "rule_id": "POL-001.approval-threshold",
                        "subject": "travel-spend",
                        "measure": "gross-amount",
                        "unit": "GBP",
                        "comparator": ">",
                        "threshold": "750",
                        "effect": "written-approval-required",
                        "conditions": [],
                        "exceptions": [],
                    },
                ),
            )
        )

    exit_code = main(
        [
            "ask",
            "Does £800 travel spend require approval?",
            "--db",
            str(database),
            "--trace-db",
            str(trace_database),
            "--actor",
            "alice",
            "--group",
            "employees",
            "--as-of",
            "2026-06-01",
            "--json",
        ]
    )
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output["answer_status"] == "published_deterministic_policy_rule"
    assert output["retrieval"]["mode"] == "policy_rule"
    assert output["trace"]["trace_id"]
