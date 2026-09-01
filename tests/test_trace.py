from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime
from pathlib import Path
from threading import Barrier

from conftest import make_record

from gkr.authority import AuthorityStore
from gkr.runtime import GovernedKnowledgeRuntime
from gkr.schemas import Actor
from gkr.trace import TraceStore


def test_query_trace_reconstructs_request_evidence_and_decision(
    store: AuthorityStore,
) -> None:
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
                    "effect": "written-approval-required-before-booking",
                    "conditions": [],
                    "exceptions": [],
                },
            ),
        )
    )
    actor = Actor("alice", ("employees",))
    known_at = datetime(2026, 6, 1, 12, tzinfo=UTC)

    with TraceStore(":memory:") as traces:
        runtime = GovernedKnowledgeRuntime(store, trace_store=traces)
        result = runtime.ask(
            "Does £800 travel spend require approval?",
            actor=actor,
            as_of=date(2026, 6, 1),
            known_at=known_at,
        )
        assert result.trace is not None
        recovered = traces.get(result.trace.trace_id)

        assert traces.count() == 1
        assert traces.verify_chain()

    assert recovered is not None
    assert recovered.to_dict() == result.trace.to_dict()
    assert recovered.question == "Does £800 travel spend require approval?"
    assert recovered.known_at == known_at
    assert recovered.authority_snapshot_id == result.evidence.authority_snapshot_id
    assert recovered.evidence_bundle_id == result.evidence.evidence_bundle_id
    assert recovered.evidence_references == ("POL-001:v1",)
    assert recovered.publication_status == "published_deterministic_policy_rule"
    assert recovered.decision_parse["status"] == "supported"
    assert recovered.resolved_principals == (
        "authenticated",
        "group:employees",
        "user:alice",
    )
    assert "alice" not in result.evidence.prompt


def test_concurrent_trace_writers_preserve_one_hash_chain(
    store: AuthorityStore,
    tmp_path: Path,
) -> None:
    store.append(make_record())
    runtime = GovernedKnowledgeRuntime(store)
    actor = Actor("alice", ("employees",))
    generated_traces = []
    for index in range(4):
        result = runtime.ask(
            f"What policy applies to request {index}?",
            actor=actor,
            as_of=date(2026, 6, 1),
        )
        assert result.trace is not None
        generated_traces.append(result.trace)

    trace_database = tmp_path / "traces.sqlite"
    barrier = Barrier(len(generated_traces))
    with TraceStore(trace_database):
        pass

    def append_trace(index: int) -> None:
        with TraceStore(trace_database) as traces:
            barrier.wait(timeout=10)
            traces.append(generated_traces[index])

    with ThreadPoolExecutor(max_workers=len(generated_traces)) as executor:
        list(executor.map(append_trace, range(len(generated_traces))))

    with TraceStore(trace_database) as verified:
        assert verified.count() == len(generated_traces)
        assert verified.verify_chain()
