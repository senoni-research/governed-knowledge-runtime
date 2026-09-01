from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime
from pathlib import Path
from threading import Barrier

import pytest
from conftest import make_record

from gkr.authority import AuthorityStore
from gkr.proposals import ChangeProposal, ProposalStore
from gkr.schemas import Actor, KnowledgeRecord


def test_proposal_never_shadows_current_authority(store: AuthorityStore) -> None:
    approved = make_record()
    store.append(approved)
    proposal = ChangeProposal(
        proposal_id="proposal-001",
        record_id=approved.record_id,
        base_reference=approved.reference,
        candidate={"statement": "Spend above £750 requires written approval."},
        proposed_by="extractor",
        proposed_at=datetime(2026, 6, 1, tzinfo=UTC),
    )

    with ProposalStore(":memory:") as proposals:
        proposals.submit(proposal)
        pending_view = store.current_records(
            actor=Actor("alice", ("employees",)),
            as_of=date(2026, 7, 1),
        )
        proposals.decide(
            proposal.proposal_id,
            decision="approved",
            reviewed_by="finance-owner",
            reviewed_at=datetime(2026, 6, 2, tzinfo=UTC),
        )
        reviewed_view = store.current_records(
            actor=Actor("alice", ("employees",)),
            as_of=date(2026, 7, 1),
        )

        assert proposals.state(proposal.proposal_id) == "approved"
        assert proposals.verify_chain()

    assert [record.reference for record in pending_view.records] == ["POL-001:v1"]
    assert [record.reference for record in reviewed_view.records] == ["POL-001:v1"]


def test_proposal_candidate_cannot_claim_authority_fields() -> None:
    proposal = ChangeProposal(
        proposal_id="proposal-002",
        record_id="POL-001",
        candidate={"statement": "candidate", "version": 2},
        proposed_by="extractor",
        proposed_at=datetime(2026, 6, 1, tzinfo=UTC),
    )

    with pytest.raises(ValueError, match="authoritative fields"):
        proposal.validate()


def test_authority_schema_rejects_workflow_status() -> None:
    value = make_record().to_dict()
    value["status"] = "proposed"

    with pytest.raises(ValueError, match="Unsupported status"):
        KnowledgeRecord.from_dict(value)


def test_rejected_proposal_does_not_shadow_current_authority(
    store: AuthorityStore,
) -> None:
    approved = make_record()
    store.append(approved)
    proposal = _proposal("proposal-rejected", approved.reference)

    with ProposalStore(":memory:") as proposals:
        proposals.submit(proposal)
        proposals.decide(
            proposal.proposal_id,
            decision="rejected",
            reviewed_by="finance-owner",
            reviewed_at=datetime(2026, 6, 2, tzinfo=UTC),
        )
        assert proposals.state(proposal.proposal_id) == "rejected"

    current = store.current_records(
        actor=Actor("alice", ("employees",)),
        as_of=date(2026, 7, 1),
    )
    assert [record.reference for record in current.records] == ["POL-001:v1"]


def test_concurrent_proposals_do_not_change_authority(
    store: AuthorityStore,
    tmp_path: Path,
) -> None:
    approved = make_record()
    store.append(approved)
    database = tmp_path / "proposals.sqlite"
    with ProposalStore(database):
        pass
    barrier = Barrier(2)

    def submit(proposal_id: str) -> None:
        with ProposalStore(database) as proposals:
            barrier.wait(timeout=10)
            proposals.submit(_proposal(proposal_id, approved.reference))

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(submit, ("proposal-a", "proposal-b")))

    with ProposalStore(database) as proposals:
        assert proposals.state("proposal-a") == "pending"
        assert proposals.state("proposal-b") == "pending"
        assert proposals.verify_chain()

    current = store.current_records(
        actor=Actor("alice", ("employees",)),
        as_of=date(2026, 7, 1),
    )
    assert [record.reference for record in current.records] == ["POL-001:v1"]


def _proposal(proposal_id: str, base_reference: str) -> ChangeProposal:
    return ChangeProposal(
        proposal_id=proposal_id,
        record_id="POL-001",
        base_reference=base_reference,
        candidate={"statement": "Spend above £750 requires written approval."},
        proposed_by="extractor",
        proposed_at=datetime(2026, 6, 1, tzinfo=UTC),
    )
