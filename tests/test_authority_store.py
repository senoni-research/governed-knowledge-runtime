from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime
from pathlib import Path
from threading import Barrier

import pytest
from conftest import make_record

from gkr.authority import AuthorityStore
from gkr.schemas import Actor


def test_temporal_resolution_uses_valid_time_and_observed_time(
    store: AuthorityStore,
) -> None:
    first = make_record()
    second = make_record(
        version=2,
        statement="Spend above £750 requires written approval.",
        valid_from=date(2026, 9, 1),
        observed_at=datetime(2026, 8, 25, 14, 31, tzinfo=UTC),
        supersedes=first.reference,
    )
    store.append_many((first, second))
    actor = Actor("alice", ("employees",))

    before_change = store.current_records(
        actor=actor,
        as_of=date(2026, 8, 31),
    )
    after_change = store.current_records(
        actor=actor,
        as_of=date(2026, 9, 1),
    )
    historically_known = store.current_records(
        actor=actor,
        as_of=date(2026, 9, 1),
        known_at=datetime(2026, 8, 1, tzinfo=UTC),
    )

    assert [record.reference for record in before_change.records] == ["POL-001:v1"]
    assert [record.reference for record in after_change.records] == ["POL-001:v2"]
    assert [record.reference for record in historically_known.records] == ["POL-001:v1"]


def test_known_at_uses_datetime_ordering_across_fractional_second_boundary(
    store: AuthorityStore,
) -> None:
    observed_at = datetime(2026, 6, 1, 12, 0, 0, 500_000, tzinfo=UTC)
    store.append(make_record(observed_at=observed_at))
    actor = Actor("alice", ("employees",))

    just_before = store.current_records(
        actor=actor,
        as_of=date(2026, 6, 1),
        known_at=datetime(2026, 6, 1, 12, 0, 0, tzinfo=UTC),
    )
    exact = store.current_records(
        actor=actor,
        as_of=date(2026, 6, 1),
        known_at=observed_at,
    )

    assert just_before.records == ()
    assert [record.reference for record in exact.records] == ["POL-001:v1"]


def test_approved_retirement_removes_current_record_but_preserves_history(
    store: AuthorityStore,
) -> None:
    approved = make_record()
    retirement = make_record(
        version=2,
        statement="This policy is retired.",
        valid_from=date(2026, 7, 1),
        observed_at=datetime(2026, 6, 20, tzinfo=UTC),
        supersedes=approved.reference,
        status="retired",
    )
    store.append_many((approved, retirement))
    actor = Actor("alice", ("employees",))

    before_retirement = store.current_records(
        actor=actor,
        as_of=date(2026, 6, 30),
    )
    after_retirement = store.current_records(
        actor=actor,
        as_of=date(2026, 7, 1),
    )

    assert [record.reference for record in before_retirement.records] == ["POL-001:v1"]
    assert after_retirement.records == ()
    assert store.get("POL-001:v1") == approved
    assert store.get("POL-001:v2") == retirement


def test_retroactive_correction_depends_on_as_of_and_known_at(
    store: AuthorityStore,
) -> None:
    original = make_record()
    correction = make_record(
        version=2,
        statement="Corrected threshold is £450.",
        valid_from=date(2026, 3, 1),
        observed_at=datetime(2026, 6, 1, tzinfo=UTC),
        supersedes=original.reference,
    )
    store.append_many((original, correction))
    actor = Actor("alice", ("employees",))

    before_correction_was_known = store.current_records(
        actor=actor,
        as_of=date(2026, 4, 1),
        known_at=datetime(2026, 5, 1, tzinfo=UTC),
    )
    after_correction_was_known = store.current_records(
        actor=actor,
        as_of=date(2026, 4, 1),
        known_at=datetime(2026, 7, 1, tzinfo=UTC),
    )
    before_business_effect = store.current_records(
        actor=actor,
        as_of=date(2026, 2, 1),
        known_at=datetime(2026, 7, 1, tzinfo=UTC),
    )

    assert [record.reference for record in before_correction_was_known.records] == [
        "POL-001:v1"
    ]
    assert [record.reference for record in after_correction_was_known.records] == [
        "POL-001:v2"
    ]
    assert [record.reference for record in before_business_effect.records] == ["POL-001:v1"]


def test_authorization_happens_after_version_resolution(store: AuthorityStore) -> None:
    first = make_record()
    restricted_update = make_record(
        version=2,
        statement="The policy is now restricted to Finance.",
        valid_from=date(2026, 7, 1),
        observed_at=datetime(2026, 6, 20, tzinfo=UTC),
        supersedes=first.reference,
        sensitivity="restricted",
        acl=("group:finance",),
    )
    store.append_many((first, restricted_update))

    employee_view = store.current_records(
        actor=Actor("alice", ("employees",)),
        as_of=date(2026, 7, 1),
    )
    finance_view = store.current_records(
        actor=Actor("bob", ("finance",)),
        as_of=date(2026, 7, 1),
    )

    assert employee_view.records == ()
    assert [record.reference for record in finance_view.records] == ["POL-001:v2"]


def test_ledger_is_hash_chained_and_database_enforces_append_only(
    store: AuthorityStore,
) -> None:
    store.append(make_record())

    assert store.verify_chain()
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        store._connection.execute(  # noqa: SLF001 - verifies the database boundary itself
            "UPDATE knowledge_ledger SET valid_from = '2030-01-01'"
        )


def test_append_many_rolls_back_as_one_transaction(store: AuthorityStore) -> None:
    first = make_record()
    invalid_sequence = make_record(
        record_id="OTHER",
        version=2,
        supersedes="OTHER:v1",
    )

    with pytest.raises(ValueError, match="first stored version"):
        store.append_many((first, invalid_sequence))

    assert store.count() == 0


def test_concurrent_writers_preserve_one_hash_chain(tmp_path: Path) -> None:
    database = tmp_path / "authority.sqlite"
    writers = 4
    barrier = Barrier(writers)
    with AuthorityStore(database):
        pass

    def append_record(index: int) -> None:
        with AuthorityStore(database) as concurrent_store:
            barrier.wait(timeout=10)
            concurrent_store.append(make_record(record_id=f"POL-{index:03d}"))

    with ThreadPoolExecutor(max_workers=writers) as executor:
        list(executor.map(append_record, range(writers)))

    with AuthorityStore(database) as verified:
        assert verified.count() == writers
        assert verified.verify_chain()
