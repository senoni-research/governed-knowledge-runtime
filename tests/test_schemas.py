from __future__ import annotations

from datetime import date

import pytest
from conftest import make_record

from gkr.schemas import Actor, KnowledgeRecord


def test_record_round_trip_preserves_temporal_and_authority_fields() -> None:
    record = make_record()

    restored = KnowledgeRecord.from_dict(record.to_dict())

    assert restored == record
    assert restored.reference == "POL-001:v1"
    assert restored.is_valid_at(date(2026, 1, 1))
    assert restored.is_permitted(Actor("alice", ("employees",)))
    assert not restored.is_permitted(Actor("mallory", ("contractors",)))


def test_non_public_record_requires_explicit_acl() -> None:
    with pytest.raises(ValueError, match="Non-public records"):
        make_record(acl=())


def test_public_record_is_visible_without_identity() -> None:
    record = make_record(sensitivity="public", acl=())

    assert record.is_permitted(Actor.anonymous())


def test_later_version_requires_valid_same_record_reference() -> None:
    with pytest.raises(ValueError, match="must name"):
        make_record(version=2)

    with pytest.raises(ValueError, match="same record_id"):
        make_record(version=2, supersedes="OTHER:v1")
