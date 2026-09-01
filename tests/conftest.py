from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from gkr.authority import AuthorityStore
from gkr.schemas import KnowledgeRecord, source_digest


@pytest.fixture
def store() -> AuthorityStore:
    authority = AuthorityStore(":memory:")
    yield authority
    authority.close()


def make_record(
    *,
    record_id: str = "POL-001",
    version: int = 1,
    statement: str = "Spend above £500 requires written approval.",
    valid_from: date = date(2026, 1, 1),
    observed_at: datetime = datetime(2026, 1, 1, tzinfo=UTC),
    supersedes: str | None = None,
    status: str = "approved",
    sensitivity: str = "internal",
    acl: tuple[str, ...] = ("group:employees",),
    relations: tuple[tuple[str, str, str], ...] = (),
    rules: tuple[dict[str, object], ...] = (),
) -> KnowledgeRecord:
    return KnowledgeRecord.from_dict(
        {
            "record_id": record_id,
            "version": version,
            "domain": "finance",
            "title": f"Policy {record_id}",
            "statement": statement,
            "valid_from": valid_from.isoformat(),
            "observed_at": observed_at.isoformat(),
            "supersedes": supersedes,
            "status": status,
            "owner": "Policy Owner",
            "source_uri": f"test://policy/{record_id}/v{version}",
            "source_hash": source_digest(statement.strip()),
            "sensitivity": sensitivity,
            "acl": list(acl),
            "aliases": ["approval threshold"],
            "relations": [list(relation) for relation in relations],
            "rules": list(rules),
            "metadata": {"hash_scope": "statement"},
        }
    )
