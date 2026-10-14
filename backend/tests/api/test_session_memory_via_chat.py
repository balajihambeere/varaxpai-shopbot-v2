import uuid

import pytest
from fastapi.testclient import TestClient

from products import PRODUCTS
from src.api.main import app
from src.ingestion.ingest import ingest_catalog
from src.memory.session import load_session

client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def ingested_catalog(live_stack):
    ingest_catalog(PRODUCTS)


def test_cache_hit_turn_still_updates_active_items(live_stack):
    """Regression test for docs/decisions/0008."""
    session_id = f"s-{uuid.uuid4()}"
    unique_query = f"p001 cache-then-memory check {uuid.uuid4()}"

    first = client.post("/v1/chat", json={"session_id": session_id, "query": unique_query}).json()
    assert first["cache_hit"] is None
    assert "p001" in first["evidence_item_ids"]

    second = client.post("/v1/chat", json={"session_id": session_id, "query": unique_query}).json()
    assert second["cache_hit"] == "exact"

    session = load_session(session_id)
    assert "p001" in session.active_item_ids
    assert len(session.recent_turns) == 4  # two full turns recorded, including the cache-hit one
