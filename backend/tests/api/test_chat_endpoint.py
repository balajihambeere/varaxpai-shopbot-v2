import uuid

import pytest
from fastapi.testclient import TestClient

from products import PRODUCTS
from src.api.main import app
from src.ingestion.ingest import ingest_catalog

client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def ingested_catalog(live_stack):
    ingest_catalog(PRODUCTS)


def _chat(session_id: str, query: str) -> dict:
    resp = client.post("/v1/chat", json={"session_id": session_id, "query": query})
    assert resp.status_code == 200
    return resp.json()


def test_literal_code_query_returns_grounded_non_refused_answer(live_stack):
    body = _chat(f"s-{uuid.uuid4()}", "p001 in mint")
    assert body["refused"] is False
    assert body["degraded"] is False
    assert "p001" in body["evidence_item_ids"]
    assert body["answer"] != ""


def test_off_topic_query_refuses_rather_than_fabricating(live_stack):
    body = _chat(f"s-{uuid.uuid4()}", "totally unrelated question about jet engine turbine maintenance")
    assert body["refused"] is True
    assert body["evidence_item_ids"] == []


def test_identical_repeat_query_is_an_exact_cache_hit(live_stack):
    session_id = f"s-{uuid.uuid4()}"
    unique_query = f"cotton kurta p001 cache test {uuid.uuid4()}"
    first = _chat(session_id, unique_query)
    assert first["cache_hit"] is None
    second = _chat(session_id, unique_query)
    assert second["cache_hit"] == "exact"
    assert second["answer"] == first["answer"]


def test_a_cached_refusal_stays_refused_on_repeat(live_stack):
    """Regression test for docs/decisions/0010."""
    session_id = f"s-{uuid.uuid4()}"
    unique_query = f"totally unrelated jet engine turbine maintenance query {uuid.uuid4()}"
    first = _chat(session_id, unique_query)
    assert first["refused"] is True

    second = _chat(session_id, unique_query)
    assert second["cache_hit"] is not None
    assert second["refused"] is True


def test_compound_query_covers_both_intents(live_stack):
    # A unique query text avoids colliding with another test's cached entry
    # for the same literal string (cache lookups happen before decomposition).
    unique_suffix = uuid.uuid4()
    body = _chat(f"s-{uuid.uuid4()}", f"something warm and something festive {unique_suffix}")
    assert len(body["sub_queries"]) == 2
    assert len(set(body["evidence_item_ids"])) >= 1  # at least some evidence returned


def test_rejects_empty_query():
    resp = client.post("/v1/chat", json={"session_id": "s", "query": ""})
    assert resp.status_code == 422


def test_session_rate_limit_returns_429_on_burst(live_stack, monkeypatch):
    from src.api import rate_limit

    monkeypatch.setattr(rate_limit, "MAX_REQUESTS_PER_WINDOW", 3)
    session_id = f"s-ratelimit-{uuid.uuid4()}"
    statuses = []
    for _ in range(5):
        resp = client.post("/v1/chat", json={"session_id": session_id, "query": "p001 overview"})
        statuses.append(resp.status_code)
    assert 429 in statuses
