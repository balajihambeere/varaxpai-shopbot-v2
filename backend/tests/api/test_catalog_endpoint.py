import uuid

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.cache import layered as cache
from src.config import settings

client = TestClient(app)

TEST_ITEM = {
    "id": "ztest-catalog-item",
    "name": "Disposable Test Item",
    "description": "A throwaway fixture item used only by automated tests.",
    "category": "test",
    "fabric": "n/a",
    "occasion": "test",
    "colors": ["red"],
    "sizes": ["one size"],
    "price": 1,
    "stock_status": "in stock",
    "return_policy": "n/a",
    "exchange_policy": "n/a",
    "care_instructions": "n/a",
    "faqs": [],
}


def test_update_without_admin_token_is_rejected():
    resp = client.post("/v1/catalog/items", json={"item": TEST_ITEM})
    assert resp.status_code == 401


def test_update_with_wrong_admin_token_is_rejected():
    resp = client.post(
        "/v1/catalog/items", json={"item": TEST_ITEM}, headers={"X-Admin-Token": "wrong-token"}
    )
    assert resp.status_code == 401


def test_update_missing_id_field_is_rejected():
    resp = client.post(
        "/v1/catalog/items",
        json={"item": {"name": "no id"}},
        headers={"X-Admin-Token": settings.catalog_admin_token},
    )
    assert resp.status_code == 400


def test_valid_update_ingests_and_evicts_affected_cache_entries(live_stack):
    item_id = TEST_ITEM["id"]
    query = f"disposable test item query {uuid.uuid4()}"
    cached = cache.CachedAnswer(answer="stale", retrieved_item_ids=[item_id], created_at=0.0)
    cache.put_exact(query, cached)
    assert cache.get_exact(query) is not None

    resp = client.post(
        "/v1/catalog/items", json={"item": TEST_ITEM}, headers={"X-Admin-Token": settings.catalog_admin_token}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["item_id"] == item_id
    assert set(body["affected_chunk_ids"]) == {f"{item_id}:overview", f"{item_id}:care"}
    assert body["cache_entries_evicted"] >= 1

    assert cache.get_exact(query) is None
