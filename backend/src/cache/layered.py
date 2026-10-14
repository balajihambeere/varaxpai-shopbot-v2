"""Three-layer answer cache: exact-match -> semantic -> (caller falls
through to the full pipeline on a miss). See docs/requirements.md C-006 /
REQ-F-006.

Both cache layers record which catalog item ids a cached answer depended
on, so `src/cache/invalidation.py` can evict precisely on a catalog change
rather than relying on TTL alone (TTL is only the backstop)."""

import hashlib
import json
import time
from dataclasses import asdict, dataclass

from src.config import settings
from src.memory.session import get_redis
from src.vectorstore import qdrant_store


@dataclass
class CachedAnswer:
    answer: str
    retrieved_item_ids: list[str]
    created_at: float
    refused: bool = False


def _exact_key(query: str) -> str:
    h = hashlib.sha256(query.strip().lower().encode()).hexdigest()
    return f"cache:exact:{h}"


def by_item_set_key(item_id: str) -> str:
    return f"cache:by_item:{item_id}"


def get_exact(query: str) -> CachedAnswer | None:
    raw = get_redis().get(_exact_key(query))
    if not raw:
        return None
    return CachedAnswer(**json.loads(raw))


def put_exact(query: str, cached: CachedAnswer) -> None:
    r = get_redis()
    key = _exact_key(query)
    r.set(key, json.dumps(asdict(cached)), ex=settings.exact_cache_ttl_seconds)
    for item_id in cached.retrieved_item_ids:
        r.sadd(by_item_set_key(item_id), key)


def get_semantic(query_vec: list[float]) -> CachedAnswer | None:
    point = qdrant_store.cache_search(query_vec, score_threshold=settings.semantic_cache_threshold)
    if point is None:
        return None
    payload = point.payload
    age = time.time() - payload["created_at"]
    if age > settings.semantic_cache_ttl_seconds:
        return None  # TTL backstop, even if an invalidation event was missed
    return CachedAnswer(
        answer=payload["answer"],
        retrieved_item_ids=payload["retrieved_item_ids"],
        created_at=payload["created_at"],
        refused=payload.get("refused", False),  # old entries predating this field default to False
    )


def put_semantic(query: str, query_vec: list[float], cached: CachedAnswer) -> None:
    qdrant_store.cache_upsert(cache_key=query, query_vec=query_vec, payload=asdict(cached))


def put_both(query: str, query_vec: list[float], answer: str, retrieved_item_ids: list[str], refused: bool = False) -> None:
    cached = CachedAnswer(answer=answer, retrieved_item_ids=retrieved_item_ids, created_at=time.time(), refused=refused)
    put_exact(query, cached)
    put_semantic(query, query_vec, cached)
