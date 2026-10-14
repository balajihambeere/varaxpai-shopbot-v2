import time
import uuid

from src.cache import layered as cache
from src.cache.invalidation import invalidate_now
from src.providers.embeddings import get_embedding_provider


def test_exact_cache_roundtrip(live_stack):
    query = f"unit-test exact cache query {uuid.uuid4()}"
    assert cache.get_exact(query) is None

    cached = cache.CachedAnswer(answer="hello world", retrieved_item_ids=["ztest-exact"], created_at=time.time())
    cache.put_exact(query, cached)

    hit = cache.get_exact(query)
    assert hit is not None
    assert hit.answer == "hello world"


def test_semantic_cache_hits_on_near_identical_query(live_stack):
    provider = get_embedding_provider()
    base_query = f"unit-test semantic cache query {uuid.uuid4()}"
    vec = provider.embed_dense([base_query])[0]
    cached = cache.CachedAnswer(answer="semantic hit answer", retrieved_item_ids=["ztest-semantic"], created_at=time.time())
    cache.put_semantic(base_query, vec, cached)

    # Query the exact same vector again — guaranteed similarity 1.0, well above threshold.
    hit = cache.get_semantic(vec)
    assert hit is not None
    assert hit.answer == "semantic hit answer"


def test_invalidation_evicts_exact_cache_entries_for_affected_item(live_stack):
    item_id = f"ztest-invalidate-{uuid.uuid4()}"
    query = f"unit-test invalidation query {uuid.uuid4()}"
    cached = cache.CachedAnswer(answer="to be evicted", retrieved_item_ids=[item_id], created_at=time.time())
    cache.put_exact(query, cached)
    assert cache.get_exact(query) is not None

    evicted = invalidate_now([item_id])
    assert evicted == 1
    assert cache.get_exact(query) is None


def test_invalidation_of_unrelated_item_does_not_evict(live_stack):
    item_id = f"ztest-keep-{uuid.uuid4()}"
    other_item_id = f"ztest-other-{uuid.uuid4()}"
    query = f"unit-test keep query {uuid.uuid4()}"
    cached = cache.CachedAnswer(answer="should survive", retrieved_item_ids=[item_id], created_at=time.time())
    cache.put_exact(query, cached)

    invalidate_now([other_item_id])
    assert cache.get_exact(query) is not None
