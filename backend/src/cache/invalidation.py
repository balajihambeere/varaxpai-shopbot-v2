"""Event-driven cache invalidation tied to catalog changes. See
docs/requirements.md C-006 / REQ-F-006, REQ-F-009.

`publish_catalog_update` is called by the catalog-update path
(`src/api/routes/catalog.py`) right after re-ingestion. `invalidate_now` does
the actual eviction and can be called either synchronously (as it is here,
for a small fixture catalog where a background worker would be
over-engineering — see framework §1.5) or from a subscriber loop in a larger
deployment."""

import json

from src.cache.layered import by_item_set_key
from src.memory.session import get_redis
from src.vectorstore import qdrant_store

CHANNEL = "catalog.updated"


def invalidate_now(affected_item_ids: list[str]) -> int:
    """Evict every cache entry (exact + semantic) that depended on any of
    `affected_item_ids`. Returns the number of exact-cache keys evicted."""
    r = get_redis()
    evicted = 0
    for item_id in affected_item_ids:
        set_key = by_item_set_key(item_id)
        keys = r.smembers(set_key)
        if keys:
            r.delete(*keys)
            evicted += len(keys)
        r.delete(set_key)
    qdrant_store.cache_delete_by_item_ids(affected_item_ids)
    return evicted


def publish_catalog_update(affected_item_ids: list[str]) -> int:
    """Announce the change on the pub/sub channel (for any future external
    subscriber) and invalidate synchronously in-process — a background
    worker subscribing to `CHANNEL` would be over-engineering at this
    catalog's scale (framework §1.5); this call is the invalidation path."""
    get_redis().publish(CHANNEL, json.dumps({"item_ids": affected_item_ids}))
    return invalidate_now(affected_item_ids)
