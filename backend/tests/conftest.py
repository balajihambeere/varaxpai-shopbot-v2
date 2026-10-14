import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def force_mock_llm_by_default():
    """The default test suite must stay free, fast, and deterministic
    (REQ-N-007) — it must NOT silently start making real, paid OpenAI calls
    just because a real OPENAI_API_KEY happens to be sitting in .env for
    production use. Force the mock provider for every test unless a human
    deliberately opts in with TEST_USE_REAL_LLM=1 (see
    docs/decisions/0011)."""
    if os.environ.get("TEST_USE_REAL_LLM") == "1":
        yield
        return

    from src.config import settings
    from src.providers.llm import reset_llm_provider_cache

    original_key = settings.openai_api_key
    settings.openai_api_key = ""
    reset_llm_provider_cache()
    yield
    settings.openai_api_key = original_key
    reset_llm_provider_cache()


def _services_reachable() -> bool:
    try:
        from src.memory.session import get_redis
        from src.vectorstore.qdrant_store import get_client

        get_client().get_collections()
        get_redis().ping()
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def live_stack():
    """Skips the test if Qdrant/Redis aren't reachable, e.g. in a CI job
    that hasn't run `docker compose up`. Integration tests depend on this.

    Also clears the answer cache (both layers) once per test session before
    anything else runs. Tests must not depend on cache state left over from
    a previous run or from manual exploration against the same local
    stack — without this, a repeat test run can spuriously hit a cache
    entry a *previous* run wrote for the same literal query text, which is
    exactly the kind of flaky, order-dependent test this project does not
    want to ship."""
    if not _services_reachable():
        pytest.skip("Qdrant/Redis not reachable — run `docker compose up -d qdrant redis rerank` first.")
    _clear_caches()
    return True


def _clear_caches() -> None:
    from qdrant_client import models

    from src.config import settings
    from src.memory.session import get_redis
    from src.vectorstore.qdrant_store import get_client

    r = get_redis()
    for key in r.scan_iter("cache:exact:*"):
        r.delete(key)
    for key in r.scan_iter("cache:by_item:*"):
        r.delete(key)

    client = get_client()
    existing = {c.name for c in client.get_collections().collections}
    if settings.answer_cache_collection in existing:
        client.delete(
            collection_name=settings.answer_cache_collection,
            points_selector=models.FilterSelector(filter=models.Filter()),  # empty filter = match all points
        )
