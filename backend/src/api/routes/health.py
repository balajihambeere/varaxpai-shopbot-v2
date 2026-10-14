import httpx
from fastapi import APIRouter

from src.config import settings
from src.memory.session import get_redis
from src.vectorstore.qdrant_store import get_client

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/ready")
def ready():
    checks = {}

    try:
        get_client().get_collections()
        checks["qdrant"] = "ok"
    except Exception as exc:  # noqa: BLE001 — readiness probe, report any failure
        checks["qdrant"] = f"error: {exc}"

    try:
        get_redis().ping()
        checks["redis"] = "ok"
    except Exception as exc:  # noqa: BLE001
        checks["redis"] = f"error: {exc}"

    try:
        # Swap only the path, not a naive string-replace on the whole URL —
        # a naive `.replace("/rerank", "/health")` silently corrupts this
        # when the hostname itself is "rerank" (exactly the case inside
        # docker-compose, where RERANK_URL is http://rerank:8001/rerank).
        from urllib.parse import urlsplit, urlunsplit

        parts = urlsplit(settings.rerank_url)
        health_url = urlunsplit((parts.scheme, parts.netloc, "/health", "", ""))
        resp = httpx.get(health_url, timeout=2.0)
        checks["rerank_service"] = "ok" if resp.status_code == 200 else f"status {resp.status_code}"
    except Exception as exc:  # noqa: BLE001
        checks["rerank_service"] = f"error: {exc}"

    all_ok = all(v == "ok" for v in checks.values())
    return {"status": "ok" if all_ok else "degraded", "checks": checks}
