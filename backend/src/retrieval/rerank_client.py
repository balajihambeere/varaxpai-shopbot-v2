import httpx

from src.config import settings
from src.domain.models import ScoredChunk


class RerankUnavailable(Exception):
    """Raised when the rerank service times out or errors. The caller must
    degrade to a safe refusal, never hang or surface a 500 to the client."""


def rerank(query: str, candidates: list[ScoredChunk], top_k: int | None = None) -> list[ScoredChunk]:
    if not candidates:
        return []
    top_k = top_k or settings.rerank_top_k
    payload = {
        "query": query,
        "candidates": [{"id": sc.chunk.id, "text": sc.chunk.text} for sc in candidates],
        "top_k": top_k,
    }
    try:
        resp = httpx.post(settings.rerank_url, json=payload, timeout=settings.rerank_timeout_seconds)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise RerankUnavailable(str(exc)) from exc

    body = resp.json()
    by_id = {sc.chunk.id: sc.chunk for sc in candidates}
    return [ScoredChunk(chunk=by_id[cid], score=score) for cid, score in zip(body["ranked_ids"], body["scores"])]
