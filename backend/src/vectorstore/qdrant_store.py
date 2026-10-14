"""Thin wrapper around the Qdrant collections this project uses.

Two collections:
  - `catalog_collection`: one point per chunk, two named vectors (`dense`,
    `sparse`), used by hybrid retrieval.
  - `answer_cache_collection`: one point per cached answer, a single dense
    vector (the query embedding), used by the semantic cache layer. Kept
    separate from the catalog collection so cache invalidation can filter by
    source chunk IDs independently of catalog search.
"""

from qdrant_client import QdrantClient, models

from src.config import settings
from src.domain.models import Chunk, ScoredChunk
from src.providers.embeddings import SparseVector

_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)
    return _client


def ensure_collections(dense_dim: int) -> None:
    client = get_client()
    existing = {c.name for c in client.get_collections().collections}

    if settings.catalog_collection not in existing:
        client.create_collection(
            collection_name=settings.catalog_collection,
            vectors_config={"dense": models.VectorParams(size=dense_dim, distance=models.Distance.COSINE)},
            sparse_vectors_config={"sparse": models.SparseVectorParams()},
        )

    if settings.answer_cache_collection not in existing:
        client.create_collection(
            collection_name=settings.answer_cache_collection,
            vectors_config=models.VectorParams(size=dense_dim, distance=models.Distance.COSINE),
        )


def upsert_chunks(chunks: list[Chunk], dense_vecs: list[list[float]], sparse_vecs: list[SparseVector]) -> None:
    client = get_client()
    points = []
    for chunk, dense, sparse in zip(chunks, dense_vecs, sparse_vecs):
        points.append(
            models.PointStruct(
                id=_stable_point_id(chunk.id),
                vector={
                    "dense": dense,
                    "sparse": models.SparseVector(indices=sparse.indices, values=sparse.values),
                },
                payload={"chunk_id": chunk.id, "item_id": chunk.item_id, "chunk_type": chunk.chunk_type, "text": chunk.text, **chunk.metadata},
            )
        )
    client.upsert(collection_name=settings.catalog_collection, points=points)


def _stable_point_id(chunk_id: str) -> str:
    import hashlib

    return hashlib.sha256(chunk_id.encode()).hexdigest()[:32]


def dense_search(query_vec: list[float], limit: int) -> list[ScoredChunk]:
    client = get_client()
    result = client.query_points(
        collection_name=settings.catalog_collection,
        query=query_vec,
        using="dense",
        limit=limit,
        with_payload=True,
    )
    return [_point_to_scored_chunk(p) for p in result.points]


def sparse_search(sparse_vec: SparseVector, limit: int) -> list[ScoredChunk]:
    client = get_client()
    result = client.query_points(
        collection_name=settings.catalog_collection,
        query=models.SparseVector(indices=sparse_vec.indices, values=sparse_vec.values),
        using="sparse",
        limit=limit,
        with_payload=True,
    )
    return [_point_to_scored_chunk(p) for p in result.points]


def _point_to_scored_chunk(point) -> ScoredChunk:
    payload = point.payload or {}
    metadata = {k: v for k, v in payload.items() if k not in ("chunk_id", "item_id", "chunk_type", "text")}
    chunk = Chunk(
        id=payload["chunk_id"],
        item_id=payload["item_id"],
        chunk_type=payload["chunk_type"],
        text=payload["text"],
        metadata=metadata,
    )
    return ScoredChunk(chunk=chunk, score=point.score)


# --- Semantic answer cache -------------------------------------------------


def cache_upsert(cache_key: str, query_vec: list[float], payload: dict) -> None:
    client = get_client()
    client.upsert(
        collection_name=settings.answer_cache_collection,
        points=[models.PointStruct(id=_stable_point_id(cache_key), vector=query_vec, payload=payload)],
    )


def cache_search(query_vec: list[float], score_threshold: float):
    client = get_client()
    result = client.query_points(
        collection_name=settings.answer_cache_collection,
        query=query_vec,
        limit=1,
        score_threshold=score_threshold,
        with_payload=True,
    )
    return result.points[0] if result.points else None


def cache_delete_by_item_ids(item_ids: list[str]) -> None:
    if not item_ids:
        return
    client = get_client()
    client.delete(
        collection_name=settings.answer_cache_collection,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[models.FieldCondition(key="retrieved_item_ids", match=models.MatchAny(any=item_ids))]
            )
        ),
    )
