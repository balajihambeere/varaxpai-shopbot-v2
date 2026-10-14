"""Hybrid retrieval: dense (semantic) + sparse (lexical) search, fused by
Reciprocal Rank Fusion. See docs/requirements.md C-001 / REQ-F-001.

RRF is implemented as a pure function (`reciprocal_rank_fusion`) precisely so
it can be unit-tested on synthetic ranked lists without touching Qdrant —
the fusion math is the thing most likely to be silently wrong."""

from collections import defaultdict

from src.config import settings
from src.domain.models import ScoredChunk
from src.providers.embeddings import get_embedding_provider
from src.vectorstore import qdrant_store


def reciprocal_rank_fusion(*ranked_lists: list[ScoredChunk], k: int) -> list[ScoredChunk]:
    """Fuse multiple ranked lists into one, by summed reciprocal rank.
    A chunk's raw score from either source list is discarded; only its
    *rank* within each list matters, which is what makes this safe to apply
    across two retrievers whose raw scores are on incomparable scales."""
    fused_scores: dict[str, float] = defaultdict(float)
    chunk_by_id = {}
    for ranked in ranked_lists:
        for rank, scored in enumerate(ranked, start=1):
            fused_scores[scored.chunk.id] += 1.0 / (k + rank)
            chunk_by_id[scored.chunk.id] = scored.chunk

    ordered = sorted(fused_scores.items(), key=lambda kv: kv[1], reverse=True)
    return [ScoredChunk(chunk=chunk_by_id[cid], score=score) for cid, score in ordered]


def retrieve_hybrid(query: str, top_n: int | None = None) -> list[ScoredChunk]:
    """Run dense and sparse retrieval against the catalog collection, fuse
    with RRF, and return the top_n fused candidates (default: HYBRID_FETCH_K)."""
    top_n = top_n or settings.hybrid_fetch_k
    provider = get_embedding_provider()

    dense_vec = provider.embed_dense([query])[0]
    sparse_vec = provider.embed_sparse([query])[0]

    dense_hits = qdrant_store.dense_search(dense_vec, limit=settings.hybrid_fetch_k)
    sparse_hits = qdrant_store.sparse_search(sparse_vec, limit=settings.hybrid_fetch_k)

    fused = reciprocal_rank_fusion(dense_hits, sparse_hits, k=settings.rrf_k)
    return fused[:top_n]
