"""Wires together decomposition, hybrid retrieval, reranking, the
active-item boost, and the confidence gate into one call. See
docs/architecture.md for the full request-flow diagram."""

from dataclasses import dataclass

from src.config import settings
from src.domain.models import ScoredChunk
from src.memory.session import Session
from src.providers.llm import LLMProvider
from src.retrieval.confidence import passes_confidence_floor
from src.retrieval.hybrid import reciprocal_rank_fusion, retrieve_hybrid
from src.retrieval.rerank_client import RerankUnavailable
from src.retrieval.rerank_client import rerank as rerank_candidates


@dataclass
class PipelineResult:
    sub_queries: list[str]
    final_chunks: list[ScoredChunk]
    top_score: float | None
    degraded: bool  # rerank service was unreachable


def apply_active_item_boost(ranked: list[ScoredChunk], session: Session | None) -> list[ScoredChunk]:
    if not session or not session.active_item_ids:
        return ranked
    boosted = []
    for sc in ranked:
        bonus = settings.active_entity_boost if sc.chunk.item_id in session.active_item_ids else 0.0
        boosted.append(ScoredChunk(chunk=sc.chunk, score=sc.score + bonus))
    boosted.sort(key=lambda sc: sc.score, reverse=True)
    return boosted


def run_retrieval(query: str, llm: LLMProvider, session: Session | None = None) -> PipelineResult:
    sub_queries = llm.decompose(query)

    per_subquery_reranked: list[list[ScoredChunk]] = []
    confident_subquery_results: list[list[ScoredChunk]] = []
    degraded = False
    for sub_query in sub_queries:
        fused = retrieve_hybrid(sub_query)
        try:
            # Fetch more than the final top_k per sub-query so a later
            # cross-sub-query merge still has real choices to rank.
            reranked = rerank_candidates(sub_query, fused, top_k=max(settings.rerank_top_k, 5))
        except RerankUnavailable:
            degraded = True
            reranked = []
        per_subquery_reranked.append(reranked)

        # Confidence is gated per sub-query, on that sub-query's own
        # cross-encoder scale, BEFORE any cross-sub-query merge — merging
        # first would mix cross-encoder scores with RRF rank-scores on an
        # incomparable scale (see docs/decisions/0003 for why this is an
        # engineering decision beyond what the source material specified
        # for the single-intent case). A sub-query whose own best evidence
        # is weak contributes nothing rather than dragging in noise.
        if passes_confidence_floor(reranked):
            confident_subquery_results.append(reranked)

    if degraded:
        return PipelineResult(sub_queries=sub_queries, final_chunks=[], top_score=None, degraded=True)

    if not confident_subquery_results:
        return PipelineResult(sub_queries=sub_queries, final_chunks=[], top_score=None, degraded=False)

    if len(confident_subquery_results) > 1:
        merged = reciprocal_rank_fusion(*confident_subquery_results, k=settings.rrf_k)
    else:
        merged = confident_subquery_results[0]

    boosted = apply_active_item_boost(merged, session)
    final_chunks = boosted[: settings.rerank_top_k]
    top_score = final_chunks[0].score if final_chunks else None
    return PipelineResult(sub_queries=sub_queries, final_chunks=final_chunks, top_score=top_score, degraded=False)
