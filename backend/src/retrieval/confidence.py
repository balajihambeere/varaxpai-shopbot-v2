"""Confidence gate: refuse rather than guess when the reranker itself is not
confident. See docs/requirements.md C-003 / REQ-F-003.

Deliberately gates on the reranker's own score (a single consistent scale
across all candidates for a given query), not on the raw retrieval
similarity — raw dense/sparse scores from two different signals are not
comparable to each other, let alone to a single global threshold."""

from src.config import settings
from src.domain.models import ScoredChunk


def passes_confidence_floor(reranked: list[ScoredChunk]) -> bool:
    if not reranked:
        return False
    top_score = reranked[0].score
    return top_score > settings.confidence_floor


def top_score(reranked: list[ScoredChunk]) -> float | None:
    return reranked[0].score if reranked else None
