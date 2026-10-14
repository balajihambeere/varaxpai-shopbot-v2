from dataclasses import dataclass, field


@dataclass
class Chunk:
    """One retrievable unit of catalog content."""

    id: str
    item_id: str
    chunk_type: str  # e.g. "overview", "sizing", "care", "faq"
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float


@dataclass
class RetrievalTrace:
    """Everything needed to audit how a final set of chunks was produced."""

    sub_queries: list[str]
    fused_top_n: list[ScoredChunk]
    reranked_top_k: list[ScoredChunk]
    top_rerank_score: float | None
