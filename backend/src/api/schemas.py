from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=128)
    query: str = Field(..., min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    answer: str
    refused: bool
    degraded: bool
    cache_hit: str | None  # "exact" | "semantic" | None
    evidence_item_ids: list[str]
    sub_queries: list[str]
    top_rerank_score: float | None
    estimated_cost: float
    request_id: str


class CatalogUpdateRequest(BaseModel):
    item: dict


class CatalogUpdateResponse(BaseModel):
    item_id: str
    affected_chunk_ids: list[str]
    cache_entries_evicted: int
