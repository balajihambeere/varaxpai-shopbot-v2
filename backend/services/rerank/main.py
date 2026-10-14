"""Standalone rerank service. Runs independently of the main API process so
the most CPU-expensive step in the pipeline (cross-encoder scoring) can be
scaled on its own — see docs/requirements.md C-009 / REQ-N-005."""

import os

from fastapi import FastAPI
from fastembed.rerank.cross_encoder import TextCrossEncoder
from pydantic import BaseModel

MODEL_NAME = os.environ.get("RERANK_MODEL_NAME", "Xenova/ms-marco-MiniLM-L-6-v2")

app = FastAPI(title="rerank-service")
_encoder = TextCrossEncoder(model_name=MODEL_NAME)


class Candidate(BaseModel):
    id: str
    text: str


class RerankRequest(BaseModel):
    query: str
    candidates: list[Candidate]
    top_k: int = 3


class RerankResponse(BaseModel):
    ranked_ids: list[str]
    scores: list[float]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/rerank", response_model=RerankResponse)
def rerank(req: RerankRequest) -> RerankResponse:
    if not req.candidates:
        return RerankResponse(ranked_ids=[], scores=[])
    texts = [c.text for c in req.candidates]
    scores = list(_encoder.rerank(req.query, texts))
    ranked = sorted(zip(req.candidates, scores), key=lambda p: p[1], reverse=True)
    top = ranked[: req.top_k]
    return RerankResponse(ranked_ids=[c.id for c, _ in top], scores=[float(s) for _, s in top])
