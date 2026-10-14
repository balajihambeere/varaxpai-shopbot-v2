import logging

from fastapi import APIRouter, Request

from src.api.rate_limit import enforce_session_rate_limit
from src.api.schemas import ChatRequest, ChatResponse
from src.cache import layered as cache
from src.cost.estimator import CostAccumulator
from src.logging_setup import log_extra
from src.memory.session import load_session, maybe_refresh_summary, record_turn, save_session
from src.providers.embeddings import get_embedding_provider
from src.providers.llm import REFUSAL_TEXT, EvidenceItem, get_llm_provider
from src.retrieval.pipeline import run_retrieval

router = APIRouter()
logger = logging.getLogger("api.chat")


@router.post("/v1/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request) -> ChatResponse:
    request_id = getattr(request.state, "request_id", "-")
    enforce_session_rate_limit(req.session_id)
    cost = CostAccumulator()
    embedding_provider = get_embedding_provider()
    llm = get_llm_provider()

    query_vec = embedding_provider.embed_dense([req.query])[0]
    cost.add_embed_call()

    # Session state must be updated on every turn, including a cache hit —
    # a turn served from cache is still a real turn in the conversation,
    # and skipping it here would silently break active-item tracking for
    # exactly the queries that are cheap to answer. See docs/decisions/0008.
    session = load_session(req.session_id)

    exact_hit = cache.get_exact(req.query)
    semantic_hit = None if exact_hit else cache.get_semantic(query_vec)
    cache_hit = exact_hit or semantic_hit

    if cache_hit:
        layer = "exact" if exact_hit else "semantic"
        log_extra(logger, logging.INFO, "cache hit", cache_layer=layer, session_id=req.session_id)
        record_turn(session, req.query, cache_hit.answer, cache_hit.retrieved_item_ids)
        maybe_refresh_summary(session, lambda turns: _summarize(llm, turns, cost))
        save_session(session)
        return _response(cache_hit.answer, cache_hit.refused, False, layer, cache_hit.retrieved_item_ids, [], None, cost, request_id)

    cost.add_llm_call("decompose")
    result = run_retrieval(req.query, llm, session)
    cost.add_rerank_call()

    if result.degraded:
        log_extra(logger, logging.WARNING, "pipeline degraded, refusing safely", session_id=req.session_id)
        return _response(REFUSAL_TEXT, True, True, None, [], result.sub_queries, None, cost, request_id)

    if not result.final_chunks:
        answer = REFUSAL_TEXT
        refused = True
        item_ids: list[str] = []
    else:
        evidence = [
            EvidenceItem(item_id=sc.chunk.item_id, chunk_type=sc.chunk.chunk_type, text=sc.chunk.text, metadata=sc.chunk.metadata)
            for sc in result.final_chunks
        ]
        recent_turns = [(t.role, t.content) for t in session.recent_turns]
        answer = llm.generate_answer(req.query, evidence, result.sub_queries, session.summary, recent_turns)
        cost.add_llm_call("generate")
        refused = False
        item_ids = list(dict.fromkeys(sc.chunk.item_id for sc in result.final_chunks))

    record_turn(session, req.query, answer, item_ids)
    maybe_refresh_summary(session, lambda turns: _summarize(llm, turns, cost))
    save_session(session)

    cache.put_both(req.query, query_vec, answer, item_ids, refused=refused)

    return _response(
        answer, refused, False, None, item_ids, result.sub_queries, result.top_score, cost, request_id
    )


def _summarize(llm, turns, cost: CostAccumulator) -> str:
    cost.add_llm_call("summarize")
    return llm.summarize(turns)


def _response(
    answer: str,
    refused: bool,
    degraded: bool,
    cache_hit: str | None,
    evidence_item_ids: list[str],
    sub_queries: list[str],
    top_score: float | None,
    cost: CostAccumulator,
    request_id: str,
) -> ChatResponse:
    return ChatResponse(
        answer=answer,
        refused=refused,
        degraded=degraded,
        cache_hit=cache_hit,
        evidence_item_ids=evidence_item_ids,
        sub_queries=sub_queries,
        top_rerank_score=top_score,
        estimated_cost=cost.total_estimated_cost,
        request_id=request_id,
    )
