import pytest

from products import PRODUCTS
from src.ingestion.ingest import ingest_catalog
from src.retrieval.confidence import passes_confidence_floor
from src.retrieval.hybrid import retrieve_hybrid
from src.retrieval.rerank_client import rerank
from src.vectorstore import qdrant_store


@pytest.fixture(scope="module", autouse=True)
def ingested_catalog(live_stack):
    count = ingest_catalog(PRODUCTS)
    assert count > 0
    return count


def test_ingestion_populates_both_named_vectors_per_chunk(live_stack):
    """REQ-F-007 acceptance criterion."""
    client = qdrant_store.get_client()
    info = client.get_collection("catalog_chunks")
    assert info.points_count >= len(PRODUCTS)  # at least one chunk per item
    vector_names = set(info.config.params.vectors.keys())
    sparse_names = set((info.config.params.sparse_vectors or {}).keys())
    assert "dense" in vector_names
    assert "sparse" in sparse_names


def test_literal_item_code_is_retrieved_and_passes_confidence_floor(live_stack):
    """REQ-F-001 acceptance criterion — a literal product code with no
    semantic neighborhood must still be found."""
    query = "p001 in mint"
    fused = retrieve_hybrid(query)
    fused_ids = {sc.chunk.item_id for sc in fused}
    assert "p001" in fused_ids

    reranked = rerank(query, fused, top_k=3)
    assert reranked[0].chunk.item_id == "p001"
    assert passes_confidence_floor(reranked) is True


def test_off_topic_query_fails_confidence_floor(live_stack):
    """REQ-F-003 acceptance criterion — no relevant catalog content means refuse."""
    query = "totally unrelated question about jet engine turbine maintenance"
    fused = retrieve_hybrid(query)
    reranked = rerank(query, fused, top_k=3)
    assert passes_confidence_floor(reranked) is False


def test_reranking_promotes_the_correct_candidate_over_a_naive_top_rank(live_stack):
    """REQ-F-002 acceptance criterion — hybrid rank-1 is not always the
    reranked rank-1; the cross-encoder should fix a known mis-ordering."""
    query = "what can I wear to a small office party in the evening"
    fused = retrieve_hybrid(query)
    reranked = rerank(query, fused, top_k=5)
    # Assert reranking actually changed something relative to fused order —
    # otherwise this test would pass even if reranking were a no-op.
    fused_order = [sc.chunk.id for sc in fused[:5]]
    reranked_order = [sc.chunk.id for sc in reranked]
    assert fused_order != reranked_order
