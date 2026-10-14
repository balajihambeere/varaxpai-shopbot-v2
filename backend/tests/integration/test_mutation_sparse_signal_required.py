"""Mutation/fault check on the core concept (framework §11): confirm hybrid
fusion is not a no-op, i.e. confirm it is actually doing the work C-001
claims. If someone deleted the sparse signal or the RRF fusion step and
silently fell back to dense-only ranking, this test is the one that would
catch it — it would stop finding any disagreement at all.

An earlier version of this test tried to show "dense-only can't find a
literal product code." That premise did not survive contact with a real
embedding model on this fixture: BAAI/bge-small-en-v1.5 turned out to
assign meaningful similarity to short literal codes once they are present
in the chunk text (see docs/decisions/0007). Asserting the disproven
premise would have been exactly the kind of fabricated/stale verification
the framework's evidence rules forbid, so the test was replaced with one
whose premise is actually true on this fixture: fusing dense and sparse
rankings changes which chunk ranks first, for at least some real queries,
relative to dense-only ranking alone."""

import pytest

from products import PRODUCTS
from src.ingestion.ingest import ingest_catalog
from src.providers.embeddings import get_embedding_provider
from src.retrieval.hybrid import retrieve_hybrid
from src.vectorstore import qdrant_store

SINGLE_INTENT_QUERIES = [
    "p001 in mint",
    "price of p015",
    "is p040 in stock",
    "care instructions for p003",
    "something good for a function",
    "what can I wear to a wedding",
    "need something for a party tonight",
    "looking for daily wear",
    "recommend a saree for a wedding",
    "cotton kurta for daily wear",
    "something for winter",
    "lightweight stole for summer",
]


@pytest.fixture(scope="module", autouse=True)
def ingested_catalog(live_stack):
    ingest_catalog(PRODUCTS)


def test_rrf_fusion_changes_top1_versus_dense_only_for_some_queries(live_stack):
    provider = get_embedding_provider()
    disagreements = 0
    for query in SINGLE_INTENT_QUERIES:
        dense_vec = provider.embed_dense([query])[0]
        dense_top1 = qdrant_store.dense_search(dense_vec, limit=1)
        fused_top1 = retrieve_hybrid(query, top_n=1)

        dense_id = dense_top1[0].chunk.id if dense_top1 else None
        fused_id = fused_top1[0].chunk.id if fused_top1 else None
        if dense_id != fused_id:
            disagreements += 1

    assert disagreements > 0, (
        "Hybrid fusion produced the exact same top-1 choice as dense-only retrieval "
        "on every single query in this set — fusion would be a no-op, which means "
        "C-001 is not actually implemented, only named."
    )
