"""Demonstrates the dual-write / shadow-read / compare discipline for
safely replacing a stateful retrieval backend. See docs/requirements.md
C-008 / REQ-F-012, and docs/decisions/0006 for why this demonstrates the
discipline via a model swap rather than a literal second vector-store
product (this project only has one Qdrant instance available).

"Old": the live `catalog_chunks` collection, dense model
BAAI/bge-small-en-v1.5 (current production config).
"New"/candidate: a second collection, same chunks, same sparse signal,
dense model swapped to sentence-transformers/all-MiniLM-L6-v2 — standing in
for "the next embedding model we're considering."

Dual-write: both collections are populated from the same source chunks.
Shadow-read: every labeled-dataset query is run against both, never
switching the "old" collection out of service.
Compare: top-3 chunk-id agreement is measured and reported; a migration
would only be declared safe once this number is high and every
disagreement is individually explained (exactly the standard this project
applies to itself, not a lower bar applied only to a toy script).

Run with: python -m scripts.shadow_compare
"""

import json
import time
from pathlib import Path

from qdrant_client import models
from qdrant_client import QdrantClient

from products import PRODUCTS
from src.config import settings
from src.ingestion.chunking import chunk_catalog
from src.providers.embeddings import FastEmbedLocalProvider

CANDIDATE_COLLECTION = "catalog_chunks_candidate"
CANDIDATE_DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DATASET_PATH = Path(__file__).resolve().parents[1] / "eval" / "labeled_dataset.jsonl"
EVIDENCE_DIR = Path(__file__).resolve().parents[1] / "docs" / "evidence"


def build_candidate_collection(client: QdrantClient, candidate_provider: FastEmbedLocalProvider) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if CANDIDATE_COLLECTION in existing:
        client.delete_collection(CANDIDATE_COLLECTION)
    client.create_collection(
        collection_name=CANDIDATE_COLLECTION,
        vectors_config={"dense": models.VectorParams(size=candidate_provider.dense_dim, distance=models.Distance.COSINE)},
        sparse_vectors_config={"sparse": models.SparseVectorParams()},
    )

    chunks = chunk_catalog(PRODUCTS)
    texts = [c.text for c in chunks]
    dense_vecs = candidate_provider.embed_dense(texts)
    sparse_vecs = candidate_provider.embed_sparse(texts)

    import hashlib

    points = []
    for chunk, dense, sparse in zip(chunks, dense_vecs, sparse_vecs):
        point_id = hashlib.sha256(chunk.id.encode()).hexdigest()[:32]
        points.append(
            models.PointStruct(
                id=point_id,
                vector={"dense": dense, "sparse": models.SparseVector(indices=sparse.indices, values=sparse.values)},
                payload={"chunk_id": chunk.id, "item_id": chunk.item_id, "chunk_type": chunk.chunk_type, "text": chunk.text, **chunk.metadata},
            )
        )
    client.upsert(collection_name=CANDIDATE_COLLECTION, points=points)


def top3_ids(client: QdrantClient, collection: str, dense_vec: list[float]) -> list[str]:
    result = client.query_points(collection_name=collection, query=dense_vec, using="dense", limit=3, with_payload=True)
    return [p.payload["chunk_id"] for p in result.points]


def main() -> None:
    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)

    old_provider = FastEmbedLocalProvider(settings.dense_model_name, settings.sparse_model_name)
    candidate_provider = FastEmbedLocalProvider(CANDIDATE_DENSE_MODEL, settings.sparse_model_name)

    print(f"Building candidate collection with dense model {CANDIDATE_DENSE_MODEL} ...")
    build_candidate_collection(client, candidate_provider)

    queries = [json.loads(line)["query"] for line in open(DATASET_PATH) if line.strip()]

    rows = []
    agree_count = 0
    for query in queries:
        old_vec = old_provider.embed_dense([query])[0]
        new_vec = candidate_provider.embed_dense([query])[0]
        old_top3 = set(top3_ids(client, settings.catalog_collection, old_vec))
        new_top3 = set(top3_ids(client, CANDIDATE_COLLECTION, new_vec))
        agreement = len(old_top3 & new_top3)
        if agreement == len(old_top3) and len(old_top3) > 0:
            agree_count += 1
        rows.append({"query": query, "old_top3": sorted(old_top3), "new_top3": sorted(new_top3), "agreement_at_3": agreement})

    report = {
        "generated_at": time.time(),
        "old_dense_model": settings.dense_model_name,
        "candidate_dense_model": CANDIDATE_DENSE_MODEL,
        "n_queries": len(queries),
        "fully_agreeing_queries": agree_count,
        "agreement_rate": round(agree_count / len(queries), 3) if queries else 0.0,
        "rows": rows,
    }

    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EVIDENCE_DIR / f"shadow_compare_{int(report['generated_at'])}.json"
    out_path.write_text(json.dumps(report, indent=2))

    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    print(f"\nFull report written to {out_path}")
    print(
        "\nThis is a discipline demonstration, not a real production cutover: "
        f"'{CANDIDATE_COLLECTION}' is never read by the live app, and a real migration "
        "would require every disagreement below to be individually explained before cutover, "
        "exactly as docs/decisions/0006 describes."
    )


if __name__ == "__main__":
    main()
