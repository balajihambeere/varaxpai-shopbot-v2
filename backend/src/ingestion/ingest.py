from src.domain.models import Chunk
from src.ingestion.chunking import chunk_catalog, chunk_item
from src.providers.embeddings import get_embedding_provider
from src.vectorstore import qdrant_store


def ingest_catalog(items: list[dict]) -> int:
    """Chunk, embed, and index a full catalog. Returns chunk count."""
    chunks = chunk_catalog(items)
    _ingest_chunks(chunks)
    return len(chunks)


def reingest_item(item: dict) -> list[str]:
    """Re-chunk and re-index a single changed item. Returns affected chunk ids."""
    chunks = chunk_item(item)
    _ingest_chunks(chunks)
    return [c.id for c in chunks]


def _ingest_chunks(chunks: list[Chunk]) -> None:
    if not chunks:
        return
    provider = get_embedding_provider()
    qdrant_store.ensure_collections(dense_dim=provider.dense_dim)
    texts = [c.text for c in chunks]
    dense_vecs = provider.embed_dense(texts)
    sparse_vecs = provider.embed_sparse(texts)
    qdrant_store.upsert_chunks(chunks, dense_vecs, sparse_vecs)
