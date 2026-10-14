"""Embedding provider abstraction.

Swapping providers is a configuration change (`EMBEDDING_PROVIDER` env var),
never a change to call sites. `FastEmbedLocalProvider` is the default and the
only one verified in this environment: it runs fully locally via ONNX
runtime, requires no API key, and produces both a dense (semantic) vector and
a sparse (lexical/BM25-style) vector per text, which is exactly what hybrid
retrieval (see `src/retrieval/hybrid.py`) needs.

A paid-API adapter can be added behind this same `EmbeddingProvider`
interface later; none is wired up here because no credentials are available
(framework default — see docs/unknowns.md item 2).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class SparseVector:
    indices: list[int]
    values: list[float]


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed_dense(self, texts: list[str]) -> list[list[float]]:
        """Return one dense vector per input text."""

    @abstractmethod
    def embed_sparse(self, texts: list[str]) -> list[SparseVector]:
        """Return one sparse vector per input text."""

    @property
    @abstractmethod
    def dense_dim(self) -> int:
        ...


class FastEmbedLocalProvider(EmbeddingProvider):
    """Local, no-API-key dense + sparse embeddings via fastembed/onnxruntime."""

    def __init__(self, dense_model_name: str, sparse_model_name: str):
        from fastembed import SparseTextEmbedding, TextEmbedding

        self._dense = TextEmbedding(model_name=dense_model_name)
        self._sparse = SparseTextEmbedding(model_name=sparse_model_name)
        self._dense_dim = self._infer_dim(dense_model_name)

    @staticmethod
    def _infer_dim(dense_model_name: str) -> int:
        from fastembed import TextEmbedding

        for m in TextEmbedding.list_supported_models():
            if m["model"] == dense_model_name:
                return int(m["dim"])
        raise ValueError(f"Unknown dense model: {dense_model_name}")

    def embed_dense(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return [vec.tolist() for vec in self._dense.embed(texts)]

    def embed_sparse(self, texts: list[str]) -> list[SparseVector]:
        if not texts:
            return []
        out = []
        for sv in self._sparse.embed(texts):
            out.append(SparseVector(indices=sv.indices.tolist(), values=sv.values.tolist()))
        return out

    @property
    def dense_dim(self) -> int:
        return self._dense_dim


_provider: EmbeddingProvider | None = None


def get_embedding_provider() -> EmbeddingProvider:
    global _provider
    if _provider is None:
        from src.config import settings

        if settings.embedding_provider != "fastembed_local":
            raise NotImplementedError(
                f"Embedding provider '{settings.embedding_provider}' is not configured in "
                "this environment (NOT-VERIFIED). Only 'fastembed_local' is wired up."
            )
        _provider = FastEmbedLocalProvider(settings.dense_model_name, settings.sparse_model_name)
    return _provider
