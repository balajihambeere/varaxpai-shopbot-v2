"""Evaluation metrics. See docs/requirements.md C-007 / REQ-F-010.

Honesty note: a real RAGAS-style evaluation uses an LLM judge for
faithfulness, answer relevance, and context precision. No real LLM is
available in this environment (framework default). Rather than fabricate an
LLM judgment, three of the four metrics below are **heuristic
approximations** computed without any LLM — word-overlap and embedding
cosine similarity, using the same real local embedding model the rest of
the system uses. They are clearly labeled as approximations in every
output. The fourth metric, context recall, needs no LLM in the first place
(it's a set-overlap against hand-labeled gold evidence) and is computed
exactly as intended.

Do not read "0.83" out of this module as comparable to a published RAGAS
score from anywhere else. It is comparable only to this project's own past
runs of this module against this dataset — which is the only comparison
a recurring week-over-week tracking cadence actually needs.
"""

import re
from dataclasses import dataclass

_STOPWORDS = {
    "the", "a", "an", "is", "are", "to", "for", "of", "in", "on", "and", "or",
    "with", "this", "that", "it", "i", "you", "your", "do", "does", "what",
    "can", "please", "have", "has", "be", "will", "at",
}


def _informative_tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


def faithfulness_heuristic(answer: str, context_texts: list[str]) -> float:
    """Approximate groundedness: fraction of the answer's informative words
    that also appear somewhere in the retrieved context. 1.0 for a refusal
    (nothing to be unfaithful to) since the answer is a fixed, pre-approved
    template, not a generated claim."""
    if not context_texts:
        # No evidence was retrieved at all -> this is the refusal path.
        # The refusal is a fixed template, not a generated claim about the
        # catalog, so there is nothing for it to be unfaithful to.
        return 1.0
    answer_tokens = _informative_tokens(answer)
    if not answer_tokens:
        return 1.0
    context_tokens: set[str] = set()
    for text in context_texts:
        context_tokens |= _informative_tokens(text)
    covered = answer_tokens & context_tokens
    return len(covered) / len(answer_tokens)


def answer_relevance_embedding(query: str, answer: str, embed_dense) -> float:
    """Approximate relevance: cosine similarity between the query's and the
    answer's dense embedding, using this project's own embedding provider."""
    import math

    q_vec, a_vec = embed_dense([query, answer])
    dot = sum(x * y for x, y in zip(q_vec, a_vec))
    norm_q = math.sqrt(sum(x * x for x in q_vec))
    norm_a = math.sqrt(sum(x * x for x in a_vec))
    if norm_q == 0 or norm_a == 0:
        return 0.0
    return dot / (norm_q * norm_a)


def context_precision_heuristic(rerank_scores: list[float], confidence_floor: float) -> float:
    """Approximate precision: fraction of the chunks actually sent to
    generation whose own cross-encoder score clears the confidence floor
    (the same bar the system uses to decide whether to trust a chunk at
    all)."""
    if not rerank_scores:
        # Correct refusal path — no chunks were sent to generation, so
        # there is nothing imprecise to measure. Vacuously perfect.
        return 1.0
    above = sum(1 for s in rerank_scores if s > confidence_floor)
    return above / len(rerank_scores)


def context_recall(gold_chunk_ids: list[str], retrieved_chunk_ids: list[str]) -> float:
    """Exact set-overlap recall against hand-labeled gold evidence. For an
    out-of-scope query (empty gold set), recall is defined as 1.0 if the
    system retrieved nothing (correct refusal) and 0.0 if it retrieved
    anything (hallucinated relevance on a query with no catalog support)."""
    if not gold_chunk_ids:
        return 1.0 if not retrieved_chunk_ids else 0.0
    found = set(gold_chunk_ids) & set(retrieved_chunk_ids)
    return len(found) / len(gold_chunk_ids)


@dataclass
class RowScore:
    id: str
    strata: str
    faithfulness: float
    answer_relevance: float
    context_precision: float
    context_recall: float


@dataclass
class EvalSummary:
    n: int
    faithfulness: float
    answer_relevance: float
    context_precision: float
    context_recall: float
    by_strata: dict[str, dict[str, float]]


def summarize(rows: list[RowScore]) -> EvalSummary:
    n = len(rows)
    if n == 0:
        return EvalSummary(0, 0.0, 0.0, 0.0, 0.0, {})

    def avg(xs):
        return sum(xs) / len(xs)

    by_strata: dict[str, list[RowScore]] = {}
    for r in rows:
        by_strata.setdefault(r.strata, []).append(r)

    strata_summary = {
        s: {
            "faithfulness": avg([r.faithfulness for r in rs]),
            "answer_relevance": avg([r.answer_relevance for r in rs]),
            "context_precision": avg([r.context_precision for r in rs]),
            "context_recall": avg([r.context_recall for r in rs]),
        }
        for s, rs in by_strata.items()
    }

    return EvalSummary(
        n=n,
        faithfulness=avg([r.faithfulness for r in rows]),
        answer_relevance=avg([r.answer_relevance for r in rows]),
        context_precision=avg([r.context_precision for r in rows]),
        context_recall=avg([r.context_recall for r in rows]),
        by_strata=strata_summary,
    )
