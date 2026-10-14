"""Evaluation CLI. Run with: python -m src.eval.run_eval

Runs the full pipeline (decompose -> retrieve -> rerank -> confidence gate
-> generate) directly against the labeled dataset, bypassing the cache (an
evaluation run always wants a fresh measurement of the live pipeline, not a
cached answer from a previous run) and the HTTP layer (so this has no
dependency on a running server process, only on Qdrant/Redis/rerank-service
being reachable).

Writes a timestamped JSON result to docs/evidence/ and prints the summary.
Run it twice in a row with nothing changed and the numbers should match —
see REQ-N-007.
"""

import json
import sys
import time
from pathlib import Path

from src.config import settings
from src.eval.metrics import RowScore, context_precision_heuristic, context_recall, faithfulness_heuristic, summarize
from src.providers.embeddings import get_embedding_provider
from src.providers.llm import EvidenceItem, get_llm_provider
from src.retrieval.pipeline import run_retrieval

DATASET_PATH = Path(__file__).resolve().parents[2] / "eval" / "labeled_dataset.jsonl"
EVIDENCE_DIR = Path(__file__).resolve().parents[2] / "docs" / "evidence"


def load_dataset(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run() -> dict:
    llm = get_llm_provider()
    embedding_provider = get_embedding_provider()
    rows = load_dataset(DATASET_PATH)

    row_scores: list[RowScore] = []
    details = []
    for row in rows:
        result = run_retrieval(row["query"], llm, session=None)

        if result.degraded or not result.final_chunks:
            answer = "I don't have enough information to answer that. Please contact support for help."
            retrieved_chunk_ids: list[str] = []
            context_texts: list[str] = []
            rerank_scores: list[float] = []
        else:
            evidence = [
                EvidenceItem(item_id=sc.chunk.item_id, chunk_type=sc.chunk.chunk_type, text=sc.chunk.text, metadata=sc.chunk.metadata)
                for sc in result.final_chunks
            ]
            answer = llm.generate_answer(row["query"], evidence, result.sub_queries, "", [])
            retrieved_chunk_ids = [sc.chunk.id for sc in result.final_chunks]
            context_texts = [sc.chunk.text for sc in result.final_chunks]
            rerank_scores = [sc.score for sc in result.final_chunks]

        faithfulness = faithfulness_heuristic(answer, context_texts)
        answer_relevance = _answer_relevance(row["query"], answer, embedding_provider)
        context_precision = context_precision_heuristic(rerank_scores, settings.confidence_floor)
        recall = context_recall(row["gold_chunk_ids"], retrieved_chunk_ids)

        row_scores.append(
            RowScore(
                id=row["id"],
                strata=row["strata"],
                faithfulness=faithfulness,
                answer_relevance=answer_relevance,
                context_precision=context_precision,
                context_recall=recall,
            )
        )
        details.append(
            {
                "id": row["id"],
                "query": row["query"],
                "strata": row["strata"],
                "answer": answer,
                "retrieved_chunk_ids": retrieved_chunk_ids,
                "gold_chunk_ids": row["gold_chunk_ids"],
                "faithfulness": faithfulness,
                "answer_relevance": answer_relevance,
                "context_precision": context_precision,
                "context_recall": recall,
            }
        )

    summary = summarize(row_scores)
    return {
        "generated_at": time.time(),
        "provider": "openai" if settings.openai_api_key else "mock",
        "note": "faithfulness/answer_relevance/context_precision are heuristic approximations computed without an LLM judge (see src/eval/metrics.py docstring); context_recall is an exact set-overlap against hand-labeled gold evidence.",
        "summary": {
            "n": summary.n,
            "faithfulness": round(summary.faithfulness, 4),
            "answer_relevance": round(summary.answer_relevance, 4),
            "context_precision": round(summary.context_precision, 4),
            "context_recall": round(summary.context_recall, 4),
        },
        "by_strata": summary.by_strata,
        "details": details,
    }


def _answer_relevance(query: str, answer: str, embedding_provider) -> float:
    from src.eval.metrics import answer_relevance_embedding

    return answer_relevance_embedding(query, answer, embedding_provider.embed_dense)


def main() -> None:
    result = run()
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EVIDENCE_DIR / f"eval_run_{int(result['generated_at'])}.json"
    out_path.write_text(json.dumps(result, indent=2))

    print(json.dumps(result["summary"], indent=2))
    print(f"\nFull result written to {out_path}")


if __name__ == "__main__":
    sys.exit(main())
