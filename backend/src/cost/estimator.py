"""Per-request cost estimate. See docs/requirements.md REQ-N-006.

The unit costs below are placeholder `ASSUMPTION` values (no real billing
account is attached to any provider in this environment — the mock LLM and
local embedding/rerank models are free to run). The point of this module is
to prove the *plumbing* for cost accounting exists and is additive across a
request, so swapping in real per-call prices later (once a real provider is
configured) is a one-line config change, not a rewrite."""

from dataclasses import dataclass, field

EMBED_CALL_COST = 0.0001  # ASSUMPTION placeholder, per embed_dense+embed_sparse call pair
LLM_CALL_COST = 0.0008  # ASSUMPTION placeholder, per decompose/generate/summarize call
RERANK_CALL_COST = 0.00005  # ASSUMPTION placeholder, per rerank batch call


@dataclass
class CostAccumulator:
    embed_calls: int = 0
    llm_calls: int = 0
    rerank_calls: int = 0
    breakdown: list[str] = field(default_factory=list)

    def add_embed_call(self) -> None:
        self.embed_calls += 1

    def add_llm_call(self, label: str) -> None:
        self.llm_calls += 1
        self.breakdown.append(label)

    def add_rerank_call(self) -> None:
        self.rerank_calls += 1

    @property
    def total_estimated_cost(self) -> float:
        return (
            self.embed_calls * EMBED_CALL_COST
            + self.llm_calls * LLM_CALL_COST
            + self.rerank_calls * RERANK_CALL_COST
        )
