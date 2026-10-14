from src.cost.estimator import EMBED_CALL_COST, LLM_CALL_COST, RERANK_CALL_COST, CostAccumulator


def test_cost_accumulates_additively():
    cost = CostAccumulator()
    cost.add_embed_call()
    cost.add_llm_call("decompose")
    cost.add_llm_call("generate")
    cost.add_rerank_call()
    expected = EMBED_CALL_COST + 2 * LLM_CALL_COST + RERANK_CALL_COST
    assert cost.total_estimated_cost == expected
    assert cost.breakdown == ["decompose", "generate"]


def test_zero_calls_is_zero_cost():
    cost = CostAccumulator()
    assert cost.total_estimated_cost == 0.0
