from src.eval.metrics import context_precision_heuristic, context_recall, faithfulness_heuristic


def test_faithfulness_is_perfect_when_no_context_was_retrieved_refusal_case():
    assert faithfulness_heuristic("I don't have enough information to answer that.", []) == 1.0


def test_faithfulness_penalizes_claims_absent_from_context():
    answer = "The widget costs fifty dollars and comes in purple."
    context = ["The widget is a small plastic tool."]
    score = faithfulness_heuristic(answer, context)
    assert 0.0 <= score < 1.0


def test_faithfulness_is_high_when_answer_is_built_from_context():
    answer = "The widget is a small plastic tool."
    context = ["The widget is a small plastic tool used for testing."]
    score = faithfulness_heuristic(answer, context)
    assert score > 0.8


def test_context_precision_vacuously_perfect_on_correct_refusal():
    assert context_precision_heuristic([], confidence_floor=-8.5) == 1.0


def test_context_precision_counts_fraction_above_floor():
    assert context_precision_heuristic([5.0, -9.0, -8.0], confidence_floor=-8.5) == 2 / 3


def test_context_recall_out_of_scope_rewards_correct_refusal():
    assert context_recall(gold_chunk_ids=[], retrieved_chunk_ids=[]) == 1.0


def test_context_recall_out_of_scope_penalizes_hallucinated_evidence():
    assert context_recall(gold_chunk_ids=[], retrieved_chunk_ids=["x:overview"]) == 0.0


def test_context_recall_partial_overlap():
    assert context_recall(gold_chunk_ids=["a", "b"], retrieved_chunk_ids=["a", "c"]) == 0.5
