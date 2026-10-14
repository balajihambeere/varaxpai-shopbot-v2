from products import PRODUCTS
from src.eval.run_eval import run
from src.ingestion.ingest import ingest_catalog


def test_eval_run_is_reproducible_against_an_unchanged_index(live_stack):
    """REQ-N-007 acceptance criterion."""
    ingest_catalog(PRODUCTS)
    first = run()
    second = run()
    assert first["summary"] == second["summary"]


def test_eval_summary_has_all_four_dimensions(live_stack):
    result = run()
    for key in ("faithfulness", "answer_relevance", "context_precision", "context_recall"):
        assert key in result["summary"]
        assert 0.0 <= result["summary"][key] <= 1.0
