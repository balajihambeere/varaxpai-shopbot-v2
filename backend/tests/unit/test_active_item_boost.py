from src.domain.models import Chunk, ScoredChunk
from src.memory.session import Session
from src.retrieval.pipeline import apply_active_item_boost


def _sc(item_id: str, score: float) -> ScoredChunk:
    return ScoredChunk(chunk=Chunk(id=f"{item_id}:overview", item_id=item_id, chunk_type="overview", text=""), score=score)


def test_no_session_returns_unchanged_order():
    ranked = [_sc("a", 1.0), _sc("b", 0.9)]
    assert apply_active_item_boost(ranked, None) == ranked


def test_active_item_breaks_a_near_tie(monkeypatch):
    from src import config

    monkeypatch.setattr(config.settings, "active_entity_boost", 0.05)
    session = Session(session_id="s", active_item_ids=["b"])
    ranked = [_sc("a", 1.00), _sc("b", 0.98)]  # "a" ahead by 0.02, boost is 0.05
    boosted = apply_active_item_boost(ranked, session)
    assert boosted[0].chunk.item_id == "b"


def test_active_item_boost_does_not_override_a_clear_winner(monkeypatch):
    from src import config

    monkeypatch.setattr(config.settings, "active_entity_boost", 0.05)
    session = Session(session_id="s", active_item_ids=["b"])
    ranked = [_sc("a", 5.0), _sc("b", 0.5)]  # gap of 4.5, far bigger than the 0.05 boost
    boosted = apply_active_item_boost(ranked, session)
    assert boosted[0].chunk.item_id == "a"


def test_empty_active_items_is_a_no_op():
    session = Session(session_id="s", active_item_ids=[])
    ranked = [_sc("a", 1.0), _sc("b", 0.9)]
    assert apply_active_item_boost(ranked, session) == ranked
