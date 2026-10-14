from src.domain.models import Chunk, ScoredChunk
from src.retrieval.confidence import passes_confidence_floor, top_score


def _sc(score: float) -> ScoredChunk:
    return ScoredChunk(chunk=Chunk(id="c", item_id="c", chunk_type="overview", text=""), score=score)


def test_empty_list_fails_floor():
    assert passes_confidence_floor([]) is False
    assert top_score([]) is None


def test_score_above_floor_passes(monkeypatch):
    from src import config

    monkeypatch.setattr(config.settings, "confidence_floor", -8.5)
    assert passes_confidence_floor([_sc(-8.0)]) is True


def test_score_below_floor_fails(monkeypatch):
    from src import config

    monkeypatch.setattr(config.settings, "confidence_floor", -8.5)
    assert passes_confidence_floor([_sc(-9.0)]) is False


def test_only_top_score_matters_not_the_rest():
    from src import config

    config.settings.confidence_floor = -8.5
    reranked = [_sc(5.0), _sc(-100.0), _sc(-100.0)]
    assert passes_confidence_floor(reranked) is True
    assert top_score(reranked) == 5.0
