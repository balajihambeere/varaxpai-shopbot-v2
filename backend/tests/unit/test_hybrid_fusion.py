from src.domain.models import Chunk, ScoredChunk
from src.retrieval.hybrid import reciprocal_rank_fusion


def _sc(chunk_id: str) -> ScoredChunk:
    return ScoredChunk(chunk=Chunk(id=chunk_id, item_id=chunk_id, chunk_type="overview", text=""), score=0.0)


def test_rrf_favors_chunk_ranked_first_in_both_lists():
    list_a = [_sc("x"), _sc("y"), _sc("z")]
    list_b = [_sc("x"), _sc("z"), _sc("y")]
    fused = reciprocal_rank_fusion(list_a, list_b, k=60)
    assert fused[0].chunk.id == "x"


def test_rrf_ignores_raw_score_uses_rank_only():
    # Raw scores differ wildly but ranks are identical — fusion should not care.
    a = ScoredChunk(chunk=Chunk(id="a", item_id="a", chunk_type="o", text=""), score=9999.0)
    b = ScoredChunk(chunk=Chunk(id="b", item_id="b", chunk_type="o", text=""), score=0.0001)
    fused = reciprocal_rank_fusion([a, b], [a, b], k=60)
    assert fused[0].chunk.id == "a"
    assert fused[0].score == 2 * (1.0 / 61)


def test_rrf_chunk_absent_from_both_lists_is_not_scored():
    list_a = [_sc("x")]
    list_b = [_sc("y")]
    fused = reciprocal_rank_fusion(list_a, list_b, k=60)
    ids = {sc.chunk.id for sc in fused}
    assert ids == {"x", "y"}
    assert "z" not in ids


def test_rrf_smaller_k_rewards_top_of_list_more_steeply():
    list_a = [_sc("top"), _sc("mid"), _sc("bottom")]
    fused_small_k = reciprocal_rank_fusion(list_a, k=1)
    fused_large_k = reciprocal_rank_fusion(list_a, k=1000)
    gap_small = fused_small_k[0].score - fused_small_k[-1].score
    gap_large = fused_large_k[0].score - fused_large_k[-1].score
    assert gap_small > gap_large
