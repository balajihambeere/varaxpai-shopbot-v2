import uuid

from src.memory.session import Session, load_session, maybe_refresh_summary, record_turn, save_session


def test_session_round_trips_through_redis(live_stack):
    session_id = f"test-{uuid.uuid4()}"
    session = load_session(session_id)
    assert session.recent_turns == []
    assert session.active_item_ids == []

    record_turn(session, "hello", "hi there", mentioned_item_ids=["p001"])
    save_session(session)

    reloaded = load_session(session_id)
    assert len(reloaded.recent_turns) == 2
    assert reloaded.active_item_ids == ["p001"]


def test_recent_turns_window_is_bounded(monkeypatch):
    from src import config

    monkeypatch.setattr(config.settings, "recent_turns_window", 2)
    session = Session(session_id=f"test-{uuid.uuid4()}")
    for i in range(5):
        record_turn(session, f"user turn {i}", f"assistant turn {i}", mentioned_item_ids=[])
    # window=2 turns => at most 4 verbatim messages (2 user + 2 assistant)
    assert len(session.recent_turns) == 4
    assert session.recent_turns[-2].content == "user turn 4"


def test_active_item_ids_are_most_recent_first_and_bounded(monkeypatch):
    from src import config

    monkeypatch.setattr(config.settings, "active_entity_limit", 3)
    session = Session(session_id="s")
    record_turn(session, "q1", "a1", mentioned_item_ids=["p1"])
    record_turn(session, "q2", "a2", mentioned_item_ids=["p2"])
    record_turn(session, "q3", "a3", mentioned_item_ids=["p3"])
    record_turn(session, "q4", "a4", mentioned_item_ids=["p4"])
    assert session.active_item_ids == ["p4", "p3", "p2"]


def test_re_mentioning_an_item_moves_it_to_front(monkeypatch):
    from src import config

    monkeypatch.setattr(config.settings, "active_entity_limit", 3)
    session = Session(session_id="s")
    record_turn(session, "q1", "a1", mentioned_item_ids=["p1"])
    record_turn(session, "q2", "a2", mentioned_item_ids=["p2"])
    record_turn(session, "q3", "a3", mentioned_item_ids=["p1"])
    assert session.active_item_ids == ["p1", "p2"]


def test_summary_refreshes_only_on_configured_cadence(monkeypatch):
    from src import config

    monkeypatch.setattr(config.settings, "summary_refresh_every_n_turns", 2)
    session = Session(session_id="s")
    calls = []

    def fake_summarize(turns):
        calls.append(turns)
        return "summary"

    record_turn(session, "q1", "a1", mentioned_item_ids=[])
    maybe_refresh_summary(session, fake_summarize)
    assert session.summary == ""  # turn_count=1, not a multiple of 2

    record_turn(session, "q2", "a2", mentioned_item_ids=[])
    maybe_refresh_summary(session, fake_summarize)
    assert session.summary == "summary"  # turn_count=2, refreshes
    assert len(calls) == 1
