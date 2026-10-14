"""Structured, bounded conversational memory. See docs/requirements.md
C-005 / REQ-F-005.

Three parts, three jobs:
  - `recent_turns`: last N turns verbatim, for immediate pronoun/reference
    resolution ("what about it in size L").
  - `summary`: a rolling, periodically-refreshed summary covering anything
    older than the verbatim window.
  - `active_item_ids`: a small bounded list of items the conversation is
    currently "about," used to bias retrieval ranking (see
    `apply_active_item_boost` in `src/retrieval/pipeline.py`), most-recent
    first.

Deliberately NOT a sliding window over the raw transcript (that either gets
expensive/noisy as conversations grow, or silently drops older-but-relevant
facts depending on the window size) — see the ADR reasoning folded into
docs/architecture.md.
"""

import json
from dataclasses import asdict, dataclass, field

import redis

from src.config import settings

_redis: redis.Redis | None = None


def get_redis() -> redis.Redis:
    global _redis
    if _redis is None:
        _redis = redis.from_url(settings.redis_url, decode_responses=True)
    return _redis


@dataclass
class Turn:
    role: str  # "user" | "assistant"
    content: str


@dataclass
class Session:
    session_id: str
    recent_turns: list[Turn] = field(default_factory=list)
    summary: str = ""
    active_item_ids: list[str] = field(default_factory=list)
    turn_count: int = 0

    def key(self) -> str:
        return f"session:{self.session_id}"


def load_session(session_id: str) -> Session:
    raw = get_redis().get(f"session:{session_id}")
    if not raw:
        return Session(session_id=session_id)
    data = json.loads(raw)
    data["recent_turns"] = [Turn(**t) for t in data["recent_turns"]]
    return Session(**data)


def save_session(session: Session) -> None:
    get_redis().set(session.key(), json.dumps(asdict(session)), ex=settings.session_ttl_seconds)


def record_turn(session: Session, user_text: str, assistant_text: str, mentioned_item_ids: list[str]) -> None:
    """Append the completed turn, refresh the rolling summary on schedule,
    and update the active-item list. Mutates `session` in place; caller is
    responsible for persisting it via `save_session`."""
    session.recent_turns.append(Turn(role="user", content=user_text))
    session.recent_turns.append(Turn(role="assistant", content=assistant_text))
    session.recent_turns = session.recent_turns[-(settings.recent_turns_window * 2) :]
    session.turn_count += 1

    for item_id in mentioned_item_ids:
        if item_id in session.active_item_ids:
            session.active_item_ids.remove(item_id)
        session.active_item_ids.insert(0, item_id)
    session.active_item_ids = session.active_item_ids[: settings.active_entity_limit]


def maybe_refresh_summary(session: Session, llm_summarize) -> None:
    if session.turn_count == 0 or session.turn_count % settings.summary_refresh_every_n_turns != 0:
        return
    turns_as_tuples = [(t.role, t.content) for t in session.recent_turns]
    session.summary = llm_summarize(turns_as_tuples)
