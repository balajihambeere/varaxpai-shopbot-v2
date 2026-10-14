"""Minimal per-session rate limit. See docs/threat-model.md threat #5.

Deliberately simple (fixed window, not sliding/token-bucket) — this is a
SHOULD, not a MUST, with no measured abuse pattern to design against yet
(framework §1.5: don't over-engineer a control nothing has justified)."""

from fastapi import HTTPException

from src.memory.session import get_redis

WINDOW_SECONDS = 60
MAX_REQUESTS_PER_WINDOW = 30


def enforce_session_rate_limit(session_id: str) -> None:
    r = get_redis()
    key = f"ratelimit:{session_id}"
    count = r.incr(key)
    if count == 1:
        r.expire(key, WINDOW_SECONDS)
    if count > MAX_REQUESTS_PER_WINDOW:
        raise HTTPException(status_code=429, detail="Too many requests for this session. Please slow down.")
