"""Regression test for a real bug caught via L2 (containerized) testing:
naive string-replace on the rerank URL corrupted it whenever the hostname
itself was literally "rerank" (exactly the docker-compose case), because
"/rerank" also matches as a substring of "//rerank:8001". See the fix in
src/api/routes/health.py."""

from urllib.parse import urlsplit, urlunsplit


def _health_url(rerank_url: str) -> str:
    parts = urlsplit(rerank_url)
    return urlunsplit((parts.scheme, parts.netloc, "/health", "", ""))


def test_health_url_correct_when_hostname_is_literally_rerank():
    assert _health_url("http://rerank:8001/rerank") == "http://rerank:8001/health"


def test_health_url_correct_for_localhost():
    assert _health_url("http://localhost:8001/rerank") == "http://localhost:8001/health"
