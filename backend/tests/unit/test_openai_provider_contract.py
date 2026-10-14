"""Contract tests for the real-provider adapter that don't require a live
API key: they stub `_chat` (the one method that actually makes a network
call) and check that decompose()/generate_answer() parse and build prompts
correctly around it. This does NOT verify real model quality — only that
the adapter's plumbing is correct. See docs/unknowns.md item 1."""

import json

from src.providers.llm import REFUSAL_TEXT, EvidenceItem, OpenAICompatibleProvider


def _provider() -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(base_url="https://example.invalid/v1", api_key="test-key", model="gpt-4o-mini")


def test_decompose_parses_single_intent_json(monkeypatch):
    provider = _provider()
    monkeypatch.setattr(provider, "_chat", lambda *a, **k: json.dumps({"sub_queries": ["cotton kurta"]}))
    assert provider.decompose("cotton kurta") == ["cotton kurta"]


def test_decompose_parses_two_intent_json(monkeypatch):
    provider = _provider()
    monkeypatch.setattr(provider, "_chat", lambda *a, **k: json.dumps({"sub_queries": ["warm", "festive"]}))
    assert provider.decompose("warm and festive") == ["warm", "festive"]


def test_decompose_fails_safe_on_malformed_json(monkeypatch):
    provider = _provider()
    monkeypatch.setattr(provider, "_chat", lambda *a, **k: "not json at all")
    assert provider.decompose("some query") == ["some query"]


def test_decompose_fails_safe_on_too_many_sub_queries(monkeypatch):
    provider = _provider()
    monkeypatch.setattr(provider, "_chat", lambda *a, **k: json.dumps({"sub_queries": ["a", "b", "c"]}))
    assert provider.decompose("some query") == ["some query"]


def test_generate_answer_refuses_without_calling_the_model_on_empty_evidence(monkeypatch):
    provider = _provider()
    called = []
    monkeypatch.setattr(provider, "_chat", lambda *a, **k: called.append(1) or "should not happen")
    answer = provider.generate_answer("q", [], ["q"], "", [])
    assert answer == REFUSAL_TEXT
    assert called == []  # no network call made for a request with no evidence


def test_generate_answer_passes_evidence_and_query_into_the_prompt(monkeypatch):
    provider = _provider()
    captured = {}

    def fake_chat(system, user, **kwargs):
        captured["system"] = system
        captured["user"] = user
        return "the answer"

    monkeypatch.setattr(provider, "_chat", fake_chat)
    evidence = [EvidenceItem(item_id="p1", chunk_type="overview", text="A red widget.", metadata={})]
    answer = provider.generate_answer("what colors?", evidence, ["what colors?"], "prior summary", [("user", "hi")])

    assert answer == "the answer"
    assert "A red widget." in captured["user"]
    assert "what colors?" in captured["user"]
    assert "prior summary" in captured["user"]
    assert "only state facts" in captured["system"].lower() or "only" in captured["system"].lower()
