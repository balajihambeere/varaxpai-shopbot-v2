# Catalog Retrieval Assistant

A retrieval-augmented question-answering service for a product catalog: hybrid (dense + lexical) retrieval fused by Reciprocal Rank Fusion, cross-encoder reranking, a calibrated confidence-based refusal gate, LLM-based multi-intent query decomposition, structured bounded conversational memory, a layered answer cache with explicit invalidation, and a labeled-dataset evaluation harness.

Per explicit instruction, this project never names the source material it was derived from, or any of that material's narrative content (its title, its characters, its internal section structure, or any coined terminology unique to it) — everywhere in this repository the domain is a generic product catalog. See `docs/decisions/0001-anonymize-source-material.md`.

## Status

Fully implemented and live-verified, including against a real OpenAI model and through the project's frontend. See `docs/final-readiness-checklist.md` for the full checklist and `docs/evidence/` for real run output (tests, evaluation, load test, migration-discipline demo).

## Setup

```bash
cp .env.example .env          # defaults work out of the box (mock LLM, local embeddings)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
docker compose up -d qdrant redis rerank
python3 -m scripts.ingest_fixtures
```

To use a real LLM instead of the deterministic mock, set `OPENAI_API_KEY` in `.env` — no other flag needed, it's picked up automatically.

## Running

```bash
docker compose up -d --build app   # full stack, including this service, on :8010
# or, for local iteration against the dockerized dependencies above:
uvicorn src.api.main:app --host 0.0.0.0 --port 8010
```

- `POST /v1/chat` — `{session_id, query}` → grounded answer or honest refusal, with evidence IDs, sub-queries, and a cost estimate.
- `POST /v1/catalog/items` — re-index one item and invalidate affected cache entries; requires `X-Admin-Token`.
- `GET /health`, `GET /ready`

## Testing

```bash
python3 -m pytest -q --cov=src --cov-report=term-missing
```

Always runs against the deterministic mock LLM provider regardless of `.env` (see `docs/decisions/0011-default-test-suite-stays-on-mock-llm.md`) — free, fast, reproducible. Requires `docker compose up -d qdrant redis rerank` first; integration tests skip cleanly if those aren't reachable.

```bash
python3 -m src.eval.run_eval        # evaluation harness against the labeled dataset
python3 -m loadtest.harness --qpm 60 --duration 30
python3 -m scripts.shadow_compare   # migration-discipline demonstration
```

## Docs

- `docs/classification.md`, `docs/requirements.md`, `docs/architecture.md`, `docs/threat-model.md` — design package
- `docs/unknowns.md` — open questions and assumptions
- `docs/decisions/` — architecture decision records, including every real bug found via live testing and how it was fixed
- `docs/final-readiness-checklist.md`, `docs/evidence/` — verification status and real run output

## Frontend

The sibling `../frontend/` project's chat widget talks to this service via `SHOPBOT_API_URL` (default `http://localhost:8010`); see `../frontend/app/api/chat/route.ts`.
