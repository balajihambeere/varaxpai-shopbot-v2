# ShopBot

An AI-powered product assistant for **Varaxpai** (Vara Shopping Expert AI), a fictional Indian clothing retailer. ShopBot answers customer questions about a 50-product catalog using retrieval-augmented generation (RAG), and says "I don't know, contact support" rather than guessing when it has no evidence for an answer.

This directory holds two independent projects, each with its own git history:

| Project | What | Start here |
| --- | --- | --- |
| `backend/` | Python/FastAPI RAG service — hybrid retrieval, reranking, decomposition, memory, caching, `/v1/chat` + `/v1/catalog/items` + `/health` + `/ready` | `backend/README.md` |
| `frontend/` | Next.js chat widget — the customer-facing UI, wired to the backend's `/v1/chat` endpoint via `app/api/chat/route.ts` | `frontend/README.md` |

## Running both together

```bash
# terminal 1 — backend and its dependencies (Qdrant, Redis, rerank service)
cd backend
cp .env.example .env   # first time only
docker compose up -d --build
# or, for local iteration once the dependency containers are up:
#   source .venv/bin/activate && uvicorn src.api.main:app --reload --port 8010

# terminal 2 — frontend
cd frontend
npm run dev
```

Then open the frontend at `http://localhost:3000` — it talks to the backend at `http://localhost:8010` by default (`SHOPBOT_API_URL` in the frontend's `.env.local` if you need to point it elsewhere).

Full architecture, API reference, deployment, and known limitations are documented in `backend/docs/` — see `backend/README.md`'s documentation map.
