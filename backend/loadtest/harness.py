"""Load-test harness. See docs/requirements.md C-009 / REQ-F-011.

Run with: python -m loadtest.harness --qpm 60 --duration 30 --base-url http://localhost:8010

Draws queries from the labeled dataset (mixing answerable and
out-of-scope/refusal queries, same as a real traffic mix would), fires them
at a configurable rate against a running instance, and reports p50/p95
latency plus a breakdown of refusal/degraded/error counts. This talks to a
real running server over HTTP — it is not a mock of load, it is load.
"""

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import httpx

DATASET_PATH = Path(__file__).resolve().parents[1] / "eval" / "labeled_dataset.jsonl"


def load_queries() -> list[str]:
    with open(DATASET_PATH) as f:
        return [json.loads(line)["query"] for line in f if line.strip()]


async def fire_one(client: httpx.AsyncClient, base_url: str, query: str, session_id: str) -> dict:
    t0 = time.perf_counter()
    try:
        resp = await client.post(f"{base_url}/v1/chat", json={"session_id": session_id, "query": query}, timeout=10.0)
        latency = time.perf_counter() - t0
        resp.raise_for_status()
        body = resp.json()
        return {"latency": latency, "refused": body["refused"], "degraded": body["degraded"], "error": False}
    except Exception:
        return {"latency": time.perf_counter() - t0, "refused": False, "degraded": False, "error": True}


async def run(base_url: str, qpm: int, duration_s: int) -> dict:
    queries = load_queries()
    interval = 60.0 / qpm
    results: list[dict] = []
    deadline = time.time() + duration_s
    i = 0

    async with httpx.AsyncClient() as client:
        tasks = []
        while time.time() < deadline:
            query = queries[i % len(queries)]
            i += 1
            tasks.append(asyncio.create_task(fire_one(client, base_url, query, f"loadtest-{i % 20}")))
            await asyncio.sleep(interval)
        results = await asyncio.gather(*tasks)

    latencies = [r["latency"] for r in results]
    p50 = statistics.median(latencies) if latencies else 0.0
    p95 = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies, default=0.0)

    return {
        "qpm": qpm,
        "duration_s": duration_s,
        "n_requests": len(results),
        "p50_ms": round(p50 * 1000, 1),
        "p95_ms": round(p95 * 1000, 1),
        "refused": sum(1 for r in results if r["refused"]),
        "degraded": sum(1 for r in results if r["degraded"]),
        "errors": sum(1 for r in results if r["error"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8010")
    parser.add_argument("--qpm", type=int, default=60)
    parser.add_argument("--duration", type=int, default=30)
    args = parser.parse_args()

    report = asyncio.run(run(args.base_url, args.qpm, args.duration))
    print(json.dumps(report, indent=2))

    out_dir = Path(__file__).resolve().parents[1] / "docs" / "evidence"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"loadtest_{int(time.time())}.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nReport written to {out_path}")


if __name__ == "__main__":
    main()
