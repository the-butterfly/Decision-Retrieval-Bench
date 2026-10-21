"""Measure d3-* response performance on the REAL payloads the benchmark sends.

The payloads are captured by intercepting the production client's `_post`, so
they are byte-identical to what goes on the wire (batching at maxq, state
trimming, provider size shaping all applied).

We then replay them against the live endpoint, one at a time, and record:
latency percentiles, request body size, the provider's `usage.input_tokens`,
and the `X-RateLimit-*` headers (so the budget state is visible).
"""
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

import aiohttp

ROOT = Path("/Volumes/Extension/WorkSpaces/WorkBuddy/SAG2-expr/code/SAG-Benchmark/sag2")
sys.path.insert(0, str(ROOT / "scripts"))

from sag2 import CorpusIndex, corpus_fingerprint, search_sag2  # noqa: E402
from benchmark_data import DatasetLoader  # noqa: E402
from sag2_vllm import TokenBudget, VllmSrJevClient  # noqa: E402

VALS = {}
for line in Path("/Volumes/Extension/WorkSpaces/WorkBuddy/SAG2-expr/code/.env").read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        VALS[k.strip()] = v.strip()
API = VALS["VLLM_SR_API"].rstrip("/")
ENDPOINT = API + "/systemone"
KEYS = {"vllm-sr/d3-turbo": "VLLM_SR_KEY", "vllm-sr/d3-flash": "VLLM_SR_KEY2", "vllm-sr/d3-mini": "VLLM_SR_KEY3"}
MODELS = list(KEYS)
DELAY = 1.0  # be gentle: the benchmark chains are already consuming the budget


class CaptureClient(VllmSrJevClient):
    """Production client with the network call replaced by a recorder."""

    def __init__(self):
        super().__init__(
            None,
            ENDPOINT,
            "vllm-sr/d3-turbo",
            "x",
            budget=TokenBudget(1e12, 1e12),
            state_chars=1500,
            state_mode="titles",
            max_questions_per_request=10,
        )
        self.captured = []

    async def _post(self, payload, key, entry):
        self.captured.append(payload)
        return {
            "answers": {k: {"type": "noul", "noul": 0.5} for k in payload["questions"]},
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "model": "capture",
        }


async def capture_academic(n_questions=3):
    dataset_dir = ROOT / "dataset"
    loader = DatasetLoader("musique", str(dataset_dir))
    loader.validate_source_pair()
    corpus = loader.load_corpus()
    records = loader.get_question_records()
    fingerprint = corpus_fingerprint(dataset_dir / "musique_corpus.json")
    index = CorpusIndex(ROOT / ".cache/sag2/musique.sqlite", corpus, fingerprint)
    client = CaptureClient()
    for record in records[:n_questions]:
        client.captured = []
        await search_sag2(
            index, client, record["question"], candidate_k=100, seed_k=10, max_hops=2,
            expansion_term_limit=20,
        )
    index.close()
    return client.captured


def load_enterprise_payloads(n):
    base = Path(
        "/Volumes/Extension/WorkSpaces/WorkBuddy/SAG2-expr/output/sag2/exp02-vllm-sr/"
        "enterprise-deadlock-20261021/requests"
    )
    out = []
    for f in sorted(base.glob("**/payload.json")):
        if len(out) >= n:
            break
        d = json.loads(f.read_text())
        if len(d.get("questions", {})) >= 8:
            out.append(d)
    return out


def pct(values, q):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


async def replay(session, model, payloads, label):
    key = VALS[KEYS[model]]
    latencies, tokens, sizes, statuses = [], [], [], []
    headers = {}
    for payload in payloads:
        body = dict(payload)
        body["model"] = model
        raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
        started = time.perf_counter()
        for attempt in range(4):
            async with session.post(
                ENDPOINT, json=body, headers={"Authorization": f"Bearer {key}"}
            ) as response:
                status = response.status
                headers = {k.lower(): v for k, v in response.headers.items() if "ratelimit" in k.lower()}
                if status == 200:
                    data = await response.json()
                    used = data.get("usage", {}).get("input_tokens", 0)
                    break
                await response.read()
                wait = float(response.headers.get("Retry-After", 5))
                await asyncio.sleep(min(wait, 20))
        elapsed = time.perf_counter() - started
        latencies.append(elapsed)
        tokens.append(used if status == 200 else 0)
        sizes.append(len(raw))
        statuses.append(status)
        await asyncio.sleep(DELAY)
    nq = [len(p.get("questions", {})) for p in payloads]
    ok = sum(1 for s in statuses if s == 200)
    print(
        f"  {model:18} {label:11} n={len(latencies):2d} ok={ok} nq={min(nq)}-{max(nq)} "
        f"body {min(sizes)/1024:5.1f}-{max(sizes)/1024:5.1f} KiB "
        f"lat p50={statistics.median(latencies):5.2f}s p90={pct(latencies,0.9):5.2f}s max={max(latencies):5.2f}s "
        f"| prov tok p50={statistics.median(tokens):,.0f} max={max(tokens):,} "
        f"| tok/s={sum(tokens)/max(sum(latencies),1e-9):,.0f}"
    )
    if headers:
        print(f"      rate-limit headers: {headers}")


async def main():
    academic = await capture_academic()
    enterprise = load_enterprise_payloads(6)
    print(f"captured {len(academic)} production-shaped academic payloads "
          f"(nq={min(len(p['questions']) for p in academic)}-{max(len(p['questions']) for p in academic)}), "
          f"{len(enterprise)} real enterprise payloads\n")
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180)) as session:
        for model in MODELS:
            await replay(session, model, academic[:6], "academic")
            await replay(session, model, enterprise, "enterprise")
            print()


if __name__ == "__main__":
    asyncio.run(main())
