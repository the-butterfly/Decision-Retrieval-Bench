"""End-to-end smoke test of the vLLM-SR transport (sag2_vllm.py) on real questions.

Runs the genuine SAG2 search path (2 hops, candidate_k=100) against the live
vLLM-SR endpoint for each model, and reports:
  * requests / judgments / provider-reported input tokens
  * how the state was trimmed and how much evidence was dropped
  * whether any request had to be split to stay under the 131,072-token limit
  * our own size estimate vs. the provider's accounting

Protocol under test: state_chars=6000, max_questions_per_request=10.
"""
import asyncio
import json
import sys
import time
from pathlib import Path

import aiohttp

ROOT = Path("/Volumes/Extension/WorkSpaces/WorkBuddy/SAG2-expr/code/SAG-Benchmark/sag2")
sys.path.insert(0, str(ROOT / "scripts"))

from enterprise_index import ReadOnlyIndex  # noqa: E402
from sag2 import search_sag2  # noqa: E402
from sag2_vllm import MODEL_LIMITS, TokenBudget, VllmSrJevClient  # noqa: E402

VALS = {}
for line in (Path("/Volumes/Extension/WorkSpaces/WorkBuddy/SAG2-expr/code/.env")).read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        VALS[k.strip()] = v.strip()

API = VALS["VLLM_SR_API"].rstrip("/")
KEY = VALS["VLLM_SR_KEY"]
ENDPOINT = API + "/systemone"
STATE_CHARS = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
MAXQ = int(sys.argv[2]) if len(sys.argv) > 2 else 10
N_QUESTIONS = int(sys.argv[3]) if len(sys.argv) > 3 else 2

build = json.loads((ROOT / "output/enterprise/index/summary.json").read_text())
questions = [
    json.loads(line)
    for line in Path(
        "/Volumes/Extension/WorkSpaces/WorkBuddy/SAG2-expr/datasets/enterprise-rag-bench/subset/questions.jsonl"
    ).read_text().splitlines()
    if json.loads(line).get("expected_doc_ids")
][:N_QUESTIONS]


async def run_model(session, model):
    limits = MODEL_LIMITS[model]
    budget = TokenBudget(limits["tokens_per_minute"], limits["requests_per_minute"], safety=0.95)
    client = VllmSrJevClient(
        session,
        ENDPOINT,
        model,
        KEY,
        budget=budget,
        state_chars=STATE_CHARS,
        max_questions_per_request=MAXQ,
        max_inflight=16,
    )
    index = ReadOnlyIndex(
        Path(build["index_path"]), build["index_signature"], Path(build["source_directory"])
    )
    started = time.perf_counter()
    total_judgments = 0
    total_tokens = 0
    total_requests = 0
    recalls = []
    try:
        for question in questions:
            result = await search_sag2(
                index,
                client,
                question["question"],
                candidate_k=100,
                seed_k=10,
                max_hops=2,
                expansion_term_limit=20,
            )
            usage = result.get("jev_usage", {})
            total_judgments += usage.get("judgments", 0)
            total_tokens += usage.get("input_tokens", 0)
            total_requests += usage.get("requests", 0)
            gold = set(question["expected_doc_ids"])
            got = {c["source_id"] for c in result["candidates"][:10]}
            recalls.append(len(gold & got) / len(gold))
    finally:
        index.close()
    elapsed = time.perf_counter() - started
    stats = dict(client.stats)
    print(
        f"\n=== {model} ===\n"
        f"  questions={len(questions)}  requests={total_requests}  judgments={total_judgments}\n"
        f"  provider input_tokens={total_tokens:,}  ({total_tokens / max(1,total_requests):,.0f}/req)\n"
        f"  elapsed={elapsed:.1f}s  R@10(sample)={sum(recalls)/len(recalls):.3f}\n"
        f"  client stats: {stats}\n"
        f"  observed remaining tokens (last window)={budget.observed_remaining_tokens}"
    )


async def main():
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as session:
        for model in ("vllm-sr/d3-turbo", "vllm-sr/d3-flash", "vllm-sr/d3-mini"):
            try:
                await run_model(session, model)
            except Exception as error:  # noqa: BLE001
                print(f"\n=== {model} === FAILED: {type(error).__name__}: {error}")


if __name__ == "__main__":
    asyncio.run(main())
