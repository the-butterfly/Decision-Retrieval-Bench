"""Do VLLM_SR_KEY / KEY2 / KEY3 have independent per-minute budgets, or one shared set?

The docs say: "If one network address creates three or more free keys within 24
hours, all its free keys share one set of these budgets, so extra keys add no
capacity." With 3 keys on one IP that may apply. This probe issues one tiny
request per key and compares the `X-RateLimit-*` headers: if the keys are
independent, each reports its own full budget; if they share, they report the
same (depleted) remaining budget.

Run it while the benchmark is actively consuming the budget for the comparison
to be meaningful.
"""
import asyncio
import json
import os
from pathlib import Path

import aiohttp

ENV = Path("/Volumes/Extension/WorkSpaces/WorkBuddy/SAG2-expr/code/.env")
vals = {}
for line in ENV.read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        vals[k.strip()] = v.strip()

API = os.getenv("VLLM_SR_API", vals["VLLM_SR_API"]).rstrip("/")
MODEL = os.getenv("VLLM_SR_MODEL", vals["VLLM_SR_MODEL"])
KEYS = [(name, vals[name]) for name in ("VLLM_SR_KEY", "VLLM_SR_KEY2", "VLLM_SR_KEY3") if name in vals]

state = json.dumps({"original_question": "What is the capital of France?"})
q = {
    "type": "noul",
    "instructions": {
        "question": "Is this candidate necessary evidence for the original question?",
        "candidate_title": "Doc",
        "candidate_passage": "The capital of France is Paris.",
    },
    "criteria": {"true": "yes", "false": "no"},
}
payload = {"model": MODEL, "state": state, "questions": {"candidate_0": q}}

HEADERS = [
    "x-ratelimit-limit-requests",
    "x-ratelimit-remaining-requests",
    "x-ratelimit-limit-tokens",
    "x-ratelimit-remaining-tokens",
    "x-ratelimit-reset",
]


async def probe(session, name, key):
    try:
        async with session.post(
            API + "/systemone", json=payload, headers={"Authorization": f"Bearer {key}"}
        ) as r:
            body = await r.text()
            h = {k: r.headers.get(k) for k in HEADERS if r.headers.get(k) is not None}
            ok = r.status == 200
            return name, r.status, h, ok, (body[:120] if not ok else "")
    except Exception as error:  # noqa: BLE001
        return name, "ERR", {}, False, f"{type(error).__name__}: {error}"


async def main():
    print(f"endpoint={API}  model={MODEL}  keys={[n for n, _ in KEYS]}\n")
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as s:
        for name, key in KEYS:
            n, status, h, ok, extra = await probe(s, name, key)
            print(f"{n:14} status={status} {h}")
            if extra:
                print(f"               {extra}")
        print("\n(repeat the same probe ~20s later: if budgets are shared, the")
        print(" remaining values will have dropped together; if independent,")
        print(" each key keeps its own allowance.)")


if __name__ == "__main__":
    asyncio.run(main())
