"""Probe vLLM-SR: per-model limits (/api/info) + per-model scoring behaviour.

For each candidate model this reports:
  * availability (does the key get a 200 for a 16-question noul payload?)
  * latency, usage.input_tokens
  * the X-RateLimit-* headers (the provider's own per-minute budget accounting)
  * whether /v1/systemone and /v1/decisions return identical noul values
    (docs: /v1/systemone fails the whole request if one question fails,
     /v1/decisions answers each question in place)
"""
import asyncio
import json
import os
import time

import aiohttp

ENV = "/Volumes/Extension/WorkSpaces/WorkBuddy/SAG2-expr/code/.env"
vals = {}
with open(ENV) as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        vals[k.strip()] = v.strip()

API = os.getenv("VLLM_SR_API", vals["VLLM_SR_API"]).rstrip("/")
KEY = os.getenv("VLLM_SR_KEY", vals["VLLM_SR_KEY"])
MODELS = ["vllm-sr/d3-turbo", "vllm-sr/d3-flash", "vllm-sr/d3-mini"]

CRITERION = (
    "Question: is this candidate itself necessary evidence for the original question? "
    "Answer yes (1.0) only if the candidate supplies a required fact; otherwise no (0.0)."
)
state = json.dumps({"original_question": "What is the capital of France?"})


def mk_q(cid, title, passage):
    return {
        "type": "noul",
        "instructions": {
            "question": CRITERION,
            "candidate_title": title,
            "candidate_passage": passage,
        },
        "criteria": {"true": "necessary evidence", "false": "not necessary"},
    }


def build(n):
    qs = {
        f"candidate_{i}": mk_q(i, f"Doc {i}", f"The capital of France is Paris. Fact number {i}.")
        for i in range(n)
    }
    return {"state": state, "questions": qs}


def hdrs(r):
    keys = [
        "x-ratelimit-limit-requests",
        "x-ratelimit-remaining-requests",
        "x-ratelimit-limit-tokens",
        "x-ratelimit-remaining-tokens",
        "x-ratelimit-reset",
        "retry-after",
    ]
    return {k: r.headers.get(k) for k in keys if r.headers.get(k) is not None}


async def info(session):
    for url in (API + "/api/info", "https://decision.vllm-sr.ai/api/info"):
        try:
            async with session.get(url, headers={"Authorization": f"Bearer {KEY}"}) as r:
                body = await r.text()
                if r.status == 200:
                    print(f"=== {url} -> 200 ===")
                    print(json.dumps(json.loads(body), indent=2)[:4000])
                    return json.loads(body)
                print(f"{url} -> {r.status} {body[:200]}")
        except Exception as e:
            print(f"{url} -> ERR {type(e).__name__}: {e}")
    return None


async def probe(session, model, n, path):
    payload = build(n)
    payload["model"] = model
    url = API + path
    t0 = time.monotonic()
    try:
        async with session.post(url, json=payload, headers={"Authorization": f"Bearer {KEY}"}) as r:
            status, h = r.status, hdrs(r)
            body = await r.text()
    except Exception as e:
        return (model, path, n, f"ERR {type(e).__name__}: {e}", None, None, None)
    dt = time.monotonic() - t0
    try:
        data = json.loads(body)
        answers = data.get("answers", {})
        vals_ = {k: v.get("noul") for k, v in answers.items()}
        usage = data.get("usage", {})
        return (model, path, n, status, dt, (vals_, usage.get("input_tokens"), data.get("model")), h)
    except Exception:
        return (model, path, n, status, dt, body[:200], h)


async def main():
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as s:
        await info(s)
        print("\n=== per-model probes (16 questions) ===")
        results = {}
        for model in MODELS:
            for path in ("/systemone", "/decisions"):
                m, p, n, status, dt, vals_, h = await probe(s, model, n=16, path=path)
                print(f"\n-- {m} {p} nq={n} -> status={status} latency={dt if dt is None else round(dt,2)}s")
                print(f"   headers: {h}")
                if isinstance(vals_, tuple):
                    scores, itok, rmodel = vals_
                    print(f"   usage.input_tokens={itok} model={rmodel} n_answers={len(scores)}")
                    print(f"   noul sample: {list(scores.values())[:5]}")
                    results[(m, p)] = scores
                else:
                    print(f"   body: {vals_}")
        print("\n=== systemone vs decisions score equality ===")
        for model in MODELS:
            a = results.get((model, "/systemone"))
            b = results.get((model, "/decisions"))
            if a and b:
                same = a == b
                print(f"{model}: identical={same}" + ("" if same else f"  a={a} b={b}"))


if __name__ == "__main__":
    asyncio.run(main())
