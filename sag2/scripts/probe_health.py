"""Minimal one-shot health probe to the vLLM-SR /v1/systemone endpoint.

Sends a tiny 4-question noul payload and measures latency with a hard 25s cap.
Used to decide whether the live benchmark run can be safely killed + resumed
(fast recovery) or must be left to self-heal (provider degraded).
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

API = vals["VLLM_SR_API"].rstrip("/")
MODEL = vals["VLLM_SR_MODEL"]
KEY = vals["VLLM_SR_KEY"]
ENDPOINT = API + "/systemone"

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
        "criteria": {
            "true": "necessary evidence",
            "false": "not necessary",
        },
    }


questions = {
    f"candidate_{i}": mk_q(i, f"Doc {i}", f"The capital of France is Paris. Fact number {i}.")
    for i in range(4)
}
payload = {"model": MODEL, "state": state, "questions": questions}


async def main():
    t0 = time.monotonic()
    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=25)
        ) as s:
            async with s.post(
                ENDPOINT,
                json=payload,
                headers={"Authorization": f"Bearer {KEY}"},
            ) as r:
                status = r.status
                body = await r.read()
        dt = time.monotonic() - t0
        try:
            data = json.loads(body)
            answers = data.get("answers", {})
            ok = set(answers) == set(questions)
            print(
                f"HEALTH_OK status={status} latency={dt:.2f}s "
                f"answers={len(answers)}/{len(questions)} match={ok} "
                f"model={data.get('model')} usage={data.get('usage')}"
            )
        except Exception as e:
            print(f"HEALTH_PARSE_FAIL status={status} latency={dt:.2f}s err={e!r} body_head={body[:200]!r}")
    except Exception as e:
        dt = time.monotonic() - t0
        print(f"HEALTH_ERROR latency={dt:.2f}s err={type(e).__name__}: {e}")


if __name__ == "__main__":
    asyncio.run(main())
