import asyncio, os, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sag2 import JevClient  # noqa: E402

API = "https://decision.vllm-sr.ai/v1/"
KEY = os.getenv("VLLM_SR_KEY")
QPER = 64  # heavy payload like the real run


def make_payload():
    questions = {}
    for i in range(QPER):
        questions[f"c{i}"] = {
            "type": "noul",
            "instructions": {
                "question": "Is this necessary evidence for the original question?",
                "candidate_title": f"t{i}",
                "candidate_passage": f"p{i} " * 20,
            },
            "criteria": {"true": "x", "false": "y"},
        }
    return {
        "model": "vllm-sr/d3-turbo",
        "state": json.dumps({"original_question": "What is the capital of France?"}),
        "questions": questions,
    }


async def drive(rpm, duration):
    import aiohttp

    async with aiohttp.ClientSession() as s:
        client = JevClient(
            s, API, "vllm-sr/d3-turbo", KEY,
            requests_per_minute=rpm, max_questions_per_request=QPER,
        )
        ok = fail = 0
        start = time.monotonic()
        i = 0
        while time.monotonic() - start < duration:
            i += 1
            try:
                await client.request(make_payload())
                ok += 1
            except Exception:  # noqa: BLE001
                fail += 1
        dt = time.monotonic() - start
        qpm = (ok + fail) * QPER / (dt / 60.0)
        print(f"[64q] RPM={rpm}: {i} attempts in {dt:.0f}s -> ok={ok} fail={fail} "
              f"({100*ok/max(1,i):.0f}% ok); ~{qpm:.0f} questions/min", flush=True)


async def main():
    # Test heavy 64-question payloads at increasing RPM to find the ceiling.
    # 90s each so enough requests accumulate to represent the per-minute rate;
    # 30s cooldown clears any provider penalty window between tests.
    for rpm in (20, 40, 60, 90):
        await drive(rpm, 90)
        await asyncio.sleep(30)


asyncio.run(main())
