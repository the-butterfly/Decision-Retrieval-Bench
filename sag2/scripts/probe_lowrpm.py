import asyncio, os, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sag2 import JevClient  # noqa: E402

API = "https://decision.vllm-sr.ai/v1/"
KEY = os.getenv("VLLM_SR_KEY")
QPER = 16  # match the real workload's lighter batches


def make_payload():
    questions = {}
    for i in range(QPER):
        questions[f"c{i}"] = {
            "type": "noul",
            "instructions": {
                "question": "Is this necessary evidence for the original question?",
                "candidate_title": f"t{i}",
                "candidate_passage": f"p{i} " * 200,  # realistic-ish length
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
        print(f"[16q] RPM={rpm}: {i} attempts in {dt:.0f}s -> ok={ok} fail={fail} "
              f"({100*ok/max(1,i):.0f}% ok)", flush=True)


async def main():
    # Find the CURRENT tolerable rate (provider has been volatile / stricter lately).
    for rpm in (20, 40):
        await drive(rpm, 75)
        await asyncio.sleep(20)


asyncio.run(main())
