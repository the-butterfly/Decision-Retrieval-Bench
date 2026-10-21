import asyncio, os, json, sys, time
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sag2 import JevClient  # noqa: E402

API = "https://decision.vllm-sr.ai/v1/"
KEY = os.getenv("VLLM_SR_KEY")
PAYLOAD = {
    "model": "vllm-sr/d3-turbo",
    "state": json.dumps({"original_question": "test question"}),
    "questions": {
        "c0": {
            "type": "noul",
            "instructions": {"question": "Is this necessary?", "candidate_title": "t", "candidate_passage": "p"},
            "criteria": {"true": "x", "false": "y"},
        }
    },
}


async def drive(rpm, duration):
    import aiohttp

    retry_after_samples = []
    async with aiohttp.ClientSession() as s:
        client = JevClient(s, API, "vllm-sr/d3-turbo", KEY, requests_per_minute=rpm)
        ok = fail = 0
        start = time.monotonic()
        i = 0
        while time.monotonic() - start < duration:
            i += 1
            try:
                await client.request(PAYLOAD)
                ok += 1
            except Exception:  # noqa: BLE001
                fail += 1
        dt = time.monotonic() - start
        # sample Retry-After directly for visibility
        try:
            async with s.post(
                client.endpoint, json=PAYLOAD,
                headers={"Authorization": f"Bearer {KEY}"},
            ) as r:
                if r.status == 429:
                    retry_after_samples.append(r.headers.get("Retry-After"))
        except Exception:  # noqa: BLE001
            pass
        print(f"RPM={rpm}: {i} attempts in {dt:.0f}s -> ok={ok} fail={fail} ({100*ok/max(1,i):.0f}% ok); Retry-After sample={retry_after_samples}")


async def main():
    # Ascending so lower-rate windows stay clean; 30s cooldown between to clear
    # any 24s provider penalty window.
    for rpm in (60, 100, 150, 200):
        await drive(rpm, 35)
        await asyncio.sleep(30)


asyncio.run(main())
