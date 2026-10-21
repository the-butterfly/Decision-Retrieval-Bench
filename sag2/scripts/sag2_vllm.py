"""vLLM-SR (d3-*) transport for SAG2: token-rate budgeting + per-model request shaping.

Authoritative limits: https://decision.vllm-sr.ai/docs#rate-limits  (and GET /api/info).

Why this module exists (the earlier request-rate limiter was wrong)
------------------------------------------------------------------
The provider meters **input tokens per minute per key per model**
(`usage.input_tokens`), not requests. d3-turbo allows 540 req/min but only
450,000 input tokens/min. A request-rate limiter therefore cannot reason about
the real budget: a 16-question request carries ~6k tokens, so 100 req/min is
~640k tokens/min (over budget) while 40 req/min is ~256k (under). This module
paces on *tokens* instead, and additionally:

* mirrors the provider's own per-request size formula so requests never trip
  `400 request_too_long` / `400 state_too_long`;
* trims the hop-1 `state` to the model's `max_state_chars`, keeping the
  highest-ranked evidence (d3-flash/d3-mini only allow 32,000 chars and count
  the state once *per question*; d3-turbo allows 131,072 and counts it once);
* caps in-flight requests (docs: 16 in flight per key);
* self-corrects from the server's own `X-RateLimit-*` headers, so the client
  follows the provider's real window instead of a guessed one.

The token budget is guarded by a `threading.Lock` (never held across an await),
so a single instance is safe to share across threads *and* event loops — the
per-question enterprise runner uses one event loop per thread, where an
`asyncio.Lock` would deadlock.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
import unicodedata
from collections import Counter
from threading import Lock, Semaphore

import aiohttp

from sag2 import JevClient, compact_json

# Per-key, per-model budgets and per-request limits (docs + /api/info, 2026-10-21).
MODEL_LIMITS = {
    "vllm-sr/d3-turbo": {
        "requests_per_minute": 540,
        "tokens_per_minute": 450_000,
        "requests_per_day": 648_000,
        "tokens_per_day": 540_000_000,
        "max_questions": 64,
        "max_state_chars": 131_072,
        "max_request_tokens": 131_072,
        "state_counted_once": True,
    },
    "vllm-sr/d3-flash": {
        "requests_per_minute": 720,
        "tokens_per_minute": 750_000,
        "requests_per_day": 432_000,
        "tokens_per_day": 450_000_000,
        "max_questions": 64,
        "max_state_chars": 32_000,
        "max_request_tokens": 131_072,
        "state_counted_once": False,
    },
    "vllm-sr/d3-mini": {
        "requests_per_minute": 1_260,
        "tokens_per_minute": 750_000,
        "requests_per_day": 756_000,
        "tokens_per_day": 450_000_000,
        "max_questions": 64,
        "max_state_chars": 32_000,
        "max_request_tokens": 131_072,
        "state_counted_once": False,
    },
}
DEFAULT_LIMITS = {
    "requests_per_minute": 540,
    "tokens_per_minute": 450_000,
    "requests_per_day": 648_000,
    "tokens_per_day": 540_000_000,
    "max_questions": 64,
    "max_state_chars": 32_000,
    "max_request_tokens": 131_072,
    "state_counted_once": False,
}

# Measured on the completed 600-question d3-turbo run: the provider's
# `usage.input_tokens` is 0.826x our tiktoken (o200k_base) estimate of the very
# same payload (83,474,443 actual vs 101,050,550 estimated over 13,339 batches;
# per-batch median 0.833). The provider's *documented* size formula is a
# deliberately conservative byte estimate and is ~4-17x larger than what it
# actually bills - it must size requests, never pace them.
PROVIDER_TOKEN_RATIO = 0.826


def _bytes(text: str) -> int:
    """Provider's byte measure: UTF-8 after Unicode NFC normalisation."""
    return len(unicodedata.normalize("NFC", text).encode("utf-8"))


def estimate_request_tokens(payload: dict, state_counted_once: bool) -> int:
    """Reproduce the provider's per-request input-token estimate.

    Per question: bytes(state) + bytes(question JSON) + 40.
    d3-turbo reads the state once, so its state bytes are charged once.
    """
    state_bytes = _bytes(payload.get("state", ""))
    total = 0
    for question in payload.get("questions", {}).values():
        total += _bytes(compact_json(question)) + 40
    total += 0 if state_counted_once else state_bytes * len(payload.get("questions", {}))
    if state_counted_once:
        total += state_bytes
    return total


def trim_state(state: str, max_chars: int, mode: str = "full") -> tuple[str, int]:
    """Shrink `previously_retrieved_evidence` until the state fits `max_chars`.

    Entries are kept in their existing (already JEV-ranked) order, so the
    surviving context is the highest-scoring evidence. With ``mode="titles"``
    each entry is first reduced to its title (and score), which keeps the
    "what has already been found" signal at a fraction of the byte cost -
    important because d3-flash/d3-mini are billed for the state *per question*.
    Returns (state, dropped).
    """
    if max_chars <= 0 or len(state) <= max_chars:
        return state, 0
    try:
        data = json.loads(state)
    except json.JSONDecodeError:
        return state[:max_chars], 0
    evidence = data.get("previously_retrieved_evidence")
    if not isinstance(evidence, list) or not evidence:
        return state[:max_chars], 0
    if mode == "titles":
        summarized = []
        for entry in evidence:
            if not isinstance(entry, dict):
                summarized.append(entry)
                continue
            item = {"title": entry.get("title", "")}
            for key in ("previous_jev_score", "jev_score"):
                if key in entry:
                    item["previous_jev_score"] = entry[key]
                    break
            summarized.append(item)
        evidence = summarized
        data["previously_retrieved_evidence"] = evidence
    kept: list = []
    for entry in evidence:
        candidate = kept + [entry]
        data["previously_retrieved_evidence"] = candidate
        if len(compact_json(data)) > max_chars:
            break
        kept = candidate
    data["previously_retrieved_evidence"] = kept
    return compact_json(data), len(evidence) - len(kept)


class TokenBudget:
    """Thread-safe per-minute budget for input tokens *and* requests.

    One instance must be shared by every client of a run: the provider meters
    the whole key, not the process. Accounting uses a `threading.Lock` that is
    never held across an await, so the same instance is safe across threads and
    across independent event loops.
    """

    def __init__(
        self,
        tokens_per_minute: float,
        requests_per_minute: float,
        *,
        safety: float = 0.95,
        burst_tokens: float | None = None,
    ):
        self.token_rate = max(tokens_per_minute, 0.0) * safety / 60.0
        self.request_rate = max(requests_per_minute, 0.0) * safety / 60.0
        self.token_capacity = float(
            burst_tokens if burst_tokens is not None else max(tokens_per_minute, 0.0) * safety
        )
        self.request_capacity = float(max(requests_per_minute, 0.0) * safety)
        self.tokens = self.token_capacity
        self.requests = self.request_capacity
        self.updated = time.monotonic()
        self._lock = Lock()
        self.observed_remaining_tokens: float | None = None
        self.observed_remaining_requests: float | None = None

    # -- accounting -------------------------------------------------------
    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self.updated
        self.updated = now
        self.tokens = min(self.token_capacity, self.tokens + elapsed * self.token_rate)
        self.requests = min(self.request_capacity, self.requests + elapsed * self.request_rate)

    def _wait_for(self, n_tokens: float, n_requests: float) -> float:
        with self._lock:
            self._refill()
            token_wait = (
                (n_tokens - self.tokens) / self.token_rate if self.token_rate > 0 else 60.0
            )
            request_wait = (
                (n_requests - self.requests) / self.request_rate
                if self.request_rate > 0
                else 60.0
            )
            wait = max(token_wait, request_wait, 0.0)
            if wait <= 0:
                self.tokens -= n_tokens
                self.requests -= n_requests
                return 0.0
        return min(wait, 5.0)

    def acquire_sync(self, n_tokens: float, n_requests: float = 1.0) -> None:
        while True:
            wait = self._wait_for(n_tokens, n_requests)
            if wait <= 0:
                return
            time.sleep(wait)

    async def acquire_async(self, n_tokens: float, n_requests: float = 1.0) -> None:
        while True:
            wait = self._wait_for(n_tokens, n_requests)
            if wait <= 0:
                return
            await asyncio.sleep(wait)

    def observe(self, headers) -> None:
        """Trust the provider's own window: only ever slow down, never speed up."""
        remaining_tokens = headers.get("X-RateLimit-Remaining-Tokens")
        remaining_requests = headers.get("X-RateLimit-Remaining-Requests")
        with self._lock:
            if remaining_tokens is not None:
                try:
                    self.observed_remaining_tokens = float(remaining_tokens)
                    self.tokens = min(self.tokens, self.observed_remaining_tokens)
                except ValueError:
                    pass
            if remaining_requests is not None:
                try:
                    self.observed_remaining_requests = float(remaining_requests)
                    self.requests = min(self.requests, self.observed_remaining_requests)
                except ValueError:
                    pass


class KeyRing:
    """Several API keys, each with its own per-minute budget, plus a shared cap.

    Measured on this account (2026-10-21): VLLM_SR_KEY / KEY2 / KEY3 each report
    their own full 450,000-token budget, so they are *independent* per key (the
    docs' "3 free keys from one address share one budget" rule does not apply
    here). The provider additionally caps tokens per minute **per IP address**,
    so the aggregate must be bounded too: 720,000/min for d3-turbo and
    1,080,000/min for d3-flash + d3-mini combined.
    """

    def __init__(
        self,
        keys: list[str],
        tokens_per_minute: float,
        requests_per_minute: float,
        *,
        aggregate_tokens_per_minute: float | None = None,
        aggregate_requests_per_minute: float | None = None,
        safety: float = 0.95,
    ):
        self.entries = [
            {
                "key": key,
                "budget": TokenBudget(
                    tokens_per_minute, requests_per_minute, safety=safety
                ),
            }
            for key in keys
        ]
        self.aggregate = (
            TokenBudget(
                aggregate_tokens_per_minute or tokens_per_minute,
                aggregate_requests_per_minute or requests_per_minute,
                safety=1.0,
            )
            if (aggregate_tokens_per_minute or tokens_per_minute)
            else None
        )
        self._lock = Lock()
        self._cursor = 0

    def pick(self) -> dict:
        """Pick the key with the most headroom (ties broken round-robin)."""
        with self._lock:
            best_index = max(
                range(len(self.entries)), key=lambda i: self.entries[i]["budget"].tokens
            )
            self._cursor = (self._cursor + 1) % len(self.entries)
            if self._cursor == best_index:
                pass
            else:
                # prefer the headroom leader, but rotate on ties
                leader_tokens = self.entries[best_index]["budget"].tokens
                tied = [
                    i
                    for i in range(len(self.entries))
                    if self.entries[i]["budget"].tokens >= leader_tokens - 1
                ]
                best_index = tied[self._cursor % len(tied)]
            return self.entries[best_index]

    def observe(self, entry: dict, headers) -> None:
        entry["budget"].observe(headers)
        if self.aggregate is not None:
            self.aggregate.observe(headers)


class VllmSrJevClient(JevClient):
    """JevClient that speaks the vLLM-SR transport contract."""

    def __init__(
        self,
        session,
        endpoint: str,
        model: str,
        api_key: str,
        *,
        budget: TokenBudget | None = None,
        api_keys: list[str] | None = None,
        keyring: "KeyRing | None" = None,
        tokens_per_minute: float | None = None,
        requests_per_minute: float | None = None,
        aggregate_tokens_per_minute: float | None = None,
        state_chars: int | None = None,
        state_mode: str = "full",
        max_inflight: int = 16,
        safety: float = 0.95,
        max_questions_per_request: int | None = None,
        **kwargs,
    ):
        limits = MODEL_LIMITS.get(model, DEFAULT_LIMITS)
        resolved_maxq = max_questions_per_request or limits["max_questions"]
        super().__init__(
            session,
            endpoint,
            model,
            api_key,
            max_questions_per_request=resolved_maxq,
            requests_per_minute=0,  # pacing is handled by the token budget
            **kwargs,
        )
        self.limits = limits
        per_key_tokens = tokens_per_minute or limits["tokens_per_minute"]
        per_key_requests = requests_per_minute or limits["requests_per_minute"]
        self.keyring = None
        if keyring is not None:
            # A shared ring (one instance for the whole run) is required when many
            # clients exist, e.g. the enterprise runner builds one client per question.
            self.keyring = keyring
            self.budget = self.keyring.aggregate or self.keyring.entries[0]["budget"]
        elif api_keys and len(api_keys) > 1:
            self.keyring = KeyRing(
                api_keys,
                per_key_tokens,
                per_key_requests,
                aggregate_tokens_per_minute=aggregate_tokens_per_minute,
                safety=safety,
            )
            self.budget = self.keyring.aggregate or self.keyring.entries[0]["budget"]
        else:
            self.budget = budget or TokenBudget(
                per_key_tokens, per_key_requests, safety=safety
            )
        # None -> only trim to the model's own limit (turbo keeps its full state).
        self.state_chars = state_chars
        self.state_mode = state_mode
        self.state_limit = limits["max_state_chars"] if state_chars is None else state_chars
        self._inflight = Semaphore(max_inflight)
        self.stats = {
            "requests": 0,
            "split_requests": 0,
            "trimmed_states": 0,
            "dropped_evidence": 0,
            "429_rate_limit_exceeded": 0,
            "429_server_overloaded": 0,
            "429_concurrency_limit": 0,
            "422_question_failed": 0,
            "bytes_trimmed": 0,
        }

    # -- shaping ----------------------------------------------------------
    def _shape(self, payload: dict) -> dict:
        shaped = dict(payload)
        state = shaped.get("state", "")
        limit = self.state_limit
        if limit and len(state) > limit:
            state, dropped = trim_state(state, limit, self.state_mode)
            shaped["state"] = state
            self.stats["trimmed_states"] += 1
            self.stats["dropped_evidence"] += dropped
        return shaped

    def estimate(self, payload: dict) -> int:
        """Provider's conservative per-request SIZE estimate (shaping only)."""
        return estimate_request_tokens(payload, self.limits["state_counted_once"])

    def billing_tokens(self, payload: dict) -> int:
        """Tokens the provider will actually charge (`usage.input_tokens`).

        This is what the per-minute token budget must be paced against; using
        the conservative size estimate here would throttle ~4-17x too hard.
        """
        encoding = self.encoding
        total = len(encoding.encode(payload.get("state", ""), disallowed_special=())) + 1024
        for question in payload.get("questions", {}).values():
            total += len(encoding.encode(compact_json(question), disallowed_special=())) + 64
        return max(1, int(total * PROVIDER_TOKEN_RATIO))

    @staticmethod
    def _error_code(body: str) -> str:
        try:
            return json.loads(body).get("error", {}).get("code", "")
        except Exception:
            return ""

    # -- transport --------------------------------------------------------
    async def request(self, payload: dict) -> dict:
        shaped = self._shape(payload)
        estimate = self.estimate(shaped)
        questions = shaped.get("questions", {})

        # Never exceed the provider's per-request size: split instead of 400.
        if estimate > self.limits["max_request_tokens"] and len(questions) > 1:
            keys = list(questions)
            middle = len(keys) // 2
            halves = [keys[:middle], keys[middle:]]
            self.stats["split_requests"] += 1
            merged = {"answers": {}, "usage": {"input_tokens": 0, "output_tokens": 0}, "model": self.model}
            for chunk in halves:
                part = dict(shaped)
                part["questions"] = {k: questions[k] for k in chunk}
                result = await self._send(part)
                merged["answers"].update(result.get("answers", {}))
                merged["usage"]["input_tokens"] += result.get("usage", {}).get("input_tokens", 0)
                merged["usage"]["output_tokens"] += result.get("usage", {}).get("output_tokens", 0)
                merged["model"] = result.get("model", self.model)
            merged["request_attempts"] = 1
            merged["http_attempts"] = 1
            merged["failed_attempts"] = {}
            return merged

        return await self._send(shaped)

    async def _send(self, payload: dict) -> dict:
        # Pace on what the provider will actually bill, not on its size estimate.
        tokens = self.billing_tokens(payload)
        entry = None
        if self.keyring is not None:
            entry = self.keyring.pick()
            await entry["budget"].acquire_async(tokens, 1.0)
            if self.keyring.aggregate is not None:
                await self.keyring.aggregate.acquire_async(tokens, 1.0)
            key = entry["key"]
        else:
            await self.budget.acquire_async(tokens, 1.0)
            key = self.api_key
        while not self._inflight.acquire(blocking=False):
            await asyncio.sleep(0.1)
        try:
            return await self._post(payload, key, entry)
        finally:
            self._inflight.release()

    async def _post(self, payload: dict, key: str, entry: dict | None) -> dict:
        failures: Counter = Counter()
        attempt = 0
        rate_retries = 0
        while True:
            attempt += 1
            if attempt > 120:
                raise RuntimeError(
                    f"VllmSrJevClient exceeded 120 attempts (failures={dict(failures)})"
                )
            status = None
            retry_after = None
            try:
                async with self.session.post(
                    self.endpoint,
                    json=payload,
                    headers={"Authorization": f"Bearer {key}"},
                ) as response:
                    status = response.status
                    retry_after = response.headers.get("Retry-After")
                    if self.keyring is not None and entry is not None:
                        self.keyring.observe(entry, response.headers)
                    else:
                        self.budget.observe(response.headers)
                    if status == 200:
                        data = await response.json()
                        data["request_attempts"] = 1
                        data["http_attempts"] = attempt
                        data["failed_attempts"] = dict(failures)
                        self.stats["requests"] += 1
                        return data
                    body = await response.text()
                    code = self._error_code(body)
                    if status == 422:
                        # One invalid question fails the whole request on /v1/systemone.
                        self.stats["422_question_failed"] += 1
                        raise ValueError(
                            f"vLLM-SR rejected the batch (422 {code}): {body[:300]}"
                        )
                    if status not in (408, 429, 499, 500, 502, 503, 504):
                        raise aiohttp.ClientError(
                            f"vLLM-SR HTTP {status} ({code}): {body[:300]}"
                        )
                    failures[f"http_{status}_{code or 'x'}"] += 1
                    if status == 429:
                        if code == "server_overloaded":
                            self.stats["429_server_overloaded"] += 1
                        elif code == "concurrency_limit":
                            self.stats["429_concurrency_limit"] += 1
                        else:
                            self.stats["429_rate_limit_exceeded"] += 1
                        rate_retries += 1
                        if rate_retries > 60:
                            raise aiohttp.ClientError(
                                f"vLLM-SR rate limited {rate_retries} times: {body[:300]}"
                            )
            except (TimeoutError, aiohttp.ClientConnectionError) as error:
                failures["timeout" if isinstance(error, TimeoutError) else "connection"] += 1
                if failures["timeout"] + failures["connection"] > 3:
                    raise
            delay = min(30.0, 2 ** min(attempt - 1, 5))
            if status == 429:
                if retry_after:
                    try:
                        delay = max(delay, float(retry_after))
                    except ValueError:
                        pass
                # Docs: server_overloaded => exponential backoff with jitter.
                delay += random.uniform(0.0, 1.5)
            await asyncio.sleep(delay)
