"""Client-side rate limiting.

OpenAI enforces both a request and a token budget per minute. Relying on 429s
and backoff alone wastes wall-clock time and can stall a long ingest, so the
client shapes its own traffic to stay just under the published ceiling.
"""

from __future__ import annotations

import asyncio
import time


class TokenBucket:
    def __init__(self, per_minute: int):
        self.capacity = float(per_minute)
        self.rate = per_minute / 60.0
        self._available = float(per_minute)
        self._updated = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, amount: float = 1.0) -> None:
        # A single request larger than the whole budget can never be satisfied;
        # let it through and allow the server's 429 handling to take over.
        amount = min(amount, self.capacity)
        while True:
            async with self._lock:
                now = time.monotonic()
                self._available = min(
                    self.capacity, self._available + (now - self._updated) * self.rate
                )
                self._updated = now
                if self._available >= amount:
                    self._available -= amount
                    return
                wait = (amount - self._available) / self.rate
            await asyncio.sleep(wait)


class RateLimiter:
    """Combined request-per-minute and token-per-minute budget."""

    def __init__(self, requests_per_minute: int, tokens_per_minute: int):
        self._requests = TokenBucket(requests_per_minute)
        self._tokens = TokenBucket(tokens_per_minute)

    async def acquire(self, tokens: int) -> None:
        await self._requests.acquire(1)
        await self._tokens.acquire(tokens)
