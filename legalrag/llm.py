"""Async OpenAI chat wrapper with structured output, retries and a response cache.

The cache is keyed by the full request, so repeating an evaluation run costs
nothing and stays byte-identical — which is what makes ablations comparable.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

from .config import Settings
from .ratelimit import RateLimiter


class ResponseCache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, body TEXT)")
        self._db.commit()

    def get(self, key: str) -> str | None:
        with self._lock:
            row = self._db.execute("SELECT body FROM responses WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def put(self, key: str, body: str) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO responses (key, body) VALUES (?, ?)", (key, body)
            )
            self._db.commit()


class LLM:
    def __init__(self, settings: Settings, model: str | None = None):
        self.model = model or settings.llm_model
        self._client = AsyncOpenAI(
            api_key=settings.openai_api_key.get_secret_value(),
            base_url=settings.openai_base_url,
            max_retries=0,
        )
        self._cache = ResponseCache(settings.cache_dir / "llm.sqlite")
        self._semaphore = asyncio.Semaphore(settings.llm_concurrency)
        self._limiter = RateLimiter(settings.llm_rpm, settings.llm_tpm)

    @retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(5), reraise=True)
    async def _call(self, payload: dict[str, Any], tokens: int) -> str:
        await self._limiter.acquire(tokens)
        async with self._semaphore:
            response = await self._client.chat.completions.create(**payload)
        return response.choices[0].message.content or ""

    async def complete(
        self,
        system: str,
        user: str,
        *,
        schema: dict[str, Any] | None = None,
        schema_name: str = "result",
        max_tokens: int = 1024,
        temperature: float | None = 0.0,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_completion_tokens": max_tokens,
        }
        # Reasoning models reject an explicit temperature; everything else wants 0.
        if temperature is not None and not self.model.startswith(("gpt-5", "o1", "o3", "o4")):
            payload["temperature"] = temperature
        if schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "strict": True, "schema": schema},
            }

        key = hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode()
        ).hexdigest()
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        # Four characters per token is close enough for budgeting purposes.
        estimated = (len(system) + len(user)) // 4 + max_tokens
        body = await self._call(payload, estimated)
        self._cache.put(key, body)
        return body

    async def complete_json(self, system: str, user: str, **kwargs: Any) -> Any:
        raw = await self.complete(system, user, **kwargs)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None
