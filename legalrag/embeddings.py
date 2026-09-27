"""OpenAI embeddings with batching, bounded concurrency and a durable cache.

Re-embedding a corpus on every ablation would dominate both wall-clock time and
cost, so every vector is cached on disk keyed by (model, text).
"""

from __future__ import annotations

import asyncio
import hashlib
import sqlite3
import threading
from pathlib import Path

import numpy as np
import tiktoken
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

from .config import Settings
from .ratelimit import RateLimiter

MAX_INPUTS_PER_REQUEST = 1024
MAX_TOKENS_PER_INPUT = 8_000


class EmbeddingCache:
    """SQLite-backed vector cache. Thread-safe, process-durable."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.execute("CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, vec BLOB)")
        self._db.commit()

    @staticmethod
    def make_key(model: str, text: str, dim: int) -> str:
        return hashlib.sha256(f"{model}\x00{dim}\x00{text}".encode()).hexdigest()

    def get_many(self, keys: list[str]) -> dict[str, np.ndarray]:
        found: dict[str, np.ndarray] = {}
        with self._lock:
            for i in range(0, len(keys), 500):
                batch = keys[i : i + 500]
                placeholders = ",".join("?" * len(batch))
                rows = self._db.execute(
                    f"SELECT key, vec FROM vectors WHERE key IN ({placeholders})", batch
                ).fetchall()
                for key, blob in rows:
                    found[key] = np.frombuffer(blob, dtype=np.float32)
        return found

    def put_many(self, items: list[tuple[str, np.ndarray]]) -> None:
        with self._lock:
            self._db.executemany(
                "INSERT OR REPLACE INTO vectors (key, vec) VALUES (?, ?)",
                [(key, vec.astype(np.float32).tobytes()) for key, vec in items],
            )
            self._db.commit()


class OpenAIEmbedder:
    def __init__(self, settings: Settings, model: str | None = None, dim: int | None = None):
        self.model = model or settings.embedding_model
        # text-embedding-3-* are Matryoshka models: a shorter vector keeps almost
        # all of the quality while cutting index memory proportionally.
        self.dim = dim or settings.embedding_dim
        self._client = AsyncOpenAI(
            api_key=settings.openai_api_key.get_secret_value(),
            base_url=settings.openai_base_url,
            max_retries=0,
        )
        self._cache = EmbeddingCache(settings.cache_dir / "embeddings.sqlite")
        self._semaphore = asyncio.Semaphore(settings.embed_concurrency)
        self._limiter = RateLimiter(settings.embed_rpm, settings.embed_tpm)
        # A single request must fit inside the per-minute token budget, with
        # headroom for requests already in flight.
        self._max_tokens_per_request = max(1_000, settings.embed_tpm // 4)
        try:
            self._encoding = tiktoken.encoding_for_model(self.model)
        except KeyError:
            self._encoding = tiktoken.get_encoding("cl100k_base")

    def _truncate(self, text: str) -> str:
        tokens = self._encoding.encode(text, disallowed_special=())
        if len(tokens) <= MAX_TOKENS_PER_INPUT:
            return text
        return self._encoding.decode(tokens[:MAX_TOKENS_PER_INPUT])

    def _token_count(self, text: str) -> int:
        return len(self._encoding.encode(text, disallowed_special=()))

    @retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(6), reraise=True)
    async def _embed_batch(self, texts: list[str], tokens: int) -> list[np.ndarray]:
        kwargs = {"model": self.model, "input": texts}
        if self.model.startswith("text-embedding-3"):
            kwargs["dimensions"] = self.dim
        await self._limiter.acquire(tokens)
        async with self._semaphore:
            response = await self._client.embeddings.create(**kwargs)
        return [np.asarray(item.embedding, dtype=np.float32) for item in response.data]

    async def embed(self, texts: list[str], *, progress=None) -> np.ndarray:
        """Return an (n, dim) float32 matrix of L2-normalized embeddings."""
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        keys = [EmbeddingCache.make_key(self.model, t, self.dim) for t in texts]
        cached = self._cache.get_many(list(dict.fromkeys(keys)))

        missing_order: list[str] = []
        missing_texts: list[str] = []
        seen: set[str] = set()
        for key, text in zip(keys, texts):
            if key in cached or key in seen:
                continue
            seen.add(key)
            missing_order.append(key)
            missing_texts.append(self._truncate(text))

        if missing_texts:
            batches: list[tuple[list[str], list[str], int]] = []
            current_keys: list[str] = []
            current_texts: list[str] = []
            current_tokens = 0
            for key, text in zip(missing_order, missing_texts):
                tokens = self._token_count(text)
                over_limit = (
                    len(current_texts) >= MAX_INPUTS_PER_REQUEST
                    or current_tokens + tokens > self._max_tokens_per_request
                )
                if current_texts and over_limit:
                    batches.append((current_keys, current_texts, current_tokens))
                    current_keys, current_texts, current_tokens = [], [], 0
                current_keys.append(key)
                current_texts.append(text)
                current_tokens += tokens
            if current_texts:
                batches.append((current_keys, current_texts, current_tokens))

            async def run(batch_keys: list[str], batch_texts: list[str], tokens: int) -> None:
                vectors = await self._embed_batch(batch_texts, tokens)
                self._cache.put_many(list(zip(batch_keys, vectors)))
                cached.update(dict(zip(batch_keys, vectors)))
                if progress:
                    progress(len(batch_texts))

            await asyncio.gather(*(run(*batch) for batch in batches))

        matrix = np.stack([cached[key] for key in keys])
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return (matrix / np.clip(norms, 1e-12, None)).astype(np.float32)

    async def embed_one(self, text: str) -> np.ndarray:
        return (await self.embed([text]))[0]
