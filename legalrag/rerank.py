"""Cross-encoder reranking.

First-stage retrieval optimizes for recall over hundreds of candidates; the
reranker reads query and passage jointly and is what turns that pool into a
usable ordering.
"""

from __future__ import annotations

import asyncio
import json
from typing import Protocol

from .config import Settings
from .llm import LLM
from .types import Scored


class Reranker(Protocol):
    async def rerank(self, query: str, candidates: list[Scored], top_k: int) -> list[Scored]: ...


class NoopReranker:
    async def rerank(self, query: str, candidates: list[Scored], top_k: int) -> list[Scored]:
        return candidates[:top_k]


class FlashRankReranker:
    """Local ONNX cross-encoder. No network, no per-query cost."""

    def __init__(self, model_name: str = "ms-marco-MiniLM-L-12-v2", cache_dir: str | None = None):
        from flashrank import Ranker

        self._ranker = Ranker(model_name=model_name, cache_dir=cache_dir)
        self._lock = asyncio.Lock()

    async def rerank(self, query: str, candidates: list[Scored], top_k: int) -> list[Scored]:
        if not candidates:
            return []
        from flashrank import RerankRequest

        passages = [
            {"id": i, "text": c.chunk.text, "meta": {}} for i, c in enumerate(candidates)
        ]
        request = RerankRequest(query=query, passages=passages)
        # flashrank's session is not thread-safe, and it is CPU-bound, so run it
        # on a worker thread under a lock rather than blocking the event loop.
        async with self._lock:
            results = await asyncio.to_thread(self._ranker.rerank, request)
        ranked = [
            Scored(chunk=candidates[int(r["id"])].chunk, score=float(r["score"])) for r in results
        ]
        return ranked[:top_k]


_LISTWISE_SYSTEM = (
    "You rank passages from legal contracts by how directly they answer a question. "
    "Judge only whether the passage text itself contains the answer, not whether it "
    "is topically related."
)

_LISTWISE_SCHEMA = {
    "type": "object",
    "properties": {
        "ranking": {
            "type": "array",
            "items": {"type": "integer"},
            "description": "Passage ids ordered from most to least relevant.",
        }
    },
    "required": ["ranking"],
    "additionalProperties": False,
}


class LLMReranker:
    """Listwise reranking in windows, for when cross-encoder quality is the bottleneck."""

    def __init__(self, settings: Settings, window: int = 20, model: str | None = None):
        self._llm = LLM(settings, model=model)
        self._window = window

    async def _rank_window(self, query: str, window: list[tuple[int, Scored]]) -> list[int]:
        passages = "\n\n".join(
            f"[{i}] {scored.chunk.text[:1200]}" for i, scored in window
        )
        result = await self._llm.complete_json(
            _LISTWISE_SYSTEM,
            f"Question: {query}\n\nPassages:\n{passages}\n\n"
            f"Return every passage id, most relevant first.",
            schema=_LISTWISE_SCHEMA,
            schema_name="ranking",
            max_tokens=512,
        )
        valid = {i for i, _ in window}
        order: list[int] = []
        seen: set[int] = set()
        for i in (result or {}).get("ranking", []):
            if isinstance(i, int) and i in valid and i not in seen:
                seen.add(i)
                order.append(i)
        order.extend(i for i in valid if i not in seen)
        return order

    async def rerank(self, query: str, candidates: list[Scored], top_k: int) -> list[Scored]:
        if not candidates:
            return []
        indexed = list(enumerate(candidates))
        windows = [indexed[i : i + self._window] for i in range(0, len(indexed), self._window)]
        orders = await asyncio.gather(*(self._rank_window(query, w) for w in windows))

        # Interleave window orderings so earlier windows (better first-stage
        # scores) still dominate the head of the final list.
        merged: list[int] = []
        for rank in range(max((len(o) for o in orders), default=0)):
            for order in orders:
                if rank < len(order):
                    merged.append(order[rank])
        return [
            Scored(chunk=candidates[i].chunk, score=1.0 / (pos + 1))
            for pos, i in enumerate(merged[:top_k])
        ]


def build_reranker(kind: str, settings: Settings) -> Reranker:
    if kind == "none":
        return NoopReranker()
    if kind == "flashrank":
        return FlashRankReranker(cache_dir=str(settings.cache_dir / "flashrank"))
    if kind == "llm":
        return LLMReranker(settings)
    raise ValueError(f"unknown reranker: {kind}")
