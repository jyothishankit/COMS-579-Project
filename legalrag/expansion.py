"""Query expansion.

Benchmark questions are phrased in plain English ("does the agreement let the
receiving party keep copies?") while the corpus is written in contract register
("Recipient shall return or destroy all Confidential Material"). Expansion
closes that vocabulary gap before retrieval, not after.
"""

from __future__ import annotations

from .config import Settings
from .llm import LLM

_HYDE_SYSTEM = (
    "You draft the contract language that would answer a question. "
    "Reply with one or two sentences of plausible clause text in formal legal register. "
    "No preamble, no explanation, no markdown."
)

_MULTI_SYSTEM = (
    "You rewrite questions about legal contracts into search queries. "
    "Produce distinct phrasings that use the terminology a contract would actually use."
)

_MULTI_SCHEMA = {
    "type": "object",
    "properties": {
        "queries": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["queries"],
    "additionalProperties": False,
}


class QueryExpander:
    def __init__(self, settings: Settings, mode: str, n: int = 3, model: str | None = None):
        self.mode = mode
        self._n = n
        self._llm = LLM(settings, model=model) if mode != "none" else None

    async def expand(self, query: str) -> list[str]:
        """Return the query plus any generated variants. The original is always first."""
        if self._llm is None or self.mode == "none":
            return [query]

        if self.mode == "hyde":
            draft = await self._llm.complete(
                _HYDE_SYSTEM, f"Question: {query}", max_tokens=220
            )
            draft = draft.strip()
            return [query, draft] if draft else [query]

        if self.mode == "multi":
            result = await self._llm.complete_json(
                _MULTI_SYSTEM,
                f"Question: {query}\n\nWrite {self._n} alternative search queries.",
                schema=_MULTI_SCHEMA,
                schema_name="queries",
                max_tokens=400,
            )
            variants = [
                q.strip()
                for q in (result or {}).get("queries", [])
                if isinstance(q, str) and q.strip()
            ]
            return [query, *variants[: self._n]]

        raise ValueError(f"unknown expansion mode: {self.mode}")
