"""Grounded answer synthesis over retrieved spans."""

from __future__ import annotations

from .config import Settings
from .llm import LLM
from .types import Document, RetrievedSnippet

_SYSTEM = (
    "You answer questions about legal documents using only the provided excerpts.\n"
    "Cite the excerpt number in square brackets after each claim, like [2].\n"
    "If the excerpts do not contain the answer, say so plainly instead of inferring one.\n"
    "Be concise and use the document's own terminology."
)


class AnswerGenerator:
    def __init__(self, settings: Settings, model: str | None = None):
        self._llm = LLM(settings, model=model)

    async def answer(
        self, query: str, snippets: list[RetrievedSnippet], docs: dict[str, Document]
    ) -> str:
        if not snippets:
            return "No relevant passages were retrieved."

        excerpts = "\n\n".join(
            f"[{i + 1}] ({s.file_path}, chars {s.span[0]}-{s.span[1]})\n"
            f"{docs[s.file_path].content[s.span[0]:s.span[1]]}"
            for i, s in enumerate(snippets)
            if s.file_path in docs
        )
        return await self._llm.complete(
            _SYSTEM, f"Question: {query}\n\nExcerpts:\n{excerpts}", max_tokens=800
        )
