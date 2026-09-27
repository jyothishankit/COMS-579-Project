"""Answer questions over a set of PDFs.

This is the entry point used by the Funix UI (`upload.py`) and is a thin adapter
over the retrieval engine in `legalrag/`.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from legalrag import RagEngine, RetrievalStrategy, get_settings
from legalrag.ingest import load_any
from legalrag.types import QueryResponse


async def answer_async(
    pdf_paths: list[str | Path], question: str, strategy: RetrievalStrategy | None = None
) -> QueryResponse:
    settings = get_settings()
    docs = [load_any(path, settings.cache_dir / "pdf") for path in pdf_paths]

    engine = RagEngine(settings, strategy or RetrievalStrategy())
    try:
        await engine.index(docs)
        return await engine.query(question)
    finally:
        engine.close()


def answer(pdf_paths: list[str | Path], question: str) -> QueryResponse:
    return asyncio.run(answer_async(pdf_paths, question))


def upload(pdf_file_1: str, pdf_file_2: str, pdf_file_3: str, question: str) -> str:
    """Backwards-compatible entry point for the original assignment demo."""
    response = answer([pdf_file_1, pdf_file_2, pdf_file_3], question)
    citations = "\n".join(
        f"- {s.file_path} [{s.span[0]}:{s.span[1]}]" for s in response.retrieved_snippets
    )
    return f"{response.answer}\n\nSources:\n{citations}"


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("usage: python main.py <question> <file.pdf> [more.pdf ...]")
        raise SystemExit(1)
    result = answer(sys.argv[2:], sys.argv[1])
    print(result.answer)
