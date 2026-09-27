"""Core data types.

Everything in this system is expressed as a character span into a source
document. LegalBench-RAG grades at character granularity, so offsets are the
primary currency: a chunk that cannot say exactly where it came from is useless.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

Span = tuple[int, int]


def merge_spans(spans: list[Span], *, max_gap: int = 0) -> list[Span]:
    """Sort spans and merge those that overlap or sit within `max_gap` chars."""
    if not spans:
        return []
    ordered = sorted(spans)
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + max_gap:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


class Document(BaseModel):
    doc_id: str
    content: str

    def slice(self, span: Span) -> str:
        return self.content[span[0] : span[1]]


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    span: Span
    text: str
    # Index of this chunk within its document, used to walk to neighbours.
    ordinal: int = 0
    section: str | None = None

    @property
    def length(self) -> int:
        return self.span[1] - self.span[0]


class Scored(BaseModel):
    chunk: Chunk
    score: float


class RetrievedSnippet(BaseModel):
    file_path: str
    span: Span
    score: float

    @property
    def length(self) -> int:
        return self.span[1] - self.span[0]


class QueryResponse(BaseModel):
    query: str
    retrieved_snippets: list[RetrievedSnippet] = Field(default_factory=list)
    answer: str | None = None

    @property
    def total_chars(self) -> int:
        return sum(s.length for s in self.retrieved_snippets)


class Snippet(BaseModel):
    file_path: str
    span: Span


class QAGroundTruth(BaseModel):
    query: str
    snippets: list[Snippet]
    tags: list[str] = Field(default_factory=list)


class Benchmark(BaseModel):
    tests: list[QAGroundTruth]
