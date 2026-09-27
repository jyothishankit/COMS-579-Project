"""Sentence-level span refinement.

LegalBench-RAG measures precision as the fraction of *returned characters* that
fall inside a ground-truth span. A 500-character chunk containing a
40-character answer therefore scores 0.08 precision no matter how well it was
ranked. Refinement cuts each retrieved region down to the sentences that
actually answer the question, which is where most of the headroom over the
published baseline lives.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .config import RetrievalStrategy, Settings
from .embeddings import OpenAIEmbedder
from .llm import LLM
from .text import sentence_spans, trim_span
from .types import Document, RetrievedSnippet, Scored, Span, merge_spans

# Sentences closer than this are treated as one span: the few characters of
# whitespace between them cost less precision than a fragmented span list.
MERGE_GAP = 4


class Region:
    """A contiguous stretch of one document that survived reranking."""

    def __init__(self, doc_id: str, span: Span, score: float, rank: int):
        self.doc_id = doc_id
        self.span = span
        self.score = score
        self.rank = rank


def build_regions(
    scored: list[Scored], docs: dict[str, Document], context_chars: int
) -> list[Region]:
    """Merge reranked chunks into disjoint regions, padded by `context_chars`.

    Padding matters because ground-truth spans straddle chunk boundaries; without
    it, a correct retrieval can still clip half the answer.
    """
    by_doc: dict[str, list[tuple[Span, float, int]]] = {}
    for rank, item in enumerate(scored):
        doc = docs.get(item.chunk.doc_id)
        if doc is None:
            continue
        start = max(0, item.chunk.span[0] - context_chars)
        end = min(len(doc.content), item.chunk.span[1] + context_chars)
        by_doc.setdefault(item.chunk.doc_id, []).append(((start, end), item.score, rank))

    regions: list[Region] = []
    for doc_id, entries in by_doc.items():
        entries.sort(key=lambda e: e[0])
        current_span, current_score, current_rank = entries[0]
        for span, score, rank in entries[1:]:
            if span[0] <= current_span[1]:
                current_span = (current_span[0], max(current_span[1], span[1]))
                current_score = max(current_score, score)
                current_rank = min(current_rank, rank)
            else:
                regions.append(Region(doc_id, current_span, current_score, current_rank))
                current_span, current_score, current_rank = span, score, rank
        regions.append(Region(doc_id, current_span, current_score, current_rank))

    regions.sort(key=lambda r: r.rank)
    return regions


def enumerate_sentences(
    regions: list[Region], docs: dict[str, Document]
) -> list[tuple[int, Region, Span, str]]:
    """Flatten regions into globally numbered sentences."""
    sentences: list[tuple[int, Region, Span, str]] = []
    for region in regions:
        content = docs[region.doc_id].content
        for span in sentence_spans(content[region.span[0] : region.span[1]], region.span[0]):
            sentences.append((len(sentences), region, span, content[span[0] : span[1]]))
    return sentences


def spans_to_snippets(
    selected: list[tuple[str, Span, float]], docs: dict[str, Document]
) -> list[RetrievedSnippet]:
    """Merge selected spans per document into disjoint, whitespace-trimmed snippets."""
    by_doc: dict[str, list[tuple[Span, float]]] = {}
    for doc_id, span, score in selected:
        by_doc.setdefault(doc_id, []).append((span, score))

    snippets: list[RetrievedSnippet] = []
    for doc_id, entries in by_doc.items():
        merged = merge_spans([span for span, _ in entries], max_gap=MERGE_GAP)
        content = docs[doc_id].content
        for span in merged:
            trimmed = trim_span(content, span)
            if trimmed[1] <= trimmed[0]:
                continue
            overlapping = [
                score
                for other, score in entries
                if other[0] < span[1] and other[1] > span[0]
            ]
            snippets.append(
                RetrievedSnippet(
                    file_path=doc_id, span=trimmed, score=max(overlapping, default=0.0)
                )
            )

    snippets.sort(key=lambda s: -s.score)
    return snippets


class SpanRefiner(Protocol):
    async def refine(
        self, query: str, scored: list[Scored], docs: dict[str, Document]
    ) -> list[RetrievedSnippet]: ...


class NoopRefiner:
    """Return whole chunks, matching the published baseline's behaviour."""

    async def refine(
        self, query: str, scored: list[Scored], docs: dict[str, Document]
    ) -> list[RetrievedSnippet]:
        return [
            RetrievedSnippet(
                file_path=item.chunk.doc_id, span=item.chunk.span, score=1.0 / (rank + 1)
            )
            for rank, item in enumerate(scored)
        ]


class EmbeddingRefiner:
    """Score each sentence against the query embedding and keep the best.

    Cheap and fully deterministic, but it judges similarity rather than whether
    the sentence answers the question.
    """

    def __init__(self, embedder: OpenAIEmbedder, strategy: RetrievalStrategy):
        self._embedder = embedder
        self._strategy = strategy

    async def refine(
        self, query: str, scored: list[Scored], docs: dict[str, Document]
    ) -> list[RetrievedSnippet]:
        regions = build_regions(
            scored[: self._strategy.refine_chunks], docs, self._strategy.refine_context_chars
        )
        sentences = enumerate_sentences(regions, docs)
        if not sentences:
            return await NoopRefiner().refine(query, scored, docs)

        matrix = await self._embedder.embed([text for _, _, _, text in sentences])
        query_vector = await self._embedder.embed_one(query)
        scores = matrix @ query_vector

        threshold = max(self._strategy.refine_min_score, float(scores.max()) * 0.85)
        selected = [
            (region.doc_id, span, float(score))
            for (_, region, span, _), score in zip(sentences, scores)
            if score >= threshold
        ]
        if not selected:
            best = int(np.argmax(scores))
            _, region, span, _ = sentences[best]
            selected = [(region.doc_id, span, float(scores[best]))]
        return spans_to_snippets(selected, docs)


_REFINE_SYSTEM = (
    "You extract the exact supporting text from legal documents.\n"
    "You are given a question and numbered sentences drawn from candidate passages.\n"
    "Select every sentence that is part of the passage a lawyer would cite as the answer, "
    "and no others.\n"
    "Rules:\n"
    "- Include all contiguous sentences that make up a complete provision, including any "
    "sentence that a selected clause depends on to be readable.\n"
    "- Exclude headings, recitals, signature blocks and merely topical text.\n"
    "- If nothing answers the question, return the single closest sentence."
)

_REFINE_SCHEMA = {
    "type": "object",
    "properties": {
        "sentence_ids": {
            "type": "array",
            "items": {"type": "integer"},
            "description": "Ids of the sentences that make up the answer.",
        }
    },
    "required": ["sentence_ids"],
    "additionalProperties": False,
}

MAX_REFINE_CHARS = 24_000


class LLMRefiner:
    """Ask a model which sentences constitute the answer.

    A cross-encoder scores similarity; this stage judges sufficiency, which is
    the property the benchmark's character spans actually encode.
    """

    def __init__(self, settings: Settings, strategy: RetrievalStrategy, model: str | None = None):
        self._llm = LLM(settings, model=model)
        self._strategy = strategy
        self._pad = strategy.refine_pad_sentences

    async def refine(
        self, query: str, scored: list[Scored], docs: dict[str, Document]
    ) -> list[RetrievedSnippet]:
        regions = build_regions(
            scored[: self._strategy.refine_chunks], docs, self._strategy.refine_context_chars
        )
        sentences = enumerate_sentences(regions, docs)
        if not sentences:
            return await NoopRefiner().refine(query, scored, docs)

        lines: list[str] = []
        budget = MAX_REFINE_CHARS
        usable = 0
        for index, _, _, text in sentences:
            line = f"[{index}] {' '.join(text.split())}"
            if budget - len(line) < 0:
                break
            budget -= len(line)
            usable += 1
            lines.append(line)

        result = await self._llm.complete_json(
            _REFINE_SYSTEM,
            f"Question: {query}\n\nSentences:\n" + "\n".join(lines),
            schema=_REFINE_SCHEMA,
            schema_name="selection",
            max_tokens=2048,
        )
        chosen = sorted(
            {i for i in (result or {}).get("sentence_ids", []) if isinstance(i, int) and 0 <= i < usable}
        )
        if self._pad:
            padded: set[int] = set()
            for i in chosen:
                region = sentences[i][1]
                for j in range(i - self._pad, i + self._pad + 1):
                    if 0 <= j < usable and sentences[j][1] is region:
                        padded.add(j)
            chosen = sorted(padded)

        if not chosen:
            return await NoopRefiner().refine(query, scored[:1], docs)

        selected = [
            (sentences[i][1].doc_id, sentences[i][2], 1.0 / (sentences[i][1].rank + 1))
            for i in chosen
        ]
        return spans_to_snippets(selected, docs)


def build_refiner(
    strategy: RetrievalStrategy, settings: Settings, embedder: OpenAIEmbedder
) -> SpanRefiner:
    if strategy.refine == "none":
        return NoopRefiner()
    if strategy.refine == "embedding":
        return EmbeddingRefiner(embedder, strategy)
    if strategy.refine == "llm":
        return LLMRefiner(settings, strategy)
    raise ValueError(f"unknown refiner: {strategy.refine}")
