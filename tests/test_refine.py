"""Span arithmetic in the refinement stage.

A bug here does not crash: it shifts offsets and quietly costs precision or
recall, which is exactly the kind of failure the benchmark would hide.
"""

from __future__ import annotations

import asyncio

import pytest

from legalrag.config import RetrievalStrategy
from legalrag.evaluation import assert_disjoint
from legalrag.refine import NoopRefiner, build_regions, enumerate_sentences, spans_to_snippets
from legalrag.types import Chunk, Document, Scored

CONTENT = (
    "Section 1. The Receiving Party shall hold the Confidential Information in strict "
    "confidence. It shall not disclose that information to any third party without prior "
    "written consent.\n\n"
    "Section 2. This Agreement shall be governed by the laws of the State of Delaware. "
    "Any dispute shall be resolved by binding arbitration in Wilmington.\n"
)

DOCS = {"nda.txt": Document(doc_id="nda.txt", content=CONTENT)}


def chunk(start: int, end: int, ordinal: int = 0) -> Chunk:
    return Chunk(
        chunk_id=f"nda.txt#{ordinal}",
        doc_id="nda.txt",
        span=(start, end),
        text=CONTENT[start:end],
        ordinal=ordinal,
    )


def test_regions_merge_adjacent_chunks() -> None:
    scored = [Scored(chunk=chunk(0, 100, 0), score=0.9), Scored(chunk=chunk(100, 200, 1), score=0.8)]
    regions = build_regions(scored, DOCS, context_chars=0)
    assert len(regions) == 1
    assert regions[0].span == (0, 200)
    assert regions[0].score == pytest.approx(0.9)


def test_regions_keep_separated_chunks_apart() -> None:
    scored = [Scored(chunk=chunk(0, 50, 0), score=0.9), Scored(chunk=chunk(200, 260, 3), score=0.8)]
    regions = build_regions(scored, DOCS, context_chars=0)
    assert [r.span for r in regions] == [(0, 50), (200, 260)]


def test_context_padding_is_clamped_to_the_document() -> None:
    scored = [Scored(chunk=chunk(10, 40, 0), score=1.0)]
    regions = build_regions(scored, DOCS, context_chars=1000)
    assert regions[0].span == (0, len(CONTENT))


def test_region_ordering_follows_rank_not_position() -> None:
    scored = [Scored(chunk=chunk(200, 260, 3), score=0.9), Scored(chunk=chunk(0, 50, 0), score=0.5)]
    regions = build_regions(scored, DOCS, context_chars=0)
    assert [r.span for r in regions] == [(200, 260), (0, 50)]


def test_enumerated_sentences_map_back_to_their_exact_text() -> None:
    regions = build_regions([Scored(chunk=chunk(0, len(CONTENT)), score=1.0)], DOCS, 0)
    sentences = enumerate_sentences(regions, DOCS)
    assert sentences
    for index, _region, span, text in sentences:
        assert CONTENT[span[0] : span[1]] == text
    assert [s[0] for s in sentences] == list(range(len(sentences)))


def test_snippets_are_disjoint_and_whitespace_trimmed() -> None:
    selected = [
        ("nda.txt", (0, 60), 1.0),
        ("nda.txt", (58, 120), 0.5),
        ("nda.txt", (300, 340), 0.25),
    ]
    snippets = spans_to_snippets(selected, DOCS)
    assert_disjoint(snippets)
    for snippet in snippets:
        text = CONTENT[snippet.span[0] : snippet.span[1]]
        assert text == text.strip()
    assert snippets[0].score == pytest.approx(1.0)


def test_snippets_are_ordered_by_score() -> None:
    selected = [("nda.txt", (200, 240), 0.2), ("nda.txt", (0, 40), 0.9)]
    snippets = spans_to_snippets(selected, DOCS)
    assert [s.span[0] for s in snippets] == [0, 200]


def test_noop_refiner_returns_the_chunk_spans_unchanged() -> None:
    scored = [Scored(chunk=chunk(0, 100, 0), score=0.9)]
    snippets = asyncio.run(NoopRefiner().refine("q", scored, DOCS))
    assert [s.span for s in snippets] == [(0, 100)]


def test_refinement_shrinks_what_a_whole_chunk_would_return() -> None:
    """The precision premise: a selected sentence is far smaller than its chunk."""
    whole = chunk(0, len(CONTENT))
    regions = build_regions([Scored(chunk=whole, score=1.0)], DOCS, 0)
    sentences = enumerate_sentences(regions, DOCS)
    one = spans_to_snippets([("nda.txt", sentences[0][2], 1.0)], DOCS)
    assert sum(s.length for s in one) < whole.length / 2


def test_strategy_is_hashable_and_serializable() -> None:
    strategy = RetrievalStrategy(name="x", refine="embedding")
    assert RetrievalStrategy.model_validate_json(strategy.key()) == strategy
