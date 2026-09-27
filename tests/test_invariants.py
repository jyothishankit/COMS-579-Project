"""Invariants that the benchmark's character-level scoring depends on."""

from __future__ import annotations

import random
import string

import pytest

from legalrag.chunking import chunk_document
from legalrag.evaluation import assert_disjoint, precision_recall
from legalrag.stores import BM25Index, rrf_fuse
from legalrag.text import sentence_spans, trim_span
from legalrag.types import Chunk, Document, RetrievedSnippet, Snippet, merge_spans

SAMPLE = """ARTICLE 1. DEFINITIONS

1.1 "Confidential Information" means all information disclosed by Discloser to
Recipient, whether orally or in writing. For clarity, Acme Inc. and its
affiliates are Discloser hereunder.

1.2 Term. This Agreement commences on the Effective Date and continues for
three (3) years.

SECTION 2. OBLIGATIONS

(a) Recipient shall not disclose Confidential Information to any third party.
(b) Recipient shall return or destroy all copies upon request.
"""


def random_document(seed: int) -> Document:
    rng = random.Random(seed)
    alphabet = string.ascii_letters + " \n.,;:!?()" + "\n\n"
    content = "".join(rng.choice(alphabet) for _ in range(rng.randint(200, 6000)))
    return Document(doc_id=f"doc-{seed}", content=content)


@pytest.mark.parametrize("strategy", ["naive", "rcts", "structural"])
@pytest.mark.parametrize("chunk_size", [100, 500, 1000])
def test_chunks_tile_the_document_exactly(strategy: str, chunk_size: int) -> None:
    for seed in range(12):
        doc = random_document(seed)
        chunks = chunk_document(doc, strategy, chunk_size)
        assert "".join(c.text for c in chunks) == doc.content
        assert chunks[0].span[0] == 0
        assert chunks[-1].span[1] == len(doc.content)
        for previous, nxt in zip(chunks, chunks[1:]):
            assert previous.span[1] == nxt.span[0]


@pytest.mark.parametrize("strategy", ["rcts", "structural"])
def test_chunk_text_matches_its_span(strategy: str) -> None:
    doc = Document(doc_id="sample.txt", content=SAMPLE)
    for chunk in chunk_document(doc, strategy, 300):
        assert chunk.text == doc.content[chunk.span[0] : chunk.span[1]]


def test_structural_chunking_respects_clause_boundaries() -> None:
    doc = Document(doc_id="sample.txt", content=SAMPLE)
    starts = [c.text.lstrip()[:12] for c in chunk_document(doc, "structural", 80)]
    assert any(s.startswith("1.1") for s in starts)
    assert any(s.startswith("(a)") for s in starts)
    assert any(s.startswith("SECTION 2") for s in starts)


def test_sentence_spans_tile_their_input() -> None:
    for seed in range(12):
        doc = random_document(seed)
        spans = sentence_spans(doc.content)
        if not spans:
            continue
        assert spans[0][0] == 0
        assert spans[-1][1] == len(doc.content)
        for previous, nxt in zip(spans, spans[1:]):
            assert previous[1] == nxt[0]


def test_sentence_splitter_keeps_legal_abbreviations_intact() -> None:
    text = 'Acme Inc. and Beta Ltd. are parties. Section 4.1 applies.'
    spans = sentence_spans(text, min_len=5)
    assert text[spans[0][0] : spans[0][1]].strip() == "Acme Inc. and Beta Ltd. are parties."


def test_trim_span_removes_whitespace_only_edges() -> None:
    text = "   hello world   "
    assert trim_span(text, (0, len(text))) == (3, 14)


def test_trim_span_clamps_out_of_range_input() -> None:
    text = "short"
    assert trim_span(text, (0, 999)) == (0, 5)
    assert trim_span(text, (900, 999)) == (5, 5)


def test_merge_spans_bridges_gaps() -> None:
    assert merge_spans([(0, 5), (5, 9), (20, 25)]) == [(0, 9), (20, 25)]
    assert merge_spans([(0, 5), (7, 9)], max_gap=2) == [(0, 9)]


def test_precision_recall_matches_hand_computed_overlap() -> None:
    retrieved = [RetrievedSnippet(file_path="a", span=(0, 100), score=1.0)]
    truth = [Snippet(file_path="a", span=(50, 150))]
    precision, recall = precision_recall(retrieved, truth)
    assert precision == pytest.approx(0.5)
    assert recall == pytest.approx(0.5)


def test_precision_recall_ignores_other_documents() -> None:
    retrieved = [RetrievedSnippet(file_path="b", span=(0, 100), score=1.0)]
    truth = [Snippet(file_path="a", span=(0, 100))]
    assert precision_recall(retrieved, truth) == (0.0, 0.0)


def test_overlapping_snippets_are_rejected() -> None:
    overlapping = [
        RetrievedSnippet(file_path="a", span=(0, 100), score=1.0),
        RetrievedSnippet(file_path="a", span=(50, 150), score=0.5),
    ]
    with pytest.raises(ValueError):
        assert_disjoint(overlapping)
    assert_disjoint([RetrievedSnippet(file_path="a", span=(0, 50), score=1.0)])


def test_bm25_ranks_the_matching_chunk_first() -> None:
    texts = [
        "The Receiving Party shall return all Confidential Information.",
        "This Agreement is governed by the laws of the State of New York.",
        "Employees are entitled to twenty days of paid leave.",
    ]
    chunks = [
        Chunk(chunk_id=str(i), doc_id="d", span=(i, i + 1), text=t, ordinal=i)
        for i, t in enumerate(texts)
    ]
    index = BM25Index()
    index.build(chunks)
    assert index.search("governing law New York", 3)[0][0] == 1
    assert index.search("unrelated vocabulary xyzzy", 3) == []


def test_corpus_contains_non_ascii_document_ids() -> None:
    """Vector ids must be ASCII, so document identity cannot be used as the id."""
    doc_id = "maud/TIFFANY_&_CO._LVMH_MOËT_HENNESSY-LOUIS_VUITTON.txt"
    chunk = Chunk(chunk_id=f"{doc_id}#0", doc_id=doc_id, span=(0, 4), text="test", ordinal=0)
    assert not chunk.chunk_id.isascii()


def test_rrf_rewards_agreement_between_rankings() -> None:
    dense = [(1, 0.9), (2, 0.8), (3, 0.7)]
    lexical = [(3, 5.0), (1, 4.0)]
    fused = dict(rrf_fuse([dense, lexical], [1.0, 1.0]))
    assert max(fused, key=lambda k: fused[k]) == 1
    assert fused[3] > fused[2]
