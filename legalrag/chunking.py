"""Chunking strategies, all of which tile a document exactly.

Exact tiling is a hard invariant: concatenating every chunk must reproduce the
document byte for byte. That is what lets a chunk carry a trustworthy character
span, and it is what the benchmark grades against.
"""

from __future__ import annotations

import re
from typing import Callable

from .types import Chunk, Document, Span

# The separator ladder used by the published LegalBench-RAG baseline.
RCTS_SEPARATORS = ["\n\n", "\n", "!", "?", ".", ":", ";", ",", " ", ""]

# Headings that signal a real clause boundary in contracts and policies.
_HEADING = re.compile(
    r"""^[ \t]*(
        (?:ARTICLE|Article|SECTION|Section|EXHIBIT|Exhibit|SCHEDULE|Schedule
          |ANNEX|Annex|APPENDIX|Appendix|PART|Part)\s+[0-9IVXLCivxlc]+
        | \d+(?:\.\d+){0,4}\.?[ \t]+\S
        | \([a-zA-Z0-9]{1,4}\)[ \t]+\S
        | [A-Z][A-Z0-9 ,'&/\-]{6,70}\s*$
    )""",
    re.MULTILINE | re.VERBOSE,
)


def _verify_tiling(doc: Document, spans: list[Span]) -> None:
    if not spans:
        raise ValueError(f"chunker produced no spans for {doc.doc_id}")
    if spans[0][0] != 0 or spans[-1][1] != len(doc.content):
        raise ValueError(f"chunks do not cover {doc.doc_id}")
    for (_, prev_end), (start, _) in zip(spans, spans[1:]):
        if prev_end != start:
            raise ValueError(f"chunks are not contiguous in {doc.doc_id}")


def _to_chunks(doc: Document, spans: list[Span], sections: list[str | None] | None = None) -> list[Chunk]:
    _verify_tiling(doc, spans)
    return [
        Chunk(
            chunk_id=f"{doc.doc_id}#{i}",
            doc_id=doc.doc_id,
            span=span,
            text=doc.content[span[0] : span[1]],
            ordinal=i,
            section=sections[i] if sections else None,
        )
        for i, span in enumerate(spans)
    ]


def rcts_spans(text: str, chunk_size: int) -> list[Span]:
    """Recursive character splitting, byte-identical to the published baseline.

    The baseline is the thing this project claims to beat, so it uses the exact
    splitter the paper used rather than a reimplementation that is merely close.
    `strip_whitespace=False` and zero overlap are what make the output tile the
    document, which is how spans are recovered from lengths alone.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        separators=RCTS_SEPARATORS,
        chunk_size=chunk_size,
        chunk_overlap=0,
        length_function=len,
        is_separator_regex=False,
        strip_whitespace=False,
    )
    spans: list[Span] = []
    cursor = 0
    for piece in splitter.split_text(text):
        spans.append((cursor, cursor + len(piece)))
        cursor += len(piece)
    return spans


def naive_spans(text: str, chunk_size: int, offset: int = 0) -> list[Span]:
    return [
        (offset + i, offset + min(i + chunk_size, len(text)))
        for i in range(0, len(text), chunk_size)
    ]


def recursive_spans(
    text: str, chunk_size: int, offset: int = 0, separators: list[str] | None = None
) -> list[Span]:
    """Recursive character splitting that returns spans instead of strings.

    Splits on the highest-priority separator present, packs the resulting pieces
    greedily up to `chunk_size`, and recurses into any piece still too large.
    """
    separators = separators if separators is not None else RCTS_SEPARATORS
    if len(text) <= chunk_size:
        return [(offset, offset + len(text))] if text else []

    separator = ""
    rest = separators
    for i, candidate in enumerate(separators):
        if candidate == "" or candidate in text:
            separator = candidate
            rest = separators[i + 1 :]
            break

    if separator == "":
        return naive_spans(text, chunk_size, offset)

    # Keep the separator attached to the piece it follows, so pieces still tile.
    pieces: list[Span] = []
    start = 0
    for match in re.finditer(re.escape(separator), text):
        end = match.end()
        pieces.append((start, end))
        start = end
    if start < len(text):
        pieces.append((start, len(text)))

    # Everything below stays in coordinates local to `text`; `offset` is applied
    # once, on return.
    spans: list[Span] = []
    buffer: Span | None = None
    for piece in pieces:
        size = piece[1] - piece[0]
        if size > chunk_size:
            if buffer:
                spans.append(buffer)
                buffer = None
            spans.extend(
                (piece[0] + a, piece[0] + b)
                for a, b in recursive_spans(text[piece[0] : piece[1]], chunk_size, 0, rest)
            )
        elif buffer is None:
            buffer = piece
        elif piece[1] - buffer[0] <= chunk_size:
            buffer = (buffer[0], piece[1])
        else:
            spans.append(buffer)
            buffer = piece
    if buffer:
        spans.append(buffer)

    return [(offset + a, offset + b) for a, b in spans]


def _section_label(text: str) -> str | None:
    line = text.strip().split("\n", 1)[0].strip()
    return line[:120] or None


def structural_spans(text: str, chunk_size: int) -> tuple[list[Span], list[str | None]]:
    """Split on legal structure first, then fall back to recursive splitting.

    Retrieving a whole clause beats retrieving 500 characters that straddle two
    unrelated obligations, and ground-truth snippets in this benchmark are
    clause-shaped.
    """
    boundaries = sorted({0, len(text)} | {m.start() for m in _HEADING.finditer(text)})
    segments = [(a, b) for a, b in zip(boundaries, boundaries[1:]) if b > a]

    # Pack consecutive short sections together; split long ones.
    spans: list[Span] = []
    buffer: Span | None = None
    for seg in segments:
        size = seg[1] - seg[0]
        if size > chunk_size:
            if buffer:
                spans.append(buffer)
                buffer = None
            spans.extend(recursive_spans(text[seg[0] : seg[1]], chunk_size, seg[0]))
        elif buffer is None:
            buffer = seg
        elif seg[1] - buffer[0] <= chunk_size:
            buffer = (buffer[0], seg[1])
        else:
            spans.append(buffer)
            buffer = seg
    if buffer:
        spans.append(buffer)

    return spans, [_section_label(text[a:b]) for a, b in spans]


def chunk_document(doc: Document, strategy: str, chunk_size: int) -> list[Chunk]:
    if strategy == "naive":
        return _to_chunks(doc, naive_spans(doc.content, chunk_size))
    if strategy == "rcts":
        return _to_chunks(doc, rcts_spans(doc.content, chunk_size))
    if strategy == "structural":
        spans, sections = structural_spans(doc.content, chunk_size)
        return _to_chunks(doc, spans, sections)
    raise ValueError(f"unknown chunking strategy: {strategy}")


def chunk_corpus(docs: list[Document], strategy: str, chunk_size: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(chunk_document(doc, strategy, chunk_size))
    return chunks


CHUNKERS: dict[str, Callable[[Document, int], list[Chunk]]] = {
    name: (lambda d, s, n=name: chunk_document(d, n, s))
    for name in ("naive", "rcts", "structural")
}
