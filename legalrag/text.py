"""Offset-preserving text operations.

Every function here returns spans into the *original* string. Nothing is
normalized away, because a normalized offset is a wrong offset.
"""

from __future__ import annotations

import re

from .types import Span

# Abbreviations that end in a period but do not end a sentence. Legal prose is
# dense with these, and splitting on them shreds clauses mid-thought.
_ABBREVIATIONS = {
    "inc", "llc", "ltd", "co", "corp", "plc", "lp", "llp", "sa", "nv", "gmbh",
    "no", "nos", "art", "arts", "sec", "secs", "para", "paras", "cl", "sched",
    "ex", "app", "fig", "figs", "vol", "cf", "viz", "eg", "ie", "etc", "al",
    "mr", "mrs", "ms", "dr", "prof", "hon", "jr", "sr", "st",
    "u.s", "u.k", "e.g", "i.e", "v", "vs", "ch", "pp", "ss",
}

_SENTENCE_END = re.compile(r"[.!?][\"')\]]*(?=\s)|\n{2,}|(?<=[;:])\s(?=[A-Z(])")
_WORD_BEFORE = re.compile(r"([A-Za-z.]+)\s*$")
_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens, used by the lexical index."""
    return _TOKEN.findall(text.lower())


def _is_abbreviation(text: str, period_index: int) -> bool:
    match = _WORD_BEFORE.search(text, 0, period_index)
    if not match:
        return False
    word = match.group(1).rstrip(".").lower()
    if word in _ABBREVIATIONS:
        return True
    # Single initials ("J. Smith") and enumerations ("Section 4.1.") .
    return len(word) <= 1


def sentence_spans(text: str, offset: int = 0, *, min_len: int = 24) -> list[Span]:
    """Split into sentence spans that tile `text` exactly, with no gaps.

    Contiguity matters: the refinement stage reassembles selected sentences into
    character spans, and a gap here would silently drop matched ground truth.
    """
    if not text.strip():
        return []

    cuts: list[int] = []
    for match in _SENTENCE_END.finditer(text):
        end = match.end()
        if match.group().startswith((".", "!", "?")) and _is_abbreviation(text, match.start()):
            continue
        cuts.append(end)

    spans: list[Span] = []
    start = 0
    for cut in cuts:
        if cut - start >= min_len:
            spans.append((start, cut))
            start = cut
    if start < len(text):
        if spans and len(text) - start < min_len:
            spans[-1] = (spans[-1][0], len(text))
        else:
            spans.append((start, len(text)))

    return [(a + offset, b + offset) for a, b in spans]


def trim_span(text: str, span: Span) -> Span:
    """Shrink a span so it excludes leading and trailing whitespace.

    Whitespace costs precision (it counts toward retrieved characters) and can
    never overlap ground truth in a way that helps recall.
    """
    start = max(0, min(span[0], len(text)))
    end = max(start, min(span[1], len(text)))
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (start, end)
