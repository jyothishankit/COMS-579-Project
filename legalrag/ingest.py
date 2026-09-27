"""Document loading.

PDFs are converted to plain text once and cached, because every downstream span
is an offset into that exact text: re-extracting with different settings would
silently invalidate stored spans.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from .types import Document

# Page breaks inserted by extraction are not part of the document's language and
# only serve to confuse sentence splitting.
_EXCESS_BLANKS = re.compile(r"\n{3,}")
_TRAILING_SPACE = re.compile(r"[ \t]+\n")


def normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _TRAILING_SPACE.sub("\n", text)
    return _EXCESS_BLANKS.sub("\n\n", text)


def extract_pdf_text(path: Path) -> str:
    try:
        import pymupdf

        with pymupdf.open(path) as pdf:
            return "\n\n".join(page.get_text("text") for page in pdf)
    except ImportError:
        from pypdf import PdfReader

        return "\n\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)


def load_pdf(path: str | Path, cache_dir: Path | None = None) -> Document:
    path = Path(path)
    doc_id = path.name

    cache_file = None
    if cache_dir is not None:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{path.stem}-{digest}.txt"
        if cache_file.exists():
            return Document(doc_id=doc_id, content=cache_file.read_text())

    content = normalize(extract_pdf_text(path))
    if cache_file is not None:
        cache_file.write_text(content)
    return Document(doc_id=doc_id, content=content)


def load_text_file(path: str | Path, doc_id: str | None = None) -> Document:
    path = Path(path)
    return Document(doc_id=doc_id or path.name, content=path.read_text())


def load_any(path: str | Path, cache_dir: Path | None = None) -> Document:
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        return load_pdf(path, cache_dir)
    return load_text_file(path)


def load_corpus(root: str | Path, doc_ids: list[str] | None = None) -> list[Document]:
    """Load corpus files, keyed by their path relative to `root`.

    The benchmark identifies documents by relative path, so that is the id.
    """
    root = Path(root)
    if doc_ids is not None:
        return [
            Document(doc_id=doc_id, content=(root / doc_id).read_text())
            for doc_id in sorted(doc_ids)
        ]
    return [
        Document(doc_id=str(p.relative_to(root)), content=p.read_text())
        for p in sorted(root.rglob("*.txt"))
    ]
