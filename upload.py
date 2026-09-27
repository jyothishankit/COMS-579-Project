"""Funix UI for the legal RAG pipeline.

Run with:  funix upload.py
"""

from __future__ import annotations

import main


def upload(
    pdf_file_name_1: str = "genemutation.pdf",
    pdf_file_name_2: str = "LLMbasedTesting.pdf",
    pdf_file_name_3: str = "psychiatry.pdf",
    question: str = "What is the relation between hypermutable brains and age?",
) -> str:
    """Index the given PDFs and answer a question with cited source spans."""
    return main.upload(pdf_file_name_1, pdf_file_name_2, pdf_file_name_3, question)
