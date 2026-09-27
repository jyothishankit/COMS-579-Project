"""An advanced retrieval-augmented generation system for legal documents.

Pipeline: PDF/text -> structure-aware chunking -> OpenAI embeddings -> Pinecone
-> hybrid recall -> cross-encoder rerank -> sentence-level span refinement -> LLM.

Evaluated against LegalBench-RAG using the published scoring code.
"""

from .config import BASELINE_RCTS_500, RetrievalStrategy, Settings, get_settings
from .engine import RagEngine
from .types import Document, QueryResponse, RetrievedSnippet

__all__ = [
    "BASELINE_RCTS_500",
    "Document",
    "QueryResponse",
    "RagEngine",
    "RetrievalStrategy",
    "RetrievedSnippet",
    "Settings",
    "get_settings",
]
