"""The retrieval engine: ingest once, query many times.

Pipeline, in order:
    expand -> dense + lexical recall -> rank fusion -> cross-encoder -> span refinement -> budget

Each stage is independently switchable through `RetrievalStrategy`, so every
number reported by the evaluation harness can be attributed to a specific
component rather than to the pipeline as a whole.
"""

from __future__ import annotations

import asyncio
import uuid

from .chunking import chunk_corpus
from .config import RetrievalStrategy, Settings
from .embeddings import OpenAIEmbedder
from .expansion import QueryExpander
from .refine import build_refiner
from .rerank import build_reranker
from .stores import BM25Index, build_vector_store, rrf_fuse
from .types import Chunk, Document, QueryResponse, RetrievedSnippet, Scored


class RagEngine:
    def __init__(self, settings: Settings, strategy: RetrievalStrategy):
        self.settings = settings
        self.strategy = strategy
        self.docs: dict[str, Document] = {}
        self.chunks: list[Chunk] = []

        self._embedder = OpenAIEmbedder(settings)
        self._expander = QueryExpander(settings, strategy.expansion)
        self._reranker = build_reranker(strategy.reranker, settings)
        self._refiner = build_refiner(strategy, settings, self._embedder)
        self._vectors = None
        self._bm25: BM25Index | None = None
        self._index_name = f"legalrag-{uuid.uuid4().hex[:12]}"

    def index_key(self) -> tuple[str, int]:
        """What the built index depends on. Strategies sharing this can share an index."""
        return (self.strategy.chunker, self.strategy.chunk_size)

    def set_strategy(self, strategy: RetrievalStrategy) -> None:
        """Swap query-time configuration while keeping the built index.

        Re-embedding and re-upserting an unchanged corpus for every ablation is
        the slowest part of a sweep, and nothing before the query stage depends
        on these knobs.
        """
        if (strategy.chunker, strategy.chunk_size) != self.index_key():
            raise ValueError("cannot reuse an index across different chunking settings")
        self.strategy = strategy
        self._expander = QueryExpander(self.settings, strategy.expansion)
        self._reranker = build_reranker(strategy.reranker, self.settings)
        self._refiner = build_refiner(strategy, self.settings, self._embedder)
        if strategy.use_bm25 and self._bm25 is None:
            self._bm25 = BM25Index()
            self._bm25.build(self.chunks)

    async def index(self, docs: list[Document], progress=None) -> None:
        self.docs = {doc.doc_id: doc for doc in docs}
        self.chunks = chunk_corpus(docs, self.strategy.chunker, self.strategy.chunk_size)

        matrix = await self._embedder.embed([c.text for c in self.chunks], progress=progress)
        self._vectors = build_vector_store(
            self.settings, self._index_name, dim=matrix.shape[1]
        )
        self._vectors.add(matrix, self.chunks)

        if self.strategy.use_bm25:
            self._bm25 = BM25Index()
            self._bm25.build(self.chunks)

    async def _recall(self, query: str) -> list[Scored]:
        """First stage: cast a wide net, optimizing for recall over ordering."""
        variants = await self._expander.expand(query)
        vectors = await self._embedder.embed(variants)

        rankings = []
        weights = []
        for i, vector in enumerate(vectors):
            rankings.append(self._vectors.search(vector, self.strategy.dense_topk))
            # Generated variants inform but must not outvote the real question.
            weights.append(self.strategy.dense_weight * (1.0 if i == 0 else 0.5))

        if self._bm25 is not None and self.strategy.use_bm25:
            rankings.append(self._bm25.search(query, self.strategy.bm25_topk))
            weights.append(self.strategy.bm25_weight)

        fused = rrf_fuse(rankings, weights, k=self.strategy.rrf_k)
        return [
            Scored(chunk=self.chunks[i], score=score)
            for i, score in fused[: self.strategy.rerank_input_topk]
        ]

    def _apply_budget(self, snippets: list[RetrievedSnippet]) -> list[RetrievedSnippet]:
        budget = self.strategy.char_budget
        if budget is None:
            return snippets
        kept: list[RetrievedSnippet] = []
        for snippet in snippets:
            if budget <= 0:
                break
            end = min(snippet.span[1], snippet.span[0] + budget)
            kept.append(
                RetrievedSnippet(
                    file_path=snippet.file_path, span=(snippet.span[0], end), score=snippet.score
                )
            )
            budget -= end - snippet.span[0]
        return kept

    async def retrieve(self, query: str) -> list[RetrievedSnippet]:
        if self._vectors is None:
            raise RuntimeError("index() must be called before retrieve()")
        candidates = await self._recall(query)
        ranked = await self._reranker.rerank(query, candidates, self.strategy.rerank_topk)
        snippets = await self._refiner.refine(query, ranked, self.docs)
        return self._apply_budget(snippets)

    async def retrieve_many(self, queries: list[str], progress=None) -> list[list[RetrievedSnippet]]:
        async def one(query: str) -> list[RetrievedSnippet]:
            result = await self.retrieve(query)
            if progress:
                progress(1)
            return result

        return await asyncio.gather(*(one(q) for q in queries))

    async def query(self, query: str, *, generate: bool = True) -> QueryResponse:
        snippets = await self.retrieve(query)
        answer = None
        if generate:
            from .generation import AnswerGenerator

            answer = await AnswerGenerator(self.settings).answer(query, snippets, self.docs)
        return QueryResponse(query=query, retrieved_snippets=snippets, answer=answer)

    def close(self) -> None:
        if self._vectors is not None:
            self._vectors.close()
            self._vectors = None
