"""Vector and lexical indexes behind a single interface.

Two dense backends are supported. Pinecone is the production path; the numpy
backend is an exact brute-force search used as an oracle in tests and as a fast
path for ablations, since approximate recall differences would otherwise be
indistinguishable from pipeline changes.
"""

from __future__ import annotations

import math
from typing import Iterable, Protocol

import numpy as np
from scipy import sparse

from .config import Settings
from .text import tokenize
from .types import Chunk

Hit = tuple[int, float]


class VectorStore(Protocol):
    def add(self, vectors: np.ndarray, chunks: list[Chunk]) -> None: ...
    def search(self, query: np.ndarray, top_k: int) -> list[Hit]: ...
    def close(self) -> None: ...


class NumpyVectorStore:
    """Exact cosine search over an in-memory matrix."""

    def __init__(self) -> None:
        self._blocks: list[np.ndarray] = []
        self._matrix: np.ndarray | None = None

    def add(self, vectors: np.ndarray, chunks: list[Chunk]) -> None:
        self._blocks.append(np.ascontiguousarray(vectors, dtype=np.float32))
        self._matrix = None

    def _materialize(self) -> np.ndarray:
        if self._matrix is None:
            self._matrix = np.vstack(self._blocks) if self._blocks else np.zeros((0, 0), np.float32)
        return self._matrix

    def search(self, query: np.ndarray, top_k: int) -> list[Hit]:
        matrix = self._materialize()
        if matrix.shape[0] == 0:
            return []
        scores = matrix @ query.astype(np.float32)
        top_k = min(top_k, scores.shape[0])
        idx = np.argpartition(-scores, top_k - 1)[:top_k]
        idx = idx[np.argsort(-scores[idx])]
        return [(int(i), float(scores[i])) for i in idx]

    def close(self) -> None:
        self._blocks.clear()
        self._matrix = None


class PineconeVectorStore:
    """Pinecone-backed search, against either Pinecone Local or the cloud.

    Chunk text is deliberately not stored as metadata: the corpus is on disk and
    addressable by span, so shipping it to the index would only cost memory.
    """

    def __init__(self, settings: Settings, index_name: str, dim: int, metric: str = "cosine"):
        from pinecone import Pinecone

        self.index_name = index_name
        self._local_host = settings.pinecone_local_host
        self._api_key = settings.pinecone_api_key.get_secret_value()
        self._count = 0
        self._spec = {"serverless": {"cloud": settings.pinecone_cloud, "region": settings.pinecone_region}}

        kwargs = {"api_key": self._api_key}
        if self._local_host:
            kwargs["host"] = self._local_host
        self._client = Pinecone(**kwargs)

        host = self._create_index(dim, metric)
        self._index = self._client.Index(host=host)

    def _rest(self, method: str, path: str, json: dict | None = None):
        import httpx

        response = httpx.request(
            method,
            f"{self._local_host}{path}",
            json=json,
            headers={"Api-Key": self._api_key, "X-Pinecone-Api-Version": "2025-04"},
            timeout=60.0,
        )
        response.raise_for_status()
        return response.json() if response.content else {}

    def _create_index(self, dim: int, metric: str) -> str:
        # Pinecone Local's control plane predates the current SDK's create-index
        # payload, so talk to it over REST. The data plane is fully compatible.
        if self._local_host:
            existing = self._rest("GET", "/indexes").get("indexes", [])
            if any(i["name"] == self.index_name for i in existing):
                self._rest("DELETE", f"/indexes/{self.index_name}")
            created = self._rest(
                "POST",
                "/indexes",
                {"name": self.index_name, "dimension": dim, "metric": metric, "spec": self._spec},
            )
            host = created["host"]
            return host if host.startswith("http") else f"http://{host}"

        from pinecone import ServerlessSpec

        if any(i["name"] == self.index_name for i in self._client.list_indexes()):
            self._client.delete_index(self.index_name)
        description = self._client.create_index(
            name=self.index_name,
            dimension=dim,
            metric=metric,
            spec=ServerlessSpec(**self._spec["serverless"]),
        )
        return description.host

    def add(self, vectors: np.ndarray, chunks: list[Chunk], batch_size: int | None = None) -> None:
        if batch_size is None:
            # Upserts are JSON and the server rejects bodies over 2 MB. A float
            # serializes to about 21 bytes, so size each batch from the vector
            # dimension rather than guessing a fixed count.
            dim = vectors.shape[1] if vectors.size else 1
            batch_size = max(1, min(200, 1_800_000 // (dim * 21)))
        records = []
        for vector, chunk in zip(vectors, chunks):
            # Vector ids must be ASCII, and corpus filenames are not (MAUD has
            # names like "TIFFANY_&_CO._LVMH_MOËT_HENNESSY"). The row index is
            # the id; the real identity travels in metadata.
            row = self._count
            self._count += 1
            records.append(
                {
                    "id": str(row),
                    "values": vector.tolist(),
                    "metadata": {
                        "doc_id": chunk.doc_id,
                        "start": chunk.span[0],
                        "end": chunk.span[1],
                    },
                }
            )
        for i in range(0, len(records), batch_size):
            self._index.upsert(vectors=records[i : i + batch_size])

    def search(self, query: np.ndarray, top_k: int) -> list[Hit]:
        response = self._index.query(
            vector=query.tolist(), top_k=top_k, include_values=False, include_metadata=False
        )
        return [(int(match["id"]), float(match["score"])) for match in response["matches"]]

    def close(self) -> None:
        try:
            if self._local_host:
                self._rest("DELETE", f"/indexes/{self.index_name}")
            else:
                self._client.delete_index(self.index_name)
        except Exception:
            pass


class BM25Index:
    """Okapi BM25 over a sparse term-document matrix.

    Legal queries carry defined terms ("Receiving Party", "Change of Control")
    that a dense model can blur together; exact lexical matching recovers the
    documents that embeddings rank just out of reach.
    """

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._vocab: dict[str, int] = {}
        self._weights: sparse.csc_matrix | None = None

    def build(self, chunks: Iterable[Chunk]) -> None:
        docs = [tokenize(chunk.text) for chunk in chunks]
        n_docs = len(docs)
        if n_docs == 0:
            return

        for tokens in docs:
            for token in tokens:
                if token not in self._vocab:
                    self._vocab[token] = len(self._vocab)

        rows: list[int] = []
        cols: list[int] = []
        freqs: list[float] = []
        lengths = np.zeros(n_docs, dtype=np.float32)
        for row, tokens in enumerate(docs):
            lengths[row] = len(tokens)
            counts: dict[int, int] = {}
            for token in tokens:
                col = self._vocab[token]
                counts[col] = counts.get(col, 0) + 1
            rows.extend([row] * len(counts))
            cols.extend(counts.keys())
            freqs.extend(counts.values())

        tf = sparse.csr_matrix(
            (np.asarray(freqs, np.float32), (rows, cols)), shape=(n_docs, len(self._vocab))
        )
        avg_len = float(lengths.mean()) or 1.0
        doc_freq = np.asarray((tf > 0).sum(axis=0)).ravel()
        idf = np.log(1.0 + (n_docs - doc_freq + 0.5) / (doc_freq + 0.5)).astype(np.float32)

        # Saturate term frequency per BM25, then scale each column by its idf.
        norm = (self.k1 * (1 - self.b + self.b * lengths / avg_len)).astype(np.float32)
        coo = tf.tocoo()
        saturated = coo.data * (self.k1 + 1) / (coo.data + norm[coo.row])
        weighted = saturated * idf[coo.col]
        self._weights = sparse.csc_matrix(
            (weighted, (coo.row, coo.col)), shape=tf.shape, dtype=np.float32
        )

    def search(self, query: str, top_k: int) -> list[Hit]:
        if self._weights is None:
            return []
        cols = [self._vocab[t] for t in tokenize(query) if t in self._vocab]
        if not cols:
            return []
        scores = np.asarray(self._weights[:, cols].sum(axis=1)).ravel()
        top_k = min(top_k, scores.shape[0])
        idx = np.argpartition(-scores, top_k - 1)[:top_k]
        idx = idx[np.argsort(-scores[idx])]
        return [(int(i), float(scores[i])) for i in idx if scores[i] > 0]


def build_vector_store(settings: Settings, index_name: str, dim: int) -> VectorStore:
    if settings.vector_backend == "pinecone":
        return PineconeVectorStore(settings, index_name, dim)
    return NumpyVectorStore()


def rrf_fuse(rankings: list[list[Hit]], weights: list[float], k: int = 60) -> list[Hit]:
    """Reciprocal rank fusion.

    Rank-based rather than score-based, because cosine similarity and BM25 live
    on incomparable scales and normalizing them is guesswork.
    """
    totals: dict[int, float] = {}
    for ranking, weight in zip(rankings, weights):
        for rank, (idx, _) in enumerate(ranking):
            totals[idx] = totals.get(idx, 0.0) + weight / (k + rank + 1)
    return sorted(totals.items(), key=lambda item: -item[1])


def softmax(scores: list[float], temperature: float = 1.0) -> list[float]:
    if not scores:
        return []
    top = max(scores)
    exps = [math.exp((s - top) / temperature) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]
