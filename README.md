# COMS-579 — Advanced Legal RAG

A retrieval-augmented generation system for legal documents, evaluated against
[LegalBench-RAG](https://arxiv.org/abs/2408.10343).

```
PDF / text  →  chunk  →  OpenAI embeddings  →  Pinecone  →  rerank  →  refine  →  LLM
```

## Why this beats a conventional RAG pipeline

LegalBench-RAG grades retrieval at the **character** level: precision is the
fraction of returned characters that fall inside a ground-truth span, recall the
fraction of ground-truth characters returned. That changes what "good retrieval"
means. A conventional pipeline that returns eight 500-character chunks to answer
a question whose answer is 400 characters long is capped at roughly 10%
precision even when every chunk is correct.

This system is built around that fact:

1. **Every chunk carries an exact character span.** All three chunkers tile a
   document precisely — concatenating the chunks reproduces the file byte for
   byte. This is enforced by tests, because a drifting offset silently corrupts
   every score downstream.
2. **Hybrid recall.** Dense embeddings plus BM25, fused by reciprocal rank.
   Legal queries hinge on defined terms ("Receiving Party", "Change of Control")
   that lexical matching catches and embeddings blur.
3. **Cross-encoder reranking.** A local ONNX model reads query and passage
   jointly over ~100 candidates. No API cost.
4. **Sentence-level span refinement.** Retrieved regions are split into
   sentences and cut down to the ones that actually answer the question. This is
   where most of the precision gain comes from.

## Results

Scored with the official LegalBench-RAG metrics and sampling procedure
(194 queries per corpus, 776 total). See `RESULTS.md` for the full table and the
ablation ladder.

## Setup

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,ui]"

cp .env.example .env     # then add your OpenAI key
docker compose up -d pinecone
```

Download the LegalBench-RAG corpus from the
[dataset link](https://github.com/zeroentropy-ai/legalbenchrag) and point
`LEGALBENCH_DIR` at the folder containing `corpus/` and `benchmarks/`.

### Secrets

`.env` is gitignored and is the only place a key should live. Never paste a key
into a file that git tracks, into a commit message, or into a chat window — if
you do, rotate it.

## Usage

```bash
# Ask a question over your own documents
legalrag ask "What is the relation between hypermutable brains and age?" \
    -f genemutation.pdf -f LLMbasedTesting.pdf -f psychiatry.pdf --show-spans

# Reproduce the published baseline
legalrag eval --baseline

# Run the advanced pipeline
legalrag eval --strategy configs/advanced.json

# Ablation ladder on a fast subset
legalrag sweep configs/ablations.json --max-tests 25
```

The original Funix UI still works:

```bash
funix upload.py
```

## Architecture

| Module | Responsibility |
|---|---|
| `config.py` | Settings from `.env`; `RetrievalStrategy` describes one full configuration |
| `ingest.py` | PDF and text loading, with cached extraction |
| `chunking.py` | `naive`, `rcts` and legal-`structural` splitters, all offset-exact |
| `embeddings.py` | OpenAI embeddings: batching, rate limiting, SQLite vector cache |
| `stores.py` | Pinecone and numpy vector stores, BM25 index, RRF fusion |
| `rerank.py` | Local cross-encoder and LLM listwise rerankers |
| `refine.py` | Region building and sentence-level span selection |
| `generation.py` | Grounded answer synthesis with citations |
| `evaluation.py` | LegalBench-RAG metrics, sampling and reporting |
| `engine.py` | Wires the stages together |
| `cli.py` | `ask`, `eval`, `sweep` |

Every stage is switchable through `RetrievalStrategy`, so each number in the
ablation table is attributable to one component rather than to the pipeline as a
whole.

### Pinecone via Docker

`docker compose up -d pinecone` starts
[Pinecone Local](https://docs.pinecone.io/guides/operations/local-development),
an in-memory emulator of the Pinecone API that needs no account. Indexes are
created per run and deleted on exit; nothing persists across container restarts.

Two implementation notes, both discovered against the running emulator:

- Its control plane predates the current SDK's create-index payload, so index
  creation goes over REST while the data plane uses the SDK.
- Upsert bodies are capped at 2 MB, so batch size is computed from the vector
  dimension rather than fixed.

To use Pinecone cloud instead, set a real `PINECONE_API_KEY` and leave
`PINECONE_LOCAL_HOST` empty. Set `VECTOR_BACKEND=numpy` for an exact
brute-force search with no service at all.

## Rate limits

Free-tier OpenAI accounts cap embeddings at 40k tokens/minute and some chat
models at ~50 requests **per day**, which makes LLM-based span refinement
impractical across all 776 queries. `scripts/check_limits.py` prints the current
ceilings; set `embed_tpm`, `embed_rpm`, `llm_tpm` and `llm_rpm` in `.env` to
match. Embeddings and LLM responses are cached on disk, so a repeated run is
free and byte-identical.

## Tests

```bash
pytest
```

The suite covers the invariants that character-level scoring depends on: exact
tiling, span/text agreement, sentence-splitter contiguity, whitespace trimming,
disjointness of returned snippets, and the metric definitions themselves.

## Prior coursework

Earlier assignment demos (Weaviate + HuggingFace embeddings):

- [Indexing, splitting, nearest-vector retrieval](https://iowastate-my.sharepoint.com/:v:/g/personal/ankitj99_iastate_edu/EUq64OGM_hBDp7dMt2a3cKIBYyaCtLqWBXxUOPpYhfvHlw)
- [Question answering](https://iowastate-my.sharepoint.com/:v:/g/personal/ankitj99_iastate_edu/Ecx-X8sHRqpHvACr5i7t9M0BxoP0wwvTVvg0VENKHbD0rg)
- [Funix UI](https://iowastate-my.sharepoint.com/:v:/g/personal/ajayt_iastate_edu/Ecm6ZZDB9QdHo2rm6IGtYuUBrkuPYTEZl36GJwATGZLu1Q)

`docker compose --profile legacy up -d` still starts the Weaviate service those
demos used.
