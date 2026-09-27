# Results

All numbers come from the official LegalBench-RAG protocol: 194 queries sampled
per corpus (776 total, `SORT_BY_DOCUMENT=True`), scored with the character-level
precision and recall definitions from `zeroentropy-ai/legalbenchrag`. The
corpus slice is 72 documents totalling 8.68M characters.

Reproduce with:

```bash
legalrag eval --baseline                          # published configuration
legalrag eval --strategy configs/advanced.json    # this system
legalrag sweep configs/ablations.json             # component ablations
```

## How to read these numbers

Precision is the fraction of *returned characters* that fall inside a
ground-truth span; recall is the fraction of ground-truth characters returned.
Because both are character counts rather than document hits, the size of what
you return matters as much as where it came from. A system returning eight
254-character chunks spends ~2,000 characters per query whether or not the
answer is 200 characters long, and its precision is capped accordingly.

That makes precision and recall trade off against each other directly, so a
single operating point proves little. The comparison below therefore reports
characters returned per query alongside both metrics.

## Baseline

RCTS chunking at 500 characters, `text-embedding-3-large`, top-8 by cosine
similarity — the configuration the paper publishes.

_(Populated by the run in progress.)_

## This system

_(Populated by the run in progress.)_

## Ablation ladder

Each row adds one component to the row above it, over identical chunks and
identical cached embeddings, so each delta is attributable to a single change.

_(Populated by the run in progress.)_

## Notes on fairness

- Both configurations use the same embedding model, the same corpus slice, the
  same sampling seed and the same scorer.
- The `rcts` chunker calls the same `RecursiveCharacterTextSplitter` the paper
  used, with the same separators, zero overlap and `strip_whitespace=False`,
  rather than a reimplementation.
- Returned snippets are merged into disjoint spans. The official scorer sums
  per-snippet overlaps without de-duplicating, so overlapping snippets would
  double-count matched characters; `assert_disjoint` fails the run rather than
  allow an inflated score.
- Embeddings and LLM responses are cached, so repeated runs are byte-identical.

## Cost and runtime

Embedding the corpus once costs roughly $0.28 and is the only slow step; on a
free-tier account its 40k tokens/minute ceiling stretches that to about an hour.
Afterwards each strategy evaluates in about two minutes, because every vector
comes from the local cache.
