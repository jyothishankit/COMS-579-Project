"""LegalBench-RAG evaluation.

The metrics and the sampling procedure here reproduce the published harness
exactly (zeroentropy-ai/legalbenchrag), so numbers produced by this project are
directly comparable to the paper's. Deviating would make any improvement
unfalsifiable.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from pydantic import BaseModel, Field

from .config import Settings
from .types import Benchmark, QAGroundTruth, RetrievedSnippet, Snippet

BENCHMARK_WEIGHTS: dict[str, float] = {
    "privacy_qa": 0.25,
    "contractnli": 0.25,
    "maud": 0.25,
    "cuad": 0.25,
}

MAX_TESTS_PER_BENCHMARK = 194


def overlap_chars(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def precision_recall(
    retrieved: list[RetrievedSnippet], ground_truth: list[Snippet]
) -> tuple[float, float]:
    """Character-level precision and recall, as defined by the official harness.

    Both are computed over raw character counts: precision is the share of
    returned characters that land inside a ground-truth span, recall the share of
    ground-truth characters that were returned.
    """
    total_retrieved = sum(s.span[1] - s.span[0] for s in retrieved)
    total_relevant = sum(s.span[1] - s.span[0] for s in ground_truth)

    matched = 0
    for snippet in retrieved:
        for truth in ground_truth:
            if snippet.file_path == truth.file_path:
                matched += overlap_chars(snippet.span, truth.span)

    precision = matched / total_retrieved if total_retrieved else 0.0
    recall = matched / total_relevant if total_relevant else 0.0
    return precision, recall


def assert_disjoint(retrieved: list[RetrievedSnippet]) -> None:
    """Guard against inflating scores by returning the same characters twice.

    The official scorer sums overlaps without de-duplicating, so overlapping
    snippets would count matched characters more than once. This system always
    returns merged, disjoint spans; this check keeps that honest.
    """
    by_doc: dict[str, list[tuple[int, int]]] = {}
    for snippet in retrieved:
        by_doc.setdefault(snippet.file_path, []).append(snippet.span)
    for doc_id, spans in by_doc.items():
        ordered = sorted(spans)
        for previous, nxt in zip(ordered, ordered[1:]):
            if nxt[0] < previous[1]:
                raise ValueError(f"overlapping snippets returned for {doc_id}: {previous} {nxt}")


class QAResult(BaseModel):
    query: str
    tags: list[str]
    precision: float
    recall: float
    retrieved_chars: int
    retrieved_snippets: list[RetrievedSnippet] = Field(default_factory=list)

    @property
    def f1(self) -> float:
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * self.precision * self.recall / (self.precision + self.recall)


class BenchmarkResult(BaseModel):
    strategy_name: str
    results: list[QAResult]
    weights: list[float]

    def _aggregate(self, tag: str | None) -> dict[str, float]:
        picked = [
            (r, w)
            for r, w in zip(self.results, self.weights)
            if tag is None or tag in r.tags
        ]
        if not picked:
            return {"precision": float("nan"), "recall": float("nan"), "f1": float("nan"), "chars": 0.0, "n": 0}
        avg_weight = sum(w for _, w in picked) / len(picked)
        precision = sum(r.precision * w / avg_weight for r, w in picked) / len(picked)
        recall = sum(r.recall * w / avg_weight for r, w in picked) / len(picked)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "chars": sum(r.retrieved_chars for r, _ in picked) / len(picked),
            "n": len(picked),
        }

    def summary(self) -> dict[str, dict[str, float]]:
        out = {name: self._aggregate(name) for name in BENCHMARK_WEIGHTS}
        out["overall"] = self._aggregate(None)
        return out

    def render(self) -> str:
        rows = self.summary()
        header = f"{'benchmark':<14}{'precision':>11}{'recall':>10}{'F1':>9}{'chars/q':>10}{'n':>6}"
        lines = [self.strategy_name, header, "-" * len(header)]
        for name, stats in rows.items():
            lines.append(
                f"{name:<14}{stats['precision']:>10.2%}{stats['recall']:>10.2%}"
                f"{stats['f1']:>9.2%}{stats['chars']:>10.0f}{stats['n']:>6.0f}"
            )
        return "\n".join(lines)


def load_tests(
    settings: Settings,
    benchmark_names: list[str] | None = None,
    max_tests: int = MAX_TESTS_PER_BENCHMARK,
    *,
    sort_by_document: bool = True,
    seed_offset: int = 0,
) -> tuple[list[QAGroundTruth], list[float]]:
    """Sample the benchmark the same way the official harness does.

    With `sort_by_document`, tests are ordered by a hash of their source file so
    the sample concentrates on few documents; that is the published setting and
    it is what makes the corpus tractable.
    """
    names = benchmark_names or list(BENCHMARK_WEIGHTS)
    tests: list[QAGroundTruth] = []
    weights: list[float] = []

    for name in names:
        path = settings.benchmarks_dir / f"{name}.json"
        benchmark = Benchmark.model_validate_json(path.read_text())
        selected = benchmark.tests

        if len(selected) > max_tests:
            if sort_by_document:
                selected = sorted(
                    selected,
                    key=lambda t: (random.seed(t.snippets[0].file_path), random.random())[1],
                )
            else:
                random.seed(name)
                random.shuffle(selected)
            selected = selected[seed_offset : seed_offset + max_tests]

        for test in selected:
            test.tags = [name]
        tests.extend(selected)
        weights.extend([BENCHMARK_WEIGHTS[name] / len(selected)] * len(selected))

    return tests, weights


def referenced_documents(tests: list[QAGroundTruth]) -> list[str]:
    return sorted({s.file_path for t in tests for s in t.snippets})


def score(
    tests: list[QAGroundTruth],
    retrievals: list[list[RetrievedSnippet]],
    weights: list[float],
    strategy_name: str,
    *,
    check_disjoint: bool = True,
) -> BenchmarkResult:
    results: list[QAResult] = []
    for test, retrieved in zip(tests, retrievals):
        if check_disjoint:
            assert_disjoint(retrieved)
        precision, recall = precision_recall(retrieved, test.snippets)
        results.append(
            QAResult(
                query=test.query,
                tags=test.tags,
                precision=precision,
                recall=recall,
                retrieved_chars=sum(s.span[1] - s.span[0] for s in retrieved),
                retrieved_snippets=retrieved,
            )
        )
    return BenchmarkResult(strategy_name=strategy_name, results=results, weights=weights)


def save_result(result: BenchmarkResult, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{result.strategy_name}.json"
    path.write_text(
        json.dumps(
            {"summary": result.summary(), "result": result.model_dump(mode="json")}, indent=2
        )
    )
    return path
