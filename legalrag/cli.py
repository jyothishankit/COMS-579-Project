"""Command line interface."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from .config import BASELINE_RCTS_500, RetrievalStrategy, get_settings
from .engine import RagEngine
from .evaluation import load_tests, referenced_documents, save_result, score
from .ingest import load_any, load_corpus

app = typer.Typer(add_completion=False, help="Advanced legal RAG over LegalBench-RAG.")
console = Console()


def _strategy_from_file(path: Path | None) -> RetrievalStrategy:
    if path is None:
        return RetrievalStrategy()
    return RetrievalStrategy.model_validate_json(path.read_text())


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to answer."),
    files: list[Path] = typer.Option(..., "--file", "-f", help="PDF or text files to search."),
    strategy_file: Path = typer.Option(None, "--strategy", help="JSON RetrievalStrategy file."),
    show_spans: bool = typer.Option(False, "--show-spans", help="Print retrieved character spans."),
) -> None:
    """Index documents and answer a question over them."""
    settings = get_settings()
    strategy = _strategy_from_file(strategy_file)

    async def run() -> None:
        docs = [load_any(f, settings.cache_dir / "pdf") for f in files]
        engine = RagEngine(settings, strategy)
        with tqdm(total=0, desc="embedding", unit="chunk") as bar:
            await engine.index(docs, progress=lambda n: (bar.update(n), bar.refresh()))
        response = await engine.query(question)
        engine.close()

        console.rule("Answer")
        console.print(response.answer or "")
        if show_spans:
            console.rule("Retrieved spans")
            lookup = {d.doc_id: d for d in docs}
            for snippet in response.retrieved_snippets:
                console.print(f"[bold]{snippet.file_path}[/] {snippet.span} score={snippet.score:.3f}")
                content = lookup[snippet.file_path].content
                console.print(content[snippet.span[0] : snippet.span[1]].strip(), style="dim")

    asyncio.run(run())


@app.command("eval")
def evaluate(
    strategy_file: Path = typer.Option(None, "--strategy", help="JSON RetrievalStrategy file."),
    baseline: bool = typer.Option(False, "--baseline", help="Run the published RCTS-500 baseline."),
    benchmarks: list[str] = typer.Option(None, "--benchmark", "-b", help="Subset to run."),
    max_tests: int = typer.Option(194, "--max-tests", help="Tests sampled per benchmark."),
    budget: int = typer.Option(None, "--budget", help="Character budget per query."),
    save: bool = typer.Option(True, "--save/--no-save", help="Write results to disk."),
) -> None:
    """Score a retrieval strategy against LegalBench-RAG."""
    settings = get_settings()
    strategy = BASELINE_RCTS_500 if baseline else _strategy_from_file(strategy_file)
    if budget is not None:
        strategy = strategy.model_copy(update={"char_budget": budget})

    async def run() -> None:
        tests, weights = load_tests(settings, benchmarks, max_tests)
        doc_ids = referenced_documents(tests)
        docs = load_corpus(settings.corpus_dir, doc_ids)
        console.print(
            f"{len(tests)} queries over {len(docs)} documents "
            f"({sum(len(d.content) for d in docs) / 1e6:.2f}M chars)"
        )

        engine = RagEngine(settings, strategy)
        with tqdm(total=0, desc="embedding", unit="chunk") as bar:
            await engine.index(docs, progress=lambda n: bar.update(n))
        console.print(f"{len(engine.chunks)} chunks indexed")

        with tqdm(total=len(tests), desc="querying", unit="q") as bar:
            retrievals = await engine.retrieve_many(
                [t.query for t in tests], progress=lambda n: bar.update(n)
            )
        engine.close()

        result = score(tests, retrievals, weights, strategy.name)
        console.print()
        console.print(result.render())
        if save:
            path = save_result(result, settings.results_dir)
            console.print(f"\nsaved -> {path}")

    asyncio.run(run())


@app.command()
def sweep(
    config: Path = typer.Argument(..., help="JSON file containing a list of strategies."),
    max_tests: int = typer.Option(50, "--max-tests"),
    benchmarks: list[str] = typer.Option(None, "--benchmark", "-b"),
) -> None:
    """Run several strategies over one shared corpus and compare them."""
    settings = get_settings()
    strategies = [RetrievalStrategy.model_validate(s) for s in json.loads(config.read_text())]

    async def run() -> None:
        tests, weights = load_tests(settings, benchmarks, max_tests)
        docs = load_corpus(settings.corpus_dir, referenced_documents(tests))
        console.print(f"{len(tests)} queries over {len(docs)} documents")

        table = Table(title="Strategy comparison")
        for column in ("strategy", "precision", "recall", "F1", "chars/q"):
            table.add_column(column)

        # Group by chunking so strategies that differ only at query time reuse
        # one built index instead of re-embedding and re-upserting the corpus.
        engine: RagEngine | None = None
        for strategy in sorted(strategies, key=lambda s: (s.chunker, s.chunk_size)):
            if engine is None or engine.index_key() != (strategy.chunker, strategy.chunk_size):
                if engine is not None:
                    engine.close()
                engine = RagEngine(settings, strategy)
                with tqdm(total=0, desc=f"{strategy.name}: embedding", unit="chunk") as bar:
                    await engine.index(docs, progress=lambda n: bar.update(n))
            else:
                engine.set_strategy(strategy)

            with tqdm(total=len(tests), desc=f"{strategy.name}: querying", unit="q") as bar:
                retrievals = await engine.retrieve_many(
                    [t.query for t in tests], progress=lambda n: bar.update(n)
                )

            result = score(tests, retrievals, weights, strategy.name)
            save_result(result, settings.results_dir)
            stats = result.summary()["overall"]
            table.add_row(
                strategy.name,
                f"{stats['precision']:.2%}",
                f"{stats['recall']:.2%}",
                f"{stats['f1']:.2%}",
                f"{stats['chars']:.0f}",
            )

        if engine is not None:
            engine.close()
        console.print()
        console.print(table)

    asyncio.run(run())


@app.command()
def prune() -> None:
    """Delete leftover Pinecone indexes from interrupted runs.

    Each run creates a uniquely named index and removes it on exit; a crash
    leaves one behind, and on Pinecone Local those keep holding memory.
    """
    import httpx

    settings = get_settings()
    if not settings.pinecone_local_host:
        console.print("Refusing to bulk-delete indexes on Pinecone cloud.")
        raise typer.Exit(1)

    headers = {
        "Api-Key": settings.pinecone_api_key.get_secret_value(),
        "X-Pinecone-Api-Version": "2025-04",
    }
    base = settings.pinecone_local_host
    indexes = httpx.get(f"{base}/indexes", headers=headers, timeout=30).json()["indexes"]
    stale = [i["name"] for i in indexes if i["name"].startswith("legalrag-")]
    for name in stale:
        httpx.delete(f"{base}/indexes/{name}", headers=headers, timeout=30)
        console.print(f"deleted {name}")
    console.print(f"{len(stale)} index(es) removed")


if __name__ == "__main__":
    app()
