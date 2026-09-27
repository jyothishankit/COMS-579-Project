"""Central configuration. Secrets come from the environment / .env, never from code."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    openai_api_key: SecretStr
    openai_base_url: str | None = None

    embedding_model: str = "text-embedding-3-large"
    embedding_dim: int = 3072
    llm_model: str = "gpt-4.1-mini"

    legalbench_dir: Path = PROJECT_ROOT.parent / "LegalBench-RAG"
    cache_dir: Path = PROJECT_ROOT / ".cache"
    results_dir: Path = PROJECT_ROOT / "benchmark_results"

    vector_backend: Literal["pinecone", "numpy"] = "numpy"
    pinecone_api_key: SecretStr = SecretStr("pclocal")
    pinecone_local_host: str | None = "http://localhost:5080"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    embed_concurrency: int = 8
    llm_concurrency: int = 16

    # Stay under the account's published ceilings. Raise these after a tier
    # upgrade; `scripts/check_limits.py` prints the current values.
    embed_rpm: int = 1800
    embed_tpm: int = 35_000
    llm_rpm: int = 400
    llm_tpm: int = 180_000

    @property
    def corpus_dir(self) -> Path:
        return self.legalbench_dir / "corpus"

    @property
    def benchmarks_dir(self) -> Path:
        return self.legalbench_dir / "benchmarks"


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()  # type: ignore[call-arg]
        _settings.cache_dir.mkdir(parents=True, exist_ok=True)
        _settings.results_dir.mkdir(parents=True, exist_ok=True)
    return _settings


class RetrievalStrategy(BaseModel):
    """A fully-specified, serializable description of one retrieval configuration.

    Every knob that changes retrieval quality lives here so that ablations are a
    matter of constructing a different object, not editing code. Deliberately not
    a settings class: a strategy must come from the caller, never from ambient
    environment variables.
    """

    model_config = ConfigDict(frozen=True)

    name: str = "advanced"

    chunker: Literal["rcts", "naive", "structural"] = "structural"
    chunk_size: int = 500

    dense_topk: int = 150
    use_bm25: bool = True
    bm25_topk: int = 150
    rrf_k: int = 60
    dense_weight: float = 1.0
    bm25_weight: float = 0.4

    expansion: Literal["none", "hyde", "multi"] = "none"

    reranker: Literal["none", "flashrank", "llm"] = "flashrank"
    rerank_input_topk: int = 100
    rerank_topk: int = 16

    # Sentence-level span refinement: the main precision lever.
    refine: Literal["none", "embedding", "llm"] = "llm"
    refine_chunks: int = 12
    refine_context_chars: int = 0
    refine_min_score: float = 0.25
    # Widen each selected sentence run by N neighbours, trading precision for recall.
    refine_pad_sentences: int = 0

    # Final answer budget in characters (None = unbounded).
    char_budget: int | None = None

    def key(self) -> str:
        """Stable identity used for on-disk artifacts."""
        return self.model_dump_json()


BASELINE_RCTS_500 = RetrievalStrategy(
    name="baseline-rcts-500",
    chunker="rcts",
    chunk_size=500,
    use_bm25=False,
    expansion="none",
    reranker="none",
    refine="none",
    rerank_topk=8,
)
