from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


BACKEND_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """All environment-backed application configuration in one place."""

    model_config = SettingsConfigDict(
        env_file=BACKEND_ROOT / ".env.local",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        case_sensitive=False,
        extra="ignore",
    )

    app_env: Literal["development", "test", "production"] = "development"
    app_host: str = "0.0.0.0"
    app_port: int = Field(default=8000, ge=1, le=65535)
    app_log_level: str = "INFO"

    frontend_origin: str = "http://localhost:3000"
    admin_token: SecretStr = SecretStr("")

    llm_provider: str = "openai_compatible"
    llm_api_style: Literal["chat_completions", "responses"] = "chat_completions"
    llm_base_url: str = ""
    llm_api_key: SecretStr = SecretStr("")
    llm_model: str = ""
    llm_temperature: float = Field(default=0.2, ge=0, le=2)
    llm_timeout_seconds: float = Field(default=60, gt=0, le=600)

    embedding_provider: Literal["api", "local"] = "api"
    embedding_base_url: str = ""
    embedding_api_key: SecretStr = SecretStr("")
    embedding_model: str = ""
    embedding_dimension: int | None = Field(default=None, ge=1)
    local_embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    chroma_path: Path = Path("./data/chroma")
    database_url: str = "postgresql://postgres:postgres@localhost:5432/ml_coach"
    upload_path: Path = Path("./data/uploads")
    chroma_distance_metric: Literal["cosine", "l2", "ip"] = "cosine"

    rag_top_k: int = Field(default=6, ge=1, le=50)
    rag_strict_mode: bool = True
    # HyDE uses a fixed three-answer contract; only the feature toggle is configurable.
    rag_hyde_enabled: bool = True
    rag_distance_threshold: float | None = Field(default=None, ge=0)
    chunk_size: int = Field(default=800, ge=200, le=5000)
    chunk_overlap: int = Field(default=120, ge=0, le=1000)
    max_upload_mb: int = Field(default=20, ge=1, le=1024)
    max_history_messages: int = Field(default=12, ge=0, le=100)
    max_message_length: int = Field(default=8000, ge=1, le=100_000)

    practice_default_count: int = Field(default=3, ge=1, le=8)
    checkpoint_default_count: int = Field(default=5, ge=1, le=8)
    practice_max_count: int = Field(default=8, ge=1, le=8)

    mastery_gate_threshold: float = Field(default=0.80, ge=0, le=1)
    prerequisite_mastery_threshold: float = Field(default=0.80, ge=0, le=1)
    mastery_min_valid_attempts: int = Field(default=3, ge=1)
    mastery_checkpoint_score: float = Field(default=80, ge=0, le=100)
    mastery_min_dimension_coverage: int = Field(default=3, ge=1)
    mastery_evidence_confidence_target: float = Field(default=6, gt=0)

    evaluator_min_confidence: float = Field(default=0.60, ge=0, le=1)
    evaluation_max_repair_attempts: int = Field(default=1, ge=0, le=1)

    practice_generation_temperature: float = Field(default=0.30, ge=0, le=2)
    evaluation_temperature: float = Field(default=0.00, ge=0, le=2)
    remediation_temperature: float = Field(default=0.20, ge=0, le=2)

    recent_question_dedup_count: int = Field(default=10, ge=1, le=100)
    question_generation_max_retries: int = Field(default=1, ge=0, le=1)

    @model_validator(mode="after")
    def validate_and_resolve(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP 必须小于 CHUNK_SIZE")
        if self.practice_default_count > self.practice_max_count:
            raise ValueError("PRACTICE_DEFAULT_COUNT 不能大于 PRACTICE_MAX_COUNT")
        if self.checkpoint_default_count > self.practice_max_count:
            raise ValueError("CHECKPOINT_DEFAULT_COUNT 不能大于 PRACTICE_MAX_COUNT")
        origins = self.cors_origins
        if self.app_env == "production" and "*" in origins:
            raise ValueError("生产环境不允许 FRONTEND_ORIGIN 使用通配符")
        if self.app_env == "production" and self.admin_token.get_secret_value() in {
            "",
            "change-this-admin-token",
        }:
            raise ValueError("生产环境必须配置安全的 ADMIN_TOKEN")
        for field_name in ("chroma_path", "upload_path"):
            value = getattr(self, field_name)
            if not value.is_absolute():
                setattr(self, field_name, (BACKEND_ROOT / value).resolve())
        return self

    @property
    def cors_origins(self) -> list[str]:
        return [item.strip() for item in self.frontend_origin.split(",") if item.strip()]

    @property
    def llm_configured(self) -> bool:
        return bool(self.llm_model.strip() and self.llm_api_key.get_secret_value().strip())

    @property
    def embedding_configured(self) -> bool:
        if self.embedding_provider == "local":
            return bool(self.local_embedding_model.strip())
        return bool(self.embedding_model.strip() and self.embedding_api_key.get_secret_value().strip())

    def ensure_data_directories(self) -> None:
        self.chroma_path.mkdir(parents=True, exist_ok=True)
        self.upload_path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
