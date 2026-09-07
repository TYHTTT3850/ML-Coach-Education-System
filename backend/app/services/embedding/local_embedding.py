from __future__ import annotations

import logging
from typing import Any

from app.core.exceptions import ConfigurationError, EmbeddingServiceError


logger = logging.getLogger(__name__)


class LocalEmbeddingClient:
    """Lazy sentence-transformers adapter so API mode has no torch startup cost."""

    def __init__(self, model_name: str, dimension: int | None = None) -> None:
        self.model_name = model_name
        self.dimension = dimension
        self._model: Any = None

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ConfigurationError(
                    "本地 Embedding 依赖未安装，请安装 requirements-local.txt",
                    internal_message=str(exc),
                ) from exc
            try:
                self._model = SentenceTransformer(self.model_name)
            except Exception as exc:
                logger.exception("加载本地 Embedding 模型失败")
                raise EmbeddingServiceError(internal_message=str(exc)) from exc
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._encode(texts)

    def embed_query(self, text: str) -> list[float]:
        if not text.strip():
            raise EmbeddingServiceError("无法向量化空查询")
        return self._encode([text])[0]

    def _encode(self, texts: list[str]) -> list[list[float]]:
        try:
            result = self._get_model().encode(
                texts,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            vectors = [[float(value) for value in vector] for vector in result]
        except (ConfigurationError, EmbeddingServiceError):
            raise
        except Exception as exc:
            logger.exception("本地 Embedding 计算失败")
            raise EmbeddingServiceError(internal_message=str(exc)) from exc
        if self.dimension is not None and any(len(vector) != self.dimension for vector in vectors):
            raise EmbeddingServiceError(
                "本地 Embedding 维度与 EMBEDDING_DIMENSION 不一致"
            )
        return vectors
