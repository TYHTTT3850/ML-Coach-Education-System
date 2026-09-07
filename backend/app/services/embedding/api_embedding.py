from __future__ import annotations

import logging

from app.core.exceptions import EmbeddingServiceError


logger = logging.getLogger(__name__)


MAX_API_BATCH_SIZE = 10


class APIEmbeddingClient:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = "",
        timeout: float = 60,
        dimension: int | None = None,
    ) -> None:
        from openai import OpenAI

        kwargs: dict[str, object] = {"api_key": api_key, "timeout": timeout}
        if base_url:
            kwargs["base_url"] = base_url.rstrip("/")
        self.client = OpenAI(**kwargs)
        self.model = model
        self.dimension = dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        if not text.strip():
            raise EmbeddingServiceError("无法向量化空查询")
        return self._embed([text])[0]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        try:
            vectors: list[list[float]] = []
            for start in range(0, len(texts), MAX_API_BATCH_SIZE):
                batch = texts[start : start + MAX_API_BATCH_SIZE]
                kwargs: dict[str, object] = {"model": self.model, "input": batch}
                if self.dimension is not None:
                    kwargs["dimensions"] = self.dimension
                response = self.client.embeddings.create(**kwargs)
                ordered = sorted(response.data, key=lambda item: item.index)
                vectors.extend(
                    [float(value) for value in item.embedding]
                    for item in ordered
                )
            self._validate(vectors, len(texts))
            return vectors
        except EmbeddingServiceError:
            raise
        except Exception as exc:
            logger.exception("Embedding API 调用失败: %s", type(exc).__name__)
            raise EmbeddingServiceError(internal_message=f"{type(exc).__name__}: {exc}") from exc

    def _validate(self, vectors: list[list[float]], expected_count: int) -> None:
        if len(vectors) != expected_count or any(not vector for vector in vectors):
            raise EmbeddingServiceError("Embedding API 返回了无效结果")
        dimensions = {len(vector) for vector in vectors}
        if len(dimensions) != 1:
            raise EmbeddingServiceError("Embedding API 返回的向量维度不一致")
        actual_dimension = next(iter(dimensions))
        if self.dimension is not None and actual_dimension != self.dimension:
            raise EmbeddingServiceError(
                "Embedding API 返回维度与 EMBEDDING_DIMENSION 不一致",
                internal_message=(
                    f"configured embedding dimension {self.dimension}, got {actual_dimension}"
                ),
            )

    def close(self) -> None:
        self.client.close()
