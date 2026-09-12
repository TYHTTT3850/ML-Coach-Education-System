from app.core.config import Settings
from app.core.exceptions import ConfigurationError
from app.services.embedding.api_embedding import APIEmbeddingClient
from app.services.embedding.base import EmbeddingClient


def create_embedding_client(settings: Settings, api_key: str, base_url: str, model: str) -> EmbeddingClient:
    api_key = api_key.strip()
    if not api_key or not base_url.strip() or not model.strip():
        raise ConfigurationError("请先在设置页填写厂商、Embedding API Key 和 Embedding 模型")
    return APIEmbeddingClient(
        api_key=api_key,
        model=model.strip(),
        base_url=base_url.strip(),
        timeout=settings.llm_timeout_seconds,
        dimension=settings.embedding_dimension,
    )
