from app.core.config import Settings
from app.core.exceptions import ConfigurationError
from app.services.embedding.api_embedding import APIEmbeddingClient
from app.services.embedding.base import EmbeddingClient
from app.services.embedding.local_embedding import LocalEmbeddingClient


def create_embedding_client(settings: Settings) -> EmbeddingClient:
    if settings.embedding_provider == "local":
        if not settings.local_embedding_model.strip():
            raise ConfigurationError("未配置 LOCAL_EMBEDDING_MODEL")
        return LocalEmbeddingClient(
            settings.local_embedding_model,
            settings.embedding_dimension,
        )
    api_key = settings.embedding_api_key.get_secret_value().strip()
    if not api_key or not settings.embedding_model.strip():
        raise ConfigurationError("未配置 EMBEDDING_API_KEY 或 EMBEDDING_MODEL")
    return APIEmbeddingClient(
        api_key=api_key,
        model=settings.embedding_model,
        base_url=settings.embedding_base_url,
        timeout=settings.llm_timeout_seconds,
        dimension=settings.embedding_dimension,
    )
