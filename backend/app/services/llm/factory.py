import inspect

from app.core.config import Settings
from app.core.exceptions import ConfigurationError
from app.services.llm.base import LLMClient
from app.services.llm.openai_compatible import OpenAICompatibleChatClient
from app.services.llm.openai_responses import OpenAIResponsesClient


class LazyLLMClient:
    """Delay configuration checks and SDK construction until generation is needed."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client: LLMClient | None = None

    async def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
    ) -> str:
        if self._client is None:
            self._client = create_llm_client(self.settings)
        return await self._client.generate(messages, temperature=temperature)

    async def close(self) -> None:
        if self._client is None:
            return
        close = getattr(self._client, "close", None)
        if close is None:
            return
        result = close()
        if inspect.isawaitable(result):
            await result


def create_llm_client(settings: Settings) -> LLMClient:
    api_key = settings.llm_api_key.get_secret_value().strip()
    if not api_key or not settings.llm_model.strip():
        raise ConfigurationError("未配置 LLM_API_KEY 或 LLM_MODEL")
    common = {
        "api_key": api_key,
        "model": settings.llm_model,
        "base_url": settings.llm_base_url,
        "timeout": settings.llm_timeout_seconds,
        "default_temperature": settings.llm_temperature,
    }
    if settings.llm_api_style == "responses" or settings.llm_provider == "openai_responses":
        return OpenAIResponsesClient(**common)
    return OpenAICompatibleChatClient(**common)
