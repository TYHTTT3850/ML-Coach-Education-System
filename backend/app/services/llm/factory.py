import inspect

from app.core.config import Settings
from app.core.exceptions import ConfigurationError
from app.services.llm.base import LLMClient
from app.services.llm.openai_compatible import OpenAICompatibleChatClient


class LazyLLMClient:
    """Delay configuration checks and SDK construction until generation is needed."""

    def __init__(self, settings: Settings, api_key: str, base_url: str, model: str) -> None:
        self.settings = settings
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self._client: LLMClient | None = None

    async def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
    ) -> str:
        if self._client is None:
            self._client = create_llm_client(self.settings, self.api_key, self.base_url, self.model)
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


def create_llm_client(settings: Settings, api_key: str, base_url: str, model: str) -> LLMClient:
    api_key = api_key.strip()
    if not api_key or not base_url.strip() or not model.strip():
        raise ConfigurationError("请先在设置页填写厂商、API Key 和模型")
    common = {
        "api_key": api_key,
        "model": model.strip(),
        "base_url": base_url.strip(),
        "timeout": settings.llm_timeout_seconds,
        "default_temperature": settings.llm_temperature,
    }
    return OpenAICompatibleChatClient(**common)
