from __future__ import annotations

import logging

from app.core.exceptions import LLMServiceError


logger = logging.getLogger(__name__)


class OpenAIResponsesClient:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = "",
        timeout: float = 60,
        default_temperature: float = 0.2,
    ) -> None:
        from openai import AsyncOpenAI

        kwargs: dict[str, object] = {"api_key": api_key, "timeout": timeout}
        if base_url:
            kwargs["base_url"] = base_url.rstrip("/")
        self.client = AsyncOpenAI(**kwargs)
        self.model = model
        self.default_temperature = default_temperature

    async def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
    ) -> str:
        try:
            response = await self.client.responses.create(
                model=self.model,
                input=messages,  # type: ignore[arg-type]
                temperature=self.default_temperature if temperature is None else temperature,
            )
            content = (response.output_text or "").strip()
            if not content:
                raise LLMServiceError("大模型返回了空回复")
            return content
        except LLMServiceError:
            raise
        except Exception as exc:
            logger.exception("Responses API 调用失败: %s", type(exc).__name__)
            raise LLMServiceError(internal_message=f"{type(exc).__name__}: {exc}") from exc

    async def close(self) -> None:
        await self.client.close()
