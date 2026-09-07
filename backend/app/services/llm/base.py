from __future__ import annotations

from typing import Protocol


class LLMClient(Protocol):
    async def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
    ) -> str:
        ...
