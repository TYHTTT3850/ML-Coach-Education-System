from __future__ import annotations

import hashlib
import math


class MockEmbeddingClient:
    def __init__(self, dimension: int = 16) -> None:
        self.dimension = dimension

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        normalized = "".join(text.lower().split())
        for index, character in enumerate(normalized):
            digest = hashlib.sha256(f"{index % 3}:{character}".encode("utf-8")).digest()
            vector[digest[0] % self.dimension] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class MockLLMClient:
    def __init__(self, answer: str = "这是基于课程资料的测试回答。[S1]") -> None:
        self.answer = answer
        self.calls: list[list[dict[str, str]]] = []

    async def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
    ) -> str:
        self.calls.append(messages)
        return self.answer
