import pytest

from app.core.types import Course
from app.rag.pipeline import RAGPipeline, STRICT_FALLBACK_ANSWER
from app.rag.retriever import Retriever
from tests.mocks import MockEmbeddingClient, MockLLMClient


class EmptyVectorStore:
    def count(self, course):
        return 0

    def query(self, course, query_embedding, top_k):
        return []


@pytest.mark.asyncio
async def test_strict_rag_falls_back_without_calling_llm() -> None:
    llm = MockLLMClient()
    pipeline = RAGPipeline(
        retriever=Retriever(EmptyVectorStore(), MockEmbeddingClient()),  # type: ignore[arg-type]
        llm_client=llm,
        strict_mode=True,
    )
    result = await pipeline.answer(
        course=Course.MACHINE_LEARNING,
        question="资料中没有的问题",
        history=[],
    )
    assert result["answer"] == STRICT_FALLBACK_ANSWER
    assert result["sources"] == []
    assert llm.calls == []


def test_citation_sanitizer_normalizes_combined_and_range_references() -> None:
    answer = "有效 [S1]；组合 [S7, S2]；范围 [S1-S9]；全无效 [S8]。"
    cleaned = RAGPipeline._remove_invalid_citations(answer, source_count=2)
    assert "[S1]" in cleaned
    assert "[S2]" in cleaned
    assert "S7" not in cleaned
    assert "S8" not in cleaned
    assert "S9" not in cleaned
