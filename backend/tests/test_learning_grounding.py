from __future__ import annotations

import pytest

from app.core.types import Course
from app.rag.vector_store import RetrievalCandidate
from app.services.learning_grounding import LearningGroundingService


class RecordingHyDE:
    def __init__(self) -> None:
        self.answers = ["假设一", "假设二", "假设三"]
        self.calls: list[dict[str, object]] = []

    async def generate(self, **kwargs):
        self.calls.append(kwargs)
        return self.answers


class RecordingRetriever:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def collection_count(self, course):
        return 1

    def retrieve(self, **kwargs):
        self.calls.append(kwargs)
        return [
            RetrievalCandidate(
                text="逻辑回归使用伯努利似然建立目标函数。",
                metadata={
                    "chunk_id": "chunk-1",
                    "document_id": "doc-1",
                    "course": Course.MACHINE_LEARNING.value,
                    "source_file": "logistic.md",
                    "title": "逻辑回归",
                    "chapter": "分类",
                    "section": "损失函数",
                    "page": -1,
                    "chunk_index": 3,
                },
                distance=0.1,
            )
        ]

    def has_sufficient_evidence(self, **kwargs):
        return bool(kwargs["candidates"])


@pytest.mark.asyncio
async def test_learning_grounding_reuses_hyde_but_snapshots_only_real_chunks() -> None:
    hyde = RecordingHyDE()
    retriever = RecordingRetriever()
    service = LearningGroundingService(
        retriever=retriever,  # type: ignore[arg-type]
        hyde_generator=hyde,  # type: ignore[arg-type]
    )

    result = await service.retrieve(
        course=Course.MACHINE_LEARNING,
        topic={
            "title": "逻辑回归",
            "aliases": ["Logistic Regression"],
            "retrieval_keywords": ["伯努利似然", "交叉熵"],
            "learning_objectives": ["解释损失函数"],
        },
        purpose="practice_generation",
        practice_goal="考察机制",
    )

    assert retriever.calls[0]["hypothetical_answers"] == hyde.answers
    assert result.source_ids == {"S1"}
    serialized = repr(result.source_snapshot)
    assert "逻辑回归使用伯努利似然" in serialized
    assert all(answer not in serialized for answer in hyde.answers)


@pytest.mark.asyncio
async def test_learning_grounding_empty_strict_library_skips_hyde() -> None:
    hyde = RecordingHyDE()
    retriever = RecordingRetriever()
    retriever.collection_count = lambda course: 0  # type: ignore[method-assign]
    service = LearningGroundingService(
        retriever=retriever,  # type: ignore[arg-type]
        hyde_generator=hyde,  # type: ignore[arg-type]
    )

    result = await service.retrieve(
        course=Course.ADVANCED_MATHEMATICS,
        topic={"title": "导数定义"},
        purpose="practice_generation",
    )

    assert result.source_snapshot == []
    assert hyde.calls == []
    assert retriever.calls == []
