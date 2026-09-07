from __future__ import annotations

import pytest

from app.core.types import Course
from app.rag.hyde import HyDEGenerator
from app.rag.pipeline import RAGPipeline
from app.rag.vector_store import RetrievalCandidate


class SequenceLLM:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[list[dict[str, str]]] = []

    async def generate(self, messages, temperature=None) -> str:
        self.calls.append(messages)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return str(outcome)


class RecordingRetriever:
    def __init__(self) -> None:
        self.retrieve_calls: list[dict[str, object]] = []
        self.candidate = RetrievalCandidate(
            text="过拟合表现为训练误差低而验证误差高。",
            metadata={
                "chunk_id": "chunk-overfit",
                "document_id": "doc-overfit",
                "course": Course.MACHINE_LEARNING.value,
                "source_file": "overfit.md",
                "title": "过拟合",
                "chapter": "模型评估",
                "section": "泛化",
                "page": -1,
                "chunk_index": 0,
            },
            distance=0.1,
        )

    def collection_count(self, course):
        return 1

    def retrieve(self, **kwargs):
        self.retrieve_calls.append(kwargs)
        return [self.candidate]

    def has_sufficient_evidence(self, *, question, history, candidates):
        return bool(candidates)


@pytest.mark.asyncio
async def test_hyde_uses_one_generation_then_final_answer_and_stays_internal() -> None:
    hypotheses = [
        "模型记住了训练样本中的噪声。",
        "模型复杂度相对训练数据过高。",
        "正则化不足会降低泛化能力。",
    ]
    llm = SequenceLLM(
        [
            "[\"模型记住了训练样本中的噪声。\", \"模型复杂度相对训练数据过高。\", "
            "\"正则化不足会降低泛化能力。\"]",
            "最终教学回答。[S1]",
        ]
    )
    retriever = RecordingRetriever()
    history = [{"role": "user", "content": "上一问谈到泛化"}]
    pipeline = RAGPipeline(retriever=retriever, llm_client=llm)  # type: ignore[arg-type]

    result = await pipeline.answer(
        course=Course.MACHINE_LEARNING,
        question="为什么会过拟合？",
        history=history,
    )

    assert len(llm.calls) == 2
    assert "HyDE" in llm.calls[0][0]["content"]
    assert retriever.retrieve_calls[0]["hypothetical_answers"] == hypotheses
    final_prompt = "\n".join(message["content"] for message in llm.calls[1])
    assert all(hypothesis not in final_prompt for hypothesis in hypotheses)
    assert result["answer"] == "最终教学回答。[S1]"
    assert result["sources"][0]["snippet"] == retriever.candidate.text
    assert history == [{"role": "user", "content": "上一问谈到泛化"}]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "first_outcome",
    [RuntimeError("HyDE unavailable"), "这是一段不符合结构要求的自由文本。"],
)
async def test_hyde_failure_or_invalid_format_falls_back_to_original_query(
    first_outcome: object,
) -> None:
    llm = SequenceLLM([first_outcome, "降级后仍能回答。[S1]"])
    retriever = RecordingRetriever()
    pipeline = RAGPipeline(retriever=retriever, llm_client=llm)  # type: ignore[arg-type]

    result = await pipeline.answer(
        course=Course.MACHINE_LEARNING,
        question="为什么会过拟合？",
        history=[],
    )

    assert len(llm.calls) == 2
    assert retriever.retrieve_calls[0]["hypothetical_answers"] == []
    assert result["answer"] == "降级后仍能回答。[S1]"


@pytest.mark.asyncio
async def test_hyde_can_be_disabled_without_extra_llm_call() -> None:
    llm = SequenceLLM(["关闭 HyDE 后的最终回答。[S1]"])
    retriever = RecordingRetriever()
    pipeline = RAGPipeline(
        retriever=retriever,  # type: ignore[arg-type]
        llm_client=llm,
        hyde_enabled=False,
    )

    result = await pipeline.answer(
        course=Course.MACHINE_LEARNING,
        question="为什么会过拟合？",
        history=[],
    )

    assert len(llm.calls) == 1
    assert retriever.retrieve_calls[0]["hypothetical_answers"] == []
    assert result["answer"] == "关闭 HyDE 后的最终回答。[S1]"


@pytest.mark.parametrize(
    "response",
    [
        '["只有一条"]',
        '["第一条", "第二条"]',
        '["第一条", "第二条", "第三条", "第四条"]',
        '["重复", "重复", "第三条"]',
        '["Same", "same", "第三条"]',
        '["第一条", "", "第三条"]',
        '{"hypotheses": ["第一条", "第二条", "第三条"]}',
        "1. 第一条\n2. 第二条\n3. 第三条",
    ],
)
def test_hyde_parser_rejects_non_exact_or_non_unique_triplets(response: str) -> None:
    assert HyDEGenerator.parse(response) == []


def test_hyde_parser_accepts_json_fenced_triplet() -> None:
    assert HyDEGenerator.parse('```json\n["甲答案", "乙答案", "丙答案"]\n```') == [
        "甲答案",
        "乙答案",
        "丙答案",
    ]
