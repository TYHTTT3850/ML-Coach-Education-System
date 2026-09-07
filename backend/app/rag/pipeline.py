from __future__ import annotations

import re

from starlette.concurrency import run_in_threadpool

from app.core.types import Course
from app.rag.hyde import HyDEGenerator
from app.rag.prompt_builder import PromptBuilder
from app.rag.retriever import Retriever
from app.rag.vector_store import RetrievalCandidate
from app.services.llm.base import LLMClient


STRICT_FALLBACK_ANSWER = (
    "当前课程知识库中没有找到足够支持该问题的资料。请补充课程材料，或尝试换一种问法。"
)


class RAGPipeline:
    def __init__(
        self,
        *,
        retriever: Retriever,
        llm_client: LLMClient,
        prompt_builder: PromptBuilder | None = None,
        strict_mode: bool = True,
        temperature: float = 0.2,
        hyde_generator: HyDEGenerator | None = None,
        hyde_enabled: bool = True,
    ) -> None:
        self.retriever = retriever
        self.llm_client = llm_client
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.strict_mode = strict_mode
        self.temperature = temperature
        self.hyde_generator = hyde_generator or HyDEGenerator(
            llm_client,
            temperature=temperature,
        )
        self.hyde_enabled = hyde_enabled

    async def answer(
        self,
        *,
        course: Course,
        question: str,
        history: list[dict[str, str]],
        retrieval_query: str | None = None,
        system_prompt: str | None = None,
        topic_ids: list[str] | None = None,
    ) -> dict[str, object]:
        effective_retrieval_query = (retrieval_query or question).strip()
        if self.strict_mode and self.retriever.collection_count(course) == 0:
            return {
                "answer": STRICT_FALLBACK_ANSWER,
                "course": course,
                "sources": [],
                "retrieval_count": 0,
            }
        hypothetical_answers = (
            await self.hyde_generator.generate(
                course=course,
                question=effective_retrieval_query,
                history=history,
            )
            if self.hyde_enabled
            else []
        )
        candidates = await run_in_threadpool(
            self.retriever.retrieve,
            course=course,
            question=effective_retrieval_query,
            history=history,
            hypothetical_answers=hypothetical_answers,
            **({"topic_ids": topic_ids} if topic_ids is not None else {}),
        )
        sufficient = self.retriever.has_sufficient_evidence(
            question=effective_retrieval_query,
            history=history,
            candidates=candidates,
        )
        if self.strict_mode and not sufficient:
            return {
                "answer": STRICT_FALLBACK_ANSWER,
                "course": course,
                "sources": [],
                "retrieval_count": 0,
            }
        messages = self.prompt_builder.build_messages(
            course=course,
            question=question,
            history=history,
            sources=candidates,
            system_prompt=system_prompt,
        )
        answer = await self.llm_client.generate(messages, temperature=self.temperature)
        answer = self._remove_invalid_citations(answer, len(candidates))
        return {
            "answer": answer,
            "course": course,
            "sources": [self._source(candidate, index) for index, candidate in enumerate(candidates, 1)],
            "retrieval_count": len(candidates),
        }

    @staticmethod
    def _remove_invalid_citations(answer: str, source_count: int) -> str:
        def normalize_group(match: re.Match[str]) -> str:
            valid: list[int] = []
            for raw_number in re.findall(r"(?i)S\s*(\d+)", match.group(1)):
                number = int(raw_number)
                if 1 <= number <= source_count and number not in valid:
                    valid.append(number)
            if not valid:
                return ""
            return "[" + ", ".join(f"S{number}" for number in valid) + "]"

        return re.sub(
            r"\[([^\[\]]*(?:S|s)\s*\d+[^\[\]]*)\]",
            normalize_group,
            answer,
        ).strip()

    @staticmethod
    def _source(candidate: RetrievalCandidate, index: int) -> dict[str, object]:
        metadata = candidate.metadata
        page_value = metadata.get("page", -1)
        page = page_value if isinstance(page_value, int) and page_value > 0 else None
        snippet = re.sub(r"\s+", " ", candidate.text).strip()
        if len(snippet) > 260:
            snippet = snippet[:257].rstrip() + "..."
        clean_optional = lambda value: str(value).strip() or None
        return {
            "source_id": f"S{index}",
            "source_course": metadata.get("course"),
            "topic_id": metadata.get("topic_id") or None,
            "document_id": str(metadata.get("document_id", "")),
            "source_file": str(metadata.get("source_file", "")),
            "title": clean_optional(metadata.get("title", "")),
            "chapter": clean_optional(metadata.get("chapter", "")),
            "section": clean_optional(metadata.get("section", "")),
            "page": page,
            "chunk_index": int(metadata.get("chunk_index", 0)),
            "snippet": snippet,
            "distance": (
                round(candidate.distance, 6)
                if candidate.distance is not None
                else None
            ),
        }
