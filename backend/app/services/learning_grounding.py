from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from starlette.concurrency import run_in_threadpool

from app.core.types import Course
from app.rag.hyde import HyDEGenerator
from app.rag.pipeline import RAGPipeline
from app.rag.retriever import Retriever
from app.rag.vector_store import RetrievalCandidate


@dataclass(frozen=True, slots=True)
class GroundingResult:
    """Real course chunks selected for one learning operation.

    HyDE text is intentionally absent.  Only immutable snapshots of genuine
    Retrieved candidates cross the boundary into question generation/evaluation.
    """

    query: str
    candidates: list[RetrievalCandidate]
    source_snapshot: list[dict[str, Any]]

    @property
    def source_ids(self) -> set[str]:
        return {str(item["source_id"]) for item in self.source_snapshot}

    @property
    def source_summary(self) -> list[dict[str, Any]]:
        return [
            {
                "source_id": item["source_id"],
                "source_file": item["source_file"],
                "title": item.get("title"),
            }
            for item in self.source_snapshot
        ]


class LearningGroundingService:
    """Course/topic-aware adapter around the existing HyDE + Retriever stack."""

    def __init__(
        self,
        *,
        retriever: Retriever,
        hyde_generator: HyDEGenerator,
        strict_mode: bool = True,
        hyde_enabled: bool = True,
    ) -> None:
        self.retriever = retriever
        self.hyde_generator = hyde_generator
        self.strict_mode = strict_mode
        self.hyde_enabled = hyde_enabled

    async def retrieve(
        self,
        *,
        course: Course,
        topic: Mapping[str, Any],
        purpose: str,
        user_question: str = "",
        error_tags: Iterable[str] = (),
        practice_goal: str = "",
        topic_ids: list[str] | None = None,
    ) -> GroundingResult:
        query = self.build_query(
            topic=topic,
            purpose=purpose,
            user_question=user_question,
            error_tags=error_tags,
            practice_goal=practice_goal,
        )
        if topic_ids == [] or (self.strict_mode and self.retriever.collection_count(course) == 0):
            return GroundingResult(query=query, candidates=[], source_snapshot=[])

        hypotheses = (
            await self.hyde_generator.generate(
                course=course,
                question=query,
                history=[],
            )
            if self.hyde_enabled
            else []
        )
        candidates = await run_in_threadpool(
            self.retriever.retrieve,
            course=course,
            question=query,
            history=[],
            hypothetical_answers=hypotheses,
            **({"topic_ids": topic_ids} if topic_ids is not None else {}),
        )
        if self.strict_mode and not self.retriever.has_sufficient_evidence(
            question=query,
            history=[],
            candidates=candidates,
        ):
            candidates = []
        snapshot = [self._snapshot(candidate, index) for index, candidate in enumerate(candidates, 1)]
        return GroundingResult(
            query=query,
            candidates=candidates,
            source_snapshot=snapshot,
        )

    @staticmethod
    def build_query(
        *,
        topic: Mapping[str, Any],
        purpose: str,
        user_question: str = "",
        error_tags: Iterable[str] = (),
        practice_goal: str = "",
    ) -> str:
        def joined(value: Any) -> str:
            if isinstance(value, str):
                return value.strip()
            if isinstance(value, (list, tuple, set)):
                return "；".join(str(item).strip() for item in value if str(item).strip())
            return ""

        parts = [
            f"知识点：{str(topic.get('title', '')).strip()}",
            f"别名：{joined(topic.get('aliases', []))}",
            f"检索关键词：{joined(topic.get('retrieval_keywords', []))}",
            f"学习目标：{joined(topic.get('learning_objectives', []))}",
            f"检索用途：{purpose}",
        ]
        if user_question.strip():
            parts.append(f"学生当前问题：{user_question.strip()}")
        normalized_errors = joined(list(error_tags))
        if normalized_errors:
            parts.append(f"当前错误标签：{normalized_errors}")
        if practice_goal.strip():
            parts.append(f"当前练习目的：{practice_goal.strip()}")
        return "\n".join(part for part in parts if not part.endswith("："))

    @staticmethod
    def _snapshot(candidate: RetrievalCandidate, index: int) -> dict[str, Any]:
        public = RAGPipeline._source(candidate, index)
        return {
            "source_id": public["source_id"],
            "source_course": public.get("source_course"),
            "topic_id": public.get("topic_id"),
            "document_id": public["document_id"],
            "source_file": public["source_file"],
            "title": public["title"],
            "chapter": public["chapter"],
            "section": public["section"],
            "page": public["page"],
            "chunk_index": public["chunk_index"],
            # Full chunk text is required internally for stable generation and
            # later grading.  It is never returned by public practice schemas.
            "content": candidate.text,
        }
