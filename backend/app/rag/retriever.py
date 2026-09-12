from __future__ import annotations

import re
import json
from dataclasses import replace
from difflib import SequenceMatcher

from app.core.types import Course
from app.core.exceptions import EmbeddingServiceError
from app.rag.bm25 import BM25Index, tokenize
from app.rag.vector_store import PgVectorStore, RetrievalCandidate
from app.services.embedding.base import EmbeddingClient


RRF_K = 60
MIN_CANDIDATE_POOL = 20


class Retriever:
    """Course-filtered vector/BM25 retrieval with reciprocal-rank fusion."""

    def __init__(
        self,
        vector_store: PgVectorStore,
        embedding_client: EmbeddingClient,
        *,
        top_k: int = 6,
        distance_threshold: float | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.embedding_client = embedding_client
        self.top_k = top_k
        self.distance_threshold = distance_threshold

    @staticmethod
    def build_query(question: str, history: list[dict[str, str]]) -> str:
        recent_user_questions = [
            item["content"].strip()
            for item in history
            if item.get("role") == "user" and item.get("content", "").strip()
        ][-2:]
        parts = [f"当前问题：{question.strip()}"]
        if recent_user_questions:
            parts.append("最近相关问题：" + "；".join(recent_user_questions))
        return "\n".join(parts)

    @classmethod
    def build_vector_queries(
        cls,
        question: str,
        history: list[dict[str, str]],
        hypothetical_answers: list[str] | None = None,
    ) -> list[str]:
        return [
            cls.build_query(question, history),
            *(hypothetical_answers or [])[:3],
        ]

    def retrieve(
        self,
        *,
        course: Course,
        question: str,
        history: list[dict[str, str]],
        hypothetical_answers: list[str] | None = None,
        topic_ids: list[str] | None = None,
    ) -> list[RetrievalCandidate]:
        if topic_ids is not None and not topic_ids:
            return []
        vector_queries = self.build_vector_queries(
            question,
            history,
            hypothetical_answers,
        )
        if len(vector_queries) == 1:
            embeddings = [self.embedding_client.embed_query(vector_queries[0])]
        else:
            embeddings = self.embedding_client.embed_documents(vector_queries)
            if len(embeddings) != len(vector_queries):
                raise EmbeddingServiceError("Embedding 返回数量与 HyDE 查询数量不一致")
        candidate_limit = max(self.top_k * 4, MIN_CANDIDATE_POOL)
        vector_candidate_lists = [
            [
                candidate
                for candidate in self.vector_store.query(course, embedding, candidate_limit, **({"topic_ids": topic_ids} if topic_ids is not None else {}))
                if candidate.distance is not None
                and self._passes_threshold(candidate.distance)
                and candidate.metadata.get("course") == course.value
                and self._matches_topics(candidate, topic_ids)
            ]
            for embedding in embeddings
        ]
        bm25_candidates = self._bm25_candidates(
            course=course,
            # HyDE is deliberately excluded from BM25 so hallucinated terms
            # cannot create false exact-match evidence.
            query=self._retrieval_intent(question, history),
            limit=candidate_limit,
            topic_ids=topic_ids,
        )
        candidates = self._fuse(vector_candidate_lists, bm25_candidates)
        accepted: list[RetrievalCandidate] = []
        for candidate in candidates:
            if candidate.metadata.get("course") != course.value:
                # A defense-in-depth check in addition to separate collections.
                continue
            if any(self._near_duplicate(candidate.text, item.text) for item in accepted):
                continue
            accepted.append(candidate)
            if len(accepted) >= self.top_k:
                break
        return accepted

    def has_sufficient_evidence(
        self,
        *,
        question: str,
        history: list[dict[str, str]],
        candidates: list[RetrievalCandidate],
    ) -> bool:
        """Decide whether retrieved material can support a strict-RAG answer.

        An explicit distance threshold applies to the vector branch, while a
        positive BM25 match remains independent lexical evidence. Without a
        threshold, a conservative lexical policy avoids treating the nearest
        nearest item as relevant merely because the collection is non-empty.
        """
        if not candidates:
            return False
        if self.distance_threshold is not None:
            # Vector candidates already passed the threshold. A positive BM25
            # score is independent lexical evidence and may also support an answer.
            return any(
                candidate.bm25_score > 0
                or (
                    candidate.distance is not None
                    and self._passes_threshold(candidate.distance)
                )
                for candidate in candidates
            )
        # Hypothetical text must not make strict-mode evidence sufficient;
        # only the real question/history and retrieved chunks may do so.
        retrieval_intent = self._retrieval_intent(question, history)
        query_terms = self._lexical_terms(retrieval_intent)
        if not query_terms:
            return False
        for candidate in candidates:
            metadata = candidate.metadata
            evidence_text = " ".join(
                (
                    candidate.text,
                    str(metadata.get("title", "")),
                    str(metadata.get("chapter", "")),
                    str(metadata.get("section", "")),
                )
            )
            candidate_terms = self._lexical_terms(evidence_text)
            if query_terms & candidate_terms:
                return True
        return False

    def _bm25_candidates(
        self,
        *,
        course: Course,
        query: str,
        limit: int,
        topic_ids: list[str] | None = None,
    ) -> list[RetrievalCandidate]:
        list_chunks = getattr(self.vector_store, "list_chunks", None)
        if not callable(list_chunks):
            return []
        corpus = [
            candidate
            for candidate in list_chunks(course)
            if candidate.metadata.get("course") == course.value
            and self._matches_topics(candidate, topic_ids)
        ]
        if not corpus:
            return []
        index = BM25Index([self._bm25_document(candidate) for candidate in corpus])
        return [
            replace(corpus[index_number], distance=None, bm25_score=score)
            for index_number, score in index.search(query, limit)
        ]

    @staticmethod
    def _matches_topics(candidate: RetrievalCandidate, topic_ids: list[str] | None) -> bool:
        if topic_ids is None:
            return True
        metadata = candidate.metadata
        raw = metadata.get("topic_ids", [])
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except (ValueError, TypeError):
                raw = []
        tagged = set(raw) if isinstance(raw, list) and all(isinstance(item, str) for item in raw) else set()
        if metadata.get("topic_id"):
            tagged.add(metadata["topic_id"])
        return bool(tagged.intersection(topic_ids))

    @classmethod
    def _fuse(
        cls,
        vector_candidate_lists: list[list[RetrievalCandidate]],
        bm25_candidates: list[RetrievalCandidate],
    ) -> list[RetrievalCandidate]:
        candidates: dict[str, RetrievalCandidate] = {}
        scores: dict[str, float] = {}
        vector_ranks: dict[str, int] = {}
        bm25_ranks: dict[str, int] = {}

        for vector_candidates in vector_candidate_lists:
            for rank, candidate in enumerate(vector_candidates, start=1):
                chunk_id = cls._candidate_id(candidate)
                existing = candidates.get(chunk_id)
                if (
                    existing is None
                    or existing.distance is None
                    or (
                        candidate.distance is not None
                        and candidate.distance < existing.distance
                    )
                ):
                    candidates[chunk_id] = candidate
                vector_ranks[chunk_id] = min(
                    rank,
                    vector_ranks.get(chunk_id, rank),
                )
                scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)

        for rank, candidate in enumerate(bm25_candidates, start=1):
            chunk_id = cls._candidate_id(candidate)
            existing = candidates.get(chunk_id)
            if existing is not None:
                candidate = replace(existing, bm25_score=candidate.bm25_score)
            candidates[chunk_id] = candidate
            bm25_ranks[chunk_id] = rank
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)

        fused = [
            replace(candidate, fusion_score=scores[chunk_id])
            for chunk_id, candidate in candidates.items()
        ]
        infinity = len(fused) + MIN_CANDIDATE_POOL
        return sorted(
            fused,
            key=lambda candidate: (
                -candidate.fusion_score,
                min(
                    vector_ranks.get(cls._candidate_id(candidate), infinity),
                    bm25_ranks.get(cls._candidate_id(candidate), infinity),
                ),
                -candidate.bm25_score,
                vector_ranks.get(cls._candidate_id(candidate), infinity),
                cls._candidate_id(candidate),
            ),
        )

    @staticmethod
    def _candidate_id(candidate: RetrievalCandidate) -> str:
        metadata = candidate.metadata
        return str(
            metadata.get("chunk_id")
            or f"{metadata.get('document_id', '')}:{metadata.get('chunk_index', 0)}"
        )

    @staticmethod
    def _bm25_document(candidate: RetrievalCandidate) -> str:
        metadata = candidate.metadata
        return " ".join(
            str(value)
            for value in (
                metadata.get("source_file", ""),
                metadata.get("title", ""),
                metadata.get("chapter", ""),
                metadata.get("section", ""),
                candidate.text,
            )
            if value
        )

    @staticmethod
    def _retrieval_intent(
        question: str,
        history: list[dict[str, str]],
        hypothetical_answers: list[str] | None = None,
    ) -> str:
        recent_user_questions = [
            item.get("content", "").strip()
            for item in history
            if item.get("role") == "user" and item.get("content", "").strip()
        ][-2:]
        return " ".join(
            [
                *recent_user_questions,
                question.strip(),
                *(hypothetical_answers or [])[:3],
            ]
        )

    def collection_count(self, course: Course) -> int | None:
        count = getattr(self.vector_store, "count", None)
        return int(count(course)) if callable(count) else None

    @staticmethod
    def _lexical_terms(text: str) -> set[str]:
        return set(tokenize(text))

    def _passes_threshold(self, distance: float) -> bool:
        # pgvector exposes distance (smaller is more similar) for cosine, L2 and IP spaces.
        # Keeping this policy here avoids scattering metric assumptions through the app.
        return self.distance_threshold is None or distance <= self.distance_threshold

    @staticmethod
    def _near_duplicate(left: str, right: str) -> bool:
        normalize = lambda value: re.sub(r"\s+", "", value).lower()
        normalized_left = normalize(left)
        normalized_right = normalize(right)
        if normalized_left == normalized_right:
            return True
        shorter = min(len(normalized_left), len(normalized_right))
        if shorter == 0:
            return True
        if shorter < 80:
            return False
        return SequenceMatcher(None, normalized_left, normalized_right).ratio() >= 0.92
