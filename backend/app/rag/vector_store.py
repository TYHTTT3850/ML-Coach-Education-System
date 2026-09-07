from __future__ import annotations

import logging
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.exceptions import VectorStoreError
from app.core.types import Course
from app.rag.chunker import Chunk


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RetrievalCandidate:
    text: str
    metadata: dict[str, Any]
    distance: float | None
    bm25_score: float = 0.0
    fusion_score: float = 0.0


class ChromaVectorStore:
    """Physically separate Chroma collections, one per registered course domain."""

    def __init__(self, path: Path, distance_metric: str = "cosine") -> None:
        self.path = Path(path)
        self.distance_metric = distance_metric
        self._client: Any = None
        self._collections: dict[Course, Any] = {}
        self.ready = False

    def initialize(self) -> None:
        try:
            import chromadb

            self.path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self.path))
            self._collections = {
                course: self._client.get_or_create_collection(
                    name=course.collection_name,
                    metadata={"hnsw:space": self.distance_metric, "course": course.value},
                    embedding_function=None,
                )
                for course in Course
            }
            self.ready = True
        except Exception as exc:
            self.ready = False
            logger.exception("初始化 Chroma 失败")
            raise VectorStoreError(internal_message=str(exc)) from exc

    def _collection(self, course: Course) -> Any:
        if not self.ready or course not in self._collections:
            raise VectorStoreError()
        return self._collections[course]

    def upsert(self, course: Course, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise VectorStoreError(
                "向量数量与文档块数量不一致",
                internal_message=f"{len(chunks)} chunks, {len(embeddings)} vectors",
            )
        if not chunks:
            return
        try:
            collection = self._collection(course)
            for start in range(0, len(chunks), 128):
                batch = chunks[start : start + 128]
                collection.upsert(
                    ids=[chunk.chunk_id for chunk in batch],
                    documents=[chunk.text for chunk in batch],
                    metadatas=[chunk.metadata for chunk in batch],
                    embeddings=embeddings[start : start + len(batch)],
                )
        except VectorStoreError:
            raise
        except Exception as exc:
            logger.exception("写入 Chroma 失败")
            raise VectorStoreError(internal_message=str(exc)) from exc

    def query(
        self,
        course: Course,
        query_embedding: list[float],
        top_k: int,
        topic_ids: list[str] | None = None,
    ) -> list[RetrievalCandidate]:
        try:
            collection = self._collection(course)
            count = int(collection.count())
            if count == 0:
                return []
            result = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, count),
                include=["documents", "metadatas", "distances"],
                **({"where": {f"topic:{topic_ids[0]}": True} if len(topic_ids) == 1 else {"$or": [{f"topic:{topic}": True} for topic in topic_ids]}} if topic_ids else {}),
            )
            documents = (result.get("documents") or [[]])[0]
            metadatas = (result.get("metadatas") or [[]])[0]
            distances = (result.get("distances") or [[]])[0]
            candidates: list[RetrievalCandidate] = []
            for text, metadata, distance in zip(documents, metadatas, distances, strict=False):
                if text and metadata:
                    candidates.append(
                        RetrievalCandidate(
                            text=str(text),
                            metadata=dict(metadata),
                            distance=float(distance),
                        )
                    )
            return candidates
        except VectorStoreError:
            raise
        except Exception as exc:
            logger.exception("查询 Chroma 失败")
            raise VectorStoreError(internal_message=str(exc)) from exc

    def tag_document(self, course: Course, document_id: str, topic_ids: list[str]) -> int:
        """Explicit metadata-only annotation; vectors and source text remain intact."""
        collection = self._collection(course)
        result = collection.get(where={"document_id": document_id}, include=["metadatas"])
        ids = list(result.get("ids") or [])
        metadatas = []
        for original in result.get("metadatas") or []:
            metadata = dict(original or {})
            for key in list(metadata):
                if key.startswith("topic:"):
                    metadata[key] = False
            metadata.update({"topic_ids": json.dumps(topic_ids), "topic_id": topic_ids[0] if topic_ids else ""})
            metadata.update({f"topic:{topic}": True for topic in topic_ids})
            metadatas.append(metadata)
        if ids:
            collection.update(ids=ids, metadatas=metadatas)
        return len(ids)

    def list_chunks(self, course: Course) -> list[RetrievalCandidate]:
        """Return every stored chunk for course-scoped lexical retrieval."""
        try:
            collection = self._collection(course)
            if int(collection.count()) == 0:
                return []
            result = collection.get(include=["documents", "metadatas"])
            ids = list(result.get("ids") or [])
            documents = list(result.get("documents") or [])
            metadatas = list(result.get("metadatas") or [])
            candidates: list[RetrievalCandidate] = []
            for chunk_id, text, metadata in zip(
                ids,
                documents,
                metadatas,
                strict=False,
            ):
                if not text or not metadata:
                    continue
                normalized_metadata = dict(metadata)
                normalized_metadata.setdefault("chunk_id", str(chunk_id))
                candidates.append(
                    RetrievalCandidate(
                        text=str(text),
                        metadata=normalized_metadata,
                        distance=None,
                    )
                )
            return candidates
        except VectorStoreError:
            raise
        except Exception as exc:
            logger.exception("读取 Chroma 文档块失败")
            raise VectorStoreError(internal_message=str(exc)) from exc

    def delete_document(self, course: Course, document_id: str) -> int:
        try:
            collection = self._collection(course)
            existing = collection.get(where={"document_id": document_id}, include=[])
            ids = list(existing.get("ids") or [])
            if ids:
                collection.delete(ids=ids)
            return len(ids)
        except VectorStoreError:
            raise
        except Exception as exc:
            logger.exception("删除 Chroma 文档块失败")
            raise VectorStoreError(internal_message=str(exc)) from exc

    def count(self, course: Course) -> int:
        try:
            return int(self._collection(course).count())
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError(internal_message=str(exc)) from exc
