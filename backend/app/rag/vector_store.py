from __future__ import annotations

import logging
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from psycopg.types.json import Json

from app.db.database import connect_postgres
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


class PgVectorStore:
    """PostgreSQL/pgvector-backed course chunk store."""

    def __init__(self, database_url: str, distance_metric: str = "cosine") -> None:
        self.database_url = database_url
        self.distance_metric = distance_metric
        self.ready = False

    def initialize(self) -> None:
        try:
            with connect_postgres(self.database_url) as connection:
                extension = connection.execute(
                    "SELECT 1 FROM pg_extension WHERE extname = 'vector'"
                ).fetchone()
                table = connection.execute("SELECT to_regclass('public.document_chunks') AS name").fetchone()
            if extension is None or table is None or table["name"] is None:
                raise VectorStoreError("PostgreSQL 向量库尚未初始化")
            self.ready = True
        except VectorStoreError:
            raise
        except Exception as exc:
            self.ready = False
            logger.exception("初始化 PostgreSQL 向量库失败")
            raise VectorStoreError(internal_message=str(exc)) from exc

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _vector_literal(embedding: list[float]) -> str:
        if not embedding or not all(math.isfinite(float(value)) for value in embedding):
            raise VectorStoreError("Embedding 包含无效数值")
        return "[" + ",".join(str(float(value)) for value in embedding) + "]"

    def _ensure_ready(self) -> None:
        if not self.ready:
            raise VectorStoreError("PostgreSQL 向量库不可用")

    def upsert(self, course: Course, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise VectorStoreError("向量数量与文档块数量不一致")
        if not chunks:
            return
        self._ensure_ready()
        now = self._now()
        rows = [
            (
                chunk.chunk_id,
                str(chunk.metadata["document_id"]),
                course.value,
                chunk.text,
                Json(chunk.metadata),
                self._vector_literal(embedding),
                now,
                now,
            )
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]
        try:
            with connect_postgres(self.database_url) as connection:
                connection.executemany(
                    """
                    INSERT INTO document_chunks
                        (id, document_id, course, content, metadata, embedding, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?::vector, ?, ?)
                    ON CONFLICT (id) DO UPDATE SET
                        document_id = EXCLUDED.document_id, course = EXCLUDED.course,
                        content = EXCLUDED.content, metadata = EXCLUDED.metadata,
                        embedding = EXCLUDED.embedding, updated_at = EXCLUDED.updated_at
                    """,
                    rows,
                )
        except Exception as exc:
            logger.exception("写入 PostgreSQL 向量失败")
            raise VectorStoreError(internal_message=str(exc)) from exc

    def query(self, course: Course, query_embedding: list[float], top_k: int, topic_ids: list[str] | None = None) -> list[RetrievalCandidate]:
        self._ensure_ready()
        if top_k < 1:
            return []
        operator = {"cosine": "<=>", "l2": "<->", "ip": "<#>"}[self.distance_metric]
        where = ["course = ?"]
        params: list[Any] = [course.value]
        if topic_ids:
            where.append("(" + " OR ".join("jsonb_exists(metadata, ?)" for _ in topic_ids) + ")")
            params.extend(f"topic:{topic_id}" for topic_id in topic_ids)
        vector = self._vector_literal(query_embedding)
        try:
            with connect_postgres(self.database_url) as connection:
                rows = connection.execute(
                    f"SELECT content, metadata, embedding {operator} ?::vector AS distance "
                    f"FROM document_chunks WHERE {' AND '.join(where)} "
                    f"ORDER BY embedding {operator} ?::vector LIMIT ?",
                    [vector, *params, vector, top_k],
                ).fetchall()
            return [
                RetrievalCandidate(text=str(row["content"]), metadata=dict(row["metadata"]), distance=float(row["distance"]))
                for row in rows
            ]
        except Exception as exc:
            logger.exception("查询 PostgreSQL 向量失败")
            raise VectorStoreError(internal_message=str(exc)) from exc

    def list_chunks(self, course: Course) -> list[RetrievalCandidate]:
        self._ensure_ready()
        try:
            with connect_postgres(self.database_url) as connection:
                rows = connection.execute(
                    "SELECT id, content, metadata FROM document_chunks WHERE course = ?", (course.value,)
                ).fetchall()
            return [
                RetrievalCandidate(
                    text=str(row["content"]),
                    metadata={**dict(row["metadata"]), "chunk_id": str(row["id"])},
                    distance=None,
                )
                for row in rows
            ]
        except Exception as exc:
            raise VectorStoreError(internal_message=str(exc)) from exc

    def tag_document(self, course: Course, document_id: str, topic_ids: list[str]) -> int:
        self._ensure_ready()
        try:
            with connect_postgres(self.database_url) as connection:
                rows = connection.execute(
                    "SELECT id, metadata FROM document_chunks WHERE course = ? AND document_id = ?",
                    (course.value, document_id),
                ).fetchall()
                updates = []
                for row in rows:
                    metadata = dict(row["metadata"])
                    for key in [key for key in metadata if key.startswith("topic:")]:
                        metadata[key] = False
                    metadata.update({"topic_ids": json.dumps(topic_ids), "topic_id": topic_ids[0] if topic_ids else ""})
                    metadata.update({f"topic:{topic_id}": True for topic_id in topic_ids})
                    updates.append((Json(metadata), str(row["id"])))
                if updates:
                    connection.executemany("UPDATE document_chunks SET metadata = ?, updated_at = ? WHERE id = ?", [(metadata, self._now(), chunk_id) for metadata, chunk_id in updates])
            return len(rows)
        except Exception as exc:
            raise VectorStoreError(internal_message=str(exc)) from exc

    def delete_document(self, course: Course, document_id: str) -> int:
        self._ensure_ready()
        try:
            with connect_postgres(self.database_url) as connection:
                cursor = connection.execute(
                    "DELETE FROM document_chunks WHERE course = ? AND document_id = ?",
                    (course.value, document_id),
                )
            return cursor.rowcount
        except Exception as exc:
            raise VectorStoreError(internal_message=str(exc)) from exc

    def count(self, course: Course) -> int:
        self._ensure_ready()
        try:
            with connect_postgres(self.database_url) as connection:
                row = connection.execute(
                    "SELECT COUNT(*) AS count FROM document_chunks WHERE course = ?", (course.value,)
                ).fetchone()
            return int(row["count"])
        except Exception as exc:
            raise VectorStoreError(internal_message=str(exc)) from exc
