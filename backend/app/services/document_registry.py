from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.exceptions import DuplicateDocumentError
from app.core.types import Course
from app.db.database import connect_sqlite


DOCUMENT_COLUMNS = (
    "id, course, original_filename, stored_filename, sha256, file_size, "
    "chunk_count, status, error_message, created_at, updated_at"
)


class DocumentRegistry:
    """Small SQLite registry; every method owns a short-lived connection."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    course TEXT NOT NULL,
                    original_filename TEXT NOT NULL,
                    stored_filename TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL,
                    error_message TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(course, sha256)
                );
                CREATE INDEX IF NOT EXISTS idx_documents_course
                    ON documents(course);
                CREATE INDEX IF NOT EXISTS idx_documents_updated_at
                    ON documents(updated_at);
                """
            )

    def _connect(self) -> sqlite3.Connection:
        return connect_sqlite(self.path)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _row(row: sqlite3.Row | None) -> dict[str, Any] | None:
        return dict(row) if row is not None else None

    def find_duplicate(self, course: Course, sha256: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                f"SELECT {DOCUMENT_COLUMNS} FROM documents WHERE course = ? AND sha256 = ?",
                (course.value, sha256),
            ).fetchone()
        return self._row(row)

    def create_document(
        self,
        *,
        course: Course,
        original_filename: str,
        stored_filename: str,
        sha256: str,
        file_size: int,
        document_id: str | None = None,
    ) -> dict[str, Any]:
        document_id = document_id or f"doc_{uuid.uuid4().hex}"
        now = self._now()
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO documents (
                        id, course, original_filename, stored_filename, sha256,
                        file_size, chunk_count, status, error_message,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, 0, 'processing', NULL, ?, ?)
                    """,
                    (
                        document_id,
                        course.value,
                        original_filename,
                        stored_filename,
                        sha256,
                        file_size,
                        now,
                        now,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise DuplicateDocumentError(internal_message=str(exc)) from exc
        created = self.get_document(document_id)
        assert created is not None
        return created

    def mark_ready(self, document_id: str, chunk_count: int) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE documents
                SET status = 'ready', chunk_count = ?, error_message = NULL, updated_at = ?
                WHERE id = ?
                """,
                (chunk_count, self._now(), document_id),
            )

    def mark_failed(self, document_id: str, error_message: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE documents
                SET status = 'failed', error_message = ?, updated_at = ?
                WHERE id = ?
                """,
                (error_message[:1000], self._now(), document_id),
            )

    def get_document(self, document_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                f"SELECT {DOCUMENT_COLUMNS} FROM documents WHERE id = ?",
                (document_id,),
            ).fetchone()
        return self._row(row)

    def list_documents(self, course: Course | None = None) -> list[dict[str, Any]]:
        query = f"SELECT {DOCUMENT_COLUMNS} FROM documents"
        params: tuple[str, ...] = ()
        if course is not None:
            query += " WHERE course = ?"
            params = (course.value,)
        query += " ORDER BY created_at DESC"
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def delete_document(self, document_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        return cursor.rowcount > 0

    def stats(self) -> dict[Course, dict[str, Any]]:
        result: dict[Course, dict[str, Any]] = {}
        with self._connect() as connection:
            for course in Course:
                row = connection.execute(
                    """
                    SELECT COUNT(*) AS document_count,
                           COALESCE(SUM(chunk_count), 0) AS chunk_count,
                           MAX(updated_at) AS latest_updated_at
                    FROM documents WHERE course = ?
                    """,
                    (course.value,),
                ).fetchone()
                result[course] = dict(row)
        return result
