from __future__ import annotations

import hashlib
import json
import logging
import mimetypes
import uuid
from dataclasses import dataclass
from pathlib import Path

from app.core.config import Settings
from app.core.exceptions import (
    AppError,
    DocumentNotFoundError,
    DocumentProcessingError,
    DuplicateDocumentError,
)
from app.core.types import Course
from app.rag.chunker import StructuredChunker
from app.rag.loaders import (
    DocumentLoader,
    display_filename,
    sanitize_filename,
    validate_mime_type,
)
from app.rag.vector_store import ChromaVectorStore
from app.services.document_registry import DocumentRegistry
from app.services.embedding.base import EmbeddingClient
from app.services.learning_repository import LearningRepository


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class IngestResult:
    document: dict[str, object]
    duplicate: bool


class KnowledgeService:
    def __init__(
        self,
        *,
        settings: Settings,
        registry: DocumentRegistry,
        vector_store: ChromaVectorStore,
        embedding_client: EmbeddingClient | None = None,
        loader: DocumentLoader | None = None,
        chunker: StructuredChunker | None = None,
    ) -> None:
        self.settings = settings
        self.registry = registry
        self.vector_store = vector_store
        self.embedding_client = embedding_client
        self.loader = loader or DocumentLoader()
        self.chunker = chunker or StructuredChunker(
            settings.chunk_size,
            settings.chunk_overlap,
        )

    def ingest_bytes(
        self,
        *,
        course: Course,
        filename: str,
        content_type: str | None,
        data: bytes,
        topic_ids: list[str] | None = None,
    ) -> IngestResult:
        topic_ids = self.validate_topic_ids(course, topic_ids or [])
        original_name = display_filename(filename)
        safe_name = sanitize_filename(original_name)
        validate_mime_type(safe_name, content_type)
        if not data:
            raise DocumentProcessingError("文件内容为空")
        max_bytes = self.settings.max_upload_mb * 1024 * 1024
        if len(data) > max_bytes:
            from app.core.exceptions import PayloadTooLargeError

            raise PayloadTooLargeError(
                f"单文件不能超过 {self.settings.max_upload_mb}MB"
            )
        file_hash = hashlib.sha256(data).hexdigest()
        duplicate = self.registry.find_duplicate(course, file_hash)
        if duplicate is not None:
            return self._duplicate_result(duplicate)

        document_id = f"doc_{uuid.uuid4().hex}"
        relative_path = Path(course.value) / f"{document_id}_{safe_name}"
        stored_path = (self.settings.upload_path / relative_path).resolve()
        upload_root = self.settings.upload_path.resolve()
        if not stored_path.is_relative_to(upload_root):
            raise DocumentProcessingError("文件保存路径无效")
        try:
            document = self.registry.create_document(
                course=course,
                original_filename=original_name,
                stored_filename=relative_path.as_posix(),
                sha256=file_hash,
                file_size=len(data),
                document_id=document_id,
            )
        except DuplicateDocumentError:
            raced_duplicate = self.registry.find_duplicate(course, file_hash)
            if raced_duplicate is not None:
                return self._duplicate_result(raced_duplicate)
            raise

        stored_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            stored_path.write_bytes(data)
            parsed = self.loader.load(stored_path, source_file=original_name)
            chunks = self.chunker.chunk_document(
                parsed,
                document_id=document_id,
                course=course,
                file_hash=file_hash,
            )
            for chunk in chunks:
                if topic_ids:
                    chunk.metadata.update({"topic_ids": json.dumps(topic_ids), "topic_id": topic_ids[0]})
                    chunk.metadata.update({f"topic:{topic}": True for topic in topic_ids})
            if not chunks:
                raise DocumentProcessingError("文档中没有可索引的文本")
            if self.embedding_client is None:
                raise DocumentProcessingError("Embedding 服务不可用")
            embeddings: list[list[float]] = []
            for start in range(0, len(chunks), 64):
                embeddings.extend(
                    self.embedding_client.embed_documents(
                        [chunk.text for chunk in chunks[start : start + 64]]
                    )
                )
            self.vector_store.upsert(course, chunks, embeddings)
            self.registry.mark_ready(document_id, len(chunks))
            ready = self.registry.get_document(document_id)
            assert ready is not None
            return IngestResult(document=ready, duplicate=False)
        except Exception as exc:
            public_error = exc.message if isinstance(exc, AppError) else "文档处理失败，请检查文件内容"
            self.registry.mark_failed(document_id, public_error)
            try:
                self.vector_store.delete_document(course, document_id)
            except Exception:
                logger.exception("回滚文档 %s 的向量失败", document_id)
            logger.exception("导入文档 %s 失败", safe_name)
            if isinstance(exc, AppError):
                raise
            raise DocumentProcessingError(internal_message=str(exc)) from exc

    @staticmethod
    def _duplicate_result(document: dict[str, object]) -> IngestResult:
        status = str(document.get("status", ""))
        if status == "ready":
            return IngestResult(document=document, duplicate=True)
        if status == "failed":
            raise DuplicateDocumentError(
                "相同文件上次导入失败，请先删除失败记录后再重试"
            )
        raise DuplicateDocumentError("相同文件正在处理中，请稍后再试")

    def ingest_path(self, *, course: Course, path: str | Path, topic_ids: list[str] | None = None) -> IngestResult:
        file_path = Path(path)
        mime, _ = mimetypes.guess_type(file_path.name)
        return self.ingest_bytes(
            course=course,
            filename=file_path.name,
            content_type=mime or "application/octet-stream",
            data=file_path.read_bytes(),
            topic_ids=topic_ids,
        )

    def validate_topic_ids(self, course: Course, topic_ids: list[str]) -> list[str]:
        repository = LearningRepository(self.registry.path)
        ids = list(dict.fromkeys(topic_ids))
        for topic_id in ids:
            topic = repository.require_topic(topic_id)
            if topic["course"] != course.value:
                raise DocumentProcessingError("知识点标签必须归属于文档所属课程")
        return ids

    def tag_document(self, document_id: str, topic_ids: list[str]) -> dict:
        document = self.registry.get_document(document_id)
        if document is None:
            raise DocumentNotFoundError()
        course = Course(document["course"])
        ids = self.validate_topic_ids(course, topic_ids)
        count = self.vector_store.tag_document(course, document_id, ids)
        return {"document_id":document_id,"topic_ids":ids,"tagged_chunks":count,"reindexed":False}

    def delete_document(self, document_id: str) -> None:
        document = self.registry.get_document(document_id)
        if document is None:
            raise DocumentNotFoundError()
        course = Course(str(document["course"]))
        self.vector_store.delete_document(course, document_id)
        stored_path = (self.settings.upload_path / str(document["stored_filename"])).resolve()
        upload_root = self.settings.upload_path.resolve()
        if not stored_path.is_relative_to(upload_root):
            raise DocumentProcessingError("文档保存路径无效")
        try:
            stored_path.unlink(missing_ok=True)
        except OSError as exc:
            logger.exception("删除原始文档失败: %s", document_id)
            raise AppError("服务器无法删除原始文件", internal_message=str(exc)) from exc
        if not self.registry.delete_document(document_id):
            raise DocumentNotFoundError()
