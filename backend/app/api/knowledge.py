from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict, Field

from app.api.dependencies import knowledge_service
from app.core.exceptions import PayloadTooLargeError
from app.core.security import require_admin_token
from app.core.types import Course
from app.rag.loaders import sanitize_filename, validate_mime_type
from app.schemas.knowledge import (
    DeleteResponse,
    DocumentListResponse,
    DocumentResponse,
    KnowledgeStatsResponse,
    UploadResponse,
)
from app.services.knowledge_service import KnowledgeService


router = APIRouter(prefix="/api/v1/knowledge", tags=["knowledge"])


@router.post(
    "/upload",
    response_model=UploadResponse,
    dependencies=[Depends(require_admin_token)],
)
async def upload(
    request: Request,
    course: Course = Form(...),
    file: UploadFile = File(...),
    topic_ids: list[str] | None = Form(default=None),
    service: KnowledgeService = Depends(knowledge_service),
) -> UploadResponse:
    safe_name = sanitize_filename(file.filename or "")
    validate_mime_type(safe_name, file.content_type)
    limit = request.app.state.settings.max_upload_mb * 1024 * 1024
    content = bytearray()
    while chunk := await file.read(1024 * 1024):
        content.extend(chunk)
        if len(content) > limit:
            raise PayloadTooLargeError(
                f"单文件不能超过 {request.app.state.settings.max_upload_mb}MB"
            )
    result = await run_in_threadpool(
        service.ingest_bytes,
        course=course,
        filename=file.filename or safe_name,
        content_type=file.content_type,
        data=bytes(content),
        topic_ids=topic_ids,
    )
    return UploadResponse(
        document=DocumentResponse.model_validate(result.document),
        duplicate=result.duplicate,
        message="文件已存在，已跳过重复导入" if result.duplicate else "文件已导入知识库",
    )


@router.get("/documents", response_model=DocumentListResponse)
def documents(request: Request, course: Course | None = None) -> DocumentListResponse:
    rows = request.app.state.registry.list_documents(course)
    return DocumentListResponse(
        documents=[DocumentResponse.model_validate(row) for row in rows]
    )


@router.delete(
    "/documents/{document_id}",
    response_model=DeleteResponse,
    dependencies=[Depends(require_admin_token)],
)
def delete_document(document_id: str, request: Request) -> DeleteResponse:
    service = KnowledgeService(
        settings=request.app.state.settings,
        registry=request.app.state.registry,
        vector_store=request.app.state.vector_store,
    )
    service.delete_document(document_id)
    return DeleteResponse(deleted=True, document_id=document_id)


@router.get("/stats", response_model=KnowledgeStatsResponse)
def stats(request: Request) -> KnowledgeStatsResponse:
    return KnowledgeStatsResponse(courses=request.app.state.registry.stats())


class DocumentTagsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    topic_ids: list[str] = Field(min_length=1, max_length=30)


@router.patch("/documents/{document_id}/topics", dependencies=[Depends(require_admin_token)])
def tag_document(document_id: str, payload: DocumentTagsRequest, request: Request):
    service = KnowledgeService(settings=request.app.state.settings,registry=request.app.state.registry,vector_store=request.app.state.vector_store)
    return service.tag_document(document_id,payload.topic_ids)
