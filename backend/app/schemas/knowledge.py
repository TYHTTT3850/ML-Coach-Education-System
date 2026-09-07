from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel

from app.core.types import Course


DocumentStatus = Literal["processing", "ready", "failed"]


class DocumentResponse(BaseModel):
    id: str
    course: Course
    original_filename: str
    stored_filename: str
    sha256: str
    file_size: int
    chunk_count: int
    status: DocumentStatus
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime


class UploadResponse(BaseModel):
    document: DocumentResponse
    duplicate: bool
    message: str


class DocumentListResponse(BaseModel):
    documents: list[DocumentResponse]


class DeleteResponse(BaseModel):
    deleted: bool
    document_id: str


class CourseStats(BaseModel):
    document_count: int
    chunk_count: int
    latest_updated_at: datetime | None


class KnowledgeStatsResponse(BaseModel):
    courses: dict[Course, CourseStats]
