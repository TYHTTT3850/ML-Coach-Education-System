from __future__ import annotations

from fastapi import APIRouter, Header, Request

from app.schemas.student import AnonymousStudentResponse
from app.services.learning_repository import LearningRepository


router = APIRouter(prefix="/api/v1/students", tags=["students"])


def learning_repository_from_request(request: Request) -> LearningRepository:
    return request.app.state.learning_repository


def require_student_from_header(
    request: Request,
    x_student_id: str = Header(..., alias="X-Student-ID"),
) -> dict:
    """Reusable dependency for every stateful learning request; also touches activity."""

    return learning_repository_from_request(request).require_student(x_student_id, touch=True)


@router.post("/anonymous", response_model=AnonymousStudentResponse, status_code=201)
def create_anonymous_student(request: Request) -> AnonymousStudentResponse:
    student = learning_repository_from_request(request).create_anonymous_student()
    return AnonymousStudentResponse(
        student_id=student["id"],
        created_at=student["created_at"],
    )
