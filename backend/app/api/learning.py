from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, ConfigDict, Field

from app.api.dependencies import (
    learning_repository_from_request,
    learning_service_from_request,
    remediation_service_from_request,
    require_student_from_header,
)
from app.core.exceptions import CourseTopicMismatchError
from app.core.types import Course
from app.schemas.practice_api import RemediationRequest, RemediationResponse
from app.services.learning_repository import LearningRepository
from app.services.learning_service import LearningService
from app.services.remediation_service import RemediationService


router = APIRouter(tags=["learning"])


class StartLearningRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    topic_id: str = Field(pattern=r"^(?:ml|math|la|prob)\.[a-z0-9_]+$")


class RemediationByAttemptRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    topic_id: str = Field(pattern=r"^(?:ml|math|la|prob)\.[a-z0-9_]+$")


def _student_id(student: dict) -> str:
    return str(student["id"])


def _require_course_topic(
    repository: LearningRepository,
    *,
    course: Course,
    topic_id: str,
    student_id: str,
) -> dict[str, Any]:
    topic = repository.require_topic(topic_id, student_id=student_id)
    if str(topic["course"]) != course.value:
        raise CourseTopicMismatchError(
            internal_message=(
                f"topic {topic_id} belongs to {topic['course']}, requested {course.value}"
            )
        )
    return topic


@router.get("/api/v1/learning/dashboard")
@router.get("/api/learning/{course}/dashboard", include_in_schema=False)
@router.get("/api/v1/learning/{course}/dashboard", include_in_schema=False)
def learning_dashboard(
    course: Course,
    student: dict = Depends(require_student_from_header),
    service: LearningService = Depends(learning_service_from_request),
) -> dict[str, Any]:
    return service.dashboard(student_id=_student_id(student), course=course)


@router.post("/api/v1/learning/topics/{topic_id}/start")
def start_topic(
    topic_id: str,
    student: dict = Depends(require_student_from_header),
    service: LearningService = Depends(learning_service_from_request),
) -> dict[str, Any]:
    return service.start_topic(student_id=_student_id(student), topic_id=topic_id)


@router.post("/api/learning/{course}/start", include_in_schema=False)
def start_topic_compatibility(
    course: Course,
    payload: StartLearningRequest,
    student: dict = Depends(require_student_from_header),
    repository: LearningRepository = Depends(learning_repository_from_request),
    service: LearningService = Depends(learning_service_from_request),
) -> dict[str, Any]:
    student_id = _student_id(student)
    _require_course_topic(
        repository,
        course=course,
        topic_id=payload.topic_id,
        student_id=student_id,
    )
    return service.start_topic(student_id=student_id, topic_id=payload.topic_id)


@router.get("/api/learning/{course}/topics/{topic_id}", include_in_schema=False)
@router.get("/api/v1/learning/{course}/topics/{topic_id}", include_in_schema=False)
def learning_topic_detail(
    course: Course,
    topic_id: str,
    student: dict = Depends(require_student_from_header),
    repository: LearningRepository = Depends(learning_repository_from_request),
    service: LearningService = Depends(learning_service_from_request),
) -> dict[str, Any]:
    student_id = _student_id(student)
    _require_course_topic(
        repository,
        course=course,
        topic_id=topic_id,
        student_id=student_id,
    )
    return service.topic_detail(student_id=student_id, topic_id=topic_id)


@router.post(
    "/api/v1/learning/remediation",
    response_model=RemediationResponse,
)
async def create_remediation(
    payload: RemediationRequest,
    student: dict = Depends(require_student_from_header),
    service: RemediationService = Depends(remediation_service_from_request),
) -> RemediationResponse:
    return await service.create(
        student_id=_student_id(student),
        topic_id=payload.topic_id,
        attempt_id=payload.attempt_id,
    )


@router.post(
    "/api/attempts/{attempt_id}/remediation",
    response_model=RemediationResponse,
    include_in_schema=False,
)
async def create_remediation_compatibility(
    attempt_id: str,
    payload: RemediationByAttemptRequest,
    student: dict = Depends(require_student_from_header),
    service: RemediationService = Depends(remediation_service_from_request),
) -> RemediationResponse:
    return await service.create(
        student_id=_student_id(student),
        topic_id=payload.topic_id,
        attempt_id=attempt_id,
    )


@router.get("/api/v1/learning/history")
@router.get("/api/learning/{course}/history", include_in_schema=False)
@router.get("/api/v1/learning/{course}/history", include_in_schema=False)
def learning_history(
    course: Course,
    limit: int = Query(default=20, ge=1, le=100),
    student: dict = Depends(require_student_from_header),
    service: LearningService = Depends(learning_service_from_request),
) -> dict[str, Any]:
    return service.history(
        student_id=_student_id(student),
        course=course,
        limit=limit,
    )
