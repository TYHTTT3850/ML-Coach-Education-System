from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.dependencies import (
    evaluation_service_from_request,
    learning_repository_from_request,
    learning_service_from_request,
    practice_service_from_request,
    require_student_from_header,
)
from app.core.exceptions import TopicLockedError
from app.schemas.practice_api import (
    CreatePracticeSetRequest,
    PracticeSetResponse,
    PracticeSubmitResponse,
    SubmitPracticeSetRequest,
)
from app.services.evaluation_service import EvaluationService
from app.services.learning_repository import LearningRepository
from app.services.learning_service import LearningService
from app.services.practice_service import PracticeService


router = APIRouter(tags=["practice"])


def _student_id(student: dict) -> str:
    return str(student["id"])


def _require_unlocked_topic(
    repository: LearningRepository,
    *,
    student_id: str,
    topic_id: str,
    checkpoint: bool,
    learning: LearningService,
) -> None:
    topic = repository.require_topic(topic_id, student_id=student_id)
    unmet = repository.get_unmet_prerequisites(student_id, topic_id)
    if unmet and checkpoint:
        names = "、".join(item["title"] for item in unmet)
        raise TopicLockedError(
            message=f"请先通过前置知识点：{names}",
            internal_message=f"practice requested for locked topic {topic_id}",
        )
    if checkpoint:
        state = repository.get_topic_state(student_id, topic_id)
        status = str((state or {}).get("status") or "not_started")
        gate = learning.evaluate_topic_gate(student_id, topic, state)
        if status not in {"ready_for_checkpoint", "mastered"} and gate.next_action != "take_checkpoint":
            raise TopicLockedError(
                message="当前练习证据尚不足，完成推荐练习后才能进行综合验收",
                internal_message=f"checkpoint requested while topic {topic_id} is {status}",
            )


@router.post(
    "/api/v1/practice/sets",
    response_model=PracticeSetResponse,
    status_code=201,
)
@router.post(
    "/api/practice-sets",
    response_model=PracticeSetResponse,
    status_code=201,
    include_in_schema=False,
)
async def create_practice_set(
    payload: CreatePracticeSetRequest,
    student: dict = Depends(require_student_from_header),
    repository: LearningRepository = Depends(learning_repository_from_request),
    service: PracticeService = Depends(practice_service_from_request),
    learning: LearningService = Depends(learning_service_from_request),
) -> PracticeSetResponse:
    student_id = _student_id(student)
    _require_unlocked_topic(
        repository,
        student_id=student_id,
        topic_id=payload.topic_id,
        checkpoint=payload.mode == "checkpoint",
        learning=learning,
    )
    return await service.create_set(student_id=student_id, request=payload)


@router.get(
    "/api/v1/practice/sets/{practice_set_id}",
    response_model=PracticeSetResponse,
)
@router.get(
    "/api/practice-sets/{practice_set_id}",
    response_model=PracticeSetResponse,
    include_in_schema=False,
)
def get_practice_set(
    practice_set_id: str,
    student: dict = Depends(require_student_from_header),
    service: PracticeService = Depends(practice_service_from_request),
) -> PracticeSetResponse:
    return service.get_set(
        student_id=_student_id(student),
        practice_set_id=practice_set_id,
    )


@router.post(
    "/api/v1/practice/sets/{practice_set_id}/submit",
    response_model=PracticeSubmitResponse,
)
@router.post(
    "/api/practice-sets/{practice_set_id}/submit",
    response_model=PracticeSubmitResponse,
    include_in_schema=False,
)
async def submit_practice_set(
    practice_set_id: str,
    payload: SubmitPracticeSetRequest,
    student: dict = Depends(require_student_from_header),
    service: EvaluationService = Depends(evaluation_service_from_request),
) -> PracticeSubmitResponse:
    return await service.submit(
        student_id=_student_id(student),
        practice_set_id=practice_set_id,
        request=payload,
    )
