from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.schemas.chat import SourceResponse
from app.schemas.learning import MasteryGateResult
from app.schemas.practice import PracticeQuestionPublic


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CreatePracticeSetRequest(StrictModel):
    topic_id: str = Field(pattern=r"^(?:ml|math|la|prob)\.[a-z0-9_]+$")
    mode: Literal["practice", "checkpoint"] = "practice"
    count: int | None = Field(default=None, ge=1, le=8)
    difficulty: Literal["adaptive", "easy", "medium", "hard"] = "adaptive"


class PracticeSourceSummary(StrictModel):
    source_id: str
    source_file: str
    title: str | None = None


class PracticeSetResponse(StrictModel):
    practice_set_id: str
    topic_id: str
    mode: Literal["practice", "checkpoint"]
    status: str = "ready"
    questions: list[PracticeQuestionPublic]
    source_summary: list[PracticeSourceSummary]


class PracticeAnswerRequest(StrictModel):
    question_id: str = Field(min_length=1, max_length=200)
    answer: str = Field(min_length=1, max_length=50_000)

    @field_validator("answer")
    @classmethod
    def strip_answer(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("答案不能为空")
        return value


class SubmitPracticeSetRequest(StrictModel):
    client_submission_id: UUID
    answers: list[PracticeAnswerRequest] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def unique_question_ids(self) -> "SubmitPracticeSetRequest":
        question_ids = [item.question_id for item in self.answers]
        if len(question_ids) != len(set(question_ids)):
            raise ValueError("同一道题不能重复提交答案")
        return self


class QuestionEvaluationResponse(StrictModel):
    question_id: str
    attempt_id: str
    score: float = Field(ge=0, le=100)
    verdict: Literal[
        "correct", "partially_correct", "incorrect", "evaluation_uncertain"
    ]
    dimension_scores: dict[str, float]
    correct_points: list[str]
    missing_points: list[str]
    misconception_tags: list[str]
    critical_misconception: bool
    feedback: str
    reference_answer: str
    sources: list[SourceResponse]
    evaluator_confidence: float = Field(ge=0, le=1)
    evaluation_reliable: bool


class PracticeSubmitResponse(StrictModel):
    practice_set_id: str
    submission_id: str
    status: Literal["evaluated", "evaluation_uncertain"]
    total_score: float = Field(ge=0, le=100)
    mastery_before: float = Field(ge=0, le=1)
    mastery_after: float = Field(ge=0, le=1)
    evaluations: list[QuestionEvaluationResponse]
    gate: MasteryGateResult
    next_action: str
    reasons: list[str]
    idempotent_replay: bool = False


class RemediationRequest(StrictModel):
    topic_id: str = Field(pattern=r"^(?:ml|math|la|prob)\.[a-z0-9_]+$")
    attempt_id: str = Field(min_length=1, max_length=200)


class RemediationResponse(StrictModel):
    topic_id: str
    diagnosis: str
    explanation: str
    contrast_example: str
    targeted_question_set_id: str
    sources: list[SourceResponse]
