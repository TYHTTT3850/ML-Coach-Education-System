from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel

from app.core.types import Course


TopicStatus = Literal[
    "locked",
    "not_started",
    "learning",
    "practicing",
    "needs_remediation",
    "ready_for_checkpoint",
    "mastered",
]


class PrerequisiteResponse(BaseModel):
    id: str
    slug: str
    title: str
    required_mastery: float
    course: Course | None = None
    relation_type: Literal["required", "recommended", "related"] = "required"
    reason: str = ""
    required_competencies: list[str] = []
    scope: Literal["topic", "lesson_section"] = "topic"
    section_id: str = ""
    rule_version: str = "content-v1"


class TopicSummaryResponse(BaseModel):
    id: str
    slug: str
    title: str
    order_index: int
    description: str
    status: TopicStatus | None = None
    mastery: float | None = None
    mastery_display: str | None = None
    confidence: float | None = None
    prerequisites: list[PrerequisiteResponse]
    locked_reason: str | None = None


class CurriculumStageResponse(BaseModel):
    stage: int
    title: str
    topics: list[TopicSummaryResponse]


class CurriculumResponse(BaseModel):
    course: Course
    stages: list[CurriculumStageResponse]


class DimensionStateResponse(BaseModel):
    dimension: str
    mastery: float | None = None
    mastery_display: str | None = None
    confidence: float | None = None
    evidence_count: int = 0


class TopicDetailResponse(TopicSummaryResponse):
    course: Course
    stage: int
    stage_title: str
    learning_objectives: list[str]
    aliases: list[str]
    retrieval_keywords: list[str]
    difficulty: Literal["easy", "medium", "hard"]
    required_dimensions: list[str]
    dimensions: list[DimensionStateResponse]
    active: bool
    created_at: datetime
    updated_at: datetime
