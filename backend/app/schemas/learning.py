"""Strict contracts shared by deterministic learning-loop services."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)


LearningStatus = Literal[
    "locked",
    "not_started",
    "learning",
    "practicing",
    "needs_remediation",
    "ready_for_checkpoint",
    "mastered",
]
LearningAction = Literal[
    "study_prerequisite",
    "start_learning",
    "continue_learning",
    "practice",
    "remediate",
    "take_checkpoint",
    "continue_to_next_topic",
    "course_completed",
]
EvaluationVerdict = Literal[
    "correct",
    "partially_correct",
    "incorrect",
    "evaluation_uncertain",
]
CapabilityDimension = Literal[
    "concept",
    "mathematics",
    "mechanism",
    "engineering",
    "application",
    "theorem_condition",
    "derivation",
    "calculation",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EvaluationResultInternal(StrictModel):
    """Validated evaluator output before persistence or mastery mutation.

    Pass the real snapshot IDs as validation context to reject fabricated
    citations::

        EvaluationResultInternal.model_validate(
            payload, context={"source_ids": {"S1", "S2"}}
        )

    Retrieval-expansion/HyDE text is intentionally absent from this contract.
    """

    score: float = Field(ge=0, le=100, allow_inf_nan=False)
    verdict: EvaluationVerdict
    dimension_scores: dict[CapabilityDimension, float] = Field(default_factory=dict)
    correct_points: list[str] = Field(default_factory=list, max_length=50)
    missing_points: list[str] = Field(default_factory=list, max_length=50)
    misconception_tags: list[str] = Field(default_factory=list, max_length=20)
    critical_misconception: bool = False
    feedback: str = Field(max_length=20_000)
    reference_answer: str = Field(max_length=20_000)
    evidence_source_ids: list[str] = Field(default_factory=list, max_length=100)
    evaluator_confidence: float = Field(ge=0, le=1, allow_inf_nan=False)
    evaluation_reliable: bool

    @field_validator("dimension_scores")
    @classmethod
    def validate_dimension_scores(
        cls, value: dict[CapabilityDimension, float]
    ) -> dict[CapabilityDimension, float]:
        for dimension, score in value.items():
            if not 0 <= score <= 100:
                raise ValueError(f"dimension score for {dimension} must be within 0..100")
            if score != score or score in (float("inf"), float("-inf")):
                raise ValueError(f"dimension score for {dimension} must be finite")
        return value

    @field_validator(
        "correct_points",
        "missing_points",
        "misconception_tags",
        "evidence_source_ids",
    )
    @classmethod
    def clean_unique_strings(cls, value: list[str]) -> list[str]:
        cleaned: list[str] = []
        seen: set[str] = set()
        for item in value:
            normalized = item.strip()
            if not normalized:
                raise ValueError("list values must not be empty")
            if normalized not in seen:
                seen.add(normalized)
                cleaned.append(normalized)
        return cleaned

    @field_validator("evidence_source_ids")
    @classmethod
    def validate_evidence_sources(
        cls,
        value: list[str],
        info: ValidationInfo,
    ) -> list[str]:
        context = info.context or {}
        allowed = context.get("source_ids")
        if allowed is None:
            return value
        unknown = sorted(set(value) - set(allowed))
        if unknown:
            raise ValueError(f"unknown evidence source ids: {', '.join(unknown)}")
        return value

    @model_validator(mode="after")
    def validate_reliability_consistency(self) -> "EvaluationResultInternal":
        uncertain = self.verdict == "evaluation_uncertain"
        if uncertain == self.evaluation_reliable:
            raise ValueError(
                "evaluation_uncertain must be unreliable and all other verdicts must be reliable"
            )
        if uncertain and self.critical_misconception:
            raise ValueError("an uncertain evaluation cannot assert a critical misconception")
        return self

    def to_public(self) -> "EvaluationResultPublic":
        return EvaluationResultPublic.model_validate(self.model_dump())


class EvaluationResultPublic(StrictModel):
    """Post-submission response; contains no hidden rubric or prompt."""

    score: float = Field(ge=0, le=100, allow_inf_nan=False)
    verdict: EvaluationVerdict
    dimension_scores: dict[CapabilityDimension, float]
    correct_points: list[str]
    missing_points: list[str]
    misconception_tags: list[str]
    critical_misconception: bool
    feedback: str
    reference_answer: str
    evidence_source_ids: list[str]
    evaluator_confidence: float = Field(ge=0, le=1, allow_inf_nan=False)
    evaluation_reliable: bool


class RequirementStatus(StrictModel):
    passed: bool
    current: float | int | bool | None
    required: float | int | bool
    details: list[str] = Field(default_factory=list)


class MasteryGateRequirements(StrictModel):
    mastery_threshold: RequirementStatus
    valid_attempts: RequirementStatus
    checkpoint_completed: RequirementStatus
    checkpoint_score: RequirementStatus
    dimension_coverage: RequirementStatus
    critical_misconceptions: RequirementStatus
    prerequisites_mastered: RequirementStatus


class MasteryGateResult(StrictModel):
    passed: bool
    mastery: float = Field(ge=0, le=1, allow_inf_nan=False)
    requirements: MasteryGateRequirements
    next_action: LearningAction


class LearningDecision(StrictModel):
    next_action: LearningAction
    reasons: list[str] = Field(min_length=1)
    recommended_topic_id: str | None = None

    @field_validator("reasons")
    @classmethod
    def clean_reasons(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value]
        if any(not item for item in cleaned):
            raise ValueError("reasons must not contain empty values")
        return cleaned


class MasteryUpdateEligibility(StrictModel):
    """Explicit safety decision used before applying Beta evidence."""

    evaluation_reliable: bool
    evaluator_confidence: float = Field(ge=0, le=1, allow_inf_nan=False)
    minimum_confidence: float = Field(ge=0, le=1, allow_inf_nan=False)
    question_valid: bool
    duplicate_submission: bool
    schema_valid: bool

    @property
    def eligible(self) -> bool:
        return (
            self.evaluation_reliable
            and self.evaluator_confidence >= self.minimum_confidence
            and self.question_valid
            and not self.duplicate_submission
            and self.schema_valid
        )

    @property
    def rejection_reasons(self) -> list[str]:
        reasons: list[str] = []
        if not self.evaluation_reliable:
            reasons.append("evaluation_unreliable")
        if self.evaluator_confidence < self.minimum_confidence:
            reasons.append("evaluator_confidence_too_low")
        if not self.question_valid:
            reasons.append("question_invalid")
        if self.duplicate_submission:
            reasons.append("duplicate_submission")
        if not self.schema_valid:
            reasons.append("evaluation_schema_invalid")
        return reasons


def json_schema_without_hidden_fields(schema: type[BaseModel]) -> dict[str, Any]:
    """Small explicit helper for structured-output LLM integrations."""

    return schema.model_json_schema()
