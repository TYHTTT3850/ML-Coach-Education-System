"""Strict internal and public practice contracts.

The models are intentionally separate rather than relying on exclusion flags at
serialization time.  A pre-submission public response can therefore never leak
an answer key or rubric through an accidentally broadened dump call.
"""

from __future__ import annotations

from hashlib import sha256
from math import isclose
import re
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from app.core.types import Course
from app.schemas.learning import CapabilityDimension


PracticeSetKind = Literal["practice", "checkpoint"]
Difficulty = Literal["easy", "medium", "hard"]
QuestionType = Literal[
    "concept_explanation",
    "mathematical_reasoning",
    "why_mechanism",
    "code_reading",
    "case_analysis",
    "theorem_condition",
    "calculation",
    "derivation",
    "proof_outline",
    "error_correction",
    "application",
    "multiple_choice",
    "numeric",
]

_MACHINE_LEARNING_QUESTION_TYPES = frozenset(
    {
        "concept_explanation",
        "mathematical_reasoning",
        "why_mechanism",
        "code_reading",
        "case_analysis",
        "multiple_choice",
        "numeric",
    }
)
_ADVANCED_MATHEMATICS_QUESTION_TYPES = frozenset(
    {
        "concept_explanation",
        "theorem_condition",
        "calculation",
        "derivation",
        "proof_outline",
        "error_correction",
        "application",
        "multiple_choice",
        "numeric",
    }
)
_MACHINE_LEARNING_DIMENSIONS = frozenset(
    {"concept", "mathematics", "mechanism", "engineering", "application"}
)
_ADVANCED_MATHEMATICS_DIMENSIONS = frozenset(
    {"concept", "theorem_condition", "derivation", "calculation", "application"}
)


def referenced_internal_source_ids(
    texts: list[str],
    source_ids: set[str],
) -> list[str]:
    """Return opaque evidence ids leaked into student-visible text."""

    visible_text = "\n".join(texts)
    leaked = {
        match.group(0).strip("[]")
        for match in re.finditer(
            r"(?<![A-Za-z0-9_])\[?S\d+\]?(?![A-Za-z0-9_])",
            visible_text,
            flags=re.IGNORECASE,
        )
    }
    leaked.update(
        source_id
        for source_id in source_ids
        if source_id
        and re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(source_id)}(?![A-Za-z0-9_])",
            visible_text,
            flags=re.IGNORECASE,
        )
    )
    return sorted(leaked)


def replace_internal_source_references(text: str, source_ids: set[str]) -> str:
    """Render historical S1/S2-style references as a student-facing label."""

    escaped = [
        re.escape(item)
        for item in sorted(source_ids, key=len, reverse=True)
        if item
    ]
    if not escaped:
        return text
    token = rf"\[?(?:{'|'.join(escaped)})\]?"
    grouped_references = re.compile(
        rf"(?<![A-Za-z0-9_]){token}"
        rf"(?:\s*(?:、|,|，|/|和|及|与)\s*{token})*"
        rf"(?![A-Za-z0-9_])",
        flags=re.IGNORECASE,
    )
    rendered = grouped_references.sub("课程资料", text)
    return re.sub(r"\s*课程资料\s*", "课程资料", rendered)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SourceSnapshotItem(StrictModel):
    source_course: Course | None = None
    topic_id: str | None = None
    source_id: str = Field(min_length=1, max_length=100)
    document_id: str = Field(min_length=1, max_length=200)
    source_file: str = Field(min_length=1, max_length=1_000)
    content: str = Field(min_length=1, max_length=100_000)
    title: str | None = Field(default=None, max_length=1_000)
    chapter: str | None = Field(default=None, max_length=1_000)
    section: str | None = Field(default=None, max_length=1_000)
    page: int | None = Field(default=None, ge=1)
    chunk_index: int = Field(ge=0)

    @field_validator("source_id", "document_id", "source_file", "content")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("source fields must not be empty")
        return normalized


class MultipleChoiceOption(StrictModel):
    option_id: str = Field(min_length=1, max_length=20)
    text: str = Field(min_length=1, max_length=10_000)

    @field_validator("option_id", "text")
    @classmethod
    def strip_option_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("choice option fields must not be empty")
        return normalized


class AnswerKeyInternal(StrictModel):
    accepted_units: list[str] = Field(default_factory=list, max_length=10)
    canonical_answer: str = Field(min_length=1, max_length=50_000)
    key_points: list[str] = Field(default_factory=list, max_length=100)
    accepted_variants: list[str] = Field(default_factory=list, max_length=100)
    numeric_answer: float | None = Field(default=None, allow_inf_nan=False)
    tolerance: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    correct_option: str | None = Field(default=None, min_length=1, max_length=20)

    @field_validator("canonical_answer")
    @classmethod
    def strip_canonical_answer(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("canonical_answer must not be empty")
        return normalized

    @field_validator("key_points", "accepted_variants")
    @classmethod
    def clean_string_lists(cls, value: list[str]) -> list[str]:
        cleaned: list[str] = []
        seen: set[str] = set()
        for item in value:
            normalized = item.strip()
            if not normalized:
                raise ValueError("answer key values must not be empty")
            if normalized not in seen:
                cleaned.append(normalized)
                seen.add(normalized)
        return cleaned


class RubricCriterionInternal(StrictModel):
    criterion: str = Field(min_length=1, max_length=10_000)
    points: float = Field(gt=0, le=100, allow_inf_nan=False)

    @field_validator("criterion")
    @classmethod
    def strip_criterion(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("rubric criterion must not be empty")
        return normalized


class PracticeQuestionInternal(StrictModel):
    question_id: str = Field(min_length=1, max_length=200)
    question_type: QuestionType
    difficulty: Difficulty
    question: str = Field(min_length=1, max_length=50_000)
    dimensions: dict[CapabilityDimension, float] = Field(min_length=1)
    options: list[MultipleChoiceOption] | None = Field(default=None, max_length=20)
    answer_key: AnswerKeyInternal
    rubric: list[RubricCriterionInternal] = Field(min_length=1, max_length=100)
    source_ids: list[str] = Field(min_length=1, max_length=100)
    position: int = Field(ge=1)
    max_score: Literal[100] = 100

    @field_validator("question_id", "question")
    @classmethod
    def strip_question_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("question fields must not be empty")
        return normalized

    @field_validator("source_ids")
    @classmethod
    def clean_source_ids(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value]
        if any(not item for item in cleaned):
            raise ValueError("source_ids must not contain empty values")
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("source_ids must be unique")
        return cleaned

    @field_validator("dimensions")
    @classmethod
    def validate_dimensions(
        cls, value: dict[CapabilityDimension, float]
    ) -> dict[CapabilityDimension, float]:
        if any(weight <= 0 or weight > 1 for weight in value.values()):
            raise ValueError("dimension weights must be within (0, 1]")
        if not isclose(sum(value.values()), 1.0, abs_tol=0.01):
            raise ValueError("dimension weights must sum to approximately 1")
        return value

    @model_validator(mode="after")
    def validate_question_contract(self) -> "PracticeQuestionInternal":
        rubric_total = sum(item.points for item in self.rubric)
        if not isclose(rubric_total, 100.0, abs_tol=1e-6):
            raise ValueError(
                f"rubric points must sum to 100 (current sum={rubric_total:g})"
            )

        answer = self.answer_key
        if self.question_type == "numeric":
            if answer.numeric_answer is None or answer.tolerance is None:
                raise ValueError("numeric questions require numeric_answer and tolerance")
            if answer.correct_option is not None or self.options is not None:
                raise ValueError("numeric questions cannot define choice options")
        else:
            if answer.numeric_answer is not None or answer.tolerance is not None:
                raise ValueError("non-numeric questions cannot define numeric answer fields")
            if not answer.key_points:
                raise ValueError("non-numeric questions require key_points")

        if self.question_type == "multiple_choice":
            if self.options is None or len(self.options) < 2:
                raise ValueError("multiple_choice questions require at least two options")
            option_ids = [item.option_id for item in self.options]
            if len(option_ids) != len(set(option_ids)):
                raise ValueError("choice option ids must be unique")
            if answer.correct_option not in option_ids:
                raise ValueError("correct_option must reference an existing option")
        elif self.options is not None or answer.correct_option is not None:
            raise ValueError("only multiple_choice questions may define choice options")

        return self

    @property
    def normalized_question_hash(self) -> str:
        normalized = " ".join(self.question.casefold().split())
        return sha256(normalized.encode("utf-8")).hexdigest()

    def to_public(self) -> "PracticeQuestionPublic":
        source_ids = set(self.source_ids)
        public_options = (
            [
                item.model_copy(
                    update={
                        "text": replace_internal_source_references(item.text, source_ids)
                    }
                )
                for item in self.options
            ]
            if self.question_type == "multiple_choice" and self.options is not None
            else None
        )
        return PracticeQuestionPublic(
            question_id=self.question_id,
            question_type=self.question_type,
            difficulty=self.difficulty,
            question=replace_internal_source_references(self.question, source_ids),
            options=public_options,
            dimensions=list(self.dimensions),
            position=self.position,
            max_score=self.max_score,
        )


class PracticeSetInternal(StrictModel):
    practice_set_id: str = Field(min_length=1, max_length=200)
    course: Course
    topic_id: str = Field(min_length=1, max_length=200)
    kind: PracticeSetKind
    required_dimensions: list[CapabilityDimension] = Field(min_length=1, max_length=8)
    questions: list[PracticeQuestionInternal] = Field(min_length=1, max_length=8)
    source_snapshot: list[SourceSnapshotItem] = Field(min_length=1, max_length=500)
    assessment_purpose: str = 'legacy'

    @model_validator(mode="after")
    def validate_set_contract(self) -> "PracticeSetInternal":
        if len(self.required_dimensions) != len(set(self.required_dimensions)):
            raise ValueError("required_dimensions must be unique")
        if self.course == Course.MACHINE_LEARNING:
            allowed_question_types = _MACHINE_LEARNING_QUESTION_TYPES
            allowed_dimensions = _MACHINE_LEARNING_DIMENSIONS
        else:
            allowed_question_types = _ADVANCED_MATHEMATICS_QUESTION_TYPES
            allowed_dimensions = _ADVANCED_MATHEMATICS_DIMENSIONS
        if self.assessment_purpose != 'legacy':
            # A curated probability scenario or matrix code-reading item is
            # valid bank content; its capability dimensions still belong to
            # the owner course. Legacy generated-set constraints stay intact.
            allowed_question_types = _MACHINE_LEARNING_QUESTION_TYPES | _ADVANCED_MATHEMATICS_QUESTION_TYPES
        invalid_required = sorted(set(self.required_dimensions) - allowed_dimensions)
        if invalid_required:
            raise ValueError(
                "required_dimensions do not belong to course: "
                + ", ".join(invalid_required)
            )
        for question in self.questions:
            if question.question_type not in allowed_question_types:
                raise ValueError(
                    f"question type {question.question_type} does not belong to course"
                )
            invalid_dimensions = sorted(set(question.dimensions) - allowed_dimensions)
            if invalid_dimensions:
                raise ValueError(
                    "question dimensions do not belong to course: "
                    + ", ".join(invalid_dimensions)
                )

        snapshot_ids = [item.source_id for item in self.source_snapshot]
        if len(snapshot_ids) != len(set(snapshot_ids)):
            raise ValueError("source_snapshot source ids must be unique")
        allowed = set(snapshot_ids)
        unknown = sorted(
            {
                source_id
                for question in self.questions
                for source_id in question.source_ids
                if source_id not in allowed
            }
        )
        if unknown:
            raise ValueError(f"questions reference unknown source ids: {', '.join(unknown)}")
        question_ids = [item.question_id for item in self.questions]
        if len(question_ids) != len(set(question_ids)):
            raise ValueError("question ids must be unique within a practice set")
        question_hashes = [item.normalized_question_hash for item in self.questions]
        if len(question_hashes) != len(set(question_hashes)):
            raise ValueError("duplicate normalized questions are not allowed")
        if self.assessment_purpose == 'legacy' and all(item.question_type == "multiple_choice" for item in self.questions):
            raise ValueError("a practice set cannot contain only multiple_choice questions")

        if self.kind == "checkpoint":
            covered = {
                dimension
                for item in self.questions
                for dimension in item.dimensions
                if dimension in self.required_dimensions
            }
            if len(covered) < 3:
                raise ValueError("checkpoint must cover at least three dimensions")
            if not any(item.difficulty in {"medium", "hard"} for item in self.questions):
                raise ValueError("checkpoint requires at least one medium or hard question")
        return self

    def to_public(self) -> "PracticeSetPublic":
        return PracticeSetPublic(
            practice_set_id=self.practice_set_id,
            course=self.course,
            topic_id=self.topic_id,
            kind=self.kind,
            questions=[item.to_public() for item in self.questions],
        )


class PracticeQuestionPublic(StrictModel):
    question_id: str
    question_type: QuestionType
    difficulty: Difficulty
    question: str
    options: list[MultipleChoiceOption] | None = Field(default=None, max_length=20)
    dimensions: list[CapabilityDimension]
    position: int = Field(ge=1)
    max_score: float = Field(gt=0, le=100, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_public_options(self) -> "PracticeQuestionPublic":
        if self.question_type == "multiple_choice":
            if self.options is None or len(self.options) < 2:
                raise ValueError("multiple_choice questions require public options")
        elif self.options is not None:
            raise ValueError("only multiple_choice questions may expose options")
        return self


class PracticeSetPublic(StrictModel):
    practice_set_id: str
    course: Course
    topic_id: str
    kind: PracticeSetKind
    questions: list[PracticeQuestionPublic] = Field(min_length=1, max_length=8)
