"""Controlled misconception taxonomy and safe normalization."""

from __future__ import annotations

from types import MappingProxyType
from typing import Iterable, Mapping

from app.core.types import Course
from app.schemas.learning import EvaluationResultInternal


MACHINE_LEARNING_MISCONCEPTION_TAGS = frozenset(
    {
        "definition_confusion",
        "formula_misuse",
        "loss_function_confusion",
        "metric_confusion",
        "optimization_confusion",
        "probability_confusion",
        "model_assumption_ignored",
        "overfitting_confusion",
        "data_leakage",
        "code_api_misuse",
        "case_reasoning_error",
        "incomplete_answer",
        "other",
    }
)

ADVANCED_MATHEMATICS_MISCONCEPTION_TAGS = frozenset(
    {
        "definition_confusion",
        "theorem_condition_missing",
        "limit_reasoning_error",
        "derivative_rule_misuse",
        "integral_rule_misuse",
        "calculation_error",
        "sign_error",
        "domain_ignored",
        "derivation_gap",
        "proof_logic_error",
        "formula_misuse",
        "incomplete_answer",
        "other",
    }
)

MISCONCEPTION_TAGS_BY_COURSE: Mapping[Course, frozenset[str]] = MappingProxyType(
    {
        Course.MACHINE_LEARNING: MACHINE_LEARNING_MISCONCEPTION_TAGS,
        Course.ADVANCED_MATHEMATICS: ADVANCED_MATHEMATICS_MISCONCEPTION_TAGS,
        Course.LINEAR_ALGEBRA: ADVANCED_MATHEMATICS_MISCONCEPTION_TAGS,
        Course.PROBABILITY_STATISTICS: ADVANCED_MATHEMATICS_MISCONCEPTION_TAGS,
    }
)


def _course(value: Course | str) -> Course:
    try:
        return value if isinstance(value, Course) else Course(value)
    except ValueError as exc:
        raise ValueError(f"unsupported course: {value}") from exc


def normalize_misconception_tag(course: Course | str, tag: object) -> str:
    """Return a controlled tag; every blank/unknown model value becomes other."""

    normalized = str(tag).strip().lower() if tag is not None else ""
    allowed = MISCONCEPTION_TAGS_BY_COURSE[_course(course)]
    return normalized if normalized in allowed else "other"


def normalize_misconception_tags(
    course: Course | str,
    tags: Iterable[object],
) -> list[str]:
    """Normalize, de-duplicate, and preserve the evaluator's original order."""

    normalized: list[str] = []
    seen: set[str] = set()
    for raw_tag in tags:
        tag = normalize_misconception_tag(course, raw_tag)
        if tag not in seen:
            normalized.append(tag)
            seen.add(tag)
    return normalized


def normalize_evaluation_misconceptions(
    course: Course | str,
    evaluation: EvaluationResultInternal,
) -> EvaluationResultInternal:
    """Return an evaluation whose tags are safe to persist as classifications."""

    return evaluation.model_copy(
        update={
            "misconception_tags": normalize_misconception_tags(
                course, evaluation.misconception_tags
            )
        }
    )
