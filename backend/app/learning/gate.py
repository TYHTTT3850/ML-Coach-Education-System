"""Pure Mastery Gate evaluation covering every required condition."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Mapping

from app.schemas.learning import (
    MasteryGateRequirements,
    MasteryGateResult,
    RequirementStatus,
)


@dataclass(frozen=True, slots=True)
class MasteryGateConfig:
    mastery_threshold: float = 0.80
    prerequisite_mastery_threshold: float = 0.80
    minimum_valid_attempts: int = 3
    checkpoint_score_threshold: float = 80.0
    minimum_dimension_coverage: int = 3

    def __post_init__(self) -> None:
        if not 0 <= self.mastery_threshold <= 1:
            raise ValueError("mastery_threshold must be within 0..1")
        if not 0 <= self.prerequisite_mastery_threshold <= 1:
            raise ValueError("prerequisite_mastery_threshold must be within 0..1")
        if self.minimum_valid_attempts < 0:
            raise ValueError("minimum_valid_attempts must not be negative")
        if not 0 <= self.checkpoint_score_threshold <= 100:
            raise ValueError("checkpoint_score_threshold must be within 0..100")
        if self.minimum_dimension_coverage < 0:
            raise ValueError("minimum_dimension_coverage must not be negative")


@dataclass(frozen=True, slots=True)
class MasteryGateInput:
    mastery: float
    valid_attempt_count: int
    checkpoint_count: int
    latest_checkpoint_score: float | None
    required_dimensions: frozenset[str]
    checkpoint_dimensions: frozenset[str] = field(default_factory=frozenset)
    unresolved_critical_misconception_count: int = 0
    prerequisite_masteries: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isfinite(self.mastery) or not 0 <= self.mastery <= 1:
            raise ValueError("mastery must be finite and within 0..1")
        if self.valid_attempt_count < 0:
            raise ValueError("valid_attempt_count must not be negative")
        if self.checkpoint_count < 0:
            raise ValueError("checkpoint_count must not be negative")
        if self.latest_checkpoint_score is not None and (
            not isfinite(self.latest_checkpoint_score)
            or not 0 <= self.latest_checkpoint_score <= 100
        ):
            raise ValueError("latest_checkpoint_score must be finite and within 0..100")
        if self.unresolved_critical_misconception_count < 0:
            raise ValueError(
                "unresolved_critical_misconception_count must not be negative"
            )
        invalid_prerequisites = [
            topic_id
            for topic_id, mastery in self.prerequisite_masteries.items()
            if not topic_id.strip() or not isfinite(mastery) or not 0 <= mastery <= 1
        ]
        if invalid_prerequisites:
            raise ValueError("prerequisite mastery values must be finite and within 0..1")


def evaluate_mastery_gate(
    state: MasteryGateInput,
    config: MasteryGateConfig | None = None,
) -> MasteryGateResult:
    """Evaluate all seven Gate requirements without changing any state."""

    rules = config or MasteryGateConfig()

    mastery_passed = state.mastery >= rules.mastery_threshold
    attempts_passed = state.valid_attempt_count >= rules.minimum_valid_attempts
    checkpoint_completed = state.checkpoint_count >= 1
    checkpoint_score_passed = (
        checkpoint_completed
        and state.latest_checkpoint_score is not None
        and state.latest_checkpoint_score >= rules.checkpoint_score_threshold
    )
    dimension_count = len(state.checkpoint_dimensions & state.required_dimensions)
    dimensions_passed = dimension_count >= rules.minimum_dimension_coverage
    misconceptions_passed = state.unresolved_critical_misconception_count == 0
    unmet_prerequisites = sorted(
        topic_id
        for topic_id, mastery in state.prerequisite_masteries.items()
        if mastery < rules.prerequisite_mastery_threshold
    )
    prerequisites_passed = not unmet_prerequisites

    requirements = MasteryGateRequirements(
        mastery_threshold=RequirementStatus(
            passed=mastery_passed,
            current=state.mastery,
            required=rules.mastery_threshold,
        ),
        valid_attempts=RequirementStatus(
            passed=attempts_passed,
            current=state.valid_attempt_count,
            required=rules.minimum_valid_attempts,
        ),
        checkpoint_completed=RequirementStatus(
            passed=checkpoint_completed,
            current=state.checkpoint_count,
            required=1,
        ),
        checkpoint_score=RequirementStatus(
            passed=checkpoint_score_passed,
            current=state.latest_checkpoint_score,
            required=rules.checkpoint_score_threshold,
            details=[] if checkpoint_completed else ["尚未完成 checkpoint"],
        ),
        dimension_coverage=RequirementStatus(
            passed=dimensions_passed,
            current=dimension_count,
            required=rules.minimum_dimension_coverage,
            details=sorted(state.checkpoint_dimensions & state.required_dimensions),
        ),
        critical_misconceptions=RequirementStatus(
            passed=misconceptions_passed,
            current=state.unresolved_critical_misconception_count,
            required=0,
        ),
        prerequisites_mastered=RequirementStatus(
            passed=prerequisites_passed,
            current=len(state.prerequisite_masteries) - len(unmet_prerequisites),
            required=len(state.prerequisite_masteries),
            details=unmet_prerequisites,
        ),
    )
    passed = all(
        (
            mastery_passed,
            attempts_passed,
            checkpoint_completed,
            checkpoint_score_passed,
            dimensions_passed,
            misconceptions_passed,
            prerequisites_passed,
        )
    )

    if passed:
        next_action = "continue_to_next_topic"
    elif not prerequisites_passed:
        next_action = "study_prerequisite"
    elif not misconceptions_passed or (checkpoint_completed and not checkpoint_score_passed):
        next_action = "remediate"
    elif attempts_passed and mastery_passed and not checkpoint_completed:
        next_action = "take_checkpoint"
    else:
        next_action = "practice"

    return MasteryGateResult(
        passed=passed,
        mastery=state.mastery,
        requirements=requirements,
        next_action=next_action,
    )
