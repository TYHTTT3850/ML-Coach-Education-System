"""Transparent Beta-evidence mastery calculations.

All functions are pure: they never mutate caller-owned state and never perform
I/O.  Persistence and evaluation-reliability checks belong to the service layer
that invokes these helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType
from typing import Mapping


DIFFICULTY_WEIGHTS: Mapping[str, float] = MappingProxyType(
    {"easy": 0.75, "medium": 1.0, "hard": 1.25}
)
CHECKPOINT_WEIGHT_MULTIPLIER = 1.5
DEFAULT_CONFIDENCE_TARGET = 6.0


@dataclass(frozen=True, slots=True)
class BetaEvidenceState:
    alpha: float = 1.0
    beta: float = 1.0

    def __post_init__(self) -> None:
        _validate_beta_parameters(self.alpha, self.beta)

    @property
    def mastery(self) -> float:
        return mastery_from_evidence(self.alpha, self.beta)

    def confidence(self, target: float = DEFAULT_CONFIDENCE_TARGET) -> float:
        return confidence_from_evidence(self.alpha, self.beta, target=target)


@dataclass(frozen=True, slots=True)
class EvidenceUpdate:
    alpha: float
    beta: float
    mastery: float
    confidence: float
    score: float
    weight: float

    @property
    def state(self) -> BetaEvidenceState:
        return BetaEvidenceState(alpha=self.alpha, beta=self.beta)


def _finite_number(value: float, *, name: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not isfinite(numeric):
        raise ValueError(f"{name} must be a finite number")
    return numeric


def _validate_beta_parameters(alpha: float, beta: float) -> tuple[float, float]:
    normalized_alpha = _finite_number(alpha, name="alpha")
    normalized_beta = _finite_number(beta, name="beta")
    if normalized_alpha <= 0 or normalized_beta <= 0:
        raise ValueError("alpha and beta must both be greater than zero")
    return normalized_alpha, normalized_beta


def clamp_score(score: float) -> float:
    """Return a finite score clipped to the inclusive 0..100 range."""

    numeric = _finite_number(score, name="score")
    return min(100.0, max(0.0, numeric))


def difficulty_weight(difficulty: str, *, checkpoint: bool = False) -> float:
    """Return the specified practice/checkpoint evidence weight."""

    normalized = str(difficulty).strip().lower()
    try:
        weight = DIFFICULTY_WEIGHTS[normalized]
    except KeyError as exc:
        allowed = ", ".join(DIFFICULTY_WEIGHTS)
        raise ValueError(f"difficulty must be one of: {allowed}") from exc
    return weight * (CHECKPOINT_WEIGHT_MULTIPLIER if checkpoint else 1.0)


def mastery_from_evidence(alpha: float, beta: float) -> float:
    normalized_alpha, normalized_beta = _validate_beta_parameters(alpha, beta)
    # Positive finite parameters make the result finite and strictly within 0..1.
    return min(1.0, max(0.0, normalized_alpha / (normalized_alpha + normalized_beta)))


def confidence_from_evidence(
    alpha: float,
    beta: float,
    *,
    target: float = DEFAULT_CONFIDENCE_TARGET,
) -> float:
    normalized_alpha, normalized_beta = _validate_beta_parameters(alpha, beta)
    normalized_target = _finite_number(target, name="target")
    if normalized_target <= 0:
        raise ValueError("target must be greater than zero")
    evidence_strength = max(0.0, normalized_alpha + normalized_beta - 2.0)
    return min(1.0, evidence_strength / normalized_target)


def update_beta_evidence(
    alpha: float,
    beta: float,
    score: float,
    difficulty: str,
    *,
    checkpoint: bool = False,
    confidence_target: float = DEFAULT_CONFIDENCE_TARGET,
    evidence_scale: float = 1.0,
) -> EvidenceUpdate:
    """Apply one valid evaluation to a Beta state and return a new snapshot.

    ``evidence_scale`` lets a caller reduce the contribution of a dimension when
    desired.  The normal topic and dimension update uses the required value 1.0.
    It may not be negative, so an update can never remove prior evidence.
    """

    normalized_alpha, normalized_beta = _validate_beta_parameters(alpha, beta)
    normalized_score = clamp_score(score)
    normalized_scale = _finite_number(evidence_scale, name="evidence_scale")
    if normalized_scale < 0:
        raise ValueError("evidence_scale must not be negative")

    weight = difficulty_weight(difficulty, checkpoint=checkpoint) * normalized_scale
    probability = normalized_score / 100.0
    updated_alpha = normalized_alpha + weight * probability
    updated_beta = normalized_beta + weight * (1.0 - probability)
    return EvidenceUpdate(
        alpha=updated_alpha,
        beta=updated_beta,
        mastery=mastery_from_evidence(updated_alpha, updated_beta),
        confidence=confidence_from_evidence(
            updated_alpha,
            updated_beta,
            target=confidence_target,
        ),
        score=normalized_score,
        weight=weight,
    )


def update_dimension_evidence(
    states: Mapping[str, BetaEvidenceState],
    dimension_scores: Mapping[str, float],
    difficulty: str,
    *,
    checkpoint: bool = False,
    confidence_target: float = DEFAULT_CONFIDENCE_TARGET,
) -> dict[str, BetaEvidenceState]:
    """Independently update only the dimensions present in an evaluation.

    Existing dimensions absent from ``dimension_scores`` are returned unchanged;
    newly scored dimensions start from the Beta(1, 1) prior.  A fresh mapping is
    always returned and none of the caller's state objects are mutated.
    """

    updated = dict(states)
    for dimension, score in dimension_scores.items():
        normalized_dimension = str(dimension).strip()
        if not normalized_dimension:
            raise ValueError("dimension names must not be empty")
        previous = states.get(normalized_dimension, BetaEvidenceState())
        result = update_beta_evidence(
            previous.alpha,
            previous.beta,
            score,
            difficulty,
            checkpoint=checkpoint,
            confidence_target=confidence_target,
        )
        updated[normalized_dimension] = result.state
    return updated
