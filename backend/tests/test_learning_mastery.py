from __future__ import annotations

import math

import pytest

from app.learning.mastery import (
    BetaEvidenceState,
    clamp_score,
    confidence_from_evidence,
    difficulty_weight,
    mastery_from_evidence,
    update_beta_evidence,
    update_dimension_evidence,
)
from app.schemas.learning import MasteryUpdateEligibility


def test_beta_evidence_formula_is_exact() -> None:
    result = update_beta_evidence(1, 1, 80, "medium")

    assert result.alpha == pytest.approx(1.8)
    assert result.beta == pytest.approx(1.2)
    assert result.mastery == pytest.approx(0.6)
    assert result.confidence == pytest.approx(1 / 6)


@pytest.mark.parametrize(
    ("difficulty", "expected"),
    [("easy", 0.75), ("medium", 1.0), ("hard", 1.25)],
)
def test_difficulty_weights(difficulty: str, expected: float) -> None:
    assert difficulty_weight(difficulty) == expected


def test_checkpoint_multiplies_difficulty_weight() -> None:
    assert difficulty_weight("easy", checkpoint=True) == pytest.approx(1.125)
    assert difficulty_weight("medium", checkpoint=True) == pytest.approx(1.5)
    assert difficulty_weight("hard", checkpoint=True) == pytest.approx(1.875)

    result = update_beta_evidence(1, 1, 100, "hard", checkpoint=True)
    assert result.alpha == pytest.approx(2.875)
    assert result.beta == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("raw", "expected"), [(-50, 0.0), (42.5, 42.5), (120, 100.0)]
)
def test_score_is_clamped_before_update(raw: float, expected: float) -> None:
    assert clamp_score(raw) == expected
    assert update_beta_evidence(1, 1, raw, "medium").score == expected


def test_mastery_and_confidence_always_remain_bounded() -> None:
    state = BetaEvidenceState()
    for index in range(200):
        result = update_beta_evidence(
            state.alpha,
            state.beta,
            -1000 if index % 2 else 1000,
            "hard",
            checkpoint=index % 5 == 0,
        )
        assert 0 <= result.mastery <= 1
        assert 0 <= result.confidence <= 1
        state = result.state
    assert 0 <= mastery_from_evidence(state.alpha, state.beta) <= 1
    assert 0 <= confidence_from_evidence(state.alpha, state.beta) <= 1


def test_dimension_states_update_independently_without_mutation() -> None:
    original = {
        "concept": BetaEvidenceState(alpha=2, beta=1),
        "mathematics": BetaEvidenceState(alpha=1, beta=3),
        "engineering": BetaEvidenceState(alpha=4, beta=2),
    }

    updated = update_dimension_evidence(
        original,
        {"concept": 100, "mathematics": 0, "mechanism": 50},
        "medium",
    )

    assert original["concept"] == BetaEvidenceState(alpha=2, beta=1)
    assert updated["concept"] == BetaEvidenceState(alpha=3, beta=1)
    assert updated["mathematics"] == BetaEvidenceState(alpha=1, beta=4)
    assert updated["engineering"] is original["engineering"]
    assert updated["mechanism"] == BetaEvidenceState(alpha=1.5, beta=1.5)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, "bad"])
def test_non_finite_or_non_numeric_scores_are_rejected(value: object) -> None:
    with pytest.raises(ValueError):
        update_beta_evidence(1, 1, value, "medium")  # type: ignore[arg-type]


def test_mastery_update_eligibility_requires_every_safety_condition() -> None:
    eligible = MasteryUpdateEligibility(
        evaluation_reliable=True,
        evaluator_confidence=0.6,
        minimum_confidence=0.6,
        question_valid=True,
        duplicate_submission=False,
        schema_valid=True,
    )
    assert eligible.eligible
    assert eligible.rejection_reasons == []

    rejected = eligible.model_copy(
        update={"evaluation_reliable": False, "duplicate_submission": True}
    )
    assert not rejected.eligible
    assert rejected.rejection_reasons == [
        "evaluation_unreliable",
        "duplicate_submission",
    ]


def test_invalid_beta_parameters_and_difficulty_are_rejected() -> None:
    with pytest.raises(ValueError):
        update_beta_evidence(0, 1, 50, "medium")
    with pytest.raises(ValueError):
        update_beta_evidence(1, 1, 50, "impossible")
