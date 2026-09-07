"""Deterministic learning-loop rules.

This package deliberately contains no database, HTTP, or LLM dependencies.  The
API/service layer can therefore reuse the rules and unit-test them in isolation.
"""

from app.learning.gate import MasteryGateConfig, MasteryGateInput, evaluate_mastery_gate
from app.learning.mastery import (
    BetaEvidenceState,
    EvidenceUpdate,
    clamp_score,
    confidence_from_evidence,
    difficulty_weight,
    mastery_from_evidence,
    update_beta_evidence,
    update_dimension_evidence,
)
from app.learning.misconceptions import (
    ADVANCED_MATHEMATICS_MISCONCEPTION_TAGS,
    MACHINE_LEARNING_MISCONCEPTION_TAGS,
    normalize_evaluation_misconceptions,
    normalize_misconception_tag,
    normalize_misconception_tags,
)
from app.learning.orchestrator import (
    LearningOrchestratorInput,
    TopicRecommendationCandidate,
    decide_next_action,
    recommend_topic,
)

__all__ = [
    "ADVANCED_MATHEMATICS_MISCONCEPTION_TAGS",
    "BetaEvidenceState",
    "EvidenceUpdate",
    "LearningOrchestratorInput",
    "MACHINE_LEARNING_MISCONCEPTION_TAGS",
    "MasteryGateConfig",
    "MasteryGateInput",
    "TopicRecommendationCandidate",
    "clamp_score",
    "confidence_from_evidence",
    "decide_next_action",
    "difficulty_weight",
    "evaluate_mastery_gate",
    "mastery_from_evidence",
    "normalize_misconception_tag",
    "normalize_misconception_tags",
    "normalize_evaluation_misconceptions",
    "recommend_topic",
    "update_beta_evidence",
    "update_dimension_evidence",
]
