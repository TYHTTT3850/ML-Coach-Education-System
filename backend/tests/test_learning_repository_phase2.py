from __future__ import annotations

import json
import sqlite3
import uuid

import pytest

from app.core.exceptions import (
    PracticeSetAlreadySubmittedError,
    PracticeSetNotFoundError,
)
from app.core.types import Course
from app.curriculum.models import CurriculumSeed
from app.schemas.learning import EvaluationResultInternal
from app.schemas.practice import PracticeSetInternal
from app.services.learning_repository import LearningRepository


def _topic(
    topic_id: str,
    order: int,
    *,
    prerequisites: list[str] | None = None,
) -> dict:
    return {
        "id": topic_id,
        "slug": topic_id.split(".", 1)[1],
        "title": f"Topic {topic_id}",
        "stage": 1,
        "stage_title": "Repository tests",
        "order_index": order,
        "description": "A persisted topic",
        "learning_objectives": ["Explain the concept"],
        "aliases": [],
        "retrieval_keywords": [topic_id],
        "difficulty": "medium",
        "required_dimensions": ["concept", "mathematics", "mechanism"],
        "active": True,
        "prerequisites": [
            {"topic_id": item, "required_mastery": 0.8}
            for item in prerequisites or []
        ],
    }


@pytest.fixture
def repository(tmp_path) -> LearningRepository:
    value = LearningRepository(tmp_path / "learning.db")
    value.initialize()
    value.upsert_curriculum(
        [
            CurriculumSeed.model_validate(
                {
                    "course": Course.MACHINE_LEARNING,
                    "topics": [
                        _topic("ml.base", 1),
                        _topic("ml.target", 2, prerequisites=["ml.base"]),
                    ],
                }
            )
        ]
    )
    return value


def _practice_set(*, set_id: str, question_id: str) -> PracticeSetInternal:
    return PracticeSetInternal.model_validate(
        {
            "practice_set_id": set_id,
            "course": Course.MACHINE_LEARNING,
            "topic_id": "ml.target",
            "kind": "checkpoint",
            "required_dimensions": ["concept", "mathematics", "mechanism"],
            "questions": [
                {
                    "question_id": question_id,
                    "question_type": "concept_explanation",
                    "difficulty": "medium",
                    "question": "Why does the persisted mechanism work?",
                    "dimensions": {
                        "concept": 0.34,
                        "mathematics": 0.33,
                        "mechanism": 0.33,
                    },
                    "options": None,
                    "answer_key": {
                        "canonical_answer": "Because the invariant is preserved.",
                        "key_points": ["invariant"],
                        "accepted_variants": [],
                        "numeric_answer": None,
                        "tolerance": None,
                        "correct_option": None,
                    },
                    "rubric": [{"criterion": "Names the invariant", "points": 100}],
                    "source_ids": ["S1"],
                    "position": 1,
                    "max_score": 100,
                }
            ],
            "source_snapshot": [
                {
                    "source_id": "S1",
                    "document_id": "doc-1",
                    "source_file": "chapter.md",
                    "content": "The immutable source snapshot explains the invariant.",
                    "title": "Invariant",
                    "chapter": "1",
                    "section": "1.1",
                    "page": 1,
                    "chunk_index": 0,
                }
            ],
        }
    )


def _uncertain_evaluation() -> EvaluationResultInternal:
    return EvaluationResultInternal(
        score=0,
        verdict="evaluation_uncertain",
        dimension_scores={},
        correct_points=[],
        missing_points=[],
        misconception_tags=[],
        critical_misconception=False,
        feedback="The evaluator could not validate this answer.",
        reference_answer="Because the invariant is preserved.",
        evidence_source_ids=["S1"],
        evaluator_confidence=0,
        evaluation_reliable=False,
    )


def _reliable_evaluation() -> EvaluationResultInternal:
    return EvaluationResultInternal(
        score=60,
        verdict="partially_correct",
        dimension_scores={"concept": 70, "mathematics": 55, "mechanism": 55},
        correct_points=["Names part of the invariant"],
        missing_points=["Does not explain preservation"],
        misconception_tags=["ml_objective_loss_confusion"],
        critical_misconception=True,
        feedback="The invariant and its preservation were conflated.",
        reference_answer="Because the invariant is preserved.",
        evidence_source_ids=["S1"],
        evaluator_confidence=0.95,
        evaluation_reliable=True,
    )


def _correct_evaluation() -> EvaluationResultInternal:
    return EvaluationResultInternal(
        score=80,
        verdict="correct",
        dimension_scores={"concept": 90, "mathematics": 90, "mechanism": 90},
        correct_points=["Explains the invariant and preservation"],
        missing_points=[],
        misconception_tags=[],
        critical_misconception=False,
        feedback="The targeted misconception has been corrected.",
        reference_answer="Because the invariant is preserved.",
        evidence_source_ids=["S1"],
        evaluator_confidence=0.99,
        evaluation_reliable=True,
    )


def test_practice_snapshot_round_trip_and_recent_dedup(
    repository: LearningRepository,
) -> None:
    student_id = repository.create_anonymous_student()["id"]
    practice_set = _practice_set(set_id="set-round-trip", question_id="question-round-trip")

    returned_id = repository.create_practice_set(
        student_id=student_id,
        practice_set=practice_set,
        requested_difficulty="hard",
        target_misconception_tags=["tag-a", "tag-a", "  tag-b  ", ""],
    )

    loaded = repository.get_practice_set_internal(returned_id, student_id=student_id)
    assert loaded == practice_set
    assert loaded.questions[0].answer_key.canonical_answer.startswith("Because")
    assert loaded.questions[0].rubric[0].points == 100
    assert repository.get_practice_set_status(returned_id) == "generated"
    assert repository.list_recent_question_texts(student_id, "ml.target") == [
        practice_set.questions[0].question
    ]
    assert repository.list_recent_question_hashes(student_id, "ml.target") == {
        practice_set.questions[0].normalized_question_hash
    }
    with repository._connect() as connection:
        row = connection.execute(
            """
            SELECT requested_difficulty, target_misconception_tags_json
            FROM practice_sets WHERE id = ?
            """,
            (returned_id,),
        ).fetchone()
    assert row["requested_difficulty"] == "hard"
    assert json.loads(row["target_misconception_tags_json"]) == ["tag-a", "tag-b"]

    with pytest.raises(PracticeSetNotFoundError):
        repository.get_practice_set_internal(returned_id, student_id=str(uuid.uuid4()))


def test_uncertain_submission_can_retry_without_mastery_update_then_locks_on_success(
    repository: LearningRepository,
) -> None:
    student_id = repository.create_anonymous_student()["id"]
    other_student_id = repository.create_anonymous_student()["id"]
    state_before = repository.ensure_topic_state(student_id, "ml.target")
    practice_set = _practice_set(set_id="retry-set", question_id="retry-question")
    repository.create_practice_set(
        student_id=student_id,
        practice_set=practice_set,
        target_misconception_tags=["ml_objective_loss_confusion"],
    )

    first_client_id = str(uuid.uuid4())
    first_submission_id = repository.reserve_submission(
        practice_set_id=practice_set.practice_set_id,
        student_id=student_id,
        client_submission_id=first_client_id,
        mastery_before=float(state_before["mastery"]),
    )
    first_attempt_id = str(uuid.uuid4())
    repository.finalize_submission(
        submission_id=first_submission_id,
        practice_set=practice_set,
        answers={"retry-question": "I am not sure"},
        evaluations=[_uncertain_evaluation()],
        attempt_ids={"retry-question": first_attempt_id},
        total_score=0,
        evaluation_status="evaluation_uncertain",
        mastery_before=0.5,
        mastery_after=0.99,
        gate_result={"passed": False},
        response_json='{"status":"evaluation_uncertain"}',
        # Even defensive callers cannot mutate Beta through an uncertain result.
        topic_changes={"alpha": 99.0, "beta": 1.0, "mastery": 0.99},
        dimension_changes={"concept": {"alpha": 99.0, "beta": 1.0}},
        mark_submitted=False,
    )

    unchanged = repository.get_topic_state(student_id, "ml.target")
    assert unchanged is not None
    assert (unchanged["alpha"], unchanged["beta"], unchanged["mastery"]) == (
        state_before["alpha"],
        state_before["beta"],
        state_before["mastery"],
    )
    assert repository.get_practice_set_status(practice_set.practice_set_id) == (
        "evaluation_uncertain"
    )
    assert repository.count_unresolved_critical_misconceptions(
        student_id, "ml.target"
    ) == 0
    assert repository.get_submission_by_client_id(
        first_client_id,
        student_id=student_id,
        practice_set_id=practice_set.practice_set_id,
    )["response_json"]
    assert repository.get_submission_by_client_id(
        first_client_id,
        student_id=other_student_id,
        practice_set_id=practice_set.practice_set_id,
    ) is None

    context = repository.get_attempt_context(first_attempt_id, student_id=student_id)
    assert context is not None
    assert context["question"]["answer_key"]["canonical_answer"]
    assert context["student_answer"] == "I am not sure"
    assert context["evaluation"]["evaluation_reliable"] is False
    assert context["source_snapshot"][0]["content"].startswith("The immutable")
    assert context["target_misconception_tags"] == [
        "ml_objective_loss_confusion"
    ]
    assert repository.get_attempt_context(
        first_attempt_id, student_id=other_student_id
    ) is None

    second_client_id = str(uuid.uuid4())
    second_submission_id = repository.reserve_submission(
        practice_set_id=practice_set.practice_set_id,
        student_id=student_id,
        client_submission_id=second_client_id,
        mastery_before=0.5,
    )
    second_attempt_id = str(uuid.uuid4())
    repository.finalize_submission(
        submission_id=second_submission_id,
        practice_set=practice_set,
        answers={"retry-question": "The invariant remains true at each step"},
        evaluations=[_reliable_evaluation()],
        attempt_ids={"retry-question": second_attempt_id},
        total_score=60,
        evaluation_status="evaluated",
        mastery_before=0.5,
        mastery_after=2 / 3,
        gate_result={"passed": False},
        response_json='{"status":"evaluated"}',
        topic_changes={
            "alpha": 2.0,
            "beta": 1.0,
            "mastery": 2 / 3,
            "confidence": 0.1,
            "evidence_count": 1,
            "attempt_count": 1,
            "checkpoint_count": 1,
            "latest_score": 60,
            "latest_checkpoint_score": 60,
            "status": "needs_remediation",
        },
        dimension_changes={
            name: {
                "alpha": 1.6,
                "beta": 1.4,
                "mastery": 1.6 / 3,
                "confidence": 0.1,
                "evidence_count": 1,
            }
            for name in ("concept", "mathematics", "mechanism")
        },
        mark_submitted=True,
    )

    assert repository.get_practice_set_status(practice_set.practice_set_id) == "submitted"
    assert repository.get_topic_state(student_id, "ml.target")["alpha"] == 2.0
    assert repository.get_latest_checkpoint_dimensions(student_id, "ml.target") == [
        "concept",
        "mathematics",
        "mechanism",
    ]
    assert repository.count_unresolved_critical_misconceptions(
        student_id, "ml.target"
    ) == 1
    with pytest.raises(PracticeSetAlreadySubmittedError):
        repository.reserve_submission(
            practice_set_id=practice_set.practice_set_id,
            student_id=student_id,
            client_submission_id=str(uuid.uuid4()),
            mastery_before=2 / 3,
        )

    remediation_set = _practice_set(
        set_id="targeted-remediation-set",
        question_id="targeted-remediation-question",
    )
    repository.create_practice_set(
        student_id=student_id,
        practice_set=remediation_set,
        target_misconception_tags=["ml_objective_loss_confusion"],
    )
    # Generating or reading remediation is not evidence that the error is fixed.
    repository.get_practice_set_internal(
        remediation_set.practice_set_id,
        student_id=student_id,
    )
    assert repository.count_unresolved_critical_misconceptions(
        student_id, "ml.target"
    ) == 1

    remediation_submission_id = repository.reserve_submission(
        practice_set_id=remediation_set.practice_set_id,
        student_id=student_id,
        client_submission_id=str(uuid.uuid4()),
        mastery_before=2 / 3,
    )
    repository.finalize_submission(
        submission_id=remediation_submission_id,
        practice_set=remediation_set,
        answers={
            "targeted-remediation-question": "The invariant is preserved at each step"
        },
        evaluations=[_correct_evaluation()],
        attempt_ids={"targeted-remediation-question": str(uuid.uuid4())},
        total_score=80,
        evaluation_status="evaluated",
        mastery_before=2 / 3,
        mastery_after=0.75,
        gate_result={"passed": False},
        response_json='{"status":"evaluated","kind":"remediation"}',
        topic_changes={
            "alpha": 3.0,
            "beta": 1.0,
            "mastery": 0.75,
            "evidence_count": 2,
            "attempt_count": 2,
            "checkpoint_count": 2,
            "latest_score": 80,
            "latest_checkpoint_score": 80,
            "status": "practicing",
        },
        dimension_changes={},
        mark_submitted=True,
    )
    assert repository.count_unresolved_critical_misconceptions(
        student_id, "ml.target"
    ) == 0


def test_history_prerequisites_and_atomic_finalize_rollback(
    repository: LearningRepository,
) -> None:
    student_id = repository.create_anonymous_student()["id"]
    repository.ensure_topic_state(student_id, "ml.base")
    repository.update_topic_state(
        student_id,
        "ml.base",
        mastery=0.91,
        status="practicing",
    )
    assert repository.get_prerequisite_masteries(student_id, "ml.target") == {
        "ml.base": 0.0
    }
    repository.update_topic_state(
        student_id,
        "ml.base",
        status="mastered",
    )
    assert repository.get_prerequisite_masteries(student_id, "ml.target") == {
        "ml.base": 0.91
    }
    session_id = repository.create_learning_session(
        student_id=student_id,
        course=Course.MACHINE_LEARNING,
        topic_id="ml.target",
        state="learning",
    )

    practice_set = _practice_set(set_id="atomic-set", question_id="atomic-question")
    repository.create_practice_set(student_id=student_id, practice_set=practice_set)
    repository.ensure_topic_state(student_id, "ml.target")
    submission_id = repository.reserve_submission(
        practice_set_id=practice_set.practice_set_id,
        student_id=student_id,
        client_submission_id=str(uuid.uuid4()),
        mastery_before=0.5,
    )
    with repository._connect() as connection:
        connection.execute(
            """
            CREATE TRIGGER reject_test_mastery_update
            BEFORE UPDATE ON student_topic_states
            WHEN NEW.topic_id = 'ml.target'
            BEGIN
                SELECT RAISE(ABORT, 'forced_atomic_rollback');
            END
            """
        )

    with pytest.raises(sqlite3.DatabaseError, match="forced_atomic_rollback"):
        repository.finalize_submission(
            submission_id=submission_id,
            practice_set=practice_set,
            answers={"atomic-question": "An answer"},
            evaluations=[_reliable_evaluation()],
            attempt_ids={"atomic-question": str(uuid.uuid4())},
            total_score=60,
            evaluation_status="evaluated",
            mastery_before=0.5,
            mastery_after=0.6,
            gate_result={"passed": False},
            response_json='{"status":"evaluated"}',
            topic_changes={"alpha": 1.5, "beta": 1.0, "mastery": 0.6},
            dimension_changes={},
            mark_submitted=True,
        )

    with repository._connect() as connection:
        counts = {
            table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("practice_attempts", "misconception_events")
        }
        submission_status = connection.execute(
            "SELECT evaluation_status FROM practice_submissions WHERE id = ?",
            (submission_id,),
        ).fetchone()["evaluation_status"]
    assert counts == {"practice_attempts": 0, "misconception_events": 0}
    assert submission_status == "processing"
    assert repository.get_practice_set_status(practice_set.practice_set_id) == "generated"

    history = repository.list_learning_history(
        student_id,
        course=Course.MACHINE_LEARNING,
        limit=10,
    )
    assert [item["event_type"] for item in history] == ["session"]
    assert history[0]["id"] == session_id
    assert repository.list_recent_submissions(
        student_id,
        course=Course.MACHINE_LEARNING,
        limit=10,
    ) == []
