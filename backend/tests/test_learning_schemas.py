from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from app.core.types import Course
from app.learning.misconceptions import (
    normalize_evaluation_misconceptions,
    normalize_misconception_tag,
    normalize_misconception_tags,
)
from app.schemas.learning import EvaluationResultInternal
from app.schemas.practice import (
    AnswerKeyInternal,
    MultipleChoiceOption,
    PracticeQuestionInternal,
    PracticeQuestionPublic,
    PracticeSetInternal,
    RubricCriterionInternal,
    SourceSnapshotItem,
    referenced_internal_source_ids,
)


def source(source_id: str = "S1") -> SourceSnapshotItem:
    return SourceSnapshotItem(
        source_id=source_id,
        document_id="doc-1",
        source_file="lesson.pdf",
        content="真实课程资料片段",
        chunk_index=0,
    )


def text_question(**overrides: object) -> PracticeQuestionInternal:
    values: dict[str, object] = {
        "question_id": "q-1",
        "question_type": "concept_explanation",
        "difficulty": "medium",
        "question": "解释这个概念。",
        "dimensions": {"concept": 1.0},
        "answer_key": AnswerKeyInternal(
            canonical_answer="参考答案",
            key_points=["关键点"],
        ),
        "rubric": [RubricCriterionInternal(criterion="概念正确", points=100)],
        "source_ids": ["S1"],
        "position": 1,
    }
    values.update(overrides)
    return PracticeQuestionInternal(**values)  # type: ignore[arg-type]


def test_public_schema_cannot_leak_internal_answer_fields() -> None:
    internal = PracticeSetInternal(
        practice_set_id="set-1",
        course=Course.MACHINE_LEARNING,
        topic_id="ml.logistic_regression",
        kind="practice",
        required_dimensions=["concept", "mathematics", "mechanism"],
        questions=[text_question()],
        source_snapshot=[source()],
    )

    public = internal.to_public()
    payload = public.model_dump(mode="json")
    serialized = json.dumps(payload)
    question_payload = payload["questions"][0]

    assert set(question_payload) == {
        "question_id",
        "question_type",
        "difficulty",
        "question",
        "options",
        "dimensions",
        "position",
        "max_score",
    }
    assert "answer_key" not in serialized
    assert "rubric" not in serialized
    assert "canonical_answer" not in serialized
    assert "key_points" not in serialized
    assert "source_snapshot" not in serialized
    assert "source_ids" not in serialized
    assert "correct_option" not in serialized

    with pytest.raises(ValidationError):
        PracticeQuestionPublic.model_validate(internal.questions[0].model_dump())


def test_rubric_must_sum_to_exactly_100() -> None:
    with pytest.raises(ValidationError, match="rubric points must sum to 100"):
        text_question(
            rubric=[RubricCriterionInternal(criterion="不完整", points=99)]
        )


def test_dimensions_must_sum_to_approximately_one() -> None:
    with pytest.raises(ValidationError, match="sum to approximately 1"):
        text_question(dimensions={"concept": 0.4, "mathematics": 0.4})


def test_numeric_question_requires_answer_and_tolerance() -> None:
    with pytest.raises(ValidationError, match="numeric_answer and tolerance"):
        text_question(
            question_type="numeric",
            answer_key=AnswerKeyInternal(canonical_answer="2"),
        )

    valid = text_question(
        question_type="numeric",
        answer_key=AnswerKeyInternal(
            canonical_answer="2",
            numeric_answer=2,
            tolerance=0.01,
        ),
    )
    assert valid.answer_key.numeric_answer == 2


def test_non_numeric_question_rejects_numeric_fields_and_requires_key_points() -> None:
    with pytest.raises(ValidationError, match="cannot define numeric"):
        text_question(
            answer_key=AnswerKeyInternal(
                canonical_answer="答案",
                key_points=["关键点"],
                numeric_answer=2,
                tolerance=0.1,
            )
        )
    with pytest.raises(ValidationError, match="require key_points"):
        text_question(answer_key=AnswerKeyInternal(canonical_answer="答案"))


def test_multiple_choice_requires_options_and_valid_correct_option() -> None:
    choices = [
        MultipleChoiceOption(option_id="A", text="选项 A"),
        MultipleChoiceOption(option_id="B", text="选项 B"),
    ]
    question = text_question(
        question_type="multiple_choice",
        options=choices,
        answer_key=AnswerKeyInternal(
            canonical_answer="A",
            key_points=["A 正确"],
            correct_option="A",
        ),
    )
    public = question.to_public()
    assert [option.option_id for option in public.options or []] == ["A", "B"]
    assert "correct_option" not in public.model_dump()

    with pytest.raises(ValidationError, match="existing option"):
        text_question(
            question_type="multiple_choice",
            options=choices,
            answer_key=AnswerKeyInternal(
                canonical_answer="C",
                key_points=["C 正确"],
                correct_option="C",
            ),
        )


def test_question_source_ids_must_exist_in_real_snapshot() -> None:
    with pytest.raises(ValidationError, match="unknown source ids"):
        PracticeSetInternal(
            practice_set_id="set-1",
            course=Course.MACHINE_LEARNING,
            topic_id="ml.logistic_regression",
            kind="practice",
            required_dimensions=["concept"],
            questions=[text_question(source_ids=["S2"])],
            source_snapshot=[source("S1")],
        )


def test_historical_internal_source_ids_are_hidden_from_public_question() -> None:
    question = text_question(question="请结合 S1 解释这个概念。")

    assert question.question == "请结合 S1 解释这个概念。"
    assert question.to_public().question == "请结合课程资料解释这个概念。"


def test_source_reference_detector_catches_unknown_s_number() -> None:
    assert referenced_internal_source_ids(
        ["请结合 S99 和 S1 回答。"],
        {"S1"},
    ) == ["S1", "S99"]


def test_checkpoint_contract_enforces_coverage_difficulty_and_question_mix() -> None:
    with pytest.raises(ValidationError, match="at least three dimensions"):
        PracticeSetInternal(
            practice_set_id="set-1",
            course=Course.MACHINE_LEARNING,
            topic_id="ml.logistic_regression",
            kind="checkpoint",
            required_dimensions=["concept", "mathematics", "mechanism"],
            questions=[text_question()],
            source_snapshot=[source()],
        )


def test_valid_checkpoint_covers_three_required_dimensions() -> None:
    questions = [
        text_question(
            question_id="q-concept",
            question="解释概念。",
            dimensions={"concept": 1.0},
            position=1,
        ),
        text_question(
            question_id="q-math",
            question="说明数学关系。",
            question_type="mathematical_reasoning",
            dimensions={"mathematics": 1.0},
            position=2,
        ),
        text_question(
            question_id="q-mechanism",
            question="说明工作机制。",
            question_type="why_mechanism",
            dimensions={"mechanism": 1.0},
            position=3,
        ),
    ]
    checkpoint = PracticeSetInternal(
        practice_set_id="checkpoint-1",
        course=Course.MACHINE_LEARNING,
        topic_id="ml.logistic_regression",
        kind="checkpoint",
        required_dimensions=["concept", "mathematics", "mechanism"],
        questions=questions,
        source_snapshot=[source()],
    )
    assert len(checkpoint.questions) == 3


def test_set_rejects_all_choice_questions_and_cross_course_types() -> None:
    choice = text_question(
        question_type="multiple_choice",
        options=[
            MultipleChoiceOption(option_id="A", text="A"),
            MultipleChoiceOption(option_id="B", text="B"),
        ],
        answer_key=AnswerKeyInternal(
            canonical_answer="A",
            key_points=["A"],
            correct_option="A",
        ),
    )
    with pytest.raises(ValidationError, match="cannot contain only"):
        PracticeSetInternal(
            practice_set_id="set-choice",
            course=Course.MACHINE_LEARNING,
            topic_id="ml.logistic_regression",
            kind="practice",
            required_dimensions=["concept"],
            questions=[choice],
            source_snapshot=[source()],
        )

    math_question = text_question(
        question_type="theorem_condition",
        dimensions={"theorem_condition": 1.0},
    )
    with pytest.raises(ValidationError, match="does not belong to course"):
        PracticeSetInternal(
            practice_set_id="set-cross-course",
            course=Course.MACHINE_LEARNING,
            topic_id="ml.logistic_regression",
            kind="practice",
            required_dimensions=["concept"],
            questions=[math_question],
            source_snapshot=[source()],
        )


def evaluation_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "score": 72,
        "verdict": "partially_correct",
        "dimension_scores": {"concept": 80, "mathematics": 65},
        "correct_points": ["概念方向正确"],
        "missing_points": ["缺少推导"],
        "misconception_tags": ["incomplete_answer"],
        "critical_misconception": False,
        "feedback": "需要补充推导。",
        "reference_answer": "参考答案",
        "evidence_source_ids": ["S1"],
        "evaluator_confidence": 0.88,
        "evaluation_reliable": True,
    }
    payload.update(overrides)
    return payload


def test_evaluation_validates_score_confidence_and_snapshot_sources() -> None:
    result = EvaluationResultInternal.model_validate(
        evaluation_payload(), context={"source_ids": {"S1", "S2"}}
    )
    assert result.evaluation_reliable

    with pytest.raises(ValidationError, match="unknown evidence source ids"):
        EvaluationResultInternal.model_validate(
            evaluation_payload(evidence_source_ids=["S9"]),
            context={"source_ids": {"S1"}},
        )
    with pytest.raises(ValidationError):
        EvaluationResultInternal.model_validate(evaluation_payload(score=float("nan")))
    with pytest.raises(ValidationError):
        EvaluationResultInternal.model_validate(
            evaluation_payload(evaluator_confidence=1.1)
        )


def test_uncertain_evaluation_is_never_reliable_or_critical() -> None:
    uncertain = EvaluationResultInternal.model_validate(
        evaluation_payload(
            verdict="evaluation_uncertain",
            evaluation_reliable=False,
            critical_misconception=False,
            evaluator_confidence=0.2,
        )
    )
    assert not uncertain.evaluation_reliable

    with pytest.raises(ValidationError, match="must be unreliable"):
        EvaluationResultInternal.model_validate(
            evaluation_payload(
                verdict="evaluation_uncertain", evaluation_reliable=True
            )
        )
    with pytest.raises(ValidationError, match="cannot assert a critical"):
        EvaluationResultInternal.model_validate(
            evaluation_payload(
                verdict="evaluation_uncertain",
                evaluation_reliable=False,
                critical_misconception=True,
            )
        )


def test_evaluation_contract_rejects_hyde_or_hidden_prompt_fields() -> None:
    with pytest.raises(ValidationError):
        EvaluationResultInternal.model_validate(
            evaluation_payload(hypothetical_answers=["伪证据"])
        )
    public = EvaluationResultInternal.model_validate(evaluation_payload()).to_public()
    assert "rubric" not in public.model_dump()
    assert "system_prompt" not in public.model_dump()


@pytest.mark.parametrize(
    ("course", "known", "unknown"),
    [
        (Course.MACHINE_LEARNING, "data_leakage", "derivative_rule_misuse"),
        (
            Course.ADVANCED_MATHEMATICS,
            "derivative_rule_misuse",
            "data_leakage",
        ),
    ],
)
def test_misconceptions_are_course_controlled_and_unknown_maps_to_other(
    course: Course, known: str, unknown: str
) -> None:
    assert normalize_misconception_tag(course, known) == known
    assert normalize_misconception_tag(course, unknown) == "other"
    assert normalize_misconception_tag(course, "invented_by_model") == "other"
    assert normalize_misconception_tags(course, [known, unknown, "bad"]) == [
        known,
        "other",
    ]


def test_full_evaluation_can_be_normalized_before_persistence() -> None:
    evaluation = EvaluationResultInternal.model_validate(
        evaluation_payload(
            misconception_tags=["data_leakage", "invented_by_model"]
        )
    )
    normalized = normalize_evaluation_misconceptions(
        Course.MACHINE_LEARNING, evaluation
    )
    assert normalized.misconception_tags == ["data_leakage", "other"]
    assert evaluation.misconception_tags == ["data_leakage", "invented_by_model"]
