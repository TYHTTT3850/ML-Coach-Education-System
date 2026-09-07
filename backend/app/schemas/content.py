"""Versioned instructional content and deliberately separate student projections."""
from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator
from app.core.types import Course
from app.schemas.practice import (AnswerKeyInternal, MultipleChoiceOption,
    RubricCriterionInternal, PracticeQuestionInternal, SourceSnapshotItem,
    referenced_internal_source_ids)

class ContentModel(BaseModel):
    model_config = ConfigDict(extra='forbid')

class Formula(ContentModel):
    expression: str
    symbols: str
    conditions: str

class Lesson(ContentModel):
    id: str
    topic_id: str
    title: str
    learning_objectives: list[str] = Field(min_length=1)
    prerequisite_summary: str
    motivating_problem: str = Field(min_length=1)
    key_concepts: list[str] = Field(min_length=1)
    formulas: list[Formula]
    worked_example_ids: list[str] = Field(min_length=1)
    practice_recommendations: list[str]
    summary: str
    source_refs: list[dict[str, Any]] = Field(default_factory=list)
    content_origin: Literal['source_derived','instructor_authored','generated_draft']
    version: int = Field(default=1, ge=1)
    status: Literal['draft','published','archived'] = 'draft'

class ExampleStep(ContentModel):
    step_id: str
    title: str
    explanation: str = Field(min_length=1)
    formula: str = ''
    substitution: str = ''
    result: str = ''
    justification: str = Field(min_length=1)

class WorkedExample(ContentModel):
    id: str
    topic_id: str
    title: str
    difficulty: Literal['easy','medium','hard']
    problem: str
    givens: dict[str, Any]
    target: str
    solution_outline: str
    steps: list[ExampleStep] = Field(min_length=2)
    final_answer: str
    verification: dict[str, Any]
    common_mistakes: list[str]
    followup_practice_ids: list[str]
    source_refs: list[dict[str, Any]] = Field(default_factory=list)
    content_origin: Literal['source_derived','instructor_authored','generated_draft']
    version: int = Field(default=1,ge=1)
    status: Literal['draft','published','archived'] = 'draft'
    family_id: str

class BankItemPublic(ContentModel):
    id: str
    version: int
    topic_ids: list[str]
    primary_course: Course
    title: str
    question_type: str
    difficulty: str
    stem: str
    options: list[MultipleChoiceOption] | None = None
    tested_competencies: list[str]
    estimated_minutes: int
    usage: list[str]
    content_origin: str
    publication_status: str
    validation_status: str

class BankItem(ContentModel):
    id: str = Field(min_length=1, max_length=200)
    version: int = Field(default=1, ge=1)
    topic_ids: list[str] = Field(min_length=1, max_length=20)
    primary_course: Course
    title: str = Field(min_length=1)
    question_type: str
    difficulty: Literal['easy','medium','hard']
    stem: str = Field(min_length=1)
    options: list[MultipleChoiceOption] | None = None
    answer_key: AnswerKeyInternal
    accepted_variants: list[str] = Field(default_factory=list)
    solution_steps: list[str] = Field(min_length=1)
    rubric: list[RubricCriterionInternal] = Field(min_length=1)
    tested_competencies: list[str] = Field(min_length=1)
    dimensions: dict[str,float] = Field(min_length=1)
    estimated_minutes: int = Field(default=3, ge=1, le=120)
    usage: list[Literal['practice','checkpoint','prerequisite_test']] = Field(min_length=1)
    family_id: str = Field(min_length=1)
    source_refs: list[dict[str,Any]] = Field(default_factory=list)
    source_snapshot: list[SourceSnapshotItem] = Field(default_factory=list)
    content_origin: Literal['source_derived','instructor_authored','generated_draft']
    validation_status: Literal['needs_review','deterministic_checks_passed','human_reviewed'] = 'needs_review'
    publication_status: Literal['draft','published','archived'] = 'draft'
    verification: dict[str,Any] = Field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None

    @model_validator(mode='after')
    def validate_content(self):
        if len(set(self.topic_ids)) != len(self.topic_ids):
            raise ValueError('知识点不能重复')
        if self.content_origin != 'instructor_authored' and not self.source_snapshot:
            raise ValueError('资料整理或生成题必须保留真实资料快照')
        if self.content_origin == 'instructor_authored' and self.source_refs:
            raise ValueError('项目自编题不能附加伪造教材引用')
        texts=[self.stem]+[o.text for o in self.options or []]
        if referenced_internal_source_ids(texts,set()):
            raise ValueError('学生题面不能出现内部来源编号')
        if self.options and len({o.text.strip().casefold() for o in self.options}) != len(self.options):
            raise ValueError('选项文字必须唯一')
        self.as_question('validate',1)
        return self

    def as_question(self, question_id: str, position: int) -> PracticeQuestionInternal:
        return PracticeQuestionInternal(
            question_id=question_id, question_type=self.question_type,
            difficulty=self.difficulty, question=self.stem, dimensions=self.dimensions,
            options=self.options,answer_key=self.answer_key.model_copy(update={
                'accepted_variants':list(dict.fromkeys([*self.answer_key.accepted_variants,*self.accepted_variants]))}),rubric=self.rubric,
            source_ids=[s.source_id for s in self.source_snapshot] or [f'authored:{self.id}:v{self.version}'],
            position=position)

    def public(self):
        return BankItemPublic.model_validate({k:getattr(self,k) for k in BankItemPublic.model_fields})

class AssessmentBlueprint(ContentModel):
    topic_ids: list[str] = Field(min_length=1,max_length=8)
    purpose: Literal['practice','checkpoint','prerequisite_test'] = 'practice'
    question_count: int = Field(default=4,ge=1,le=8)
    difficulty_distribution: dict[Literal['easy','medium','hard'],int] = Field(default_factory=dict)
    required_competencies: list[str] = Field(default_factory=list)
    allowed_question_types: list[str] = Field(default_factory=list)
    exclude_seen: bool = True
    exclude_exposed_families: bool = True
    bank_item_ids: list[str] = Field(default_factory=list,max_length=8)

    @model_validator(mode='after')
    def valid_blueprint(self):
        if len(set(self.topic_ids)) != len(self.topic_ids):
            raise ValueError('知识点不能重复')
        if self.difficulty_distribution and (sum(self.difficulty_distribution.values()) != self.question_count or any(v<0 for v in self.difficulty_distribution.values())):
            raise ValueError('难度分布必须恰好覆盖题目数量')
        if self.purpose != 'practice' and self.bank_item_ids:
            raise ValueError('正式测试不接受客户端指定试题')
        return self

class PrerequisiteTestRequest(ContentModel):
    topic_id: str
    prerequisite_topic_id: str | None = None
    question_count: int = Field(default=3,ge=3,le=5)
    generation_mode: Literal['dynamic', 'bank'] = 'dynamic'

class ExampleAsk(ContentModel):
    version: int = Field(ge=1)
    step_id: str
    question: str = Field(default='为什么这一步可以这样做？',min_length=1,max_length=2000)

class ExampleExposure(ContentModel):
    version: int = Field(ge=1)

class QuestionHelp(ContentModel):
    question_id: str
    attempt: str = Field(default='',max_length=50000)

class ImportItems(ContentModel):
    items: list[BankItem] = Field(min_length=1,max_length=1000)

class PublishRequest(ContentModel):
    human_reviewed: bool = False
    review_note: str = ''
