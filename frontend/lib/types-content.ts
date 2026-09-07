import type { Course, PublicPracticeSet, PracticeSubmitResult, Source, TopicStatus } from "./types";

export interface OutlineTopic {
  id: string; topic_id: string; owner_course: Course; title: string; description: string;
  learning_objectives: string[]; order_index: number; is_reference: boolean;
  status: TopicStatus; mastery: number | null; prerequisite_gap: { id: string; title: string; reason: string }[];
  content_status: "ready" | "missing"; can_checkpoint: boolean;
  has_learning_content?: boolean; readiness_reason?: string;
}
export interface OutlineGroup { id: string; title: string; parent_group_id: string | null; order_index: number; topics: OutlineTopic[] }
export interface CourseOutline { course: Course; title: string; groups: OutlineGroup[]; recommended_topic_id: string | null }
export interface PrerequisiteEdge {
  id: string; prerequisite_topic_id: string; dependent_topic_id: string; title: string; course: Course;
  relation_type: "required" | "recommended" | "related"; reason: string; required_competencies: string[];
  scope: "topic" | "lesson_section"; section_id?: string; satisfied: boolean; verified_competencies: string[];
}
export interface Prerequisites {
  topic_id: string; owner_course: Course; rule_version: string; can_browse: boolean; can_study: boolean; can_checkpoint: boolean;
  required: PrerequisiteEdge[]; recommended: PrerequisiteEdge[]; related: PrerequisiteEdge[]; unmet: PrerequisiteEdge[];
  successors: { id: string; title: string; course: Course; relation_type: string; reason: string }[];
}
export interface Lesson {
  id: string; topic_id: string; title: string; learning_objectives: string[]; prerequisite_summary: string;
  motivating_problem: string; key_concepts: string[]; formulas: { expression: string; symbols: string; conditions: string }[];
  worked_example_ids: string[]; practice_recommendations: string[]; summary: string; source_refs: Source[];
  content_origin: string; version: number; status: string;
}
export interface ExampleStep { step_id: string; title: string; explanation: string; formula: string; substitution: string; result: string; justification: string }
export interface WorkedExampleData {
  id: string; topic_id: string; title: string; difficulty: string; problem: string; givens: Record<string, unknown>; target: string;
  solution_outline: string; steps: ExampleStep[]; final_answer: string; verification: Record<string, unknown>;
  common_mistakes: string[]; followup_practice_ids: string[]; source_refs: Source[]; content_origin: string; version: number; status: string; family_id: string;
}
export interface PublicBankItem {
  id: string; version: number; topic_ids: string[]; primary_course: Course; title: string; question_type: string; difficulty: string;
  stem: string; options: { option_id: string; text: string }[] | null; tested_competencies: string[]; estimated_minutes: number;
  usage: string[]; content_origin: string; publication_status: string; validation_status: string;
}
export interface ContentCoverage {
  topic_id: string; title: string; owner_course: Course; lesson_count: number; example_count: number; practice_count: number;
  checkpoint_count: number; published_count: number; needs_review_count: number; prerequisites_ready: boolean; full_flow_ready: boolean;
  independent_practice_count?: number; independent_checkpoint_count?: number; readiness_gaps?: string[];
}
export interface BankQueryResult { items: PublicBankItem[]; checkpoint_count: number; checkpoint_difficulty_distribution?: Record<string, number>; coverage?: ContentCoverage[] }
export interface Assessment extends PublicPracticeSet { purpose?: "practice" | "checkpoint" | "prerequisite_test"; result?: PracticeSubmitResult | null }
export interface AssessmentBlueprint {
  topic_ids: string[]; purpose: "practice" | "checkpoint"; question_count: number; bank_item_ids?: string[];
  exclude_seen?: boolean; exclude_exposed_families?: boolean; required_competencies?: string[];
  allowed_question_types?: string[]; difficulty_distribution?: Record<string, number>;
}
