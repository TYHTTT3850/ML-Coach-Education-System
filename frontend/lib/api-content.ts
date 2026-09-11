import { request, sanitizePublicPracticeSet } from "./api";
import type { Course, PracticeAnswer, PracticeSubmitResult, Source } from "./types";
import type { Assessment, AssessmentBlueprint, BankQueryResult, ContentCoverage, CourseOutline, Lesson, Prerequisites, WorkedExampleData } from "./types-content";

const student = (id: string) => ({ Authorization: `Bearer ${id}` });
const jsonStudent = (id: string) => ({ ...student(id), "Content-Type": "application/json" });
export const getCourseOutline = (course: Course, id: string) => request<CourseOutline>(`/api/v1/courses/${course}/outline`, { headers: student(id) });
export const getPrerequisites = (topicId: string, id: string) => request<Prerequisites>(`/api/v1/topics/${encodeURIComponent(topicId)}/prerequisites`, { headers: student(id) });
export const getLesson = (topicId: string) => request<Lesson>(`/api/v1/topics/${encodeURIComponent(topicId)}/lesson`);
export const getExamples = (topicId: string) => request<{ examples: WorkedExampleData[] }>(`/api/v1/topics/${encodeURIComponent(topicId)}/examples`);
export const exposeExample = (example: WorkedExampleData, id: string) => request(`/api/v1/examples/${encodeURIComponent(example.id)}/exposure`, { method: "POST", headers: jsonStudent(id), body: JSON.stringify({ version: example.version }) });
export const askExample = (example: WorkedExampleData, stepId: string, question: string, id: string) => request<{ answer: string; sources: Source[] }>(`/api/v1/examples/${encodeURIComponent(example.id)}/ask`, { method: "POST", headers: jsonStudent(id), body: JSON.stringify({ version: example.version, step_id: stepId, question }) });
export const getQuestionBank = (filters: Record<string, string>) => request<BankQueryResult>(`/api/v1/question-bank?${new URLSearchParams(Object.entries(filters).filter(([, value]) => value))}`);
export const getContentCoverage = () => request<{ topics: ContentCoverage[] }>("/api/v1/content/coverage");
function publicAssessment(payload: unknown): Assessment {
  const body = payload as { purpose?: Assessment["purpose"]; result?: PracticeSubmitResult | null };
  return { ...sanitizePublicPracticeSet(payload), purpose: body.purpose, result: body.result ?? null };
}
export async function createAssessment(blueprint: AssessmentBlueprint, id: string): Promise<Assessment> {
  return publicAssessment(await request("/api/v1/assessments", { method: "POST", headers: jsonStudent(id), body: JSON.stringify(blueprint) }));
}
export async function getAssessment(assessmentId: string, id: string): Promise<Assessment> {
  return publicAssessment(await request(`/api/v1/assessments/${encodeURIComponent(assessmentId)}`, { headers: student(id) }));
}
export const submitAssessment = (assessmentId: string, payload: { client_submission_id: string; answers: PracticeAnswer[] }, id: string) => request<PracticeSubmitResult>(`/api/v1/assessments/${encodeURIComponent(assessmentId)}/submit`, { method: "POST", headers: jsonStudent(id), body: JSON.stringify(payload) });
export async function createPrerequisiteTest(topicId: string, id: string, prerequisiteTopicId?: string): Promise<Assessment> {
  return publicAssessment(await request("/api/v1/prerequisite-tests", { method: "POST", headers: jsonStudent(id), body: JSON.stringify({ topic_id: topicId, prerequisite_topic_id: prerequisiteTopicId, question_count: 3 }) }));
}
export const getAssessmentHint = (assessmentId: string, questionId: string, id: string) => request<{ hint: string }>(`/api/v1/assessments/${encodeURIComponent(assessmentId)}/hint`, { method: "POST", headers: jsonStudent(id), body: JSON.stringify({ question_id: questionId }) });
export const getAssessmentSolution = (assessmentId: string, questionId: string, attempt: string, id: string) => request<{ answer: string; solution_steps: string[] }>(`/api/v1/assessments/${encodeURIComponent(assessmentId)}/solution`, { method: "POST", headers: jsonStudent(id), body: JSON.stringify({ question_id: questionId, attempt }) });
export const adminBankRequest = <T,>(path: string, token: string, method = "GET", payload?: unknown) => request<T>(`/api/v1/admin/question-bank${path}`, { method, headers: { "X-Admin-Token": token, ...(payload ? { "Content-Type": "application/json" } : {}) }, ...(payload ? { body: JSON.stringify(payload) } : {}) });
