import type {
  AuthenticatedStudent,
  ChatRequest,
  ChatResponse,
  Course,
  CourseStats,
  CreatePracticeSetRequest,
  CurriculumResponse,
  KnowledgeDocument,
  KnowledgeStats,
  KnowledgeUploadResult,
  LearningDashboard,
  LearningHistoryResponse,
  PracticeAnswer,
  PracticeOption,
  PracticeSubmitResult,
  PublicPracticeQuestion,
  PublicPracticeSet,
  RemediationRequest,
  RemediationResponse,
  CurriculumTopic,
} from "@/lib/types";
import { getAiModel, getEmbeddingBaseUrl, getEmbeddingModel, getLlmBaseUrl, getUserEmbeddingApiKey, getUserLlmApiKey, type AiFeature } from "@/lib/llm-settings";

const configuredBaseUrl =
  process.env.NEXT_PUBLIC_API_BASE_URL ??
  process.env.NEXT_PUBLIC_API_URL ??
  "http://localhost:8000";
export const API_BASE_URL = configuredBaseUrl.replace(/\/$/, "");

export class ApiError extends Error {
  constructor(
    message: string,
    public readonly status?: number,
    public readonly code?: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

function errorCode(payload: unknown): string | undefined {
  if (!payload || typeof payload !== "object") return undefined;
  const body = payload as {
    code?: unknown;
    detail?: unknown;
    error?: { code?: unknown };
  };
  if (typeof body.error?.code === "string") return body.error.code;
  if (typeof body.code === "string") return body.code;
  if (body.detail && typeof body.detail === "object" && "code" in body.detail) {
    const code = (body.detail as { code?: unknown }).code;
    return typeof code === "string" ? code : undefined;
  }
  return undefined;
}

function errorMessage(payload: unknown, fallback: string): string {
  if (!payload || typeof payload !== "object") return fallback;
  const body = payload as {
    detail?: unknown;
    message?: unknown;
    error?: { message?: unknown };
  };
  if (typeof body.error?.message === "string") return body.error.message;
  if (typeof body.detail === "string") return body.detail;
  if (body.detail && typeof body.detail === "object" && "message" in body.detail) {
    const message = (body.detail as { message?: unknown }).message;
    if (typeof message === "string") return message;
  }
  if (typeof body.message === "string") return body.message;
  if (Array.isArray(body.detail)) {
    return body.detail
      .map((item) => {
        if (item && typeof item === "object" && "msg" in item) {
          return String((item as { msg: unknown }).msg);
        }
        return String(item);
      })
      .join("；");
  }
  return fallback;
}

export async function request<T>(
  path: string,
  init: RequestInit = {},
  aiFeature?: AiFeature,
): Promise<T> {
  const userLlmApiKey = getUserLlmApiKey();
  const userEmbeddingApiKey = getUserEmbeddingApiKey();
  const llmBaseUrl = getLlmBaseUrl();
  const embeddingBaseUrl = getEmbeddingBaseUrl();
  const llmModel = aiFeature ? getAiModel(aiFeature) : null;
  const embeddingModel = getEmbeddingModel();
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      headers: {
        Accept: "application/json",
        ...(userLlmApiKey ? { "X-LLM-API-Key": userLlmApiKey } : {}),
        ...(userEmbeddingApiKey ? { "X-Embedding-API-Key": userEmbeddingApiKey } : {}),
        ...(llmBaseUrl ? { "X-LLM-Base-URL": llmBaseUrl } : {}),
        ...(embeddingBaseUrl ? { "X-Embedding-Base-URL": embeddingBaseUrl } : {}),
        ...(llmModel ? { "X-LLM-Model": llmModel } : {}),
        ...(embeddingModel ? { "X-Embedding-Model": embeddingModel } : {}),
        ...init.headers,
      },
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") throw error;
    throw new ApiError("无法连接课程服务，请确认后端已经启动。", 0);
  }

  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    throw new ApiError(
      errorMessage(payload, `请求失败（HTTP ${response.status}）`),
      response.status,
      errorCode(payload),
    );
  }
  return payload as T;
}

export function sendChat(
  payload: ChatRequest,
  signal?: AbortSignal,
  studentId?: string,
): Promise<ChatResponse> {
  return request<ChatResponse>("/api/v1/chat", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(studentId ? { Authorization: `Bearer ${studentId}` } : {}),
    },
    body: JSON.stringify(payload),
    signal,
  }, payload.learning_mode === "free_qa" ? "free_qa" : "teaching");
}

function studentHeaders(studentId: string): HeadersInit {
  return { Authorization: `Bearer ${studentId}` };
}

function jsonStudentHeaders(studentId: string): HeadersInit {
  return { "Content-Type": "application/json", ...studentHeaders(studentId) };
}

export function isStudentNotFound(error: unknown): boolean {
  return error instanceof ApiError && error.code?.toUpperCase() === "STUDENT_NOT_FOUND";
}

export function registerStudent(payload: { username: string; password: string; display_name: string }): Promise<AuthenticatedStudent> {
  return request<AuthenticatedStudent>("/api/v1/students/register", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export function loginStudent(payload: { username: string; password: string }): Promise<AuthenticatedStudent> {
  return request<AuthenticatedStudent>("/api/v1/students/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export function logoutStudent(accessToken: string): Promise<void> {
  return request<void>("/api/v1/students/logout", {
    method: "POST",
    headers: studentHeaders(accessToken),
  });
}

export function getCurriculum(
  course: Course,
  studentId?: string,
): Promise<CurriculumResponse> {
  return request<CurriculumResponse>(
    `/api/v1/curriculum/topics?course=${encodeURIComponent(course)}`,
    studentId ? { headers: studentHeaders(studentId) } : {},
  );
}

function normalizeTopic(payload: unknown): CurriculumTopic {
  const body = asRecord(payload);
  const rawDimensions = body.dimensions;
  const dimensions = Array.isArray(rawDimensions)
    ? rawDimensions.map((item) => {
        const dimension = asRecord(item);
        return {
          dimension: String(dimension.dimension ?? ""),
          label: typeof dimension.label === "string" ? dimension.label : undefined,
          mastery: typeof dimension.mastery === "number" ? dimension.mastery : null,
          confidence: typeof dimension.confidence === "number" ? dimension.confidence : null,
          evidence_count: typeof dimension.evidence_count === "number" ? dimension.evidence_count : 0,
        };
      })
    : Object.entries(asRecord(rawDimensions)).map(([name, value]) => {
        const dimension = asRecord(value);
        return {
          dimension: name,
          mastery: typeof dimension.mastery === "number" ? dimension.mastery : null,
          confidence: typeof dimension.confidence === "number" ? dimension.confidence : null,
          evidence_count: typeof dimension.evidence_count === "number" ? dimension.evidence_count : 0,
        };
      });
  return {
    id: String(body.id ?? ""),
    course: String(body.course ?? "machine_learning") as Course,
    slug: typeof body.slug === "string" ? body.slug : undefined,
    title: String(body.title ?? ""),
    stage: Number(body.stage ?? 0),
    stage_title: String(body.stage_title ?? ""),
    order_index: typeof body.order_index === "number" ? body.order_index : undefined,
    description: typeof body.description === "string" ? body.description : undefined,
    learning_objectives: strings(body.learning_objectives),
    aliases: strings(body.aliases),
    retrieval_keywords: strings(body.retrieval_keywords),
    difficulty: String(body.difficulty ?? "medium"),
    required_dimensions: strings(body.required_dimensions),
    prerequisites: Array.isArray(body.prerequisites)
      ? body.prerequisites.map((item) => {
          const prerequisite = asRecord(item);
          return {
            id: typeof prerequisite.id === "string" ? prerequisite.id : undefined,
            topic_id: typeof prerequisite.topic_id === "string" ? prerequisite.topic_id : undefined,
            title: String(prerequisite.title ?? prerequisite.id ?? "前置知识点"),
            required_mastery: typeof prerequisite.required_mastery === "number" ? prerequisite.required_mastery : undefined,
            status: typeof prerequisite.status === "string" ? prerequisite.status as CurriculumTopic["status"] : undefined,
            mastery: typeof prerequisite.mastery === "number" ? prerequisite.mastery : null,
          };
        })
      : [],
    status: (typeof body.status === "string" ? body.status : "not_started") as CurriculumTopic["status"],
    mastery: typeof body.mastery === "number" ? body.mastery : null,
    mastery_display: typeof body.mastery_display === "string" ? body.mastery_display : null,
    confidence: typeof body.confidence === "number" ? body.confidence : null,
    evidence_count: typeof body.evidence_count === "number" ? body.evidence_count : undefined,
    attempt_count: typeof body.attempt_count === "number" ? body.attempt_count : undefined,
    latest_score: typeof body.latest_score === "number" ? body.latest_score : null,
    locked_reason: typeof body.locked_reason === "string" ? body.locked_reason : null,
    dimensions,
    active: typeof body.active === "boolean" ? body.active : undefined,
    created_at: typeof body.created_at === "string" ? body.created_at : undefined,
    updated_at: typeof body.updated_at === "string" ? body.updated_at : undefined,
  };
}

export async function getTopic(
  topicId: string,
  studentId?: string,
): Promise<CurriculumTopic> {
  const response = await request<unknown>(
    `/api/v1/curriculum/topics/${encodeURIComponent(topicId)}`,
    studentId ? { headers: studentHeaders(studentId) } : {},
  );
  return normalizeTopic(response);
}

export async function getLearningDashboard(
  course: Course,
  studentId: string,
): Promise<LearningDashboard> {
  const response = await request<unknown>(
    `/api/v1/learning/dashboard?course=${encodeURIComponent(course)}`,
    { headers: studentHeaders(studentId) },
  );
  const body = asRecord(response);
  const rawRecent = Array.isArray(body.recent_practice)
    ? body.recent_practice
    : Array.isArray(body.recent_attempts)
      ? body.recent_attempts
      : [];
  return {
    course,
    total_topics: Number(body.total_topics ?? 0),
    mastered_count: Number(body.mastered_count ?? 0),
    learning_count: Number(body.learning_count ?? 0),
    not_started_count: Number(body.not_started_count ?? 0),
    locked_count: Number(body.locked_count ?? 0),
    current_topic: body.current_topic && typeof body.current_topic === "object"
      ? body.current_topic as LearningDashboard["current_topic"]
      : null,
    recommended_topic_id: typeof body.recommended_topic_id === "string" ? body.recommended_topic_id : null,
    next_action: typeof body.next_action === "string" ? body.next_action : "start_learning",
    reasons: strings(body.reasons),
    recent_practice: rawRecent.map((item) => {
      const attempt = asRecord(item);
      return {
        id: typeof attempt.id === "string" ? attempt.id : undefined,
        attempt_id: typeof attempt.attempt_id === "string" ? attempt.attempt_id : undefined,
        topic_id: String(attempt.topic_id ?? ""),
        topic_title: String(attempt.topic_title ?? attempt.title ?? attempt.topic_id ?? "练习记录"),
        mode: attempt.mode === "checkpoint" ? "checkpoint" : "practice",
        score: typeof attempt.score === "number"
          ? attempt.score
          : typeof attempt.total_score === "number"
            ? attempt.total_score
            : null,
        status: typeof attempt.status === "string" ? attempt.status : undefined,
        created_at: String(attempt.created_at ?? attempt.submitted_at ?? ""),
      };
    }),
    stages: Array.isArray(body.stages) ? body.stages as LearningDashboard["stages"] : [],
  };
}

export async function startTopic(
  topicId: string,
  studentId: string,
): Promise<CurriculumTopic> {
  const response = await request<unknown>(
    `/api/v1/learning/topics/${encodeURIComponent(topicId)}/start`,
    { method: "POST", headers: studentHeaders(studentId) },
  );
  const body = asRecord(response);
  return normalizeTopic(body.topic && typeof body.topic === "object" ? body.topic : body);
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" ? (value as Record<string, unknown>) : {};
}

function strings(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === "string")
    : [];
}

function sanitizeOptions(value: unknown): PracticeOption[] | undefined {
  if (!Array.isArray(value)) return undefined;
  const options = value.flatMap((item, index) => {
    if (typeof item === "string") return [{ value: item, label: item }];
    const option = asRecord(item);
    const labelCandidate = option.label ?? option.text ?? option.content ?? option.value;
    const valueCandidate = option.value ?? option.option_id ?? option.key ?? option.id ?? String.fromCharCode(65 + index);
    if (typeof labelCandidate !== "string") return [];
    return [{ value: String(valueCandidate), label: labelCandidate }];
  });
  return options.length > 0 ? options : undefined;
}

function sanitizeQuestion(value: unknown, index: number): PublicPracticeQuestion {
  const question = asRecord(value);
  const dimensionNames = Array.isArray(question.dimension_names)
    ? strings(question.dimension_names)
    : Array.isArray(question.dimensions)
      ? strings(question.dimensions)
      : Object.keys(asRecord(question.dimensions));
  return {
    question_id: String(question.question_id ?? question.id ?? ""),
    position: Number(question.position ?? index + 1),
    question_type: String(question.question_type ?? "concept_explanation"),
    difficulty: String(question.difficulty ?? "medium"),
    question: String(question.question ?? ""),
    options: sanitizeOptions(question.options),
    dimension_names: dimensionNames,
    max_score: typeof question.max_score === "number" ? question.max_score : undefined,
    required: typeof question.required === "boolean" ? question.required : true,
  };
}

/** Explicit projection prevents hidden grading fields from entering React state. */
export function sanitizePublicPracticeSet(payload: unknown): PublicPracticeSet {
  const body = asRecord(payload);
  if (String(body.status ?? "").toLowerCase() === "knowledge_insufficient") {
    throw new ApiError(
      typeof body.message === "string"
        ? body.message
        : "当前知识库中没有足够资料生成该知识点的可靠练习，请补充课程材料。",
      422,
      "KNOWLEDGE_INSUFFICIENT",
    );
  }
  const questions = Array.isArray(body.questions)
    ? body.questions.map(sanitizeQuestion)
    : [];
  const sourceSummary = Array.isArray(body.source_summary)
    ? body.source_summary.map((item) => {
        const source = asRecord(item);
        return {
          source_id: String(source.source_id ?? ""),
          source_file: String(source.source_file ?? ""),
          title: typeof source.title === "string" ? source.title : null,
        };
      })
    : [];
  const result: PublicPracticeSet = {
    practice_set_id: String(body.practice_set_id ?? ""),
    topic_id: String(body.topic_id ?? ""),
    mode: body.mode === "checkpoint" ? "checkpoint" : "practice",
    questions,
    source_summary: sourceSummary,
    status: typeof body.status === "string" ? body.status : undefined,
    created_at: typeof body.created_at === "string" ? body.created_at : undefined,
  };
  if (!result.practice_set_id || !result.topic_id || result.questions.some((item) => !item.question_id)) {
    throw new ApiError("服务端返回了无效的练习内容。", 502, "INVALID_PRACTICE_RESPONSE");
  }
  return result;
}

export async function createPracticeSet(
  payload: CreatePracticeSetRequest,
  studentId: string,
): Promise<PublicPracticeSet> {
  const response = await request<unknown>("/api/v1/practice/sets", {
    method: "POST",
    headers: jsonStudentHeaders(studentId),
    body: JSON.stringify(payload),
  }, "practice");
  return sanitizePublicPracticeSet(response);
}

export async function getPracticeSet(
  practiceSetId: string,
  studentId: string,
): Promise<PublicPracticeSet> {
  const response = await request<unknown>(
    `/api/v1/practice/sets/${encodeURIComponent(practiceSetId)}`,
    { headers: studentHeaders(studentId) },
  );
  return sanitizePublicPracticeSet(response);
}

export function submitPracticeSet(
  practiceSetId: string,
  payload: { client_submission_id: string; answers: PracticeAnswer[] },
  studentId: string,
): Promise<PracticeSubmitResult> {
  return request<PracticeSubmitResult>(
    `/api/v1/practice/sets/${encodeURIComponent(practiceSetId)}/submit`,
    {
      method: "POST",
      headers: jsonStudentHeaders(studentId),
      body: JSON.stringify(payload),
    },
  );
}

export function requestRemediation(
  payload: RemediationRequest,
  studentId: string,
): Promise<RemediationResponse> {
  return request<RemediationResponse>("/api/v1/learning/remediation", {
    method: "POST",
    headers: jsonStudentHeaders(studentId),
    body: JSON.stringify(payload),
  }, "practice");
}

export function getLearningHistory(
  course: Course,
  studentId: string,
  limit = 20,
): Promise<LearningHistoryResponse> {
  return request<LearningHistoryResponse>(
    `/api/v1/learning/history?course=${encodeURIComponent(course)}&limit=${limit}`,
    { headers: studentHeaders(studentId) },
  );
}

function adminHeaders(token?: string): HeadersInit {
  return token?.trim() ? { "X-Admin-Token": token.trim() } : {};
}

export async function listDocuments(
  course: Course,
  token?: string,
): Promise<KnowledgeDocument[]> {
  const payload = await request<KnowledgeDocument[] | { documents: KnowledgeDocument[] }>(
    `/api/v1/knowledge/documents?course=${encodeURIComponent(course)}`,
    { headers: adminHeaders(token) },
  );
  return Array.isArray(payload) ? payload : payload.documents;
}

export async function getKnowledgeStats(token?: string): Promise<KnowledgeStats> {
  const payload = await request<
    | KnowledgeStats
    | { courses: KnowledgeStats }
    | { stats: CourseStats[] }
    | CourseStats[]
  >("/api/v1/knowledge/stats", { headers: adminHeaders(token) });

  if (Array.isArray(payload)) {
    return Object.fromEntries(payload.map((item) => [item.course, item])) as KnowledgeStats;
  }
  if ("stats" in payload) {
    return Object.fromEntries(payload.stats.map((item) => [item.course, item])) as KnowledgeStats;
  }
  if ("courses" in payload) return payload.courses;
  return payload;
}

export function uploadDocument(
  course: Course,
  file: File,
  token: string,
  onProgress: (percent: number) => void,
  topicIds: string[] = [],
): Promise<KnowledgeUploadResult> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_BASE_URL}/api/v1/knowledge/upload`);
    xhr.setRequestHeader("Accept", "application/json");
    xhr.setRequestHeader("X-Admin-Token", token.trim());

    xhr.upload.addEventListener("progress", (event) => {
      if (event.lengthComputable) {
        onProgress(Math.round((event.loaded / event.total) * 100));
      }
    });
    xhr.addEventListener("error", () => {
      reject(new ApiError("上传失败，请检查网络与后端服务。", 0));
    });
    xhr.addEventListener("load", () => {
      let payload: unknown = null;
      try {
        payload = JSON.parse(xhr.responseText) as unknown;
      } catch {
        // A non-JSON error is normalized below.
      }
      if (xhr.status >= 200 && xhr.status < 300) {
        onProgress(100);
        if (!payload || typeof payload !== "object") {
          reject(new ApiError("服务端返回了无效的上传结果。", xhr.status));
          return;
        }
        const body = payload as Partial<KnowledgeUploadResult>;
        if (!body.document || typeof body.duplicate !== "boolean") {
          reject(new ApiError("服务端返回了无效的上传结果。", xhr.status));
          return;
        }
        resolve({
          document: body.document,
          duplicate: body.duplicate,
          message: typeof body.message === "string" ? body.message : "文件已导入知识库",
        });
        return;
      }
      reject(
        new ApiError(
          errorMessage(payload, `上传失败（HTTP ${xhr.status}）`),
          xhr.status,
        ),
      );
    });

    const formData = new FormData();
    formData.append("course", course);
    formData.append("file", file);
    topicIds.forEach((topicId) => formData.append("topic_ids", topicId));
    xhr.send(formData);
  });
}

export function deleteDocument(documentId: string, token: string): Promise<void> {
  return request<void>(
    `/api/v1/knowledge/documents/${encodeURIComponent(documentId)}`,
    {
      method: "DELETE",
      headers: adminHeaders(token),
    },
  );
}

export function tagDocumentTopics(documentId: string, topicIds: string[], token: string): Promise<unknown> {
  return request(`/api/v1/knowledge/documents/${encodeURIComponent(documentId)}/topics`, {
    method: "PATCH", headers: { ...adminHeaders(token), "Content-Type": "application/json" },
    body: JSON.stringify({ topic_ids: topicIds }),
  });
}
