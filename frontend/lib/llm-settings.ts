export const LLM_API_KEY_STORAGE_KEY = "course-coach:llm-api-key";
export const EMBEDDING_API_KEY_STORAGE_KEY = "course-coach:embedding-api-key";
export const LLM_PROVIDER_STORAGE_KEY = "course-coach:llm-provider";
export const EMBEDDING_PROVIDER_STORAGE_KEY = "course-coach:embedding-provider";
export const CUSTOM_LLM_API_BASE_URL_STORAGE_KEY = "course-coach:custom-llm-api-base-url";
export const CUSTOM_EMBEDDING_API_BASE_URL_STORAGE_KEY = "course-coach:custom-embedding-api-base-url";
export const EMBEDDING_MODEL_STORAGE_KEY = "course-coach:embedding-model";

export type AiFeature = "free_qa" | "teaching" | "practice";
export type LlmProviderId = "deepseek" | "qwen" | "siliconflow" | "moonshot" | "zhipu" | "baichuan" | "hunyuan" | "volcengine" | "qianfan" | "minimax" | "modelscope" | "custom";
export type EmbeddingProviderId = "qwen" | "siliconflow" | "zhipu" | "volcengine" | "modelscope" | "custom";

interface AiProvider {
  label: string;
  baseUrl: string;
}

export const LLM_PROVIDERS: Record<LlmProviderId, AiProvider> = {
  deepseek: { label: "DeepSeek", baseUrl: "https://api.deepseek.com/v1" },
  qwen: { label: "通义千问", baseUrl: "https://dashscope.aliyuncs.com/compatible-mode/v1" },
  siliconflow: { label: "SiliconFlow", baseUrl: "https://api.siliconflow.cn/v1" },
  moonshot: { label: "月之暗面 Kimi", baseUrl: "https://api.moonshot.cn/v1" },
  zhipu: { label: "智谱 AI", baseUrl: "https://open.bigmodel.cn/api/paas/v4" },
  baichuan: { label: "百川智能", baseUrl: "https://api.baichuan-ai.com/v1" },
  hunyuan: { label: "腾讯混元", baseUrl: "https://api.hunyuan.cloud.tencent.com/v1" },
  volcengine: { label: "火山方舟", baseUrl: "https://ark.cn-beijing.volces.com/api/v3" },
  qianfan: { label: "百度千帆", baseUrl: "https://qianfan.baidubce.com/v2" },
  minimax: { label: "MiniMax", baseUrl: "https://api.minimaxi.com/v1" },
  modelscope: { label: "魔搭 ModelScope", baseUrl: "https://api-inference.modelscope.cn/v1" },
  custom: { label: "自定义兼容服务", baseUrl: "" },
};

export const EMBEDDING_PROVIDERS: Record<EmbeddingProviderId, AiProvider> = {
  qwen: LLM_PROVIDERS.qwen,
  siliconflow: LLM_PROVIDERS.siliconflow,
  zhipu: LLM_PROVIDERS.zhipu,
  volcengine: LLM_PROVIDERS.volcengine,
  modelscope: LLM_PROVIDERS.modelscope,
  custom: LLM_PROVIDERS.custom,
};

const FEATURE_MODEL_STORAGE_KEYS: Record<AiFeature, string> = {
  free_qa: "course-coach:model-free-qa",
  teaching: "course-coach:model-teaching",
  practice: "course-coach:model-practice",
};

function getUserApiKey(storageKey: string): string | null {
  if (typeof window === "undefined") return null;
  const apiKey = window.localStorage.getItem(storageKey)?.trim();
  return apiKey || null;
}

export function getUserLlmApiKey(): string | null {
  return getUserApiKey(LLM_API_KEY_STORAGE_KEY);
}

export function getUserEmbeddingApiKey(): string | null {
  return getUserApiKey(EMBEDDING_API_KEY_STORAGE_KEY);
}

export function getLlmProvider(): LlmProviderId {
  if (typeof window === "undefined") return "deepseek";
  const provider = window.localStorage.getItem(LLM_PROVIDER_STORAGE_KEY);
  return provider && provider in LLM_PROVIDERS ? provider as LlmProviderId : "deepseek";
}

export function getLlmBaseUrl(): string | null {
  const provider = getLlmProvider();
  const configuredUrl = provider === "custom"
    ? window.localStorage.getItem(CUSTOM_LLM_API_BASE_URL_STORAGE_KEY)
    : LLM_PROVIDERS[provider].baseUrl;
  return configuredUrl?.trim() || null;
}

export function getEmbeddingProvider(): EmbeddingProviderId {
  if (typeof window === "undefined") return "siliconflow";
  const provider = window.localStorage.getItem(EMBEDDING_PROVIDER_STORAGE_KEY);
  return provider && provider in EMBEDDING_PROVIDERS ? provider as EmbeddingProviderId : "siliconflow";
}

export function getEmbeddingBaseUrl(): string | null {
  const provider = getEmbeddingProvider();
  const configuredUrl = provider === "custom"
    ? window.localStorage.getItem(CUSTOM_EMBEDDING_API_BASE_URL_STORAGE_KEY)
    : EMBEDDING_PROVIDERS[provider].baseUrl;
  return configuredUrl?.trim() || null;
}

export function getAiModel(feature: AiFeature): string | null {
  if (typeof window === "undefined") return null;
  return window.localStorage.getItem(FEATURE_MODEL_STORAGE_KEYS[feature])?.trim() || null;
}

export function setAiModel(feature: AiFeature, model: string): void {
  window.localStorage.setItem(FEATURE_MODEL_STORAGE_KEYS[feature], model);
}

export function getEmbeddingModel(): string | null {
  if (typeof window === "undefined") return null;
  return window.localStorage.getItem(EMBEDDING_MODEL_STORAGE_KEY)?.trim() || null;
}