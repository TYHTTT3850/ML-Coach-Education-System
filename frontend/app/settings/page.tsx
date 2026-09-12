"use client";

import { useEffect, useRef, useState } from "react";
import LearningShell from "@/components/LearningShell";
import { useTheme } from "@/components/ThemeProvider";
import {
  EMBEDDING_PROVIDERS,
  EMBEDDING_PROVIDER_STORAGE_KEY,
  CUSTOM_EMBEDDING_API_BASE_URL_STORAGE_KEY,
  CUSTOM_LLM_API_BASE_URL_STORAGE_KEY,
  EMBEDDING_API_KEY_STORAGE_KEY,
  LLM_PROVIDERS,
  LLM_PROVIDER_STORAGE_KEY,
  LLM_API_KEY_STORAGE_KEY,
  getEmbeddingProvider,
  getLlmProvider,
} from "@/lib/llm-settings";

export default function SettingsPage() {
  const { theme, setTheme } = useTheme();
  const llmProviderInputRef = useRef<HTMLSelectElement>(null);
  const embeddingProviderInputRef = useRef<HTMLSelectElement>(null);
  const apiKeyInputRef = useRef<HTMLInputElement>(null);
  const embeddingApiKeyInputRef = useRef<HTMLInputElement>(null);
  const customLlmBaseUrlInputRef = useRef<HTMLInputElement>(null);
  const customEmbeddingBaseUrlInputRef = useRef<HTMLInputElement>(null);
  const [saveStatus, setSaveStatus] = useState<"saved" | "cleared" | null>(null);

  useEffect(() => {
    if (llmProviderInputRef.current) {
      llmProviderInputRef.current.value = getLlmProvider();
    }
    if (embeddingProviderInputRef.current) {
      embeddingProviderInputRef.current.value = getEmbeddingProvider();
    }
    if (apiKeyInputRef.current) {
      apiKeyInputRef.current.value = window.localStorage.getItem(LLM_API_KEY_STORAGE_KEY) ?? "";
    }
    if (embeddingApiKeyInputRef.current) {
      embeddingApiKeyInputRef.current.value = window.localStorage.getItem(EMBEDDING_API_KEY_STORAGE_KEY) ?? "";
    }
    if (customLlmBaseUrlInputRef.current) {
      customLlmBaseUrlInputRef.current.value = window.localStorage.getItem(CUSTOM_LLM_API_BASE_URL_STORAGE_KEY) ?? "";
    }
    if (customEmbeddingBaseUrlInputRef.current) {
      customEmbeddingBaseUrlInputRef.current.value = window.localStorage.getItem(CUSTOM_EMBEDDING_API_BASE_URL_STORAGE_KEY) ?? "";
    }
  }, []);

  function saveApiKeys() {
    const llmProvider = llmProviderInputRef.current?.value ?? "deepseek";
    window.localStorage.setItem(LLM_PROVIDER_STORAGE_KEY, llmProvider);
    const embeddingProvider = embeddingProviderInputRef.current?.value ?? "siliconflow";
    window.localStorage.setItem(EMBEDDING_PROVIDER_STORAGE_KEY, embeddingProvider);
    const normalizedKey = apiKeyInputRef.current?.value.trim() ?? "";
    if (normalizedKey) {
      window.localStorage.setItem(LLM_API_KEY_STORAGE_KEY, normalizedKey);
    } else {
      window.localStorage.removeItem(LLM_API_KEY_STORAGE_KEY);
    }
    if (apiKeyInputRef.current) apiKeyInputRef.current.value = normalizedKey;
    const normalizedEmbeddingKey = embeddingApiKeyInputRef.current?.value.trim() ?? "";
    if (normalizedEmbeddingKey) {
      window.localStorage.setItem(EMBEDDING_API_KEY_STORAGE_KEY, normalizedEmbeddingKey);
    } else {
      window.localStorage.removeItem(EMBEDDING_API_KEY_STORAGE_KEY);
    }
    if (embeddingApiKeyInputRef.current) embeddingApiKeyInputRef.current.value = normalizedEmbeddingKey;
    const customLlmBaseUrl = customLlmBaseUrlInputRef.current?.value.trim() ?? "";
    if (customLlmBaseUrl) {
      window.localStorage.setItem(CUSTOM_LLM_API_BASE_URL_STORAGE_KEY, customLlmBaseUrl);
    } else {
      window.localStorage.removeItem(CUSTOM_LLM_API_BASE_URL_STORAGE_KEY);
    }
    const customEmbeddingBaseUrl = customEmbeddingBaseUrlInputRef.current?.value.trim() ?? "";
    if (customEmbeddingBaseUrl) {
      window.localStorage.setItem(CUSTOM_EMBEDDING_API_BASE_URL_STORAGE_KEY, customEmbeddingBaseUrl);
    } else {
      window.localStorage.removeItem(CUSTOM_EMBEDDING_API_BASE_URL_STORAGE_KEY);
    }
    setSaveStatus(normalizedKey || normalizedEmbeddingKey ? "saved" : "cleared");
  }

  return (
    <LearningShell>
      <section className="mx-auto max-w-3xl px-4 py-10 sm:px-6 lg:px-8">
        <div className="border-b border-slate-200 pb-6">
          <h1 className="text-2xl font-bold text-slate-900">设置</h1>
          <p className="mt-2 text-sm text-slate-500">调整 Course Coach 的显示偏好。</p>
        </div>

        <div className="mt-8 border border-slate-200 bg-white p-5 shadow-soft sm:p-6">
          <h2 className="text-base font-bold text-slate-900">外观</h2>
          <div className="mt-5 flex items-center justify-between gap-6">
            <div>
              <p className="text-sm font-semibold text-slate-800">界面主题</p>
              <p className="mt-1 text-sm text-slate-500">选择浅色或深色显示模式。</p>
            </div>
            <div className="inline-flex shrink-0 rounded-lg border border-slate-200 bg-slate-50 p-1" role="group" aria-label="界面主题">
              {(["light", "dark"] as const).map((option) => (
                <button
                  key={option}
                  type="button"
                  onClick={() => setTheme(option)}
                  aria-pressed={theme === option}
                  className={`rounded-md px-3 py-2 text-sm font-semibold transition ${theme === option ? "bg-white text-indigo-700 shadow-sm" : "text-slate-500 hover:text-slate-800"}`}
                >
                  {option === "light" ? "Light" : "Dark"}
                </button>
              ))}
            </div>
          </div>
        </div>

        <div className="mt-6 border border-slate-200 bg-white p-5 shadow-soft sm:p-6">
          <h2 className="text-base font-bold text-slate-900">模型服务</h2>
          <div className="mt-5 max-w-xl">
            <label htmlFor="llm-provider" className="text-sm font-semibold text-slate-800">
              对话模型厂商
            </label>
            <select
              id="llm-provider"
              ref={llmProviderInputRef}
              defaultValue="deepseek"
              onChange={() => setSaveStatus(null)}
              className="mt-2 w-full border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-indigo-500 focus:ring-2 focus:ring-indigo-100"
            >
              {Object.entries(LLM_PROVIDERS).map(([id, provider]) => (
                <option key={id} value={id}>{provider.label}</option>
              ))}
            </select>
            <label htmlFor="custom-llm-api-base-url" className="mt-4 block text-sm font-semibold text-slate-800">
              自定义对话 API Base URL
            </label>
            <input
              id="custom-llm-api-base-url"
              type="url"
              ref={customLlmBaseUrlInputRef}
              onChange={() => setSaveStatus(null)}
              autoComplete="off"
              spellCheck={false}
              placeholder="仅自定义兼容服务时填写，例如 https://api.example.com/v1"
              className="mt-2 w-full border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 outline-none transition placeholder:text-slate-400 focus:border-indigo-500 focus:ring-2 focus:ring-indigo-100"
            />
            <label htmlFor="embedding-provider" className="mt-4 block text-sm font-semibold text-slate-800">
              Embedding 模型厂商
            </label>
            <select
              id="embedding-provider"
              ref={embeddingProviderInputRef}
              defaultValue="siliconflow"
              onChange={() => setSaveStatus(null)}
              className="mt-2 w-full border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-indigo-500 focus:ring-2 focus:ring-indigo-100"
            >
              {Object.entries(EMBEDDING_PROVIDERS).map(([id, provider]) => (
                <option key={id} value={id}>{provider.label}</option>
              ))}
            </select>
            <label htmlFor="custom-embedding-api-base-url" className="mt-4 block text-sm font-semibold text-slate-800">
              自定义 Embedding API Base URL
            </label>
            <input
              id="custom-embedding-api-base-url"
              type="url"
              ref={customEmbeddingBaseUrlInputRef}
              onChange={() => setSaveStatus(null)}
              autoComplete="off"
              spellCheck={false}
              placeholder="仅自定义兼容服务时填写，例如 https://api.example.com/v1"
              className="mt-2 w-full border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 outline-none transition placeholder:text-slate-400 focus:border-indigo-500 focus:ring-2 focus:ring-indigo-100"
            />
            <label htmlFor="llm-api-key" className="text-sm font-semibold text-slate-800">
              LLM API Key
            </label>
            <p className="mt-1 text-sm text-slate-500">
              密钥仅保存在当前浏览器中，用于你的模型请求。
            </p>
            <div className="mt-3 space-y-3">
              <input
                id="llm-api-key"
                type="password"
                ref={apiKeyInputRef}
                onChange={() => setSaveStatus(null)}
                autoComplete="off"
                spellCheck={false}
                placeholder="输入你的 API Key"
                className="min-w-0 flex-1 border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 outline-none transition placeholder:text-slate-400 focus:border-indigo-500 focus:ring-2 focus:ring-indigo-100"
              />
              <label htmlFor="embedding-api-key" className="block text-sm font-semibold text-slate-800">
                Embedding API Key
              </label>
              <input
                id="embedding-api-key"
                type="password"
                ref={embeddingApiKeyInputRef}
                onChange={() => setSaveStatus(null)}
                autoComplete="off"
                spellCheck={false}
                placeholder="输入 Embedding API Key"
                className="min-w-0 flex-1 border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 outline-none transition placeholder:text-slate-400 focus:border-indigo-500 focus:ring-2 focus:ring-indigo-100"
              />
            </div>
            {saveStatus && (
              <p className="mt-2 text-sm text-emerald-700" role="status">
                {saveStatus === "saved" ? "API Key 已保存。" : "API Key 已清除。"}
              </p>
            )}
            <div className="mt-5 flex justify-end">
              <button
                type="button"
                onClick={saveApiKeys}
                className="bg-indigo-600 px-4 py-2 text-sm font-semibold text-white transition hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2"
              >
                保存
              </button>
            </div>
          </div>
        </div>
      </section>
    </LearningShell>
  );
}