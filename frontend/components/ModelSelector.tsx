"use client";

import { getAiModel, setAiModel, type AiFeature } from "@/lib/llm-settings";

export default function ModelSelector({ feature, label }: { feature: AiFeature; label: string }) {
  return (
    <label className="flex items-center gap-2 text-xs font-medium text-slate-500">
      <span>{label}</span>
      <input
        type="text"
        defaultValue={getAiModel(feature) ?? ""}
        onChange={(event) => setAiModel(feature, event.target.value)}
        placeholder="填写模型名称"
        autoComplete="off"
        spellCheck={false}
        className="min-w-0 border border-slate-200 bg-white px-2 py-1 text-xs font-medium text-slate-700 outline-none focus:border-indigo-500 focus:ring-2 focus:ring-indigo-100"
      />
    </label>
  );
}