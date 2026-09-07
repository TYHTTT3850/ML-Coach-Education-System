"use client";
import { useState } from "react";
import { tagDocumentTopics } from "@/lib/api";
export default function DocumentTopicTagger({ documentId, adminToken }: { documentId: string; adminToken: string }) {
  const [tags, setTags] = useState(""); const [message, setMessage] = useState(""); const [busy, setBusy] = useState(false);
  async function save() {
    const ids = tags.split(/[,，\s]+/).filter(Boolean);
    if (!ids.length) { setMessage("请先填写至少一个知识点 ID。"); return; }
    setBusy(true); setMessage("");
    try { await tagDocumentTopics(documentId, ids, adminToken); setMessage("知识点标签已保存，原有向量未重新计算。"); }
    catch (cause) { setMessage(cause instanceof Error ? cause.message : "补标失败，请检查知识点归属与 Token 后重试。"); }
    finally { setBusy(false); }
  }
  return <details className="mt-2 text-xs"><summary className="cursor-pointer text-indigo-700">补标知识点</summary><div className="mt-2 min-w-52 rounded-xl border border-slate-200 bg-white p-3"><label>Canonical 知识点 ID<input value={tags} onChange={(event) => setTags(event.target.value)} placeholder="多个 ID 用逗号分隔" className="mt-2 w-full rounded-lg border border-slate-200 p-2" /></label><p className="mt-2 leading-5 text-slate-400">为该文件所有资料片段保存指定标签，保持原有向量与文件。</p><button type="button" disabled={busy || !adminToken} onClick={() => void save()} className="mt-2 rounded-lg bg-indigo-50 px-3 py-2 font-semibold text-indigo-700 disabled:opacity-50">{busy ? "保存中…" : "保存标签"}</button>{message && <p role="status" className="mt-2 leading-5">{message}</p>}</div></details>;
}
