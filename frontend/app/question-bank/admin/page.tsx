"use client";
import { useState } from "react";
import Link from "next/link";
import MarkdownRenderer from "@/components/MarkdownRenderer";
import ContentCoverageTable from "@/components/ContentCoverageTable";
import { adminBankRequest, getContentCoverage } from "@/lib/api-content";
import { COURSES, COURSE_META } from "@/lib/types";
import type { ContentCoverage, PublicBankItem } from "@/lib/types-content";

type AdminItem = PublicBankItem & { answer_key: unknown; rubric: unknown; solution_steps: unknown; source_refs: unknown };
export default function AdminQuestionBankPage() {
  const [token, setToken] = useState("");
  const [course, setCourse] = useState(""); const [topic, setTopic] = useState(""); const [type, setType] = useState(""); const [difficulty, setDifficulty] = useState(""); const [publication, setPublication] = useState("");
  const [items, setItems] = useState<AdminItem[]>([]); const [coverage, setCoverage] = useState<ContentCoverage[]>([]);
  const [importText, setImportText] = useState(""); const [busy, setBusy] = useState(false); const [error, setError] = useState(""); const [message, setMessage] = useState("");
  const [reviewed, setReviewed] = useState<Record<string, boolean>>({});
  const [reviewNotes, setReviewNotes] = useState<Record<string, string>>({});
  async function load() {
    if (!token.trim()) { setError("请在此页面填写管理员 Token，或使用 README 中的本地 CLI。"); return; }
    setBusy(true); setError("");
    try {
      const filters = new URLSearchParams(Object.entries({ course, topic_id: topic, question_type: type, difficulty, publication_status: publication }).filter(([, value]) => value));
      const value = await adminBankRequest<{ items: AdminItem[] } | AdminItem[]>(`?${filters}`, token);
      setItems(Array.isArray(value) ? value : value.items);
      setCoverage((await getContentCoverage()).topics);
    } catch (cause) { setItems([]); setError(cause instanceof Error ? cause.message : "题库管理暂不可用，请检查 Token 后重试。"); }
    finally { setBusy(false); }
  }
  async function action(item: AdminItem, operation: string) {
    setBusy(true); setError(""); setMessage("");
    try { const result = await adminBankRequest<unknown>(`/${encodeURIComponent(item.id)}/${item.version}/${operation}`, token, "POST", operation === "publish" ? { human_reviewed: Boolean(reviewed[`${item.id}:${item.version}`]), review_note: reviewNotes[`${item.id}:${item.version}`] || "" } : undefined); setMessage(`${item.title}：${operation === "validate" ? "校验完成" : operation === "publish" ? "发布完成" : "已归档"}。${operation === "validate" ? JSON.stringify(result) : ""}`); await load(); }
    catch (cause) { setError(cause instanceof Error ? cause.message : "操作失败，请检查内容后重试。"); }
    finally { setBusy(false); }
  }
  async function importItems() {
    setBusy(true); setError(""); setMessage("");
    try {
      let parsed: unknown;
      try { parsed = JSON.parse(importText); } catch { parsed = importText.split(/\r?\n/).filter((line) => line.trim()).map((line) => JSON.parse(line)); }
      const values = Array.isArray(parsed) ? parsed : parsed && typeof parsed === "object" && "items" in parsed ? (parsed as { items: unknown }).items : [parsed];
      const result = await adminBankRequest<unknown>("/import", token, "POST", { items: values });
      setMessage(`导入完成：${JSON.stringify(result)}`); setImportText(""); await load();
    } catch (cause) { setError(cause instanceof Error ? cause.message : "JSON/JSONL 格式无效，请检查并重试。"); }
    finally { setBusy(false); }
  }
  return <div className="mx-auto max-w-7xl space-y-6 px-4 py-8 sm:px-7">
    <Link href="/question-bank" className="text-sm font-semibold text-indigo-700">← 返回学生题库</Link><h1 className="text-3xl font-bold">题库管理</h1><p className="text-sm leading-7 text-slate-500">查看答案、导入、校验、发布与归档。已发布版本修改需新版本，历史测试保留原始题目快照。</p>
    <form onSubmit={(event) => { event.preventDefault(); void load(); }} className="space-y-4 rounded-2xl border border-slate-200 bg-white p-5"><label className="block text-xs font-semibold text-slate-600">管理员 Token（仅当前页面内存，不写入浏览器持久存储）<input type="password" autoComplete="off" value={token} onChange={(event) => { setToken(event.target.value); setItems([]); }} className="mt-2 block w-full max-w-xl rounded-lg border border-slate-200 p-2 text-sm" /></label><div className="grid gap-3 sm:grid-cols-3 lg:grid-cols-5"><select aria-label="管理课程" value={course} onChange={(event) => setCourse(event.target.value)} className="rounded-lg border p-2 text-sm"><option value="">全部课程</option>{COURSES.map((value) => <option key={value} value={value}>{COURSE_META[value].label}</option>)}</select><input aria-label="管理知识点 ID" value={topic} onChange={(event) => setTopic(event.target.value)} placeholder="知识点 ID" className="rounded-lg border p-2 text-sm" /><input aria-label="管理题型" value={type} onChange={(event) => setType(event.target.value)} placeholder="题型，如 numeric" className="rounded-lg border p-2 text-sm" /><select aria-label="管理难度" value={difficulty} onChange={(event) => setDifficulty(event.target.value)} className="rounded-lg border p-2 text-sm"><option value="">全部难度</option><option value="easy">基础</option><option value="medium">中等</option><option value="hard">进阶</option></select><select aria-label="发布状态" value={publication} onChange={(event) => setPublication(event.target.value)} className="rounded-lg border p-2 text-sm"><option value="">全部状态</option><option value="draft">草稿</option><option value="published">已发布</option><option value="archived">已归档</option></select></div><button type="submit" disabled={busy || !token.trim()} className="rounded-xl bg-slate-900 px-4 py-2 text-sm font-semibold text-white disabled:opacity-50">{busy ? "处理中…" : "授权读取与筛选"}</button></form>
    {error && <p role="alert" className="rounded-xl bg-rose-50 p-4 text-sm text-rose-800">{error}</p>}{message && <p role="status" className="break-words rounded-xl bg-emerald-50 p-4 text-sm text-emerald-800">{message}</p>}
    <details className="rounded-2xl border border-slate-200 bg-white p-5"><summary className="cursor-pointer font-semibold">导入 JSON / JSONL 题库</summary><p className="mt-3 text-xs leading-6 text-slate-500">导入不会等同于审核。生成题先保留为草稿；确定性校验或人工审核满足发布规则后才可发布。</p><input type="file" accept=".json,.jsonl" aria-label="选择题库文件" onChange={(event) => { const file = event.target.files?.[0]; if (file) void file.text().then(setImportText).catch(() => setError("无法读取文件，请重新选择。")); }} className="mt-3 block text-xs" /><textarea aria-label="题库导入 JSON" rows={8} value={importText} onChange={(event) => setImportText(event.target.value)} className="mt-3 w-full rounded-xl border border-slate-200 p-3 font-mono text-xs" placeholder="粘贴题目数组、{ items: [...] } 或逐行 JSON" /><button type="button" disabled={busy || !token.trim() || !importText.trim()} onClick={() => void importItems()} className="mt-3 rounded-lg bg-indigo-600 px-4 py-2 text-xs font-semibold text-white disabled:opacity-50">校验结构并导入</button></details>
    <section className="space-y-4">{items.map((item) => <article key={`${item.id}:${item.version}`} className="rounded-2xl border border-slate-200 bg-white p-5">
      <div className="flex flex-wrap justify-between gap-3"><h2 className="font-bold">{item.title}</h2><span className="text-xs text-slate-500">v{item.version} · {item.publication_status} · {item.validation_status}</span></div>
      <div className="mt-3"><MarkdownRenderer content={item.stem} /></div>
      <details className="mt-4"><summary className="cursor-pointer text-sm font-semibold text-indigo-700">管理员答案、解析与评分规则</summary>
        <pre className="mt-3 overflow-x-auto whitespace-pre-wrap break-words rounded-xl bg-slate-50 p-4 text-xs">{JSON.stringify({ answer_key: item.answer_key, solution_steps: item.solution_steps, rubric: item.rubric, source_refs: item.source_refs }, null, 2)}</pre>
        {item.publication_status === "draft" && <div className="mt-4 rounded-xl border border-amber-200 bg-amber-50 p-4">
          <label className="flex items-start gap-2 text-xs leading-6 text-amber-900"><input type="checkbox" checked={Boolean(reviewed[`${item.id}:${item.version}`])} onChange={(event) => setReviewed((current) => ({ ...current, [`${item.id}:${item.version}`]: event.target.checked }))} className="mt-1" />我已实际核对该版本的答案、解法、评分规则和来源，愿意记录这次人工审核。</label>
          <label className="mt-3 block text-xs text-slate-600">审核说明<textarea value={reviewNotes[`${item.id}:${item.version}`] || ""} onChange={(event) => setReviewNotes((current) => ({ ...current, [`${item.id}:${item.version}`]: event.target.value }))} rows={2} className="mt-2 w-full rounded-lg border border-slate-200 bg-white p-2 text-sm" placeholder="写明人工核对依据；此框不会自动代表已审核。" /></label>
        </div>}
      </details>
      <div className="mt-4 flex gap-3">{["validate", ...(item.publication_status === "draft" ? ["publish"] : []), ...(item.publication_status !== "archived" ? ["archive"] : [])].map((operation) => <button key={operation} type="button" disabled={busy} onClick={() => void action(item, operation)} className="rounded-lg border border-slate-200 px-3 py-2 text-xs font-semibold disabled:opacity-50">{operation === "validate" ? "运行校验" : operation === "publish" ? "发布此版本" : "归档此版本"}</button>)}</div>
    </article>)}</section>
    {coverage.length > 0 && <section className="rounded-2xl border border-slate-200 bg-white p-5"><h2 className="text-xl font-bold">题库覆盖</h2><ContentCoverageTable topics={coverage.filter((item) => !course || item.owner_course === course)} /></section>}
  </div>;
}
