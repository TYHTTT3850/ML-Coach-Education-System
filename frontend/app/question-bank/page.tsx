"use client";
import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import MarkdownRenderer from "@/components/MarkdownRenderer";
import PracticePanel from "@/components/PracticePanel";
import ContentCoverageTable from "@/components/ContentCoverageTable";
import { useStudent } from "@/components/StudentProvider";
import { getCurriculum, getTopic } from "@/lib/api";
import { createAssessment, getAssessment, getContentCoverage, getQuestionBank } from "@/lib/api-content";
import { COURSES, COURSE_META, type Course, type CurriculumTopic, type CurriculumTopicSummary } from "@/lib/types";
import type { Assessment, BankQueryResult, ContentCoverage, PublicBankItem } from "@/lib/types-content";

const TYPES = { multiple_choice: "单选题", numeric: "数值计算", concept_explanation: "概念解释", derivation: "推导", code_reading: "代码阅读", case_analysis: "场景分析", theorem_condition: "定理条件", error_correction: "错误改正" };
export default function QuestionBankPage() {
  const { withStudent, studentId } = useStudent();
  const [course, setCourse] = useState<Course>("machine_learning");
  const [topicId, setTopicId] = useState("");
  const [topics, setTopics] = useState<CurriculumTopicSummary[]>([]);
  const [type, setType] = useState(""); const [difficulty, setDifficulty] = useState(""); const [query, setQuery] = useState("");
  const [data, setData] = useState<BankQueryResult>({ items: [], checkpoint_count: 0 });
  const [coverage, setCoverage] = useState<ContentCoverage[]>([]);
  const [error, setError] = useState(""); const [busy, setBusy] = useState(false);
  const [selected, setSelected] = useState<{ topic: CurriculumTopic; assessment: Assessment } | null>(null);
  const search = useCallback(async () => {
    setBusy(true); setError("");
    try { setData(await getQuestionBank({ course, topic_id: topicId, question_type: type, difficulty, q: query })); }
    catch (cause) { setError(cause instanceof Error ? cause.message : "题库暂不可用，请重试。"); }
    finally { setBusy(false); }
  }, [course, topicId, type, difficulty, query]);
  useEffect(() => { const timer = window.setTimeout(() => void search(), 250); return () => window.clearTimeout(timer); }, [search]);
  useEffect(() => { let cancelled = false; void getCurriculum(course).then((value) => { if (!cancelled) setTopics(value.stages.flatMap((stage) => stage.topics)); }).catch(() => {}); return () => { cancelled = true; }; }, [course]);
  useEffect(() => { let cancelled = false; void getContentCoverage().then((value) => { if (!cancelled) setCoverage(value.topics); }).catch(() => {}); return () => { cancelled = true; }; }, []);
  useEffect(() => {
    if (!studentId) return;
    let cancelled = false;
    const timer = window.setTimeout(() => {
      let saved: { topicId?: string; assessmentId?: string } = {};
      try { saved = JSON.parse(window.localStorage.getItem(`course-coach:bank-active:${studentId}`) || "{}"); } catch { return; }
      if (!saved.topicId || !saved.assessmentId) return;
      void withStudent(async (id) => ({ topic: await getTopic(saved.topicId!, id), assessment: await getAssessment(saved.assessmentId!, id) })).then((value) => { if (!cancelled) setSelected(value); }).catch(() => { if (!cancelled) setError("上次单题练习暂时无法恢复，可以重试或重新选择题目。"); });
    }, 0);
    return () => { cancelled = true; window.clearTimeout(timer); };
  }, [studentId, withStudent]);
  async function practice(item: PublicBankItem) {
    setBusy(true); setError("");
    try {
      const value = await withStudent(async (id) => ({ topic: await getTopic(item.topic_ids[0], id), assessment: await createAssessment({ topic_ids: [item.topic_ids[0]], purpose: "practice", question_count: 1, bank_item_ids: [item.id], exclude_seen: false, exclude_exposed_families: false }, id) }));
      setSelected(value);
      if (studentId) { try { window.localStorage.setItem(`course-coach:bank-active:${studentId}`, JSON.stringify({ topicId: value.topic.id, assessmentId: value.assessment.practice_set_id })); } catch { /* Optional recovery location. */ } }
    } catch (cause) { setError(cause instanceof Error ? cause.message : "暂时无法开始这道题，请重试。"); }
    finally { setBusy(false); }
  }
  return <div className="mx-auto max-w-7xl space-y-7 px-4 py-8 sm:px-7">
    <div className="flex flex-wrap items-end justify-between gap-4"><div><p className="text-xs font-semibold text-indigo-600">持久化课程题库</p><h1 className="mt-2 text-3xl font-bold text-slate-900">自己练习，独立检验</h1><p className="mt-3 text-sm leading-7 text-slate-500">浏览保留的已发布练习题，自选辅助学习。课程页的练习与测试已恢复由 AI 按课程资料动态出题；本页题库及历史测试保持可用。</p></div><Link href="/question-bank/admin" prefetch={false} className="rounded-xl border border-slate-200 bg-white px-4 py-2 text-xs font-semibold text-slate-600">管理员题库管理</Link></div>
    <form onSubmit={(event) => { event.preventDefault(); void search(); }} className="grid gap-3 rounded-2xl border border-slate-200 bg-white p-4 sm:grid-cols-2 lg:grid-cols-5">
      <label className="text-xs text-slate-500">课程<select aria-label="题库课程" value={course} onChange={(event) => { setCourse(event.target.value as Course); setTopicId(""); }} className="mt-2 w-full rounded-lg border border-slate-200 p-2 text-sm">{COURSES.map((item) => <option key={item} value={item}>{COURSE_META[item].label}</option>)}</select></label>
      <label className="text-xs text-slate-500">知识点<select value={topicId} onChange={(event) => setTopicId(event.target.value)} className="mt-2 w-full rounded-lg border border-slate-200 p-2 text-sm"><option value="">全部知识点</option>{topics.map((topic) => <option key={topic.id} value={topic.id}>{topic.title}</option>)}</select></label>
      <label className="text-xs text-slate-500">题型<select value={type} onChange={(event) => setType(event.target.value)} className="mt-2 w-full rounded-lg border border-slate-200 p-2 text-sm"><option value="">全部题型</option>{Object.entries(TYPES).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>
      <label className="text-xs text-slate-500">难度<select value={difficulty} onChange={(event) => setDifficulty(event.target.value)} className="mt-2 w-full rounded-lg border border-slate-200 p-2 text-sm"><option value="">全部难度</option><option value="easy">基础</option><option value="medium">中等</option><option value="hard">进阶</option></select></label>
      <label className="text-xs text-slate-500">关键词<input type="search" value={query} onChange={(event) => setQuery(event.target.value)} className="mt-2 w-full rounded-lg border border-slate-200 p-2 text-sm" placeholder="搜索题干或标题" /></label>
    </form>
    {error && <div role="alert" className="rounded-xl bg-rose-50 p-4 text-sm text-rose-800">{error}<button type="button" onClick={() => void search()} className="ml-3 underline">重试查询</button></div>}
    <p className="text-sm text-slate-500">{busy ? "正在读取…" : `可见练习 ${data.items.length} 题 · 正式题库 ${data.checkpoint_count} 题（题干分配后展示）`}</p>
    {data.checkpoint_difficulty_distribution && <p className="text-xs text-slate-500">正式池难度：{Object.entries(data.checkpoint_difficulty_distribution).map(([level, count]) => `${level === "easy" ? "基础" : level === "medium" ? "中等" : "进阶"} ${count}`).join(" · ")}</p>}
    {selected && <><Link href={`/learn/${selected.topic.course}/${encodeURIComponent(selected.topic.id)}`} className="block text-sm font-semibold text-indigo-700">前往{selected.topic.title}学习、查看前置与参加正式测试 →</Link><PracticePanel key={selected.assessment.practice_set_id} topic={selected.topic} bankMode practiceOnly initialAssessment={selected.assessment} onLearningUpdated={async () => {}} /></>}
    <section aria-label="已发布练习题" className="grid gap-4 lg:grid-cols-2">{data.items.map((item) => <article key={`${item.id}:${item.version}`} className="rounded-2xl border border-slate-200 bg-white p-5"><div className="flex flex-wrap justify-between gap-2 text-xs text-slate-500"><span>{TYPES[item.question_type as keyof typeof TYPES] || item.question_type} · {item.difficulty} · 约 {item.estimated_minutes} 分钟</span><span>版本 {item.version}</span></div><h2 className="mt-3 text-base font-bold text-slate-900">{item.title}</h2><div className="mt-3"><MarkdownRenderer content={item.stem} /></div><button type="button" disabled={busy} onClick={() => void practice(item)} className="mt-4 rounded-xl bg-indigo-600 px-4 py-2 text-xs font-semibold text-white disabled:opacity-50">练习这道题</button></article>)}</section>
    {!busy && !data.items.length && <p className="rounded-2xl border border-dashed border-slate-300 p-6 text-sm text-slate-500">当前筛选没有已发布练习。可清空筛选，或查看下方覆盖报告。</p>}
    <section className="rounded-3xl border border-slate-200 bg-white p-5"><h2 className="text-xl font-bold">内容覆盖报告</h2><p className="mt-2 text-xs text-slate-500">数字来自保存的内容版本。未完成节点明确标注“内容待补充”。</p><ContentCoverageTable topics={coverage.filter((topic) => topic.owner_course === course)} /></section>
  </div>;
}
