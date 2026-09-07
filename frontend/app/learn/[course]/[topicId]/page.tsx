"use client";

import Link from "next/link";
import { useParams, useSearchParams } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
import { useCourseOutline, contentState } from "@/components/CourseLayout";
import { LoadingBlock, MasteryBar, TopicStatusBadge } from "@/components/LearningWidgets";
import LessonContent, { WorkedExample } from "@/components/LessonContent";
import TopicPrerequisiteCard from "@/components/TopicPrerequisiteCard";
import PracticePanel from "@/components/PracticePanel";
import TopicTutor from "@/components/TopicTutor";
import { useStudent } from "@/components/StudentProvider";
import { ApiError, getLearningHistory, getTopic, isStudentNotFound, startTopic } from "@/lib/api";
import { getExamples, getLesson, getPrerequisites } from "@/lib/api-content";
import { hasValidTopicEvidence, scoreLabel } from "@/lib/learning";
import { COURSE_META, COURSES, type Course, type CurriculumTopic, type RecentLearningAttempt } from "@/lib/types";
import type { Assessment, Lesson, Prerequisites, WorkedExampleData } from "@/lib/types-content";

export default function TopicLearningPage() {
  const params = useParams<{ course: string; topicId: string }>();
  return <TopicContent key={`${params.course}:${params.topicId}`} ownerCourse={params.course as Course} topicId={decodeURIComponent(params.topicId)} />;
}

function TopicContent({ ownerCourse, topicId }: { ownerCourse: Course; topicId: string }) {
  const { withStudent, status } = useStudent();
  const { outline, course, topicHref, refresh } = useCourseOutline();
  const searchParams = useSearchParams();
  const [topic, setTopic] = useState<CurriculumTopic | null>(null);
  const [lesson, setLesson] = useState<Lesson | null>(null);
  const [examples, setExamples] = useState<WorkedExampleData[]>([]);
  const [prerequisites, setPrerequisites] = useState<Prerequisites | null>(null);
  const [history, setHistory] = useState<RecentLearningAttempt[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [contentError, setContentError] = useState("");
  const [starting, setStarting] = useState(false);
  const [assessment, setAssessment] = useState<Assessment | null>(null);
  const [formalActive, setFormalActive] = useState(false);
  const load = useCallback(async () => {
    setError("");
    setPrerequisites(null);
    try {
      const values = await withStudent(async (id) => {
        const results = await Promise.allSettled([
          getTopic(topicId, id), getPrerequisites(topicId, id), getLesson(topicId), getExamples(topicId), getLearningHistory(ownerCourse, id),
        ]);
        // Preserve the provider's single retry when one request finds a stale identity.
        for (const result of results) if (result.status === "rejected" && isStudentNotFound(result.reason)) throw result.reason;
        return results;
      });
      if (values[0].status === "rejected") throw values[0].reason;
      if (values[0].value.course !== ownerCourse) throw new Error("该地址与知识点归属不一致，请从课程目录重新进入。");
      setTopic(values[0].value);
      if (values[1].status === "fulfilled") setPrerequisites(values[1].value);
      else {
        const cause = values[1].reason;
        const details = cause instanceof ApiError ? `${cause.message}${cause.code && /^[A-Z0-9_]{1,80}$/i.test(cause.code) ? `（${cause.code}）` : ""}` : "无法完成前置能力检查。";
        setError(`前置能力检查暂时不可用：${details} 正式测试暂不开放。请重试；讲解和例题仍可阅读。`);
      }
      if (values[2].status === "fulfilled") { setLesson(values[2].value); setContentError(""); }
      else if (values[2].reason instanceof ApiError && values[2].reason.status === 404) { setLesson(null); setContentError(""); }
      else setContentError("课程正文暂时加载失败，请重新加载内容。");
      if (values[3].status === "fulfilled") setExamples(values[3].value.examples);
      else setContentError("例题暂时加载失败，请重新加载内容。");
      if (values[4].status === "fulfilled") setHistory(values[4].value.events.filter((event) => event.topic_id === topicId));
    } catch (cause) { setError(cause instanceof Error ? cause.message : "知识点暂时无法加载，请重试。"); }
    finally { setLoading(false); }
  }, [topicId, ownerCourse, withStudent]);
  useEffect(() => { if (status !== "ready") return; const timer = window.setTimeout(() => void load(), 0); return () => window.clearTimeout(timer); }, [status, load]);
  const onLearningUpdated = useCallback(async () => { await Promise.all([load(), refresh()]); }, [load, refresh]);
  async function begin() {
    setStarting(true);
    try { await withStudent((id) => startTopic(topicId, id)); await onLearningUpdated(); document.getElementById("lesson")?.scrollIntoView({ behavior: "smooth" }); }
    catch (cause) { setError(cause instanceof Error ? cause.message : "暂时无法记录开始学习。你仍可阅读课程内容。"); }
    finally { setStarting(false); }
  }
  if (!COURSES.includes(ownerCourse)) return <div className="p-8"><h1>课程地址无效</h1><Link href="/learn" className="text-indigo-700">返回课程列表</Link></div>;
  if (loading) return <div className="p-5 sm:p-8"><LoadingBlock label="正在读取课程正文与学习状态…" /></div>;
  if (!topic) return <div role="alert" className="m-5 rounded-2xl bg-rose-50 p-6 text-rose-800"><h1 className="font-bold">知识点无法加载</h1><p className="mt-2 text-sm">{error}</p><button type="button" onClick={() => void load()} className="mt-4 underline">重新加载</button></div>;
  const ordered = outline?.groups.flatMap((group) => group.topics) ?? [];
  const position = ordered.findIndex((item) => item.topic_id === topicId);
  const previous = position > 0 ? ordered[position - 1] : null;
  const next = position >= 0 ? ordered[position + 1] : null;
  const placed = ordered[position];
  const currentGroup = outline?.groups.find((group) => group.topics.some((item) => item.topic_id === topicId));
  const returnTopic = searchParams.get("return_topic");
  return <div className="space-y-6 px-4 py-6 sm:px-7 lg:px-8">
    {course && course !== ownerCourse && <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-indigo-200 bg-indigo-50 px-4 py-3 text-xs text-indigo-800"><span>从{COURSE_META[course].label}补学 · 当前知识点归属于{COURSE_META[ownerCourse].label}，学习状态共用。</span><Link href={returnTopic ? `/learn/${course}/${encodeURIComponent(returnTopic)}` : `/learn/${course}`} className="font-bold">{returnTopic ? "返回原机器学习知识点" : "返回原课程"} →</Link></div>}
    <section className="rounded-3xl border border-slate-200 bg-white p-5 shadow-sm sm:p-7">
      <div className="flex flex-wrap items-center gap-2 text-xs text-slate-500"><Link href={`/learn/${course || ownerCourse}`} className="hover:text-indigo-700">{COURSE_META[course || ownerCourse].label}</Link><span>/</span><span>{currentGroup?.title || topic.stage_title}</span><TopicStatusBadge status={topic.status} />{placed && placed.content_status !== "ready" && <span className="rounded-full bg-slate-100 px-3 py-1">□ {contentState(placed).label}</span>}</div>
      <h1 className="mt-4 text-2xl font-bold tracking-tight text-slate-950 sm:text-3xl">{topic.title}</h1>
      {placed?.content_status === "missing" && placed.readiness_reason && <p className="mt-3 rounded-xl bg-slate-50 px-3 py-2 text-xs leading-6 text-slate-500">{placed.has_learning_content ? "已有讲解可读；" : ""}{placed.readiness_reason}</p>}
      <h2 className="mt-5 text-sm font-bold text-slate-900">为什么学它</h2><p className="mt-2 text-sm leading-7 text-slate-600">{topic.description}</p>
      <h2 className="mt-5 text-sm font-bold text-slate-900">学习目标</h2><ul className="mt-3 grid gap-2 xl:grid-cols-2">{topic.learning_objectives.map((objective) => <li key={objective} className="rounded-xl bg-slate-50 px-3 py-2 text-sm leading-6 text-slate-600">✓ {objective}</li>)}</ul>
      <div className="mt-6 flex flex-wrap items-center justify-between gap-4 border-t border-slate-100 pt-4"><button type="button" disabled={starting} onClick={() => void begin()} className="rounded-xl bg-indigo-600 px-4 py-2.5 text-sm font-semibold text-white disabled:opacity-50">{starting ? "正在记录…" : topic.status === "not_started" ? "开始学习" : "继续学习"}</button><div className="flex flex-wrap gap-4 text-xs font-semibold text-indigo-700">{previous && <Link href={topicHref(previous)} prefetch={false}>← 上一个：{previous.title}</Link>}{next && <Link href={topicHref(next)} prefetch={false}>下一个：{next.title} →</Link>}</div></div>
    </section>
    {error && <div role="alert" className="rounded-xl bg-amber-50 p-4 text-sm text-amber-900">{error}<button type="button" onClick={() => void load()} className="ml-3 font-semibold underline">重试</button></div>}
    <nav aria-label="知识点内容分区" className="flex flex-wrap gap-2 rounded-2xl border border-slate-200 bg-white p-3">{[["lesson", "1 知识讲解"], ["examples", "2 分步例题"], ["practice", "3 练习与测试"], ["history", "4 学习记录"]].map(([id, label]) => <a key={id} href={`#${id}`} className="rounded-lg px-3 py-2 text-xs font-semibold text-indigo-700 hover:bg-indigo-50">{label}</a>)}</nav>
    {prerequisites && <TopicPrerequisiteCard data={prerequisites} onAssessment={setAssessment} />}
    {contentError && <div role="alert" className="rounded-xl bg-rose-50 p-4 text-sm text-rose-800">{contentError}<button type="button" onClick={() => void load()} className="ml-3 underline">重新加载内容</button></div>}
    {lesson ? <LessonContent lesson={lesson} /> : <section id="lesson" className="rounded-3xl border border-dashed border-slate-300 bg-white p-7"><h2 className="text-lg font-bold text-slate-700">□ 固定课程正文待补充</h2><p className="mt-3 text-sm leading-7 text-slate-500">这个知识点尚无已发布的固定课程正文，但仍可根据已有课程资料进行 AI 讲解、分步例题与动态练习。不需要等固定内容补齐后才开始学习。</p><a href="#topic-tutor" className="mt-4 inline-block text-sm font-semibold text-indigo-700">使用 AI 讲解与例题 ↓</a></section>}
    <section id="examples" className="space-y-4 scroll-mt-24"><h2 className="text-xl font-bold text-slate-900">分步例题</h2>{examples.length > 0 ? examples.map((example) => <WorkedExample key={`${example.id}:${example.version}`} example={example} askDisabled={formalActive} />) : <p className="rounded-2xl border border-dashed border-slate-300 p-5 text-sm leading-7 text-slate-500">暂无固定例题。点击下方“AI 讲例题”，按本知识点课程资料讲解 2 道具体例题及其步骤。</p>}
      {formalActive ? <section id="topic-tutor" className="rounded-2xl border border-amber-200 bg-amber-50 p-5 text-sm text-amber-900">正式测试进行中，AI 例题、课程答疑与例题步骤提问暂时关闭。提交后可继续使用辅助学习。</section> : <TopicTutor course={ownerCourse} topicId={topic.id} topicTitle={topic.title} />}
    </section>
    <PracticePanel topic={topic} checkpointAllowed={prerequisites?.can_checkpoint ?? false} checkpointBlockedReason={!prerequisites ? "前置能力状态尚未确认，请点击上方重试检查。仍可阅读讲解和进行辅助练习，正式测试暂不开放。" : prerequisites.unmet.length ? "先补齐必需前置能力，或通过前置小测证明对应技能。仍可继续阅读和辅助练习。" : "前置已满足；请先完成无提示练习，积累现有 Mastery Gate 所需的有效证据，再参加正式测试。"} initialAssessment={assessment} onLearningUpdated={onLearningUpdated} onActiveCheckpoint={setFormalActive} />
    <section id="history" className="scroll-mt-24 rounded-3xl border border-slate-200 bg-white p-5 sm:p-7"><h2 className="text-lg font-bold">学习记录</h2><div className="mt-4 max-w-md"><MasteryBar value={hasValidTopicEvidence(topic) ? topic.mastery : null} /></div>{history.length ? <ul className="mt-5 divide-y divide-slate-100">{history.map((event, index) => <li key={event.id || event.attempt_id || index} className="flex flex-wrap justify-between gap-2 py-3 text-sm text-slate-600"><span>{event.mode === "checkpoint" ? "正式测试" : "练习"} · {scoreLabel(event.score)}</span><time className="text-xs text-slate-400">{new Date(event.created_at).toLocaleString("zh-CN")}</time></li>)}</ul> : <p className="mt-4 text-sm text-slate-500">还没有本知识点的作答记录。阅读例题不会自动成为正式通过证据。</p>}</section>
    {prerequisites && prerequisites.successors.length > 0 && <section className="rounded-3xl border border-slate-200 bg-white p-5 sm:p-7"><h2 className="text-lg font-bold">学完以后可以学什么</h2><div className="mt-4 grid gap-3 xl:grid-cols-2">{prerequisites.successors.map((item) => <Link key={item.id} href={topicHref({ topic_id: item.id, owner_course: item.course })} className="rounded-xl bg-slate-50 p-4"><p className="text-sm font-semibold text-indigo-700">{item.title} →</p><p className="mt-2 text-xs leading-6 text-slate-500">{item.reason}</p></Link>)}</div></section>}
  </div>;
}
