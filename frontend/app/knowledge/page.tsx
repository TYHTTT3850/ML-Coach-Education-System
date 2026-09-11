"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";
import CourseSelector from "@/components/CourseSelector";
import {
  AlertIcon,
  ArrowLeftIcon,
  BookIcon,
  DatabaseIcon,
  EyeIcon,
  EyeOffIcon,
  FileIcon,
  RefreshIcon,
  SparklesIcon,
  TrashIcon,
} from "@/components/Icons";
import KnowledgeUploader from "@/components/KnowledgeUploader";
import DocumentTopicTagger from "@/components/DocumentTopicTagger";
import { deleteDocument, getKnowledgeStats, listDocuments } from "@/lib/api";
import {
  COURSE_META,
  COURSES,
  type Course,
  type KnowledgeDocument,
  type KnowledgeStats,
} from "@/lib/types";

const ADMIN_TOKEN_KEY = "course-coach:admin-token";

function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes < 0) return "—";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
}

function formatDate(value?: string | null): string {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat("zh-CN", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

function statusStyle(status: string): { label: string; className: string } {
  switch (status.toLowerCase()) {
    case "ready":
    case "completed":
    case "success":
      return { label: "可用", className: "border-emerald-200 bg-emerald-50 text-emerald-700" };
    case "processing":
    case "pending":
      return { label: "处理中", className: "border-amber-200 bg-amber-50 text-amber-700" };
    case "failed":
    case "error":
      return { label: "失败", className: "border-rose-200 bg-rose-50 text-rose-700" };
    default:
      return { label: status, className: "border-slate-200 bg-slate-50 text-slate-600" };
  }
}

export default function KnowledgePage() {
  const [course, setCourse] = useState<Course>("machine_learning");
  const [adminToken, setAdminToken] = useState("");
  const [tokenReady, setTokenReady] = useState(false);
  const [showToken, setShowToken] = useState(false);
  const [documents, setDocuments] = useState<KnowledgeDocument[]>([]);
  const [stats, setStats] = useState<KnowledgeStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      try {
        setAdminToken(window.sessionStorage.getItem(ADMIN_TOKEN_KEY) ?? "");
      } finally {
        setTokenReady(true);
      }
    }, 0);
    return () => window.clearTimeout(timer);
  }, []);

  const refresh = useCallback(async () => {
    if (!tokenReady) return;
    setLoading(true);
    setError(null);
    try {
      const [nextDocuments, nextStats] = await Promise.all([
        listDocuments(course, adminToken),
        getKnowledgeStats(adminToken),
      ]);
      setDocuments(nextDocuments);
      setStats(nextStats);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "知识库数据加载失败。\n");
    } finally {
      setLoading(false);
    }
  }, [adminToken, course, tokenReady]);

  useEffect(() => {
    const timer = window.setTimeout(() => void refresh(), 0);
    return () => window.clearTimeout(timer);
  }, [refresh]);

  function updateToken(value: string) {
    setAdminToken(value);
    try {
      if (value) window.sessionStorage.setItem(ADMIN_TOKEN_KEY, value);
      else window.sessionStorage.removeItem(ADMIN_TOKEN_KEY);
    } catch {
      // The page still works if browser storage is unavailable.
    }
  }

  async function handleDelete(document: KnowledgeDocument) {
    if (!adminToken.trim()) {
      setError("请先填写管理员 Token，再删除文档。");
      return;
    }
    if (!window.confirm(`确定删除“${document.original_filename}”及其全部向量片段吗？`)) return;
    setDeletingId(document.id);
    setError(null);
    try {
      await deleteDocument(document.id, adminToken);
      setDocuments((current) => current.filter((item) => item.id !== document.id));
      await refresh();
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "删除失败，请稍后重试。");
    } finally {
      setDeletingId(null);
    }
  }

  const selectedStats = stats?.[course];
  const totalDocuments = useMemo(
    () => (stats ? COURSES.reduce((total, item) => total + (stats[item]?.document_count ?? 0), 0) : 0),
    [stats],
  );
  const totalChunks = useMemo(
    () => (stats ? COURSES.reduce((total, item) => total + (stats[item]?.chunk_count ?? 0), 0) : 0),
    [stats],
  );

  return (
    <main className="min-h-svh bg-canvas">
      <header className="sticky top-0 z-20 border-b border-slate-200/80 bg-white/90 backdrop-blur-xl">
        <div className="mx-auto flex max-w-7xl items-center justify-between gap-4 px-4 py-3 sm:px-6 lg:px-8">
          <div className="flex min-w-0 items-center gap-3">
            <Link
              href="/?view=chat"
              className="grid size-9 shrink-0 place-items-center rounded-xl border border-slate-200 bg-white text-slate-600 transition hover:border-indigo-200 hover:text-indigo-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500"
              aria-label="返回课程问答"
            >
              <ArrowLeftIcon className="size-4" />
            </Link>
            <span className="hidden size-9 place-items-center rounded-xl bg-gradient-to-br from-indigo-600 to-violet-600 text-white sm:grid">
              <SparklesIcon className="size-4" />
            </span>
            <div className="min-w-0">
              <h1 className="truncate text-base font-semibold tracking-tight text-slate-900">知识库管理</h1>
              <p className="hidden text-[11px] text-slate-400 sm:block">Course Coach · 课程资料与向量索引</p>
            </div>
          </div>
          <Link
            href="/?view=chat"
            className="rounded-xl bg-slate-900 px-3.5 py-2 text-xs font-semibold text-white transition hover:bg-slate-800 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-slate-500 focus-visible:ring-offset-2 sm:text-sm"
          >
            返回问答
          </Link>
        </div>
      </header>

      <div className="mx-auto max-w-7xl px-4 py-7 sm:px-6 sm:py-10 lg:px-8">
        <div className="flex flex-col justify-between gap-5 lg:flex-row lg:items-end">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.18em] text-indigo-600">Knowledge Base</p>
            <h2 className="mt-2 text-2xl font-semibold tracking-tight text-slate-900 sm:text-3xl">管理你的课程资料</h2>
            <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-500">
              上传后服务会自动解析、切分并向量化。文档严格归属所选课程，聊天检索不会跨课程混用。
            </p>
          </div>
          <button
            type="button"
            onClick={() => void refresh()}
            disabled={loading}
            className="inline-flex w-fit items-center gap-2 rounded-xl border border-slate-200 bg-white px-4 py-2.5 text-sm font-medium text-slate-600 shadow-sm transition hover:border-indigo-200 hover:text-indigo-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 disabled:cursor-wait disabled:opacity-60"
          >
            <RefreshIcon className={`size-4 ${loading ? "animate-spin" : ""}`} />
            刷新数据
          </button>
        </div>

        {error && (
          <div className="mt-5 flex items-start gap-2 rounded-2xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-800" role="alert">
            <AlertIcon className="mt-0.5 size-4 shrink-0" />
            <span className="flex-1">{error}</span>
            <button type="button" onClick={() => setError(null)} className="text-xs font-semibold text-rose-600 hover:text-rose-800">关闭</button>
          </div>
        )}

        <section className="mt-7 grid gap-3 sm:grid-cols-2 lg:grid-cols-4" aria-label="知识库统计">
          <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium text-slate-500">全部文档</span>
              <DatabaseIcon className="size-4 text-indigo-500" />
            </div>
            <p className="mt-3 text-2xl font-semibold tabular-nums text-slate-900">{loading && !stats ? "—" : totalDocuments}</p>
            <p className="mt-1 text-[11px] text-slate-400">两个课程合计</p>
          </div>
          <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium text-slate-500">全部片段</span>
              <BookIcon className="size-4 text-violet-500" />
            </div>
            <p className="mt-3 text-2xl font-semibold tabular-nums text-slate-900">{loading && !stats ? "—" : totalChunks}</p>
            <p className="mt-1 text-[11px] text-slate-400">可检索的 chunk 数量</p>
          </div>
          {COURSES.map((item) => (
            <button
              key={item}
              type="button"
              onClick={() => setCourse(item)}
              className={`rounded-2xl border p-4 text-left shadow-sm transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 ${
                course === item
                  ? "border-indigo-200 bg-indigo-50/70"
                  : "border-slate-200 bg-white hover:border-indigo-200"
              }`}
            >
              <span className="text-xs font-medium text-slate-500">{COURSE_META[item].label}</span>
              <div className="mt-3 flex items-baseline gap-2">
                <span className="text-2xl font-semibold tabular-nums text-slate-900">{stats?.[item]?.document_count ?? (loading ? "—" : 0)}</span>
                <span className="text-[11px] text-slate-400">文档</span>
                <span className="ml-auto text-xs tabular-nums text-slate-500">{stats?.[item]?.chunk_count ?? 0} chunks</span>
              </div>
            </button>
          ))}
        </section>

        <div className="mt-7 grid gap-6 lg:grid-cols-[minmax(0,0.85fr)_minmax(0,1.4fr)]">
          <div className="space-y-6">
            <section className="rounded-3xl border border-slate-200 bg-white p-5 shadow-soft sm:p-6">
              <div className="flex items-center gap-3">
                <span className="grid size-9 place-items-center rounded-xl bg-slate-100 text-slate-600">
                  <DatabaseIcon className="size-4" />
                </span>
                <div>
                  <h3 className="text-sm font-semibold text-slate-900">管理员凭证</h3>
                  <p className="text-[11px] text-slate-400">只保存在当前浏览器标签会话</p>
                </div>
              </div>
              <label htmlFor="admin-token" className="mt-5 block text-xs font-medium text-slate-600">Admin Token</label>
              <div className="relative mt-2">
                <input
                  id="admin-token"
                  type={showToken ? "text" : "password"}
                  value={adminToken}
                  onChange={(event) => updateToken(event.target.value)}
                  autoComplete="off"
                  placeholder="输入后端配置的 ADMIN_TOKEN"
                  className="h-11 w-full rounded-xl border border-slate-200 bg-slate-50 px-3 pr-11 text-sm text-slate-800 outline-none transition placeholder:text-slate-400 focus:border-indigo-300 focus:bg-white focus:ring-4 focus:ring-indigo-100/70"
                />
                <button
                  type="button"
                  onClick={() => setShowToken((visible) => !visible)}
                  className="absolute right-1.5 top-1.5 grid size-8 place-items-center rounded-lg text-slate-400 transition hover:bg-slate-100 hover:text-slate-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500"
                  aria-label={showToken ? "隐藏 Token" : "显示 Token"}
                >
                  {showToken ? <EyeOffIcon className="size-4" /> : <EyeIcon className="size-4" />}
                </button>
              </div>
              <p className="mt-2 text-[11px] leading-5 text-slate-400">
                不会写入代码或 localStorage，关闭标签页后由浏览器清除。
              </p>
            </section>

            <section className="rounded-3xl border border-slate-200 bg-white p-5 shadow-soft sm:p-6">
              <div className="mb-5">
                <h3 className="text-sm font-semibold text-slate-900">上传到课程</h3>
                <p className="mt-1 text-xs text-slate-500">先确认课程，再选择资料文件。</p>
              </div>
              <CourseSelector value={course} onChange={setCourse} compact className="mb-5" />
              <KnowledgeUploader
                course={course}
                adminToken={adminToken}
                onBatchComplete={refresh}
              />
            </section>
          </div>

          <section className="min-w-0 rounded-3xl border border-slate-200 bg-white shadow-soft">
            <div className="flex flex-col gap-3 border-b border-slate-100 px-5 py-5 sm:flex-row sm:items-center sm:justify-between sm:px-6">
              <div>
                <h3 className="text-sm font-semibold text-slate-900">{COURSE_META[course].label}文档</h3>
                <p className="mt-1 text-xs text-slate-400">
                  {documents.length} 个文档 · {selectedStats?.chunk_count ?? 0} 个片段
                </p>
              </div>
              <div className="text-[11px] text-slate-400">
                最近更新：{formatDate(selectedStats?.latest_updated_at)}
              </div>
            </div>

            {loading ? (
              <div className="grid min-h-72 place-items-center p-8" role="status">
                <div className="text-center">
                  <RefreshIcon className="mx-auto size-6 animate-spin text-indigo-500" />
                  <p className="mt-3 text-sm text-slate-500">正在读取知识库…</p>
                </div>
              </div>
            ) : documents.length === 0 ? (
              <div className="grid min-h-72 place-items-center p-8 text-center">
                <div>
                  <span className="mx-auto grid size-14 place-items-center rounded-2xl bg-slate-100 text-slate-400">
                    <FileIcon className="size-6" />
                  </span>
                  <p className="mt-4 text-sm font-semibold text-slate-700">这个课程还没有已上传文档</p>
                  <p className="mt-1.5 max-w-xs text-xs leading-5 text-slate-400">从左侧上传第一份资料，处理完成后即可用于聊天检索。</p>
                </div>
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full min-w-[760px] text-left">
                  <thead>
                    <tr className="border-b border-slate-100 bg-slate-50/70 text-[11px] font-semibold uppercase tracking-wider text-slate-400">
                      <th className="px-6 py-3">文档</th>
                      <th className="px-4 py-3">课程</th>
                      <th className="px-4 py-3">大小</th>
                      <th className="px-4 py-3">Chunks</th>
                      <th className="px-4 py-3">状态</th>
                      <th className="px-4 py-3">上传时间</th>
                      <th className="px-6 py-3 text-right">操作</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {documents.map((document) => {
                      const status = statusStyle(document.status);
                      return (
                        <tr key={document.id} className="group transition hover:bg-slate-50/70">
                          <td className="max-w-64 px-6 py-4">
                            <div className="flex items-center gap-3">
                              <span className="grid size-9 shrink-0 place-items-center rounded-xl bg-indigo-50 text-indigo-600">
                                <FileIcon className="size-4" />
                              </span>
                              <div className="min-w-0">
                                <p className="truncate text-sm font-medium text-slate-800" title={document.original_filename}>{document.original_filename}</p>
                                {document.error_message && (
                                  <p className="mt-1 line-clamp-1 text-[11px] text-rose-500" title={document.error_message}>{document.error_message}</p>
                                )}
                              </div>
                            </div>
                          </td>
                          <td className="px-4 py-4 text-xs text-slate-500">{COURSE_META[document.course]?.label ?? document.course}</td>
                          <td className="px-4 py-4 text-xs tabular-nums text-slate-500">{formatBytes(document.file_size)}</td>
                          <td className="px-4 py-4 text-xs tabular-nums text-slate-600">{document.chunk_count}</td>
                          <td className="px-4 py-4">
                            <span className={`inline-flex rounded-full border px-2 py-1 text-[10px] font-semibold ${status.className}`}>{status.label}</span>
                          </td>
                          <td className="whitespace-nowrap px-4 py-4 text-xs text-slate-500">{formatDate(document.created_at)}</td>
                          <td className="px-6 py-4 text-right">
                            <button
                              type="button"
                              onClick={() => void handleDelete(document)}
                              disabled={deletingId === document.id}
                              className="inline-flex items-center gap-1.5 rounded-lg px-2.5 py-2 text-xs font-medium text-slate-400 transition hover:bg-rose-50 hover:text-rose-600 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-rose-400 disabled:cursor-wait disabled:opacity-50"
                            >
                              <TrashIcon className="size-3.5" />
                              {deletingId === document.id ? "删除中" : "删除"}
                            </button>
                            <DocumentTopicTagger documentId={document.id} adminToken={adminToken} />
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </section>
        </div>
      </div>
    </main>
  );
}
