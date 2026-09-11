"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import type { ReactNode } from "react";
import {
  AlertIcon,
  BookIcon,
  DatabaseIcon,
  SparklesIcon,
  TargetIcon,
} from "@/components/Icons";
import { useStudent } from "@/components/StudentProvider";

const NAV_ITEMS = [
  { href: "/?view=chat", label: "自由问答", icon: SparklesIcon },
  { href: "/learn", label: "课程学习", icon: TargetIcon },
  { href: "/knowledge", label: "知识库", icon: DatabaseIcon },
  { href: "/question-bank", label: "题库", icon: BookIcon },
];

export default function LearningShell({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  const { status, error, retryInitialization } = useStudent();

  return (
    <main className="min-h-svh bg-canvas">
      <header className="sticky top-0 z-30 border-b border-slate-200/80 bg-white/90 backdrop-blur-xl">
        <div className="mx-auto flex max-w-7xl items-center justify-between gap-4 px-4 py-3 sm:px-6 lg:px-8">
          <Link
            href="/learn"
            className="flex min-w-0 items-center gap-3 rounded-xl focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:ring-offset-2"
          >
            <span className="grid size-10 shrink-0 place-items-center rounded-2xl bg-gradient-to-br from-indigo-600 to-violet-600 text-white shadow-md shadow-indigo-200">
              <BookIcon className="size-5" />
            </span>
            <span className="min-w-0">
              <span className="block truncate text-sm font-bold tracking-tight text-slate-900 sm:text-base">Course Coach</span>
              <span className="hidden text-[11px] text-slate-400 sm:block">可测量学习闭环</span>
            </span>
          </Link>

          <nav className="flex items-center gap-1 rounded-2xl border border-slate-200 bg-slate-50/80 p-1" aria-label="主导航">
            {NAV_ITEMS.map((item) => {
              const active = item.href.startsWith("/?")
                ? pathname === "/"
                : item.href === "/learn"
                  ? pathname.startsWith("/learn")
                  : pathname === item.href;
              const Icon = item.icon;
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  aria-current={active ? "page" : undefined}
                  className={`inline-flex items-center gap-2 rounded-xl px-2.5 py-2 text-xs font-semibold transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 sm:px-3 ${
                    active
                      ? "bg-white text-indigo-700 shadow-sm"
                      : "text-slate-500 hover:bg-white/70 hover:text-slate-800"
                  }`}
                >
                  <Icon className="size-4" />
                  <span className="hidden md:inline">{item.label}</span>
                </Link>
              );
            })}
          </nav>

          <div className="hidden min-w-28 justify-end sm:flex">
            <span
              className={`inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-[11px] font-medium ${
                status === "ready"
                  ? "border-emerald-200 bg-emerald-50 text-emerald-700"
                  : status === "error"
                    ? "border-rose-200 bg-rose-50 text-rose-700"
                    : "border-slate-200 bg-slate-50 text-slate-500"
              }`}
            >
              <span className={`size-1.5 rounded-full ${status === "ready" ? "bg-emerald-500" : status === "error" ? "bg-rose-500" : "animate-pulse bg-slate-400"}`} />
              {status === "ready" ? "学习记录已连接" : status === "error" ? "连接失败" : "正在连接"}
            </span>
          </div>
        </div>
      </header>

      {error && (
        <div className="mx-auto max-w-7xl px-4 pt-4 sm:px-6 lg:px-8">
          <div className="flex items-start gap-3 rounded-2xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-800" role="alert">
            <AlertIcon className="mt-0.5 size-4 shrink-0" />
            <span className="flex-1">{error}</span>
            <button
              type="button"
              onClick={() => void retryInitialization()}
              className="shrink-0 rounded-lg px-2 py-1 text-xs font-semibold text-rose-700 transition hover:bg-rose-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-rose-500"
            >
              重试
            </button>
          </div>
        </div>
      )}

      {children}
    </main>
  );
}
