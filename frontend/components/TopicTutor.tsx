"use client";

import { useEffect, useRef, useState, type FormEvent } from "react";
import { AlertIcon, MessageIcon, SendIcon, SparklesIcon } from "@/components/Icons";
import MarkdownRenderer from "@/components/MarkdownRenderer";
import SourcePanel from "@/components/SourcePanel";
import ModelSelector from "@/components/ModelSelector";
import { useStudent } from "@/components/StudentProvider";
import { sendChat } from "@/lib/api";
import type { ChatMessage, Course } from "@/lib/types";

function messageId(): string {
  return typeof crypto !== "undefined" && "randomUUID" in crypto
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

export default function TopicTutor({
  course,
  topicId,
  topicTitle,
}: {
  course: Course;
  topicId: string;
  topicTitle: string;
}) {
  const { withStudent } = useStudent();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  useEffect(() => () => abortRef.current?.abort(), []);

  async function ask(content: string, mode: "free_qa" | "topic_teaching" | "worked_example") {
    if (!content.trim() || loading) return;
    const userMessage: ChatMessage = {
      id: messageId(),
      role: "user",
      content: content.trim(),
      createdAt: new Date().toISOString(),
    };
    const previous = messages;
    setMessages((current) => [...current, userMessage]);
    setQuestion("");
    setError(null);
    setLoading(true);
    const controller = new AbortController();
    abortRef.current = controller;
    try {
      const response = await withStudent((studentId) =>
        sendChat(
          {
            course,
            message: userMessage.content,
            topic_id: topicId,
            learning_mode: mode,
            history: previous.slice(-8).map(({ role, content: messageContent }) => ({
              role,
              content: messageContent,
            })),
          },
          controller.signal,
          studentId,
        ),
      );
      setMessages((current) => [
        ...current,
        {
          id: messageId(),
          role: "assistant",
          content: response.answer,
          sources: response.sources,
          createdAt: new Date().toISOString(),
        },
      ]);
    } catch (cause) {
      if (!(cause instanceof DOMException && cause.name === "AbortError")) {
        setError(cause instanceof Error ? cause.message : "Tutor 暂时无法回答，请稍后重试。");
      }
    } finally {
      setLoading(false);
      abortRef.current = null;
    }
  }

  function handleSubmit(event: FormEvent) {
    event.preventDefault();
    void ask(question, "free_qa");
  }

  return (
    <section id="topic-tutor" className="scroll-mt-24 rounded-3xl border border-slate-200 bg-white shadow-soft">
      <div className="flex flex-col gap-4 border-b border-slate-100 p-5 sm:flex-row sm:items-center sm:justify-between sm:p-6">
        <div className="flex items-center gap-3">
          <span className="grid size-10 place-items-center rounded-2xl bg-indigo-50 text-indigo-700">
            <MessageIcon className="size-5" />
          </span>
          <div>
            <h2 className="text-base font-bold text-slate-900">AI 例题与课程答疑</h2>
            <p className="mt-0.5 text-xs leading-5 text-slate-500">围绕“{topicTitle}”检索真实课程资料；点击后才生成，不计入考试成绩</p>
            <div className="mt-2"><ModelSelector feature="teaching" label="讲解模型" /></div>
          </div>
        </div>
        <div className="flex flex-wrap gap-2">
        <button
          type="button"
          onClick={() => void ask(`请根据课程资料，为“${topicTitle}”出 2 道由浅入深的教学例题。每道都给出具体题干与已知条件、所求问题、分步思路、代入计算或具体推理、结论及易错点。先展示题干，再讲解答案，最后留一个简短的举一反三问题让我自己尝试。`, "worked_example")}
          disabled={loading}
          className="inline-flex w-fit items-center gap-2 rounded-xl bg-indigo-600 px-4 py-2.5 text-sm font-semibold text-white shadow-sm transition hover:bg-indigo-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:ring-offset-2 disabled:cursor-wait disabled:opacity-60"
        >
          <SparklesIcon className="size-4" />
          AI 讲例题
        </button>
        <button
          type="button"
          onClick={() => void ask(`请围绕“${topicTitle}”的学习目标进行结构化讲解，并说明关键概念、常见误区和一个例子。`, "topic_teaching")}
          disabled={loading}
          className="inline-flex w-fit items-center gap-2 rounded-xl border border-indigo-200 bg-indigo-50 px-4 py-2.5 text-sm font-semibold text-indigo-700 transition hover:bg-indigo-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:ring-offset-2 disabled:cursor-wait disabled:opacity-60"
        >
          <SparklesIcon className="size-4" />
          进入知识讲解
        </button>
        </div>
      </div>

      <div className="max-h-[560px] space-y-5 overflow-y-auto p-5 sm:p-6">
        {messages.length === 0 ? (
          <div className="rounded-2xl border border-dashed border-slate-200 bg-slate-50/70 px-5 py-8 text-center">
            <SparklesIcon className="mx-auto size-6 text-indigo-400" />
            <p className="mt-3 text-sm font-semibold text-slate-700">让 AI 讲两道例题，边看步骤边理解</p>
            <p className="mt-1 text-xs leading-5 text-slate-400">也可以先听知识讲解，或继续追问某一步为什么这样做。资料不足时会提示补充资料。</p>
          </div>
        ) : (
          messages.map((message) => (
            <article key={message.id} className={message.role === "user" ? "ml-auto max-w-[88%]" : "max-w-full"}>
              {message.role === "user" ? (
                <div className="rounded-2xl rounded-br-md bg-indigo-600 px-4 py-3 text-sm leading-6 text-white">{message.content}</div>
              ) : (
                <div className="rounded-2xl rounded-tl-md border border-slate-200 bg-white px-4 py-4 shadow-sm sm:px-5">
                  <MarkdownRenderer content={message.content} />
                  {message.sources && message.sources.length > 0 && (
                    <details className="mt-4 border-t border-slate-100 pt-3">
                      <summary className="cursor-pointer text-xs font-semibold text-indigo-600">查看 {message.sources.length} 条课程资料来源</summary>
                      <div className="mt-3"><SourcePanel sources={message.sources} /></div>
                    </details>
                  )}
                </div>
              )}
            </article>
          ))
        )}
        {loading && (
          <div className="flex items-center gap-2 text-xs text-slate-500" role="status">
            <span className="size-2 animate-pulse rounded-full bg-indigo-500" /> Tutor 正在检索课程资料并组织回答…
          </div>
        )}
      </div>

      <form onSubmit={handleSubmit} className="border-t border-slate-100 p-4 sm:p-5">
        {error && (
          <div className="mb-3 flex items-start gap-2 rounded-xl border border-rose-200 bg-rose-50 px-3 py-2 text-xs text-rose-700" role="alert">
            <AlertIcon className="mt-0.5 size-3.5 shrink-0" />
            <span>{error}</span>
          </div>
        )}
        <div className="flex items-end gap-2 rounded-2xl border border-slate-200 bg-slate-50 p-2 transition focus-within:border-indigo-300 focus-within:bg-white focus-within:ring-4 focus-within:ring-indigo-100/60">
          <label htmlFor="topic-question" className="sr-only">围绕当前知识点自由提问</label>
          <textarea
            id="topic-question"
            rows={2}
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder={`围绕“${topicTitle}”自由提问…`}
            disabled={loading}
            className="max-h-40 min-h-12 flex-1 resize-y bg-transparent px-2 py-2 text-sm leading-6 text-slate-800 outline-none placeholder:text-slate-400 disabled:opacity-60"
          />
          <button
            type="submit"
            disabled={loading || !question.trim()}
            className="grid size-10 shrink-0 place-items-center rounded-xl bg-slate-900 text-white transition hover:bg-slate-800 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-slate-500 disabled:cursor-not-allowed disabled:opacity-35"
            aria-label="发送问题"
          >
            <SendIcon className="size-4" />
          </button>
        </div>
      </form>
    </section>
  );
}
