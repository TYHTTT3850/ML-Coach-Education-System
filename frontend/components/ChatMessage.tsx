import MarkdownRenderer from "@/components/MarkdownRenderer";
import SourcePanel from "@/components/SourcePanel";
import { SparklesIcon } from "@/components/Icons";
import type { ChatMessage as ChatMessageType } from "@/lib/types";

function formatTime(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  return new Intl.DateTimeFormat("zh-CN", {
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

export default function ChatMessage({ message }: { message: ChatMessageType }) {
  const isUser = message.role === "user";

  if (isUser) {
    return (
      <article className="ml-auto flex max-w-[88%] animate-fade-up flex-col items-end sm:max-w-[78%]">
        <div className="rounded-2xl rounded-br-md bg-indigo-600 px-4 py-3 text-[15px] leading-6 text-white shadow-sm">
          <p className="whitespace-pre-wrap break-words">{message.content}</p>
        </div>
        <time className="mt-1.5 px-1 text-[11px] text-slate-400" dateTime={message.createdAt}>
          {formatTime(message.createdAt)}
        </time>
      </article>
    );
  }

  return (
    <article className="flex animate-fade-up items-start gap-3">
      <div className="mt-0.5 hidden size-9 shrink-0 place-items-center rounded-xl bg-gradient-to-br from-indigo-600 to-violet-600 text-white shadow-sm sm:grid">
        <SparklesIcon className="size-4.5" />
      </div>
      <div className="min-w-0 flex-1">
        <div className="rounded-2xl rounded-tl-md border border-slate-200 bg-white px-4 py-3.5 shadow-soft sm:px-5 sm:py-4">
          <div className="mb-2 flex items-center justify-between border-b border-slate-100 pb-2 sm:hidden">
            <span className="inline-flex items-center gap-1.5 text-xs font-semibold text-indigo-700">
              <SparklesIcon className="size-3.5" />
              Course Tutor
            </span>
          </div>
          <MarkdownRenderer content={message.content} />
        </div>
        <div className="mt-1.5 flex items-center gap-2 px-1 text-[11px] text-slate-400">
          <span>Course Tutor</span>
          <span aria-hidden="true">·</span>
          <time dateTime={message.createdAt}>{formatTime(message.createdAt)}</time>
        </div>

        {message.sources && (
          <details className="group mt-3 xl:hidden">
            <summary className="flex cursor-pointer list-none items-center gap-2 rounded-xl border border-indigo-100 bg-indigo-50/70 px-3 py-2 text-xs font-semibold text-indigo-700 transition hover:bg-indigo-50 [&::-webkit-details-marker]:hidden">
              查看本次回答的 {message.sources.length} 条资料来源
              <span className="ml-auto transition group-open:rotate-180" aria-hidden="true">⌄</span>
            </summary>
            <div className="mt-2">
              <SourcePanel sources={message.sources} compact />
            </div>
          </details>
        )}
      </div>
    </article>
  );
}
