"use client";

import { useRef, useState, type KeyboardEvent } from "react";
import { SendIcon } from "@/components/Icons";

const MAX_LENGTH = 6000;

interface ChatInputProps {
  onSend: (message: string) => void;
  disabled?: boolean;
  placeholder?: string;
}

export default function ChatInput({ onSend, disabled = false, placeholder }: ChatInputProps) {
  const [value, setValue] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  function submit() {
    const normalized = value.trim();
    if (!normalized || disabled) return;
    onSend(normalized);
    setValue("");
    if (textareaRef.current) textareaRef.current.style.height = "auto";
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      submit();
    }
  }

  return (
    <div className="mx-auto w-full max-w-4xl">
      <div className="rounded-2xl border border-slate-200 bg-white p-2 shadow-lg shadow-slate-200/60 transition focus-within:border-indigo-300 focus-within:ring-4 focus-within:ring-indigo-100/70">
        <div className="flex items-end gap-2">
          <textarea
            ref={textareaRef}
            value={value}
            onChange={(event) => {
              setValue(event.target.value);
              event.currentTarget.style.height = "auto";
              event.currentTarget.style.height = `${Math.min(event.currentTarget.scrollHeight, 160)}px`;
            }}
            onKeyDown={handleKeyDown}
            maxLength={MAX_LENGTH}
            rows={1}
            disabled={disabled}
            placeholder={placeholder ?? "输入课程问题…"}
            aria-label="课程问题"
            className="max-h-40 min-h-11 flex-1 resize-none bg-transparent px-3 py-2.5 text-[15px] leading-6 text-slate-800 outline-none placeholder:text-slate-400 disabled:cursor-not-allowed disabled:opacity-60"
          />
          <button
            type="button"
            onClick={submit}
            disabled={disabled || !value.trim()}
            className="grid size-11 shrink-0 place-items-center rounded-xl bg-indigo-600 text-white shadow-sm transition hover:bg-indigo-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:bg-slate-200 disabled:text-slate-400 disabled:shadow-none"
            aria-label="发送问题"
          >
            <SendIcon className="size-4.5" />
          </button>
        </div>
        <div className="flex items-center justify-between px-3 pb-1 pt-1 text-[11px] text-slate-400">
          <span>Enter 发送 · Shift + Enter 换行</span>
          {value.length > MAX_LENGTH * 0.8 && <span>{value.length}/{MAX_LENGTH}</span>}
        </div>
      </div>
    </div>
  );
}
