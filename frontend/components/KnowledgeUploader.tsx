"use client";

import { useRef, useState, type DragEvent } from "react";
import {
  AlertIcon,
  CheckIcon,
  FileIcon,
  RefreshIcon,
  UploadIcon,
} from "@/components/Icons";
import { uploadDocument } from "@/lib/api";
import type { Course, KnowledgeDocument } from "@/lib/types";

const ACCEPTED_EXTENSIONS = [".md", ".txt", ".pdf", ".docx"];

interface KnowledgeUploaderProps {
  course: Course;
  adminToken: string;
  onUploaded?: (document: KnowledgeDocument) => void;
  onBatchComplete: () => void | Promise<void>;
}

type UploadStatus =
  | "queued"
  | "uploading"
  | "processing"
  | "success"
  | "duplicate"
  | "error";

interface UploadItem {
  id: string;
  file: File;
  status: UploadStatus;
  progress: number;
  message: string;
}

function isAccepted(file: File): boolean {
  const filename = file.name.toLowerCase();
  return ACCEPTED_EXTENSIONS.some((extension) => filename.endsWith(extension));
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
}

function statusMeta(status: UploadStatus): { label: string; className: string } {
  switch (status) {
    case "queued":
      return { label: "等待上传", className: "bg-slate-100 text-slate-600" };
    case "uploading":
      return { label: "上传中", className: "bg-indigo-50 text-indigo-700" };
    case "processing":
      return { label: "建立索引中", className: "bg-violet-50 text-violet-700" };
    case "success":
      return { label: "上传成功", className: "bg-emerald-50 text-emerald-700" };
    case "duplicate":
      return { label: "重复已跳过", className: "bg-amber-50 text-amber-700" };
    case "error":
      return { label: "上传失败", className: "bg-rose-50 text-rose-700" };
  }
}

export default function KnowledgeUploader({
  course,
  adminToken,
  onUploaded,
  onBatchComplete,
}: KnowledgeUploaderProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [batchRunning, setBatchRunning] = useState(false);
  const [items, setItems] = useState<UploadItem[]>([]);
  const [topicTags, setTopicTags] = useState("");

  function updateItem(id: string, changes: Partial<Omit<UploadItem, "id" | "file">>) {
    setItems((current) =>
      current.map((item) => (item.id === id ? { ...item, ...changes } : item)),
    );
  }

  async function beginBatch(files: File[]) {
    if (batchRunning || files.length === 0) return;

    // Keep one batch in the course/token selected when it starts, even if the
    // parent page is re-rendered while a long document is being indexed.
    const targetCourse = course;
    const targetAdminToken = adminToken;
    const targetTopics = topicTags.split(/[,，\s]+/).map((item) => item.trim()).filter(Boolean);
    const batchId = Date.now();
    const nextItems: UploadItem[] = files.map((file, index) => {
      const valid = isAccepted(file);
      return {
        id: `${batchId}-${index}-${file.lastModified}`,
        file,
        status: valid ? "queued" : "error",
        progress: 0,
        message: valid ? "等待上传" : "不支持该格式，仅支持 MD、TXT、PDF 和 DOCX。",
      };
    });

    setItems(nextItems);
    setBatchRunning(true);

    try {
      const uploadableItems = nextItems.filter((item) => item.status === "queued");
      if (!targetAdminToken.trim()) {
        setItems((current) =>
          current.map((item) =>
            item.status === "queued"
              ? { ...item, status: "error", message: "请先填写管理员 Token。" }
              : item,
          ),
        );
        return;
      }

      for (const item of uploadableItems) {
        updateItem(item.id, { status: "uploading", progress: 0, message: "正在上传…" });
        try {
          const result = await uploadDocument(
            targetCourse,
            item.file,
            targetAdminToken,
            (progress) => {
              updateItem(item.id, {
                status: progress < 100 ? "uploading" : "processing",
                progress,
                message: progress < 100 ? `正在上传 ${progress}%` : "正在解析并建立索引…",
              });
            },
            targetTopics,
          );
          if (result.duplicate) {
            updateItem(item.id, {
              status: "duplicate",
              progress: 100,
              message: "已跳过重复导入",
            });
          } else {
            updateItem(item.id, {
              status: "success",
              progress: 100,
              message: result.message || "文件已导入知识库",
            });
          }
          onUploaded?.(result.document);
        } catch (error) {
          updateItem(item.id, {
            status: "error",
            message: error instanceof Error ? error.message : "上传失败，请稍后重试。",
          });
        }
      }
    } finally {
      setBatchRunning(false);
      if (inputRef.current) inputRef.current.value = "";
      await onBatchComplete();
    }
  }

  function handleDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDragging(false);
    if (batchRunning) return;
    const files = Array.from(event.dataTransfer.files);
    if (files.length > 0) void beginBatch(files);
  }

  const completedCount = items.filter((item) =>
    ["success", "duplicate", "error"].includes(item.status),
  ).length;
  const successCount = items.filter((item) => item.status === "success").length;
  const duplicateCount = items.filter((item) => item.status === "duplicate").length;
  const errorCount = items.filter((item) => item.status === "error").length;

  return (
    <div>
      <label className="mb-4 block text-xs font-semibold text-slate-600">知识点标签（可选，多个 canonical topic_id 用逗号分隔）<input value={topicTags} onChange={(event) => setTopicTags(event.target.value)} disabled={batchRunning} placeholder="例如 la.dot_product" className="mt-2 block w-full rounded-xl border border-slate-200 px-3 py-2 text-sm font-normal" /><span className="mt-2 block text-[11px] font-normal leading-5 text-slate-400">标签必须属于当前课程。按前置知识点补学时只检索有明确标签的资料，不会扩大到整库。</span></label>
      <div
        onDragEnter={(event) => { event.preventDefault(); setDragging(true); }}
        onDragOver={(event) => event.preventDefault()}
        onDragLeave={(event) => {
          if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setDragging(false);
        }}
        onDrop={handleDrop}
        className={`relative rounded-2xl border-2 border-dashed px-6 py-9 text-center transition ${
          dragging
            ? "border-indigo-400 bg-indigo-50"
            : "border-slate-200 bg-slate-50/60 hover:border-indigo-300 hover:bg-indigo-50/40"
        } ${batchRunning ? "cursor-wait opacity-80" : "cursor-pointer"}`}
        role="button"
        tabIndex={batchRunning ? -1 : 0}
        aria-disabled={batchRunning}
        onClick={() => !batchRunning && inputRef.current?.click()}
        onKeyDown={(event) => {
          if (!batchRunning && (event.key === "Enter" || event.key === " ")) {
            event.preventDefault();
            inputRef.current?.click();
          }
        }}
      >
        <input
          ref={inputRef}
          type="file"
          multiple
          disabled={batchRunning}
          accept=".md,.txt,.pdf,.docx,text/markdown,text/plain,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document"
          className="sr-only"
          onChange={(event) => {
            const files = Array.from(event.target.files ?? []);
            if (files.length > 0) void beginBatch(files);
          }}
        />
        <span className="mx-auto grid size-12 place-items-center rounded-2xl bg-white text-indigo-600 shadow-sm">
          {batchRunning ? <RefreshIcon className="size-5 animate-spin" /> : <UploadIcon className="size-5" />}
        </span>
        <p className="mt-4 text-sm font-semibold text-slate-800">
          {batchRunning
            ? `正在顺序处理 ${Math.min(completedCount + 1, items.length)}/${items.length}`
            : "拖拽多个文件到这里，或点击选择"}
        </p>
        <p className="mt-1.5 text-xs text-slate-500">
          支持 .md、.txt、.pdf、.docx · 可一次选择多个文件
        </p>
      </div>

      {items.length > 0 && (
        <div className="mt-4 space-y-2" aria-live="polite">
          <div className="flex items-center justify-between px-1 text-[11px] text-slate-400">
            <span>本次共 {items.length} 个文件</span>
            <span>
              {batchRunning
                ? `已处理 ${completedCount}/${items.length}`
                : `成功 ${successCount} · 跳过 ${duplicateCount} · 失败 ${errorCount}`}
            </span>
          </div>
          {items.map((item) => {
            const meta = statusMeta(item.status);
            return (
              <div
                key={item.id}
                className="rounded-xl border border-slate-200 bg-white px-3 py-3 shadow-sm"
              >
                <div className="flex items-start gap-3">
                  <span className={`mt-0.5 grid size-8 shrink-0 place-items-center rounded-lg ${
                    item.status === "error"
                      ? "bg-rose-50 text-rose-600"
                      : item.status === "success"
                        ? "bg-emerald-50 text-emerald-600"
                        : "bg-slate-100 text-slate-500"
                  }`}>
                    {item.status === "error" ? (
                      <AlertIcon className="size-4" />
                    ) : item.status === "success" || item.status === "duplicate" ? (
                      <CheckIcon className="size-4" />
                    ) : (
                      <FileIcon className="size-4" />
                    )}
                  </span>
                  <div className="min-w-0 flex-1">
                    <div className="flex items-start justify-between gap-2">
                      <div className="min-w-0">
                        <p className="truncate text-xs font-semibold text-slate-700" title={item.file.name}>
                          {item.file.name}
                        </p>
                        <p className="mt-0.5 text-[10px] text-slate-400">{formatBytes(item.file.size)}</p>
                      </div>
                      <span className={`shrink-0 rounded-full px-2 py-1 text-[10px] font-semibold ${meta.className}`}>
                        {meta.label}
                      </span>
                    </div>
                    <p className={`mt-2 text-[11px] leading-4 ${item.status === "error" ? "text-rose-600" : "text-slate-500"}`}>
                      {item.message}
                    </p>
                    {(item.status === "uploading" || item.status === "processing") && (
                      <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-slate-100">
                        <div
                          className="h-full rounded-full bg-indigo-600 transition-all duration-300"
                          style={{ width: `${item.progress}%` }}
                        />
                      </div>
                    )}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {items.length === 0 && (
        <div className="mt-3 flex items-center gap-2 text-xs text-slate-400">
          <FileIcon className="size-3.5" />
          文件会按顺序上传；单个文件失败不会影响同批次的其他文件。
        </div>
      )}
    </div>
  );
}
