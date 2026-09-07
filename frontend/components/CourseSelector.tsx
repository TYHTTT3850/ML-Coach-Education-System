"use client";

import { COURSE_META, COURSES, type Course } from "@/lib/types";

interface CourseSelectorProps {
  value: Course;
  onChange: (course: Course) => void;
  compact?: boolean;
  className?: string;
}

export default function CourseSelector({
  value,
  onChange,
  compact = false,
  className = "",
}: CourseSelectorProps) {
  return (
    <div
      className={`grid grid-cols-2 gap-2 ${compact ? "" : "lg:grid-cols-1"} ${className}`}
      role="radiogroup"
      aria-label="选择课程"
    >
      {COURSES.map((course) => {
        const meta = COURSE_META[course];
        const active = value === course;
        return (
          <button
            key={course}
            type="button"
            role="radio"
            aria-checked={active}
            onClick={() => onChange(course)}
            className={`group flex min-w-0 items-center gap-3 rounded-2xl border px-3 py-3 text-left transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:ring-offset-2 ${
              active
                ? "border-indigo-200 bg-indigo-50 text-indigo-950 shadow-sm"
                : "border-transparent text-slate-600 hover:border-slate-200 hover:bg-white hover:text-slate-900"
            }`}
          >
            <span
              className={`grid size-9 shrink-0 place-items-center rounded-xl text-sm font-bold transition ${
                active
                  ? course === "machine_learning"
                    ? "bg-indigo-600 text-white"
                    : "bg-cyan-600 text-white"
                  : "bg-slate-100 text-slate-500 group-hover:bg-slate-200"
              }`}
            >
              {meta.shortLabel}
            </span>
            <span className="min-w-0">
              <span className="block truncate text-sm font-semibold">{meta.label}</span>
              {!compact && (
                <span className="mt-0.5 block truncate text-xs text-slate-500">
                  {meta.description}
                </span>
              )}
            </span>
          </button>
        );
      })}
    </div>
  );
}
