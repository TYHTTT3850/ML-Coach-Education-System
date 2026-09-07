"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { createAnonymousStudent, isStudentNotFound } from "@/lib/api";

export const STUDENT_STORAGE_KEY = "course_coach_student_id";

type StudentStatus = "initializing" | "ready" | "error";

interface StudentContextValue {
  studentId: string | null;
  status: StudentStatus;
  error: string | null;
  withStudent: <T>(request: (studentId: string) => Promise<T>) => Promise<T>;
  retryInitialization: () => Promise<void>;
}

const StudentContext = createContext<StudentContextValue | null>(null);
let pendingStudentCreation: Promise<string> | null = null;

async function createAndPersistStudent(): Promise<string> {
  if (!pendingStudentCreation) {
    pendingStudentCreation = createAnonymousStudent()
      .then((student) => {
        if (!student.student_id) throw new Error("服务端没有返回匿名学生 ID。");
        try {
          window.localStorage.setItem(STUDENT_STORAGE_KEY, student.student_id);
        } catch {
          // The current session can still learn if browser storage is unavailable.
        }
        return student.student_id;
      })
      .finally(() => {
        pendingStudentCreation = null;
      });
  }
  return pendingStudentCreation;
}

export function StudentProvider({ children }: { children: ReactNode }) {
  const [studentId, setStudentId] = useState<string | null>(null);
  const [status, setStatus] = useState<StudentStatus>("initializing");
  const [error, setError] = useState<string | null>(null);
  const studentIdRef = useRef<string | null>(null);

  const rememberStudent = useCallback((id: string | null) => {
    studentIdRef.current = id;
    setStudentId(id);
  }, []);

  const initialize = useCallback(async () => {
    setStatus("initializing");
    setError(null);
    try {
      let storedId: string | null = null;
      try {
        storedId = window.localStorage.getItem(STUDENT_STORAGE_KEY);
      } catch {
        // Fall through to a session-only anonymous student.
      }
      const id = storedId?.trim() || (await createAndPersistStudent());
      rememberStudent(id);
      setStatus("ready");
    } catch (cause) {
      rememberStudent(null);
      setStatus("error");
      setError(cause instanceof Error ? cause.message : "匿名学习身份创建失败，请稍后重试。");
    }
  }, [rememberStudent]);

  useEffect(() => {
    const timer = window.setTimeout(() => void initialize(), 0);
    return () => window.clearTimeout(timer);
  }, [initialize]);

  const ensureStudent = useCallback(async (): Promise<string> => {
    const current = studentIdRef.current;
    if (current) return current;
    const id = await createAndPersistStudent();
    rememberStudent(id);
    setStatus("ready");
    setError(null);
    return id;
  }, [rememberStudent]);

  const withStudent = useCallback(
    async <T,>(request: (id: string) => Promise<T>): Promise<T> => {
      const id = await ensureStudent();
      try {
        return await request(id);
      } catch (cause) {
        if (!isStudentNotFound(cause)) throw cause;

        // The backend is authoritative. Replace a stale local ID and retry once.
        try {
          window.localStorage.removeItem(STUDENT_STORAGE_KEY);
        } catch {
          // Continue with a new session-only identity.
        }
        rememberStudent(null);
        const replacementId = await createAndPersistStudent();
        rememberStudent(replacementId);
        setStatus("ready");
        setError(null);
        return request(replacementId);
      }
    },
    [ensureStudent, rememberStudent],
  );

  const value = useMemo<StudentContextValue>(
    () => ({
      studentId,
      status,
      error,
      withStudent,
      retryInitialization: initialize,
    }),
    [error, initialize, status, studentId, withStudent],
  );

  return <StudentContext.Provider value={value}>{children}</StudentContext.Provider>;
}

export function useStudent(): StudentContextValue {
  const context = useContext(StudentContext);
  if (!context) throw new Error("useStudent 必须在 StudentProvider 内使用。");
  return context;
}
