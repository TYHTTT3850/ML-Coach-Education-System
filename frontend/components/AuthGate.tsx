"use client";

import { useState, type FormEvent, type ReactNode } from "react";
import { loginStudent, registerStudent } from "@/lib/api";
import { useStudent } from "@/components/StudentProvider";

export default function AuthGate({ children }: { children: ReactNode }) {
  const { status, error, completeLogin } = useStudent();
  const [registering, setRegistering] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [formError, setFormError] = useState("");

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setSubmitting(true); setFormError("");
    const data = new FormData(event.currentTarget);
    try {
      const account = registering
        ? await registerStudent({ username: String(data.get("username")), password: String(data.get("password")), display_name: String(data.get("display_name")) })
        : await loginStudent({ username: String(data.get("username")), password: String(data.get("password")) });
      completeLogin(account);
    } catch (cause) {
      setFormError(cause instanceof Error ? cause.message : "暂时无法完成登录，请稍后重试。");
    } finally { setSubmitting(false); }
  }

  if (status === "ready") return <>{children}</>;
  if (status === "initializing") return <main className="grid min-h-svh place-items-center bg-canvas text-sm text-slate-500">正在恢复登录状态...</main>;
  return <main className="app-grid-bg grid min-h-svh place-items-center p-4">
    <section className="w-full max-w-md border border-slate-200 bg-white p-6 shadow-xl sm:p-8">
      <p className="text-xs font-bold uppercase tracking-[0.16em] text-indigo-600">Course Coach</p>
      <h1 className="mt-3 text-2xl font-bold text-slate-950">{registering ? "创建学习账户" : "登录学习空间"}</h1>
      <form className="mt-7 space-y-4" onSubmit={submit}>
        {registering && <label className="block text-sm font-medium text-slate-700">姓名<input required name="display_name" maxLength={64} className="mt-1.5 w-full border border-slate-300 px-3 py-2.5 outline-none focus:border-indigo-500" /></label>}
        <label className="block text-sm font-medium text-slate-700">用户名<input required name="username" minLength={3} maxLength={32} pattern="[A-Za-z0-9_.-]+" autoComplete="username" className="mt-1.5 w-full border border-slate-300 px-3 py-2.5 outline-none focus:border-indigo-500" /></label>
        <label className="block text-sm font-medium text-slate-700">密码<input required name="password" type="password" minLength={8} maxLength={128} autoComplete={registering ? "new-password" : "current-password"} className="mt-1.5 w-full border border-slate-300 px-3 py-2.5 outline-none focus:border-indigo-500" /></label>
        {(formError || error) && <p role="alert" className="text-sm text-rose-700">{formError || error}</p>}
        <button disabled={submitting} type="submit" className="w-full bg-indigo-600 px-4 py-2.5 text-sm font-semibold text-white transition hover:bg-indigo-700 disabled:cursor-wait disabled:opacity-60">{submitting ? "正在处理..." : registering ? "创建并登录" : "登录"}</button>
      </form>
      <button type="button" onClick={() => { setRegistering((value) => !value); setFormError(""); }} className="mt-5 text-sm font-semibold text-indigo-700 hover:text-indigo-900">{registering ? "已有账户？登录" : "没有账户？创建账户"}</button>
    </section>
  </main>;
}