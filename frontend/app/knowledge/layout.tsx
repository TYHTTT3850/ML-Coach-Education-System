import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "知识库管理",
  description: "上传和管理机器学习与高等数学课程资料",
};

export default function KnowledgeLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return children;
}
