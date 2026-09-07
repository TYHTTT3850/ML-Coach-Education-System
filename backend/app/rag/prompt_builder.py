from __future__ import annotations

from app.core.types import Course
from app.prompts.advanced_mathematics import ADVANCED_MATHEMATICS_TUTOR_PROMPT
from app.prompts.machine_learning import MACHINE_LEARNING_TUTOR_PROMPT
from app.rag.vector_store import RetrievalCandidate


class PromptBuilder:
    def build_messages(
        self,
        *,
        course: Course,
        question: str,
        history: list[dict[str, str]],
        sources: list[RetrievalCandidate],
        system_prompt: str | None = None,
    ) -> list[dict[str, str]]:
        selected_system_prompt = system_prompt or (
            MACHINE_LEARNING_TUTOR_PROMPT
            if course is Course.MACHINE_LEARNING
            else ADVANCED_MATHEMATICS_TUTOR_PROMPT
        )
        if course in (Course.LINEAR_ALGEBRA, Course.PROBABILITY_STATISTICS):
            selected_system_prompt = selected_system_prompt.replace("高等数学", course.display_name)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": selected_system_prompt}
        ]
        messages.extend(
            {"role": item["role"], "content": item["content"]}
            for item in history
            if item.get("role") in {"user", "assistant"} and item.get("content", "").strip()
        )
        context = self._format_sources(sources)
        messages.append(
            {
                "role": "user",
                "content": (
                    "下面是本次可用的课程资料。资料之间彼此独立，方括号编号是唯一可引用来源。\n\n"
                    f"{context}\n\n"
                    f"学生问题：{question}\n\n"
                    "请给出适合教学的中文回答，并在相关陈述后标注真实来源编号。"
                ),
            }
        )
        return messages

    @staticmethod
    def _format_sources(sources: list[RetrievalCandidate]) -> str:
        if not sources:
            return "（未提供课程资料）"
        blocks: list[str] = []
        for index, source in enumerate(sources, start=1):
            metadata = source.metadata
            location = [metadata.get("chapter", ""), metadata.get("section", "")]
            location_text = " / ".join(item for item in location if item) or "未标注章节"
            page = metadata.get("page", -1)
            page_text = f"；页码：{page}" if isinstance(page, int) and page > 0 else ""
            blocks.append(
                f"[S{index}]\n"
                f"文件：{metadata.get('source_file', '未知文件')}；章节：{location_text}{page_text}\n"
                f"内容：{source.text.strip()}"
            )
        return "\n\n".join(blocks)
