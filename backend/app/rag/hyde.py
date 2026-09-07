from __future__ import annotations

import json
import logging
import re

from app.core.types import Course
from app.services.llm.base import LLMClient


logger = logging.getLogger(__name__)

MAX_HYPOTHETICAL_ANSWERS = 3
MAX_HYPOTHETICAL_ANSWER_LENGTH = 280


class HyDEGenerator:
    """Generate short hypothetical answers solely to expand retrieval queries."""

    def __init__(self, llm_client: LLMClient, *, temperature: float = 0.2) -> None:
        self.llm_client = llm_client
        self.temperature = min(max(temperature, 0.0), 0.3)

    async def generate(
        self,
        *,
        course: Course,
        question: str,
        history: list[dict[str, str]],
    ) -> list[str]:
        messages = self._messages(course=course, question=question, history=history)
        try:
            response = await self.llm_client.generate(
                messages,
                temperature=self.temperature,
            )
        except Exception as exc:
            # HyDE is an optional retrieval enhancement. Its failure must never
            # prevent retrieval with the original student question.
            logger.warning("HyDE 生成失败，已回退原查询: %s", type(exc).__name__)
            return []
        hypotheses = self.parse(response)
        if not hypotheses:
            logger.warning("HyDE 返回格式无效，已回退原查询")
        return hypotheses

    @staticmethod
    def _messages(
        *,
        course: Course,
        question: str,
        history: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        recent_user_questions = [
            item.get("content", "").strip()
            for item in history
            if item.get("role") == "user" and item.get("content", "").strip()
        ][-2:]
        history_hint = "；".join(recent_user_questions) or "无"
        return [
            {
                "role": "system",
                "content": (
                    "你是课程检索查询扩展器。请针对学生问题生成恰好 3 条彼此互补、"
                    "简短且可能出现在课程教材中的假设答案。它们只用于 HyDE 检索，"
                    "不会展示给学生。不得输出解释、Markdown、来源编号或其他字段。"
                    "只输出 JSON 字符串数组，例如 [\"假设答案一\", \"假设答案二\", \"假设答案三\"]。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"课程：{course.display_name}\n"
                    f"最近相关问题：{history_hint}\n"
                    f"当前问题：{question.strip()}"
                ),
            },
        ]

    @classmethod
    def parse(cls, response: str) -> list[str]:
        text = (response or "").strip()
        if not text or len(text) > 5000:
            return []
        fenced = re.fullmatch(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
        if fenced:
            text = fenced.group(1).strip()

        try:
            decoded = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return []
        if not isinstance(decoded, list) or len(decoded) != MAX_HYPOTHETICAL_ANSWERS:
            return []
        normalized: list[str] = []
        seen: set[str] = set()
        for value in decoded:
            if not isinstance(value, str):
                return []
            item = re.sub(r"\s+", " ", value).strip().strip("\"'")
            if not item or len(item) > MAX_HYPOTHETICAL_ANSWER_LENGTH:
                return []
            identity = item.casefold()
            if identity not in seen:
                normalized.append(item)
                seen.add(identity)
        return normalized if len(normalized) == MAX_HYPOTHETICAL_ANSWERS else []
