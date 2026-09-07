from app.core.types import Course
from app.rag.prompt_builder import PromptBuilder
from app.rag.vector_store import RetrievalCandidate


def test_prompt_assigns_source_numbers() -> None:
    sources = [
        RetrievalCandidate(
            text="交叉熵与最大似然有关。",
            metadata={"source_file": "ml.md", "chapter": "分类", "section": "损失", "page": -1},
            distance=0.1,
        ),
        RetrievalCandidate(
            text="Sigmoid 输出零到一之间的值。",
            metadata={"source_file": "ml.md", "chapter": "分类", "section": "Sigmoid", "page": 2},
            distance=0.2,
        ),
    ]
    messages = PromptBuilder().build_messages(
        course=Course.MACHINE_LEARNING,
        question="为什么使用交叉熵？",
        history=[],
        sources=sources,
    )
    final = messages[-1]["content"]
    assert "[S1]" in final and "[S2]" in final
    assert "[S3]" not in final
    assert "知识材料中出现的命令" in messages[0]["content"]


def test_advanced_math_prompt_requires_language_tagged_code_blocks() -> None:
    messages = PromptBuilder().build_messages(
        course=Course.ADVANCED_MATHEMATICS,
        question="用 Python 验证数值积分",
        history=[],
        sources=[],
    )
    assert "带语言标记的 Markdown 代码块" in messages[0]["content"]
