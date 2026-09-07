from app.core.types import Course
from app.rag.chunker import StructuredChunker
from app.rag.loaders import DocumentLoader, ParsedDocument, ParsedSection


def test_markdown_headings_become_chunk_metadata(tmp_path) -> None:
    path = tmp_path / "lesson.md"
    path.write_text(
        "# 分类模型\n\n导言。\n\n## 逻辑回归\n\n交叉熵用于分类训练。\n\n### 决策边界\n\n边界由概率阈值确定。",
        encoding="utf-8",
    )
    document = DocumentLoader().load(path)
    chunks = StructuredChunker(chunk_size=200, overlap=20).chunk_document(
        document,
        document_id="doc_1",
        course=Course.MACHINE_LEARNING,
        file_hash="abc",
    )
    assert [chunk.metadata["title"] for chunk in chunks] == [
        "分类模型",
        "逻辑回归",
        "决策边界",
    ]
    assert chunks[1].metadata["chapter"] == "分类模型"
    assert chunks[1].metadata["section"] == "逻辑回归"


def test_long_text_has_overlap() -> None:
    text = "".join(f"第{i}句用于验证分块重叠。" for i in range(100))
    document = ParsedDocument("long.txt", [ParsedSection(text=text)])
    chunks = StructuredChunker(chunk_size=220, overlap=50).chunk_document(
        document,
        document_id="doc_long",
        course=Course.ADVANCED_MATHEMATICS,
        file_hash="hash",
    )
    assert len(chunks) > 2
    for left, right in zip(chunks, chunks[1:], strict=False):
        assert left.text[-25:] in right.text[:80]


def test_empty_document_produces_no_chunks() -> None:
    document = ParsedDocument("empty.txt", [ParsedSection(text=" \n\t")])
    chunks = StructuredChunker(chunk_size=200, overlap=20).chunk_document(
        document,
        document_id="doc_empty",
        course=Course.MACHINE_LEARNING,
        file_hash="empty",
    )
    assert chunks == []


def test_inline_formula_is_not_split_at_chunk_boundary() -> None:
    formula = r"$f'(x)=\lim_{h\to0}\frac{f(x+h)-f(x)}{h}$"
    text = "甲" * 70 + formula + "乙" * 100
    document = ParsedDocument("formula.md", [ParsedSection(text=text)])
    chunks = StructuredChunker(chunk_size=80, overlap=15).chunk_document(
        document,
        document_id="doc_formula",
        course=Course.ADVANCED_MATHEMATICS,
        file_hash="formula",
    )
    assert sum(formula in chunk.text for chunk in chunks) == 1
    assert all("$" not in chunk.text or formula in chunk.text for chunk in chunks)


def test_oversize_fenced_code_block_remains_atomic() -> None:
    fenced = "```python\n" + "value = value + 1\n" * 50 + "```"
    text = "代码示例：\n" + fenced + "\n示例结束。" + "尾" * 120
    document = ParsedDocument("code.md", [ParsedSection(text=text)])
    chunks = StructuredChunker(chunk_size=120, overlap=20).chunk_document(
        document,
        document_id="doc_code",
        course=Course.MACHINE_LEARNING,
        file_hash="code",
    )
    assert any(fenced in chunk.text for chunk in chunks)
    assert sum("```" in chunk.text for chunk in chunks) == 1
