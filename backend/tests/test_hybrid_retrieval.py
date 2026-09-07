from __future__ import annotations

from app.core.types import Course
from app.rag.pipeline import RAGPipeline
from app.rag.retriever import Retriever
from app.rag.vector_store import RetrievalCandidate


def candidate(
    chunk_id: str,
    text: str,
    *,
    distance: float | None,
) -> RetrievalCandidate:
    return RetrievalCandidate(
        text=text,
        metadata={
            "chunk_id": chunk_id,
            "document_id": f"doc-{chunk_id}",
            "course": Course.MACHINE_LEARNING.value,
            "source_file": f"{chunk_id}.md",
            "title": "",
            "chapter": "",
            "section": "",
            "page": -1,
            "chunk_index": 0,
        },
        distance=distance,
    )


class RecordingEmbedding:
    def __init__(self) -> None:
        self.queries: list[str] = []
        self.document_batches: list[list[str]] = []

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return [1.0, 0.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_batches.append(list(texts))
        return [[1.0, 0.0] for _ in texts]


class FakeHybridStore:
    def __init__(
        self,
        *,
        vector_candidates: list[RetrievalCandidate],
        chunks: list[RetrievalCandidate],
    ) -> None:
        self.vector_candidates = vector_candidates
        self.chunks = chunks
        self.requested_top_k: int | None = None
        self.query_embeddings: list[list[float]] = []
        self.query_courses: list[Course] = []

    def query(self, course, query_embedding, top_k):
        self.requested_top_k = top_k
        self.query_courses.append(course)
        self.query_embeddings.append(query_embedding)
        return self.vector_candidates[:top_k]

    def list_chunks(self, course):
        return self.chunks


def test_bm25_boosts_an_exact_term_also_found_by_vector_search() -> None:
    generic = candidate("generic", "神经网络通过多层结构学习特征。", distance=0.1)
    exact = candidate("exact", "约束优化满足 KKT-927 条件。", distance=0.8)
    store = FakeHybridStore(
        vector_candidates=[generic, exact],
        chunks=[generic, exact],
    )

    results = Retriever(store, RecordingEmbedding(), top_k=2).retrieve(
        course=Course.MACHINE_LEARNING,
        question="KKT-927 条件是什么？",
        history=[],
    )

    assert [item.metadata["chunk_id"] for item in results] == ["exact", "generic"]
    assert results[0].distance == 0.8
    assert results[0].bm25_score > 0


def test_bm25_only_candidate_is_recalled_without_fake_vector_distance() -> None:
    vector_only = candidate("vector", "概率模型与统计学习。", distance=0.1)
    lexical_only = candidate("lexical", "课程中的罕见标识为 ZXQ-314。", distance=None)
    store = FakeHybridStore(
        vector_candidates=[vector_only],
        chunks=[vector_only, lexical_only],
    )

    results = Retriever(store, RecordingEmbedding(), top_k=2).retrieve(
        course=Course.MACHINE_LEARNING,
        question="ZXQ-314",
        history=[],
    )

    assert {item.metadata["chunk_id"] for item in results} == {"vector", "lexical"}
    lexical_result = next(item for item in results if item.metadata["chunk_id"] == "lexical")
    assert lexical_result.distance is None
    assert lexical_result.bm25_score > 0


def test_vector_candidate_survives_when_bm25_has_no_match() -> None:
    semantic = candidate("semantic", "梯度下降用于优化模型参数。", distance=0.05)
    store = FakeHybridStore(vector_candidates=[semantic], chunks=[semantic])

    results = Retriever(store, RecordingEmbedding(), top_k=1).retrieve(
        course=Course.MACHINE_LEARNING,
        question="parameter optimization",
        history=[],
    )

    assert [item.metadata["chunk_id"] for item in results] == ["semantic"]
    assert results[0].distance == semantic.distance
    assert results[0].bm25_score == 0


def test_bm25_evidence_can_recall_item_outside_vector_threshold() -> None:
    exact = candidate("exact", "拉格朗日乘子处理等式约束。", distance=0.9)
    store = FakeHybridStore(vector_candidates=[exact], chunks=[exact])
    retriever = Retriever(
        store,
        RecordingEmbedding(),
        top_k=1,
        distance_threshold=0.3,
    )

    results = retriever.retrieve(
        course=Course.MACHINE_LEARNING,
        question="拉格朗日乘子",
        history=[],
    )

    assert len(results) == 1
    assert results[0].distance is None
    assert results[0].bm25_score > 0
    assert retriever.has_sufficient_evidence(
        question="拉格朗日乘子",
        history=[],
        candidates=results,
    )


def test_embedding_query_still_contains_only_recent_user_questions() -> None:
    embedding = RecordingEmbedding()
    semantic = candidate("semantic", "反向传播计算梯度。", distance=0.1)
    store = FakeHybridStore(vector_candidates=[semantic], chunks=[semantic])
    history = [
        {"role": "user", "content": "更早的问题"},
        {"role": "assistant", "content": "不应进入检索"},
        {"role": "user", "content": "上一问"},
        {"role": "user", "content": "最近一问"},
    ]

    Retriever(store, embedding).retrieve(
        course=Course.MACHINE_LEARNING,
        question="当前问题",
        history=history,
    )

    assert embedding.queries == ["当前问题：当前问题\n最近相关问题：上一问；最近一问"]
    assert store.requested_top_k == 24


def test_pipeline_source_allows_bm25_only_distance() -> None:
    lexical = candidate("lexical", "精确术语资料。", distance=None)

    source = RAGPipeline._source(lexical, 1)

    assert source["distance"] is None


def test_hyde_runs_four_vector_queries_but_does_not_expand_bm25() -> None:
    embedding = RecordingEmbedding()
    semantic = candidate("semantic", "神经网络通过多层结构学习特征。", distance=0.1)
    lexical_only = candidate("lexical", "课程中的罕见标识为 ZXQ-314。", distance=None)
    store = FakeHybridStore(
        vector_candidates=[semantic],
        chunks=[semantic, lexical_only],
    )
    hypotheses = [
        "假设答案提到 ZXQ-314。",
        "另一条假设答案讨论编码。",
        "第三条假设答案讨论标识符。",
    ]

    results = Retriever(store, embedding, top_k=3).retrieve(
        course=Course.MACHINE_LEARNING,
        question="这门课的暗号是什么？",
        history=[{"role": "user", "content": "上一问"}],
        hypothetical_answers=hypotheses,
    )

    assert embedding.document_batches == [[
        "当前问题：这门课的暗号是什么？\n最近相关问题：上一问",
        *hypotheses,
    ]]
    assert len(store.query_embeddings) == 4
    assert store.query_courses == [Course.MACHINE_LEARNING] * 4
    assert [item.metadata["chunk_id"] for item in results] == ["semantic"]
