from app.core.types import Course
from app.rag.chunker import Chunk
from app.rag.retriever import Retriever
from app.rag.vector_store import ChromaVectorStore
from tests.mocks import MockEmbeddingClient


def make_chunk(course: Course, document_id: str, text: str) -> Chunk:
    chunk_id = f"{course.value}_{document_id}"
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        metadata={
            "chunk_id": chunk_id,
            "document_id": document_id,
            "course": course.value,
            "source_file": f"{document_id}.md",
            "title": "标题",
            "chapter": "章节",
            "section": "小节",
            "page": -1,
            "chunk_index": 0,
            "file_hash": "hash",
        },
    )


def test_course_collections_are_isolated_and_metadata_complete(tmp_path) -> None:
    embedding = MockEmbeddingClient()
    store = ChromaVectorStore(tmp_path / "chroma")
    store.initialize()
    ml_chunk = make_chunk(Course.MACHINE_LEARNING, "ml", "逻辑回归使用交叉熵。")
    math_chunk = make_chunk(Course.ADVANCED_MATHEMATICS, "math", "导数表示切线斜率。")
    store.upsert(Course.MACHINE_LEARNING, [ml_chunk], embedding.embed_documents([ml_chunk.text]))
    store.upsert(
        Course.ADVANCED_MATHEMATICS,
        [math_chunk],
        embedding.embed_documents([math_chunk.text]),
    )

    retriever = Retriever(store, embedding, top_k=6)
    ml_results = retriever.retrieve(
        course=Course.MACHINE_LEARNING,
        question="导数和逻辑回归是什么？",
        history=[],
    )
    math_results = retriever.retrieve(
        course=Course.ADVANCED_MATHEMATICS,
        question="导数和逻辑回归是什么？",
        history=[],
    )
    assert {item.metadata["document_id"] for item in ml_results} == {"ml"}
    assert {item.metadata["document_id"] for item in math_results} == {"math"}
    required = {
        "chunk_id",
        "document_id",
        "course",
        "source_file",
        "title",
        "chapter",
        "section",
        "page",
        "chunk_index",
        "file_hash",
    }
    assert required <= ml_results[0].metadata.keys()


def test_bm25_chunks_are_persistent_course_scoped_and_follow_deletes(tmp_path) -> None:
    embedding = MockEmbeddingClient()
    path = tmp_path / "hybrid-chroma"
    store = ChromaVectorStore(path)
    store.initialize()
    exact = make_chunk(
        Course.MACHINE_LEARNING,
        "exact",
        "约束优化中的罕见标识是 KKT-927。",
    )
    other_course = make_chunk(
        Course.ADVANCED_MATHEMATICS,
        "math",
        "拉格朗日中值定理属于高等数学。",
    )
    store.upsert(
        Course.MACHINE_LEARNING,
        [exact],
        embedding.embed_documents([exact.text]),
    )
    store.upsert(
        Course.ADVANCED_MATHEMATICS,
        [other_course],
        embedding.embed_documents([other_course.text]),
    )

    restarted = ChromaVectorStore(path)
    restarted.initialize()
    ml_chunks = restarted.list_chunks(Course.MACHINE_LEARNING)

    assert [item.metadata["document_id"] for item in ml_chunks] == ["exact"]
    assert ml_chunks[0].distance is None
    assert restarted.delete_document(Course.MACHINE_LEARNING, "exact") == 1
    assert restarted.list_chunks(Course.MACHINE_LEARNING) == []
    assert {
        item.metadata["document_id"]
        for item in restarted.list_chunks(Course.ADVANCED_MATHEMATICS)
    } == {"math"}
