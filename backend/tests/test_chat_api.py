from fastapi.testclient import TestClient

from app.main import create_app
from app.core.types import Course
from app.rag.chunker import Chunk
from app.rag.pipeline import STRICT_FALLBACK_ANSWER
from app.services.llm.factory import LazyLLMClient
from tests.mocks import MockEmbeddingClient, MockLLMClient


class CloseableEmbeddingClient(MockEmbeddingClient):
    def __init__(self) -> None:
        super().__init__()
        self.closed = False

    def close(self) -> None:
        self.closed = True


class CloseableLLMClient(MockLLMClient):
    def __init__(self) -> None:
        super().__init__()
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def test_chat_api_validates_course_message_and_history(client) -> None:
    assert client.post(
        "/api/v1/chat",
        json={"course": "unknown", "message": "问题", "history": []},
    ).status_code == 422
    assert client.post(
        "/api/v1/chat",
        json={"course": "machine_learning", "message": "   ", "history": []},
    ).status_code == 422
    too_many = [{"role": "user", "content": str(index)} for index in range(13)]
    assert client.post(
        "/api/v1/chat",
        json={"course": "machine_learning", "message": "问题", "history": too_many},
    ).status_code == 422


def test_health_does_not_expose_secrets(client) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "api_key" not in response.text.lower()
    assert "test-admin-token" not in response.text


def test_strict_chat_fallback_does_not_require_llm_configuration(
    settings, mock_embedding
) -> None:
    application = create_app(settings, embedding_client=mock_embedding, llm_client=None)
    with TestClient(application) as no_llm_client:
        first_response = no_llm_client.post(
            "/api/v1/chat",
            json={
                "course": "machine_learning",
                "message": "知识库为空时怎么办？",
                "history": [],
            },
        )
        second_response = no_llm_client.post(
            "/api/v1/chat",
            json={
                "course": "machine_learning",
                "message": "仍然没有知识资料吗？",
                "history": [],
            },
        )
    assert first_response.status_code == 200, first_response.text
    assert first_response.json()["answer"] == STRICT_FALLBACK_ANSWER
    assert first_response.json()["retrieval_count"] == 0
    assert second_response.status_code == 200
    assert application.state.llm_client is None


def test_lazy_llm_proxy_is_cached_after_nonempty_retrieval(settings, mock_embedding) -> None:
    application = create_app(settings, embedding_client=mock_embedding, llm_client=None)
    chunk = Chunk(
        chunk_id="cached_llm_chunk",
        text="逻辑回归使用交叉熵进行训练。",
        metadata={
            "chunk_id": "cached_llm_chunk",
            "document_id": "cached_llm_doc",
            "course": Course.MACHINE_LEARNING.value,
            "source_file": "cached.md",
            "title": "逻辑回归",
            "chapter": "分类",
            "section": "交叉熵",
            "page": -1,
            "chunk_index": 0,
            "file_hash": "cached_hash",
        },
    )
    with TestClient(application) as no_llm_client:
        application.state.vector_store.upsert(
            Course.MACHINE_LEARNING,
            [chunk],
            mock_embedding.embed_documents([chunk.text]),
        )
        payload = {
            "course": "machine_learning",
            "message": "逻辑回归为什么使用交叉熵？",
            "history": [],
        }
        first = no_llm_client.post("/api/v1/chat", json=payload)
        cached_proxy = application.state.llm_client
        second = no_llm_client.post("/api/v1/chat", json=payload)
        assert application.state.llm_client is cached_proxy
    assert first.status_code == 503
    assert second.status_code == 503
    assert isinstance(cached_proxy, LazyLLMClient)


def test_strict_chat_falls_back_for_nonempty_but_irrelevant_library(
    client, mock_llm
) -> None:
    uploaded = client.post(
        "/api/v1/knowledge/upload",
        headers={"X-Admin-Token": "test-admin-token"},
        data={"course": "machine_learning"},
        files={
            "file": (
                "regression.md",
                "# 逻辑回归\n\n逻辑回归用 Sigmoid 输出分类概率，并以交叉熵训练。",
                "text/markdown",
            )
        },
    )
    assert uploaded.status_code == 200, uploaded.text
    response = client.post(
        "/api/v1/chat",
        json={
            "course": "machine_learning",
            "message": "如何证明拉格朗日中值定理？",
            "history": [],
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["answer"] == STRICT_FALLBACK_ANSWER
    assert response.json()["sources"] == []
    assert len(mock_llm.calls) == 1


def test_strict_chat_answers_when_nonempty_library_is_lexically_relevant(
    client, mock_llm
) -> None:
    uploaded = client.post(
        "/api/v1/knowledge/upload",
        headers={"X-Admin-Token": "test-admin-token"},
        data={"course": "machine_learning"},
        files={
            "file": (
                "regression.md",
                "# 逻辑回归\n\n逻辑回归使用 Sigmoid 输出分类概率，并使用交叉熵训练。",
                "text/markdown",
            )
        },
    )
    assert uploaded.status_code == 200, uploaded.text
    response = client.post(
        "/api/v1/chat",
        json={
            "course": "machine_learning",
            "message": "为什么逻辑回归采用交叉熵？",
            "history": [],
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["answer"] == "这是基于课程资料的测试回答。[S1]"
    assert response.json()["sources"][0]["source_id"] == "S1"
    assert len(mock_llm.calls) == 2


def test_topic_chat_enriches_existing_hybrid_retrieval_and_validates_course(
    client, mock_llm
) -> None:
    uploaded = client.post(
        "/api/v1/knowledge/upload",
        headers={"X-Admin-Token": "test-admin-token"},
        data={"course": "machine_learning"},
        files={
            "file": (
                "topic-chat.md",
                "# 逻辑回归\n\n逻辑回归使用 Sigmoid 和交叉熵完成二分类。",
                "text/markdown",
            )
        },
    )
    assert uploaded.status_code == 200, uploaded.text

    mismatch = client.post(
        "/api/v1/chat",
        json={
            "course": "advanced_mathematics",
            "topic_id": "ml.logistic_regression",
            "learning_mode": "topic_teaching",
            "message": "请讲解这个知识点",
            "history": [],
        },
    )
    assert mismatch.status_code == 422
    assert mismatch.json()["error"]["code"] == "COURSE_TOPIC_MISMATCH"

    response = client.post(
        "/api/v1/chat",
        json={
            "course": "machine_learning",
            "topic_id": "ml.logistic_regression",
            "learning_mode": "topic_teaching",
            "message": "逻辑回归为什么使用交叉熵？",
            "history": [],
        },
    )
    assert response.status_code == 200, response.text
    hyde_prompt = "\n".join(item["content"] for item in mock_llm.calls[-2])
    assert "知识点：逻辑回归" in hyde_prompt
    assert "检索用途：topic_teaching" in hyde_prompt
    assert "课程知识点 Tutor" in mock_llm.calls[-1][0]["content"]
    assert response.json()["sources"][0]["source_file"] == "topic-chat.md"


def test_topic_chat_rejects_unknown_optional_student_but_free_chat_does_not_require_it(
    client,
) -> None:
    invalid_student = client.post(
        "/api/v1/chat",
        headers={"X-Student-ID": "6ab2192e-e096-4018-bc08-f452230c5ca7"},
        json={
            "course": "machine_learning",
            "topic_id": "ml.machine_learning_basics",
            "message": "什么是机器学习？",
            "history": [],
        },
    )
    assert invalid_student.status_code == 404
    assert invalid_student.json()["error"]["code"] == "STUDENT_NOT_FOUND"

    free_chat = client.post(
        "/api/v1/chat",
        json={
            "course": "machine_learning",
            "message": "什么是机器学习？",
            "history": [],
        },
    )
    assert free_chat.status_code == 200


def test_route_uses_configured_message_limit(settings, mock_embedding, mock_llm) -> None:
    settings.max_message_length = 10
    application = create_app(
        settings,
        embedding_client=mock_embedding,
        llm_client=mock_llm,
    )
    with TestClient(application) as limited_client:
        response = limited_client.post(
            "/api/v1/chat",
            json={
                "course": "machine_learning",
                "message": "12345678901",
                "history": [],
            },
        )
    assert response.status_code == 422
    assert "10" in response.json()["error"]["message"]


def test_lifespan_closes_managed_sdk_clients(settings) -> None:
    embedding = CloseableEmbeddingClient()
    llm = CloseableLLMClient()
    application = create_app(settings, embedding_client=embedding, llm_client=llm)
    with TestClient(application) as managed_client:
        assert managed_client.get("/health").status_code == 200
        assert embedding.closed is False
        assert llm.closed is False
    assert embedding.closed is True
    assert llm.closed is True


def test_invalid_chat_payload_stays_422_when_services_are_unconfigured(settings) -> None:
    settings.embedding_provider = "api"
    settings.max_message_length = 5
    application = create_app(settings, embedding_client=None, llm_client=None)
    with TestClient(application) as unconfigured_client:
        invalid_course = unconfigured_client.post(
            "/api/v1/chat",
            json={"course": "not-a-course", "message": "问题", "history": []},
        )
        empty_message = unconfigured_client.post(
            "/api/v1/chat",
            json={"course": "machine_learning", "message": "  ", "history": []},
        )
        too_long = unconfigured_client.post(
            "/api/v1/chat",
            json={"course": "machine_learning", "message": "123456", "history": []},
        )
        too_much_history = unconfigured_client.post(
            "/api/v1/chat",
            json={
                "course": "machine_learning",
                "message": "问题",
                "history": [
                    {"role": "user", "content": str(index)} for index in range(13)
                ],
            },
        )
    assert invalid_course.status_code == 422
    assert empty_message.status_code == 422
    assert too_long.status_code == 422
    assert too_much_history.status_code == 422


def test_empty_strict_collection_needs_neither_embedding_nor_llm(settings) -> None:
    settings.embedding_provider = "api"
    application = create_app(settings, embedding_client=None, llm_client=None)
    with TestClient(application) as unconfigured_client:
        response = unconfigured_client.post(
            "/api/v1/chat",
            json={
                "course": "advanced_mathematics",
                "message": "什么是导数？",
                "history": [],
            },
        )
    assert response.status_code == 200, response.text
    assert response.json()["answer"] == STRICT_FALLBACK_ANSWER
    assert application.state.embedding_client is None
    assert application.state.llm_client is None


def test_nonempty_collection_without_embedding_returns_503(settings) -> None:
    settings.embedding_provider = "api"
    application = create_app(settings, embedding_client=None, llm_client=None)
    seed_embedding = MockEmbeddingClient()
    chunk = Chunk(
        chunk_id="seed_chunk",
        text="导数表示函数的瞬时变化率。",
        metadata={
            "chunk_id": "seed_chunk",
            "document_id": "seed_doc",
            "course": Course.ADVANCED_MATHEMATICS.value,
            "source_file": "seed.md",
            "title": "导数",
            "chapter": "微分学",
            "section": "定义",
            "page": -1,
            "chunk_index": 0,
            "file_hash": "seed_hash",
        },
    )
    with TestClient(application) as unconfigured_client:
        application.state.vector_store.upsert(
            Course.ADVANCED_MATHEMATICS,
            [chunk],
            seed_embedding.embed_documents([chunk.text]),
        )
        response = unconfigured_client.post(
            "/api/v1/chat",
            json={
                "course": "advanced_mathematics",
                "message": "什么是导数？",
                "history": [],
            },
        )
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "service_not_configured"
