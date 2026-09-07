"""Dynamic prerequisite generation preserves canonical, tagged retrieval scope.

These tests use actual temporary SQLite/Chroma and the normal grounding/retriever
adapter. Only embedding and model calls are replaced; production data is unused.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from tests.mocks import MockEmbeddingClient
from tests.test_dynamic_assessments import generated, headers
from tests.test_learning_loop_e2e import QueueLLM


@pytest.fixture
def scoped_app(settings):
    configured = settings.model_copy(update={
        "rag_hyde_enabled": False,
        "question_generation_max_retries": 0,
        "evaluation_max_repair_attempts": 0,
    })
    llm = QueueLLM([])
    app = create_app(configured, embedding_client=MockEmbeddingClient(), llm_client=llm)
    with TestClient(app) as client:
        sid = client.post("/api/v1/students/anonymous").json()["student_id"]
        yield client, app.state.learning_repository, llm, sid


def upload(client, name, *, course="linear_algebra", topic_ids=(), text=None):
    data = {"course": course}
    if topic_ids:
        data["topic_ids"] = list(topic_ids)
    result = client.post("/api/v1/knowledge/upload",
        headers={"X-Admin-Token": "test-admin-token"}, data=data,
        files={"file": (name, (text or (
            "# 向量内积\n内积逐项相乘再求和：向量 (6,0) 与 (7,2) 的内积是 "
            "6×7+0×2=42。线性回归的预测可写成权重与特征的内积。"
        )).encode(), "text/markdown")})
    assert result.status_code == 200, result.text
    return result.json()["document"]["id"]


@pytest.mark.parametrize("course,tags", [
    ("linear_algebra", ()),
    ("linear_algebra", ("la.vectors",)),
    ("machine_learning", ("ml.linear_regression",)),
])
def test_dynamic_prerequisite_cannot_borrow_untagged_other_topic_or_other_course(scoped_app, course, tags):
    client, repository, llm, sid = scoped_app
    # Deliberately strong lexical/vector evidence is still outside the declared
    # canonical target. A query-only implementation would incorrectly use it.
    upload(client, "out-of-scope.md", course=course, topic_ids=tags)
    result = client.post("/api/v1/prerequisite-tests", headers=headers(sid), json={
        "topic_id": "ml.linear_regression", "prerequisite_topic_id": "la.dot_product",
    })
    assert result.status_code == 422, result.text
    assert result.json()["error"]["code"] == "KNOWLEDGE_INSUFFICIENT"
    assert not llm.calls
    with repository._connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM practice_sets").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM prerequisite_verifications").fetchone()[0] == 0


def test_dynamic_prerequisite_uses_only_tagged_canonical_sources(scoped_app):
    client, repository, llm, sid = scoped_app
    upload(client, "untagged.md", text="# 向量内积\n未标注资料：内积逐项相乘再求和，用于线性回归预测。")
    upload(client, "other-topic.md", topic_ids=("la.vectors",), text="# 向量内积\n其他知识点资料：内积把向量映射为一个实数。")
    document_id = upload(client, "canonical-dot-product.md", topic_ids=("la.dot_product",))
    competency = "la.dot_product.foundation"
    llm.answers.append(generated("计算内积的局部能力验证", competency=competency))
    result = client.post("/api/v1/prerequisite-tests", headers=headers(sid), json={
        "topic_id": "ml.linear_regression", "prerequisite_topic_id": "la.dot_product",
    })
    assert result.status_code == 201, result.text
    body = result.json()
    assert body["purpose"] == "prerequisite_test" and body["topic_id"] == "la.dot_product"
    assert {source["source_file"] for source in body["source_summary"]} == {"canonical-dot-product.md"}
    internal = repository.get_practice_set_internal(body["id"], student_id=sid)
    assert {source.document_id for source in internal.source_snapshot} == {document_id}
    assert all(source.topic_id == "la.dot_product" and source.source_course.value == "linear_algebra"
               for source in internal.source_snapshot)
    assert len(llm.calls) == 1
    prompt = str(llm.calls[0])
    assert "canonical-dot-product.md" in prompt
    assert "未标注资料" not in prompt and "其他知识点资料" not in prompt


def test_ordinary_dynamic_practice_still_accepts_untagged_owner_course_material(scoped_app):
    client, _, llm, sid = scoped_app
    upload(client, "ml-untagged.md", course="machine_learning", text=(
        "# 机器学习基本概念\n机器学习通过训练数据学习规律，监督学习使用样本标签训练预测模型。"
        "六组样本每组七个，总共有42个训练样本。"
    ))
    llm.answers.append(generated("机器学习基本概念练习"))
    result = client.post("/api/v1/practice/sets", headers=headers(sid), json={
        "topic_id": "ml.machine_learning_basics", "mode": "practice", "count": 3,
    })
    assert result.status_code == 201, result.text
    assert {source["source_file"] for source in result.json()["source_summary"]} == {"ml-untagged.md"}
    assert len(llm.calls) == 1
