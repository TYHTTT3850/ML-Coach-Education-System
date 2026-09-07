"""AI teaching examples use the existing course-isolated RAG, never exam keys."""
from __future__ import annotations

import pytest

from app.core.types import Course
from app.prompts.learning import get_learning_prompt
from app.rag.pipeline import STRICT_FALLBACK_ANSWER


@pytest.mark.parametrize("course", list(Course))
def test_example_and_teaching_prompts_require_concrete_grounded_steps(course):
    example = get_learning_prompt(course, "worked_example")
    teaching = get_learning_prompt(course, "topic_teaching")
    assert "默认提供两道" in example
    assert "逐步解答" in example and "验算" in example
    assert "基于课程资料原理编写的教学例题" in example
    assert "HyDE" in example and "不得编造" in example
    assert "不读取考试答案" in example
    assert "至少包含一道具体的完整分步例题" in teaching
    if course in {Course.LINEAR_ALGEBRA, Course.PROBABILITY_STATISTICS}:
        assert course.display_name in example


def test_example_request_requires_canonical_topic_and_matching_course(client, mock_llm):
    missing = client.post("/api/v1/chat", json={
        "course": "machine_learning", "message": "给我两道例题", "learning_mode": "worked_example",
    })
    assert missing.status_code == 422
    mismatch = client.post("/api/v1/chat", json={
        "course": "advanced_mathematics", "topic_id": "ml.machine_learning_basics",
        "message": "给我两道例题", "learning_mode": "worked_example",
    })
    assert mismatch.status_code == 422
    assert mismatch.json()["error"]["code"] == "COURSE_TOPIC_MISMATCH"
    assert not mock_llm.calls


def test_example_without_course_material_does_not_call_llm_or_borrow_other_course(client, mock_llm):
    uploaded = client.post("/api/v1/knowledge/upload",
        headers={"X-Admin-Token": "test-admin-token"}, data={"course": "machine_learning"},
        files={"file": ("ml-only.md", "# 机器学习基本概念\n机器学习从数据中学习规律，用学到的模型对新输入作预测。", "text/markdown")})
    assert uploaded.status_code == 200
    response = client.post("/api/v1/chat", json={
        "course": "linear_algebra", "topic_id": "la.vectors",
        "message": "给我两道向量例题", "learning_mode": "worked_example",
    })
    assert response.status_code == 200
    assert response.json()["answer"] == STRICT_FALLBACK_ANSWER
    assert response.json()["sources"] == []
    assert not mock_llm.calls


def test_dynamic_examples_use_real_retrieved_sources_and_do_not_change_mastery(client, mock_llm):
    uploaded = client.post("/api/v1/knowledge/upload",
        headers={"X-Admin-Token": "test-admin-token"}, data={"course": "machine_learning"},
        files={"file": ("basics-examples.md", "# 机器学习基本概念\n机器学习从训练数据中学习规律，利用模型预测新数据。监督学习用带标签的样本训练，分类预测类别，回归预测连续数值。", "text/markdown")})
    assert uploaded.status_code == 200
    student_id = client.post("/api/v1/students/anonymous").json()["student_id"]
    repository = client.app.state.learning_repository
    with repository._connect() as connection:
        before = {name: connection.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
            for name in ("student_topic_states", "practice_sets", "practice_attempts", "worked_examples", "question_bank_items")}
    response = client.post("/api/v1/chat", headers={"X-Student-ID": student_id}, json={
        "course": "machine_learning", "topic_id": "ml.machine_learning_basics",
        "message": "请根据机器学习基本概念，给两道分类和回归的分步例题。", "learning_mode": "worked_example",
    })
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["sources"][0]["source_file"] == "basics-examples.md"
    assert all(source["source_course"] == "machine_learning" for source in body["sources"])
    assert body["retrieval_count"] > 0
    assert len(mock_llm.calls) == 2  # existing HyDE plus teaching, no extra generator
    assert "检索用途：worked_example" in str(mock_llm.calls[0])
    assert "课程分步例题 Tutor" in mock_llm.calls[-1][0]["content"]
    assert "机器学习从训练数据中学习规律" in mock_llm.calls[-1][-1]["content"]
    with repository._connect() as connection:
        after = {name: connection.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0] for name in before}
        session = connection.execute("SELECT state FROM learning_sessions WHERE student_id=? ORDER BY created_at DESC", (student_id,)).fetchone()
    assert after == before
    assert session["state"] == "worked_example"
