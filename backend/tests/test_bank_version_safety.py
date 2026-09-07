"""Version/exposure regression checks using test-only content and mock clients."""
import pytest

from app.schemas.content import BankItem
from tests.test_content_assessment import content_app, create, submit, headers


def clone_item(content, *, usage="practice"):
    original = next(item for item in content.bank(topic_id="la.vectors", usage=usage, publication_status="published") if item.question_type == "numeric")
    raw = original.model_dump(mode="json")
    raw.update(id="version-safety-fixture", family_id="version-safety-fixture", version=1)
    content.import_items([BankItem.model_validate(raw)])
    content.transition(raw["id"], 1, "publish", reviewed=True, review_note="模拟管理员审核的自动化测试夹具")
    return raw


def test_new_draft_and_archived_draft_do_not_hide_published_coverage(content_app):
    _, _, content, _, _, _ = content_app
    raw = clone_item(content)
    before = next(row for row in content.coverage()["topics"] if row["topic_id"] == "la.vectors")
    draft = {**raw, "version": 2, "stem": raw["stem"] + "（新版本草稿）"}
    content.import_items([BankItem.model_validate(draft)])
    published = [item for item in content.bank(publication_status="published") if item.id == raw["id"]]
    assert len(published) == 1 and published[0].version == 1
    after = next(row for row in content.coverage()["topics"] if row["topic_id"] == "la.vectors")
    assert after["practice_count"] == before["practice_count"]
    assert after["published_count"] == before["published_count"]
    assert after["needs_review_count"] == before["needs_review_count"] + 1
    content.transition(raw["id"], 2, "archive")
    after_archive = next(row for row in content.coverage()["topics"] if row["topic_id"] == "la.vectors")
    assert after_archive["published_count"] == before["published_count"]


def test_published_new_answer_keeps_allocated_snapshot_and_historical_score(content_app):
    client, repository, content, student, _, _ = content_app
    raw = clone_item(content)
    old = create(client, student, count=1, bank_item_ids=[raw["id"]])
    internal_before = repository.get_practice_set_internal(old["id"], student_id=student)
    answer = {**raw["answer_key"], "numeric_answer": raw["answer_key"]["numeric_answer"] + 100}
    answer["canonical_answer"] = str(answer["numeric_answer"])
    updated = {**raw, "version": 2, "answer_key": answer, "stem": raw["stem"] + " 新条件：结果加100。", "solution_steps": [answer["canonical_answer"]]}
    content.import_items([BankItem.model_validate(updated)])
    content.transition(raw["id"], 2, "publish", reviewed=True, review_note="模拟审核的新答案测试夹具")
    assert repository.get_practice_set_internal(old["id"], student_id=student).model_dump() == internal_before.model_dump()
    result, _ = submit(client, repository, student, old)
    assert result["total_score"] == 100
    latest = create(client, student, count=1, bank_item_ids=[raw["id"]])
    assert repository.get_practice_set_internal(latest["id"], student_id=student).questions[0].answer_key.numeric_answer == answer["numeric_answer"]


def test_version_change_cannot_launder_seen_formal_item_id(content_app):
    client, repository, content, student, llm, _ = content_app
    practice = create(client, student)
    submit(client, repository, student, practice)
    original = content.bank(topic_id="la.vectors", usage="checkpoint", publication_status="published")[0]
    content.exposure(student, original.id, original.version, original.family_id, "feedback")
    revised = original.model_dump(mode="json")
    revised.update(version=2, family_id=original.family_id + "-renamed")
    content.import_items([BankItem.model_validate(revised)])
    content.transition(original.id, 2, "publish", reviewed=True, review_note="模拟版本变动测试夹具")
    response = client.post("/api/v1/assessments", headers=headers(student), json={"topic_ids":["la.vectors"],"purpose":"checkpoint","question_count":5})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "QUESTION_BANK_INSUFFICIENT"
    assert not llm.calls
