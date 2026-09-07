"""Readiness means the saved pools can support an independent learning flow."""
from app.schemas.content import BankItem
from tests.test_content_assessment import content_app


def coverage_row(content, topic_id="la.vectors"):
    return next(row for row in content.coverage()["topics"] if row["topic_id"] == topic_id)


def outline_topic(client, topic_id="la.vectors", course="linear_algebra"):
    response = client.get(f"/api/v1/courses/{course}/outline")
    assert response.status_code == 200, response.text
    return next(topic for group in response.json()["groups"] for topic in group["topics"] if topic["id"] == topic_id)


def test_all_twenty_published_core_units_have_independent_full_flow_inventory(content_app):
    _, _, content, _, llm, _ = content_app
    rows = content.coverage()["topics"]
    published = [row for row in rows if row["lesson_count"]]
    assert len(published) == 20
    assert all(row["independent_practice_count"] >= 4 for row in published)
    assert all(row["independent_checkpoint_count"] >= 5 for row in published)
    assert all(row["checkpoint_blueprint_ready"] for row in published)
    assert all(row["prerequisites_ready"] and row["full_flow_ready"] for row in published), [
        (row["topic_id"], row["readiness_gaps"]) for row in published if not row["full_flow_ready"]
    ]
    assert sum(row["full_flow_ready"] for row in rows) == 20
    assert not llm.calls


def test_example_family_variants_do_not_make_independent_practice_ready(content_app):
    client, _, content, _, llm, _ = content_app
    before = coverage_row(content)
    practices = content.bank(topic_id="la.vectors", publication_status="published", usage="practice")
    template = content.examples("la.vectors")[0].model_dump(mode="json")
    # Arrange additional published worked examples covering every practice family.
    examples = [{**template, "id": f"readiness-example-{index}", "family_id": item.family_id}
                for index, item in enumerate(practices)]
    content.seed({"lessons": [], "examples": examples, "questions": []})
    row = coverage_row(content)
    assert row["practice_count"] == before["practice_count"] >= 4
    assert row["checkpoint_count"] >= 5
    assert row["independent_practice_count"] == 0
    assert not row["local_flow_ready"] and not row["full_flow_ready"]
    topic = outline_topic(client)
    assert topic["content_status"] == "missing" and topic["has_learning_content"]
    assert "独立练习" in topic["readiness_reason"]
    assert not llm.calls


def test_checkpoint_total_is_not_enough_when_dimension_coverage_is_missing(content_app):
    _, _, content, _, _, _ = content_app
    checkpoints = content.bank(topic_id="la.vectors", publication_status="published", usage="checkpoint")
    revised = [BankItem.model_validate({**item.model_dump(mode="json"), "version": item.version + 1,
                "dimensions": {"concept": 1.0}}) for item in checkpoints]
    # Test-only saved content with declared publication status; no production seed mutation.
    content.import_items(revised, trusted_seed=True)
    row = coverage_row(content)
    assert row["checkpoint_count"] >= 5 and row["independent_checkpoint_count"] >= 5
    assert row["checkpoint_dimension_count"] == 1
    assert not row["checkpoint_blueprint_ready"] and not row["full_flow_ready"]


def test_checkpoint_families_and_question_type_diversity_are_required(content_app):
    _, _, content, _, _, _ = content_app
    checkpoints = content.bank(topic_id="la.vectors", publication_status="published", usage="checkpoint")
    numeric = next(item for item in checkpoints if item.question_type == "numeric")
    revised = [BankItem.model_validate({**numeric.model_dump(mode="json"), "id": item.id,
                "version": item.version + 1, "family_id": item.family_id}) for item in checkpoints]
    content.import_items(revised, trusted_seed=True)
    row = coverage_row(content)
    assert row["independent_checkpoint_count"] >= 5
    assert row["checkpoint_question_type_count"] == 1
    assert not row["checkpoint_blueprint_ready"]
    same_family = [item.model_copy(update={"version": item.version + 1, "family_id": "one-shared-formal-template"}) for item in revised]
    content.import_items(same_family, trusted_seed=True)
    row = coverage_row(content)
    assert row["checkpoint_count"] >= 5 and row["independent_checkpoint_count"] == 1
    assert not row["full_flow_ready"]


def test_published_lesson_alone_never_claims_complete_outline_readiness(content_app):
    client, _, content, _, _, _ = content_app
    lesson = content.lesson("la.vectors").model_dump(mode="json")
    lesson.update(id="lesson-only-fixture", topic_id="ml.machine_learning_basics")
    content.seed({"lessons": [lesson], "examples": [], "questions": []})
    row = coverage_row(content, "ml.machine_learning_basics")
    assert row["lesson_count"] == 1 and row["has_learning_content"]
    assert not row["full_flow_ready"]
    topic = outline_topic(client, "ml.machine_learning_basics", "machine_learning")
    assert topic["content_status"] != "ready"
    assert topic["has_learning_content"] and topic["readiness_reason"]


def test_required_prerequisite_content_gap_prevents_full_flow_claim(content_app):
    client, repository, content, _, _, _ = content_app
    with repository._connect() as connection:
        connection.execute("UPDATE lessons SET status='archived' WHERE topic_id='la.dot_product'")
    row = coverage_row(content, "ml.linear_regression")
    assert not row["prerequisites_ready"] and not row["full_flow_ready"]
    assert any("前置内容待补充" in gap for gap in row["readiness_gaps"])
    assert outline_topic(client, "ml.linear_regression", "machine_learning")["content_status"] != "ready"
