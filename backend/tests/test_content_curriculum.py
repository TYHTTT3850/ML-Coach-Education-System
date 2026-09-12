from __future__ import annotations

import json

import pytest
import psycopg
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.course_content import router
from app.core.config import Settings
from app.core.types import Course
from app.curriculum.models import CurriculumSeed
from app.curriculum.seed import seed_curriculum
from app.curriculum.validation import CurriculumCycleError, validate_curriculum
from app.rag.pipeline import RAGPipeline
from app.rag.retriever import Retriever
from app.rag.vector_store import RetrievalCandidate
from app.services.learning_repository import LearningRepository
from app.services.learning_service import LearningService
from tests.mocks import MockEmbeddingClient
from tests.test_curriculum import _topic


@pytest.fixture
def repository(postgres_test_database_url):
    repo = LearningRepository(postgres_test_database_url)
    seed_curriculum(repo)
    return repo


def test_global_required_cycle_rejected_but_related_network_allowed():
    ml = CurriculumSeed.model_validate({"course":"machine_learning","topics":[_topic("ml.a",1,["math.a"])]})
    mathematics = CurriculumSeed.model_validate({"course":"advanced_mathematics","topics":[_topic("math.a",1,["ml.a"])]})
    with pytest.raises(CurriculumCycleError):
        validate_curriculum([ml,mathematics])
    mathematics.topics[0].prerequisites[0].relation_type = "related"
    validate_curriculum([ml,mathematics])


def test_database_guards_global_cycle_and_registered_course(repository):
    with repository._connect() as connection:
        with pytest.raises(psycopg.DatabaseError, match="cycle"):
            connection.execute("INSERT INTO topic_prerequisites(topic_id,prerequisite_topic_id) VALUES ('la.vectors','ml.linear_regression')")
        with pytest.raises(psycopg.DatabaseError):
            connection.execute("INSERT INTO topic_prerequisites(topic_id,prerequisite_topic_id) VALUES ('la.vectors','la.vectors')")
        with pytest.raises(psycopg.DatabaseError, match="unregistered_course"):
            connection.execute("UPDATE curriculum_topics SET course='unregistered' WHERE id='la.vectors'")


def test_recommended_and_section_requirements_do_not_block_preview(repository):
    student = repository.create_anonymous_student()["id"]
    service = LearningService(settings=Settings(_env_file=None),repository=repository)
    started = service.start_topic(student_id=student,topic_id="ml.linear_regression")
    assert started["status"] == "learning"
    assert repository.get_unmet_prerequisites(student,"ml.linear_regression")[0]["id"] == "la.dot_product"
    assert set(repository.get_prerequisite_masteries(student,"ml.linear_regression")) == {"la.dot_product"}
    assert not repository.get_unmet_prerequisites(student,"ml.dataset_split")
    assert repository.get_topic_state(student,"la.dot_product") is None


def test_local_verified_competency_reused_for_same_canonical_skill_not_whole_topic(repository):
    student = repository.create_anonymous_student()["id"]
    with repository._connect() as connection:
        connection.execute("""INSERT INTO practice_sets(id,student_id,topic_id,course,mode,status,requested_difficulty,source_snapshot_json,created_at)
            VALUES ('local-test',?,'ml.linear_regression','machine_learning','practice','submitted','easy','[]','now')""", (student,))
        connection.execute("""INSERT INTO prerequisite_verifications
            VALUES (?, 'ml.linear_regression', 'la.dot_product', 'la.dot_product.foundation', 'content-v1', 'local-test', 'now')""", (student,))
    assert repository.get_unmet_prerequisites(student,"ml.linear_regression") == []
    assert repository.get_prerequisite_masteries(student,"ml.linear_regression") == {"la.dot_product":1.0}
    assert not any(edge["id"]=="la.dot_product" for edge in repository.get_unmet_prerequisites(student,"ml.logistic_regression"))
    assert any(edge["id"]=="ml.sigmoid" for edge in repository.get_unmet_prerequisites(student,"ml.logistic_regression"))
    assert repository.get_topic_state(student,"la.dot_product") is None
    # A genuinely different competency or new rule version still needs proof.
    with repository._connect() as connection:
        connection.execute("UPDATE topic_prerequisites SET required_competencies_json='[\"la.dot_product.matrix_path\"]' WHERE topic_id='ml.logistic_regression' AND prerequisite_topic_id='la.dot_product'")
    assert any(edge["id"]=="la.dot_product" for edge in repository.get_unmet_prerequisites(student,"ml.logistic_regression"))
    with repository._connect() as connection:
        connection.execute("UPDATE topic_prerequisites SET rule_version='content-v2' WHERE topic_id='ml.linear_regression' AND prerequisite_topic_id='la.dot_product'")
    assert repository.get_unmet_prerequisites(student,"ml.linear_regression")


def test_canonical_outline_reference_shares_state_and_keeps_missing_content_separate(repository):
    student = repository.create_anonymous_student()["id"]
    repository.update_topic_state(student,"math.gradient",status="learning",evidence_count=1,mastery=.67)
    app = FastAPI()
    app.state.learning_repository = repository
    app.state.settings = Settings(_env_file=None)
    app.include_router(router)
    client = TestClient(app)
    headers={"X-Student-ID":student}
    ml = client.get("/api/v1/courses/machine_learning/outline",headers=headers).json()
    math = client.get("/api/v1/courses/advanced_mathematics/outline",headers=headers).json()
    ml_topic = next(t for group in ml["groups"] for t in group["topics"] if t["topic_id"]=="math.gradient")
    math_topic = next(t for group in math["groups"] for t in group["topics"] if t["topic_id"]=="math.gradient")
    assert ml_topic["is_reference"] and not math_topic["is_reference"]
    assert ml_topic["owner_course"] == "advanced_mathematics"
    assert ml_topic["status"] == math_topic["status"] == "learning"
    assert ml_topic["mastery"] == math_topic["mastery"] == .67
    assert ml_topic["content_status"] == "missing"
    assert len([g for g in ml["groups"] if g["id"].startswith("stage-")]) == 7
    prerequisites = client.get("/api/v1/topics/ml.linear_regression/prerequisites",headers=headers).json()
    assert prerequisites["can_browse"] and prerequisites["can_study"]
    assert not prerequisites["can_checkpoint"]
    assert prerequisites["recommended"] and prerequisites["required"]
    assert client.post("/api/v1/topics/ml.linear_regression/prerequisites/prob.bayes/ask",json={"message":"任意补学"}).status_code == 403


def test_core_required_closure_has_only_ready_candidate_topics(repository):
    core = {"la.vectors","la.dot_product","la.matrix_multiplication","prob.basics","prob.conditional_probability","prob.distributions","prob.expectation_variance","prob.bernoulli","math.functions","math.exponential_logarithmic","math.derivative_definition","math.derivative_rules","math.chain_rule","math.partial_derivatives","math.gradient","ml.linear_regression","ml.gradient_descent","ml.logistic_regression","ml.cross_entropy","ml.sigmoid"}
    for topic_id in core:
        required = {p["id"] for p in repository.list_prerequisites(topic_id) if p["relation_type"]=="required" and p["scope"]=="topic"}
        assert required <= core


def test_topic_scoped_retrieval_fails_closed_for_untagged_chunks():
    def candidate(chunk_id,course,topic_id=None):
        return RetrievalCandidate("条件概率 基础公式 P(A|B) 需要 P(B)>0",{"chunk_id":chunk_id,"document_id":"doc","course":course,"topic_id":topic_id or "","source_file":"课件"},.1)
    chunks=[candidate("tagged","probability_statistics","prob.conditional_probability"),candidate("untagged","probability_statistics"),candidate("other","probability_statistics","prob.bayes"),candidate("foreign","machine_learning","prob.conditional_probability")]
    class Store:
        def query(self,course,embedding,limit,**kwargs):
            return chunks
        def list_chunks(self,course):
            return chunks
    retriever = Retriever(Store(),MockEmbeddingClient())
    scoped = retriever.retrieve(course=Course.PROBABILITY_STATISTICS,question="条件概率",history=[],topic_ids=["prob.conditional_probability"])
    assert [c.metadata["chunk_id"] for c in scoped] == ["tagged"]
    assert retriever.retrieve(course=Course.PROBABILITY_STATISTICS,question="条件概率",history=[],topic_ids=["prob.independence"]) == []
    source = RAGPipeline._source(scoped[0],1)
    assert source["source_course"] == "probability_statistics"
    assert source["topic_id"] == "prob.conditional_probability"


def test_upload_tags_and_explicit_annotation_do_not_reembed(client, mock_embedding):
    response=client.post("/api/v1/knowledge/upload",headers={"X-Admin-Token":"test-admin-token"},
        data={"course":"linear_algebra","topic_ids":["la.dot_product"]},
        files={"file":("dot-product.md","向量内积是逐项相乘再求和，用于模型预测。".encode(),"text/markdown")})
    assert response.status_code==200,response.text
    document_id=response.json()["document"]["id"]
    store=client.app.state.vector_store
    before=store.list_chunks(Course.LINEAR_ALGEBRA)
    assert before and before[0].metadata["topic_id"]=="la.dot_product"
    changed=client.patch(f"/api/v1/knowledge/documents/{document_id}/topics",headers={"X-Admin-Token":"test-admin-token"},json={"topic_ids":["la.vectors"]})
    assert changed.status_code==200,changed.text
    assert changed.json()["reindexed"] is False
    after=store.list_chunks(Course.LINEAR_ALGEBRA)
    assert before[0].text==after[0].text
    assert after[0].metadata["topic_id"]=="la.vectors"
    assert after[0].metadata["topic:la.dot_product"] is False
    forbidden=client.patch(f"/api/v1/knowledge/documents/{document_id}/topics",json={"topic_ids":["la.dot_product"]})
    assert forbidden.status_code in (401,403)
    wrong_course=client.patch(f"/api/v1/knowledge/documents/{document_id}/topics",headers={"X-Admin-Token":"test-admin-token"},json={"topic_ids":["math.gradient"]})
    assert wrong_course.status_code==422


def test_authorized_supplement_cannot_use_untagged_course_chunks(client):
    uploaded=client.post("/api/v1/knowledge/upload",headers={"X-Admin-Token":"test-admin-token"},
        data={"course":"linear_algebra"},files={"file":("vectors.md","向量内积是逐项相乘再求和。".encode(),"text/markdown")})
    assert uploaded.status_code==200
    answer=client.post("/api/v1/topics/ml.linear_regression/prerequisites/la.dot_product/ask",json={"message":"如何计算向量内积？"})
    assert answer.status_code==200,answer.text
    assert answer.json()["sources"]==[]
    assert answer.json()["retrieval_count"]==0


