"""Stable canonical curriculum navigation and prerequisite explanations."""
from __future__ import annotations

from fastapi import APIRouter, Header, Request, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from app.core.types import Course
from app.services.learning_service import LearningService
from app.api.dependencies import build_rag_pipeline
from app.schemas.chat import ChatResponse
from app.rag.pipeline import STRICT_FALLBACK_ANSWER
from app.services.content_repository import ContentRepository

router = APIRouter(prefix="/api/v1", tags=["course-content"])


def _can_checkpoint(request: Request, student_id: str | None, topic: dict) -> bool:
    if student_id is None:
        return False
    repository = request.app.state.learning_repository
    if repository.get_unmet_prerequisites(student_id, topic["id"]):
        return False
    state = repository.get_topic_state(student_id, topic["id"])
    service = LearningService(settings=request.app.state.settings, repository=repository)
    gate = service.evaluate_topic_gate(student_id, topic, state)
    return bool(gate.next_action == "take_checkpoint" or (state and state["status"] in ("ready_for_checkpoint", "mastered")))


@router.get("/courses")
def registered_courses(request: Request):
    with request.app.state.learning_repository._connect() as connection:
        return {"courses": [dict(row) for row in connection.execute("SELECT * FROM courses WHERE active=1 ORDER BY id")]}


@router.get("/courses/{course}/outline")
def outline(course: Course, request: Request, x_student_id: str | None = Header(default=None, alias="X-Student-ID")):
    repository = request.app.state.learning_repository
    if x_student_id:
        repository.require_student(x_student_id)
    with repository._connect() as connection:
        placements = [dict(row) for row in connection.execute("""SELECT p.*,t.title,t.description,t.course AS owner_course
            FROM topic_placements p JOIN curriculum_topics t ON t.id=p.topic_id
            WHERE p.course_id=? AND t.active=1 ORDER BY p.is_reference DESC,p.order_index""", (course.value,))]
    coverage={row["topic_id"]:row for row in ContentRepository(repository).coverage()["topics"]}
    ready={topic_id for topic_id,row in coverage.items() if row["full_flow_ready"]}
    groups: dict[str, dict] = {}
    if any(item["is_reference"] for item in placements):
        groups["math-preparation"] = {"id":"math-preparation","title":"数学准备","parent_group_id":None,"order_index":0,"topics":[]}
    recommended = None
    for placement in placements:
        topic_id = placement["topic_id"]
        topic = repository.require_topic(topic_id)
        state = repository.get_topic_state(x_student_id, topic_id) if x_student_id else None
        unmet = repository.get_unmet_prerequisites(x_student_id, topic_id) if x_student_id else [edge for edge in repository.list_prerequisites(topic_id) if edge["relation_type"]=="required" and edge["scope"]=="topic"]
        group_id = placement["parent_group_id"]
        group = groups.setdefault(group_id, {"id":group_id,"title":placement["group_title"],"parent_group_id":"math-preparation" if placement["is_reference"] else None,"order_index":len(groups),"topics":[]})
        status = state["status"] if state and state["status"]!="locked" else "not_started"
        group["topics"].append({"id":topic_id,"topic_id":topic_id,"owner_course":placement["owner_course"],
            "title":topic["title"],"description":topic["description"],"learning_objectives":topic["learning_objectives"],
            "order_index":placement["order_index"],"is_reference":bool(placement["is_reference"]),"status":status,
            "mastery":float(state["mastery"]) if state and state["evidence_count"] else None,
            "prerequisite_gap":[{"id":edge["id"],"title":edge["title"],"reason":edge["reason"]} for edge in unmet],
            "content_status":"ready" if topic_id in ready else "missing",
            "has_learning_content":coverage.get(topic_id,{}).get("has_learning_content",False),
            "readiness_reason":"；".join(coverage.get(topic_id,{}).get("readiness_gaps",[]))})
        if recommended is None and topic_id in ready and status!="mastered" and not unmet:
            recommended = topic_id
    return {"course":course.value,"title":course.display_name,"groups":list(groups.values()),"recommended_topic_id":recommended}


@router.get("/topics/{topic_id}/prerequisites")
def prerequisites(topic_id: str, request: Request, x_student_id: str | None = Header(default=None, alias="X-Student-ID")):
    repository = request.app.state.learning_repository
    if x_student_id:
        repository.require_student(x_student_id)
    topic = repository.require_topic(topic_id)
    edges = repository.prerequisite_status(x_student_id, topic_id)
    with repository._connect() as connection:
        successors = [dict(row) for row in connection.execute("""SELECT t.id,t.title,t.course,p.relation_type,p.reason
            FROM topic_prerequisites p JOIN curriculum_topics t ON t.id=p.topic_id
            WHERE p.prerequisite_topic_id=? ORDER BY t.course,t.order_index""", (topic_id,))]
    return {"topic_id":topic_id,"owner_course":topic["course"],"rule_version":"content-v1","can_browse":True,"can_study":True,
        "can_checkpoint":_can_checkpoint(request,x_student_id,topic),
        **{kind:[edge for edge in edges if edge["relation_type"]==kind] for kind in ("required","recommended","related")},
        "unmet":[edge for edge in edges if edge["relation_type"]=="required" and edge["scope"]=="topic" and not edge["satisfied"]],
        "successors":successors}


class SupplementQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message: str = Field(min_length=1, max_length=8000)


@router.post("/topics/{topic_id}/prerequisites/{prerequisite_id}/ask", response_model=ChatResponse)
async def ask_prerequisite(topic_id: str, prerequisite_id: str, payload: SupplementQuestion, request: Request):
    repository = request.app.state.learning_repository
    repository.require_topic(topic_id)
    declared = repository.list_prerequisites(topic_id)
    if not any(edge["id"] == prerequisite_id for edge in declared):
        raise HTTPException(403, detail={"code":"RETRIEVAL_SCOPE_FORBIDDEN","message":"该知识点未声明这一补学范围，请从前置知识卡片进入。"})
    topic = repository.require_topic(prerequisite_id)
    course = Course(topic["course"])
    if request.app.state.vector_store.count(course) == 0:
        return ChatResponse(answer=STRICT_FALLBACK_ANSWER,course=course,sources=[],retrieval_count=0)
    pipeline = build_rag_pipeline(request)
    return ChatResponse.model_validate(await pipeline.answer(
        course=course,question=payload.message,history=[],topic_ids=[prerequisite_id],
        retrieval_query=f"知识点：{topic['title']}\n检索用途：prerequisite_supplement\n{payload.message}"))
