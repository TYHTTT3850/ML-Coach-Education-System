"""Trusted, offline arithmetic and course coverage acceptance tests.

The tests intentionally do not open production data or construct a Retriever.
They check the authored payload rather than evaluating student expressions.
"""

import json
import math
from pathlib import Path
import re
from types import SimpleNamespace
import uuid

import pytest

from app.content.seeds.core import TOPICS, build_content, calculate, verify_content
from app.schemas.content import BankItem, Lesson, WorkedExample


def _example(suffix):
    return next(e for e in build_content()["examples"] if e["id"].endswith(suffix))


def test_required_example_a_linear_regression_training_step():
    """Independently compute every requested A value from the original data."""
    xs, ys, w, b, eta = [1, 2], [2, 4], 1, 0, 0.1
    predictions = [w*x+b for x in xs]
    residuals = [p-y for p,y in zip(predictions,ys,strict=True)]
    loss = sum(r*r for r in residuals) / (2*len(xs))
    dw = sum(r*x for r,x in zip(residuals,xs,strict=True)) / len(xs)
    db = sum(residuals) / len(xs)
    next_w, next_b = w-eta*dw, b-eta*db
    next_loss = sum((next_w*x+next_b-y)**2 for x,y in zip(xs,ys,strict=True)) / (2*len(xs))
    assert predictions == [1,2]
    assert (loss,dw,db,next_w,next_b,next_loss) == pytest.approx((1.25,-2.5,-1.5,1.25,0.15,0.545625))
    example = _example("A_training_step")
    assert example["givens"]["normalization"] == "1/(2n)"
    assert "0.545625" in example["final_answer"]
    assert len(example["steps"]) >= 5


def test_required_example_b_partial_derivatives_and_gradient():
    x,y = 1,2
    gradient = (2*x,6*y)
    function_value = x*x+3*y*y
    assert gradient == (2,12)
    assert function_value == 13
    example = _example("B_partial_gradient")
    assert "(2,12)" in example["final_answer"]
    assert "13" in example["final_answer"]


def test_required_example_c_dot_bias_sigmoid():
    x,w,b = (2,1),(0.5,-1),0.5
    dot = sum(a*c for a,c in zip(x,w,strict=True))
    z = dot+b
    probability = 1/(1+math.exp(-z))
    assert dot == 0
    assert z == 0.5
    assert probability == pytest.approx(0.6224593312018546)
    assert "0.622459" in _example("C_dot_sigmoid")["final_answer"]


def test_required_example_d_conditional_probability():
    joint, condition = 0.2,0.5
    assert 0 <= joint <= condition <= 1
    assert joint/condition == pytest.approx(0.4)
    assert "0.4" in _example("D_conditional")["final_answer"]


def test_required_example_e_binary_cross_entropy():
    y,p = 1,0.8
    loss = -(y*math.log(p)+(1-y)*math.log(1-p))
    assert loss == pytest.approx(0.2231435513142097)
    assert loss < -math.log(0.4)
    assert "0.223144" in _example("E_positive_label")["final_answer"]


def test_all_authored_payloads_pass_strict_internal_schemas():
    data = build_content()
    models = {"lessons":Lesson,"examples":WorkedExample,"questions":BankItem}
    for group,items in data.items():
        for item in items:
            models[group].model_validate(item)
    assert verify_content(data) == []
    assert build_content() == data  # no random regeneration on refresh
    data["lessons"][0]["learning_objectives"].append("mutated copy")
    assert build_content()["lessons"][0]["learning_objectives"] != data["lessons"][0]["learning_objectives"]


def test_numeric_checker_detects_wrong_answer_not_only_valid_json():
    data = build_content()
    question = next(q for q in data["questions"] if q["question_type"]=="numeric")
    question["answer_key"]["numeric_answer"] += 1
    assert any(question["id"] in error and "numeric answer" in error for error in verify_content(data))


def test_independent_choice_catalogue_detects_changed_correct_option():
    data = build_content()
    question = next(q for q in data["questions"] if q["question_type"]=="multiple_choice")
    original = question["answer_key"]["correct_option"]
    question["answer_key"]["correct_option"] = next(o["option_id"] for o in question["options"] if o["option_id"]!=original)
    assert any(question["id"] in error and "conceptual key" in error for error in verify_content(data))


def _curriculum_topics():
    seed_path = Path(__file__).resolve().parents[1]/"app"/"curriculum"/"seeds"
    topics = {}
    for name in ("machine_learning.json","advanced_mathematics.json"):
        topics.update({t["id"]:t for t in json.loads((seed_path/name).read_text(encoding="utf-8"))["topics"]})
    config = json.loads((seed_path/"content_curriculum.json").read_text(encoding="utf-8"))
    for course in config["additional_courses"]:
        topics.update({t["id"]:t for t in course["topics"]})
    return topics,config


@pytest.mark.parametrize("topic_id", list(TOPICS))
def test_each_core_unit_has_real_content_and_independent_published_pool(topic_id):
    data=build_content()
    lessons=[l for l in data["lessons"] if l["topic_id"]==topic_id]
    examples=[e for e in data["examples"] if e["topic_id"]==topic_id]
    questions=[q for q in data["questions"] if q["topic_ids"]==[topic_id]]
    published=[q for q in questions if q["publication_status"]=="published"]
    practice=[q for q in published if "practice" in q["usage"]]
    checkpoint=[q for q in published if "checkpoint" in q["usage"]]
    assert len(lessons)>=1 and len(examples)>=2
    assert len(practice)>=4 and len(checkpoint)>=5
    assert all(len(e["steps"])>=3 for e in examples)
    assert len({q["question_type"] for q in checkpoint})>=2
    assert any(q["difficulty"] in {"medium","hard"} for q in checkpoint)
    assert len({q["family_id"] for q in checkpoint})==len(checkpoint)
    exposed_families={e["family_id"] for e in data["examples"]}
    exposed_families.update(q["family_id"] for q in data["questions"] if "practice" in q["usage"])
    assert not exposed_families & {q["family_id"] for q in checkpoint}
    curriculum,_ = _curriculum_topics()
    required=set(curriculum[topic_id]["required_dimensions"])
    actual={d for q in checkpoint for d in q["dimensions"]}
    assert len(required & actual)>=3, (topic_id,required,actual)
    assert all(q["tested_competencies"]==[topic_id+".foundation"] for q in questions)


def test_required_core_prerequisite_closure_is_not_empty():
    data=build_content()
    _,config=_curriculum_topics()
    available={l["topic_id"] for l in data["lessons"]}
    required=set()
    pending=list(available)
    while pending:
        target=pending.pop()
        for relation in config["relations"].get(target,[]):
            if relation.get("relation_type","required")!="required" or relation.get("scope","topic")!="topic":
                continue
            prerequisite=relation.get("topic_id",relation.get("prerequisite_topic_id"))
            if prerequisite not in required:
                required.add(prerequisite)
                pending.append(prerequisite)
    assert required <= available
    for topic_id in required:
        pool=[q for q in data["questions"] if q["topic_ids"]==[topic_id] and "prerequisite_test" in q["usage"] and q["publication_status"]=="published"]
        teaching_families={e['family_id'] for e in data['examples']}
        teaching_families.update(q['family_id'] for q in data['questions'] if 'practice' in q['usage'])
        independent=[q for q in pool if q['family_id'] not in teaching_families]
        assert len(independent)>=3, topic_id
        assert {q["question_type"] for q in independent}=={"numeric","multiple_choice"}


def test_authored_sources_are_honest_and_open_questions_remain_draft():
    data=build_content()
    for group in data.values():
        for item in group:
            assert item["content_origin"]=="instructor_authored"
            assert item["source_refs"]==[]
            assert item.get("source_snapshot",[])==[]
            assert item.get("validation_status")!="human_reviewed"
    questions=data["questions"]
    subjective=[q for q in questions if q["question_type"] not in {"numeric","multiple_choice"}]
    assert len(subjective)==20
    assert len({q["question_type"] for q in subjective})>=5
    assert all(q["publication_status"]=="draft" and q["validation_status"]=="needs_review" for q in subjective)
    assert not any(re.search(r"\bS\d+\b",q["stem"]) for q in questions)


def test_calculations_do_not_execute_unregistered_operations():
    with pytest.raises(ValueError,match="Unregistered"):
        calculate("__import__('os').system('anything')",[])


def test_public_bank_schema_hides_answers_and_verification():
    for item in build_content()["questions"]:
        public=BankItem.model_validate(item).public().model_dump()
        assert not {"answer_key","solution_steps","rubric","verification","source_snapshot"} & public.keys()


@pytest.mark.parametrize("topic_id", list(TOPICS))
def test_real_read_examples_practice_prerequisite_checkpoint_flow(topic_id, settings):
    """Exercise each actual student flow using only the tmp_path SQLite DB.

    This catches a bank that looks sufficient in counts but becomes unusable
    once the student has followed the intended path and read worked examples.
    """
    from fastapi.testclient import TestClient
    from app.main import create_app
    from app.services.content_repository import ContentRepository

    class ForbiddenLLM:
        async def generate(self, *args, **kwargs):
            raise AssertionError("The authored assessment workflow must not call an LLM")

    app=create_app(settings,llm_client=ForbiddenLLM(),embedding_client=object(),vector_store=SimpleNamespace(ready=True))
    with TestClient(app) as client:
        repository=app.state.learning_repository
        ContentRepository(repository).seed(build_content())
        student_id=client.post('/api/v1/students/anonymous').json()['student_id']
        auth={'X-Student-ID':student_id}
        assert client.get(f'/api/v1/topics/{topic_id}/lesson').status_code==200
        examples=client.get(f'/api/v1/topics/{topic_id}/examples').json()['examples']
        for example in examples:
            response=client.post(f"/api/v1/examples/{example['id']}/exposure",headers=auth,json={'version':example['version']})
            assert response.status_code==200,response.text

        def answer_assessment(assessment):
            internal=repository.get_practice_set_internal(assessment['id'],student_id=student_id)
            response=client.post(f"/api/v1/assessments/{assessment['id']}/submit",headers=auth,json={
                'client_submission_id':str(uuid.uuid4()),
                'answers':[{'question_id':q.question_id,'answer':q.answer_key.correct_option or str(q.answer_key.numeric_answer)} for q in internal.questions]})
            assert response.status_code==200,(topic_id,response.text)
            value=response.json()
            assert value['status']=='evaluated' and value['total_score']==100,(topic_id,value)
            return value

        response=client.post('/api/v1/assessments',headers=auth,json={'topic_ids':[topic_id],'purpose':'practice','question_count':4})
        assert response.status_code==201,(topic_id,'practice-after-examples',response.text)
        practice=answer_assessment(response.json())
        assert practice['mastery_after']>practice['mastery_before'],(topic_id,practice)

        for required in repository.get_unmet_prerequisites(student_id,topic_id):
            # Students may read the prerequisite's examples before deciding
            # they already know the skill and taking its independent mini-test.
            prerequisite_examples=client.get(f"/api/v1/topics/{required['id']}/examples").json()['examples']
            for example in prerequisite_examples:
                exposed=client.post(f"/api/v1/examples/{example['id']}/exposure",headers=auth,json={'version':example['version']})
                assert exposed.status_code==200,exposed.text
            response=client.post('/api/v1/prerequisite-tests',headers=auth,json={'topic_id':topic_id,'prerequisite_topic_id':required['id'],'question_count':3,'generation_mode':'bank'})
            assert response.status_code==201,(topic_id,required['id'],response.text)
            prerequisite=answer_assessment(response.json())
            assert prerequisite['mastery_after']==prerequisite['mastery_before']
        assert not repository.get_unmet_prerequisites(student_id,topic_id)

        response=client.post('/api/v1/assessments',headers=auth,json={'topic_ids':[topic_id],'purpose':'checkpoint','question_count':5})
        assert response.status_code==201,(topic_id,'checkpoint',response.text)
        checkpoint=answer_assessment(response.json())
        assert checkpoint['gate']['passed'],(topic_id,checkpoint)
