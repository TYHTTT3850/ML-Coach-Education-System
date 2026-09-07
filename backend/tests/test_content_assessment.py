from __future__ import annotations
import json
import uuid
from types import SimpleNamespace
import pytest
from fastapi.testclient import TestClient
from app.main import create_app
from app.content.seeds.core import build_content
from app.content.importing import parse_items
from app.schemas.content import BankItem, AssessmentBlueprint
from app.services.content_repository import ContentRepository,ContentError
from app.services.assessment_service import AssessmentService
from app.services.evaluation_service import score_numeric

class NoNetworkLLM:
    def __init__(self):self.calls=[]
    async def generate(self,messages,temperature=None):
        self.calls.append(messages)
        return '根据保存例题，这一步使用内积定义逐项相乘并求和。'

@pytest.fixture
def content_app(settings):
    llm=NoNetworkLLM()
    app=create_app(settings,llm_client=llm,embedding_client=object(),vector_store=SimpleNamespace(ready=True))
    with TestClient(app) as client:
        repo=app.state.learning_repository
        content=ContentRepository(repo);content.seed(build_content())
        sid=client.post('/api/v1/students/anonymous').json()['student_id']
        yield client,repo,content,sid,llm,settings

def headers(sid):return {'X-Student-ID':sid}

def create(client,sid,topic='la.vectors',purpose='practice',count=4,**kwargs):
    response=client.post('/api/v1/assessments',headers=headers(sid),json={
        'topic_ids':[topic],'purpose':purpose,'question_count':count,**kwargs})
    assert response.status_code==201,response.text
    return response.json()

def submit(client,repo,sid,value,overrides=None,client_id=None):
    internal=repo.get_practice_set_internal(value['practice_set_id'],student_id=sid)
    answers=[{'question_id':q.question_id,'answer':(overrides or {}).get(q.question_id,
        q.answer_key.correct_option or str(q.answer_key.numeric_answer))} for q in internal.questions]
    payload={'client_submission_id':client_id or str(uuid.uuid4()),'answers':answers}
    response=client.post('/api/v1/assessments/'+value['practice_set_id']+'/submit',headers=headers(sid),json=payload)
    assert response.status_code==200,response.text
    return response.json(),payload

def assert_public(value):
    text=json.dumps(value)
    for forbidden in ('answer_key','rubric','canonical_answer','correct_option','numeric_answer','solution_steps','source_snapshot','verification'):
        assert forbidden not in text

def test_saved_lesson_and_public_bank_no_answers(content_app):
    client,repo,content,sid,llm,_=content_app
    first=client.get('/api/v1/topics/ml.linear_regression/lesson')
    assert first.status_code==200
    assert client.get('/api/v1/topics/ml.linear_regression/lesson').json()==first.json()
    bank=client.get('/api/v1/question-bank',params={'topic_id':'ml.linear_regression'}).json()
    assert len(bank['items'])>=4 and bank['checkpoint_count']==5
    assert_public(bank)
    assert not llm.calls

def test_example_ask_uses_saved_version_and_does_not_increase_mastery(content_app):
    client,repo,content,sid,llm,_=content_app
    value=content.examples('la.dot_product')[0]
    path=f'/api/v1/examples/{value.id}'
    assert client.post(path+'/exposure',headers=headers(sid),json={'version':value.version}).json()['mastery_updated'] is False
    asked=client.post(path+'/ask',headers=headers(sid),json={'version':value.version,'step_id':value.steps[0].step_id,'question':'解释这个步骤'})
    assert asked.status_code==200
    data=json.loads(llm.calls[-1][-1]['content'])
    assert data['selected_step']==value.steps[0].model_dump()
    assert data['version']==value.version
    assert repo.get_topic_state(sid,'la.dot_product') is None
    assert client.post(path+'/ask',headers=headers(sid),json={'version':999,'step_id':'fake'}).status_code==404

def test_practice_then_independent_checkpoint_and_idempotency(content_app):
    client,repo,content,sid,llm,_=content_app
    practice=create(client,sid)
    assert_public(practice)
    assert client.get('/api/v1/assessments/'+practice['id'],headers=headers(sid)).json()==practice
    result,payload=submit(client,repo,sid,practice)
    assert result['total_score']==100 and result['mastery_after']>result['mastery_before']
    before=repo.get_topic_state(sid,'la.vectors')
    replay=client.post('/api/v1/assessments/'+practice['id']+'/submit',headers=headers(sid),json=payload)
    assert replay.json()['idempotent_replay'] is True
    assert repo.get_topic_state(sid,'la.vectors')['evidence_count']==before['evidence_count']
    checkpoint=create(client,sid,purpose='checkpoint',count=5)
    assert len({q['question_type'] for q in checkpoint['questions']})>=2
    assert_public(checkpoint)
    assert client.post('/api/v1/assessments/'+checkpoint['id']+'/solution',headers=headers(sid),json={'question_id':checkpoint['questions'][0]['question_id'],'attempt':'test'}).status_code==422
    result,_=submit(client,repo,sid,checkpoint)
    assert result['total_score']==100 and result['gate']['passed']
    assert not llm.calls

def test_hint_and_repeat_suppress_mastery(content_app):
    client,repo,content,sid,llm,_=content_app
    practice=create(client,sid)
    response=client.post('/api/v1/assessments/'+practice['id']+'/hint',headers=headers(sid),json={'question_id':practice['questions'][0]['question_id']})
    assert response.status_code==200
    result,_=submit(client,repo,sid,practice)
    assert result['total_score']==100 and result['mastery_after']==result['mastery_before']==0.5
    with repo._connect() as c:
        ids=[r[0] for r in c.execute('SELECT bank_item_id FROM practice_questions WHERE practice_set_id=? ORDER BY position',(practice['id'],))]
    repeated=create(client,sid,bank_item_ids=ids)
    result,_=submit(client,repo,sid,repeated)
    assert result['mastery_after']==0.5
    assert repo.get_topic_state(sid,'la.vectors')['evidence_count']==0

def test_uncertain_then_visible_answer_retry_never_adds_evidence(content_app):
    client,repo,content,sid,llm,_=content_app
    value=create(client,sid)
    numeric=next(q for q in value['questions'] if q['question_type']=='numeric')
    result,_=submit(client,repo,sid,value,{numeric['question_id']:'NaN'})
    assert result['status']=='evaluation_uncertain' and result['mastery_after']==0.5
    result,_=submit(client,repo,sid,value)
    assert result['status']=='evaluated' and result['mastery_after']==0.5

def test_local_prerequisite_test_not_whole_course_mastery(content_app):
    client,repo,content,sid,llm,_=content_app
    response=client.post('/api/v1/prerequisite-tests',headers=headers(sid),json={'topic_id':'ml.linear_regression','prerequisite_topic_id':'la.dot_product','generation_mode':'bank'})
    assert response.status_code==201,response.text
    result,_=submit(client,repo,sid,response.json())
    assert result['total_score']==100 and result['mastery_after']==0.5
    unmet=repo.get_unmet_prerequisites(sid,'ml.linear_regression')
    assert 'la.dot_product' not in {r['id'] for r in unmet}
    assert repo.get_topic_state(sid,'la.dot_product')['status']!='mastered'
    assert repo.get_topic_state(sid,'math.functions') is None
    with repo._connect() as c:
        assert c.execute('SELECT COUNT(*) FROM prerequisite_verifications WHERE student_id=?',(sid,)).fetchone()[0]>0

def test_ownership_and_tampered_question_rejected(content_app):
    client,repo,content,sid,llm,_=content_app
    value=create(client,sid)
    other=client.post('/api/v1/students/anonymous').json()['student_id']
    assert client.get('/api/v1/assessments/'+value['id'],headers=headers(other)).status_code==404
    bad={'client_submission_id':str(uuid.uuid4()),'answers':[{'question_id':'someone-else','answer':'A'}]}
    assert client.post('/api/v1/assessments/'+value['id']+'/submit',headers=headers(sid),json=bad).status_code==422
    assert client.post('/api/v1/assessments/'+value['id']+'/submit',headers=headers(other),json=bad).status_code==404

def test_bank_insufficient_no_generation_and_blueprint_checks(content_app):
    client,repo,content,sid,llm,settings=content_app
    response=client.post('/api/v1/assessments',headers=headers(sid),json={'topic_ids':['la.vectors'],'question_count':8})
    assert response.status_code==422 and response.json()['error']['code']=='QUESTION_BANK_INSUFFICIENT'
    create(client,sid)
    response=client.post('/api/v1/assessments',headers=headers(sid),json={'topic_ids':['la.vectors']})
    assert response.json()['error']['code']=='QUESTION_BANK_INSUFFICIENT'
    assert not llm.calls

def test_json_jsonl_idempotent_import_and_immutable_versions(content_app):
    client,repo,content,sid,llm,_=content_app
    item=content.bank(topic_id='la.vectors',usage='practice',publication_status='published')[0]
    raw=item.model_dump(mode='json');raw.update(id='own-test',family_id='own-test',version=1)
    imported=parse_items(json.dumps([raw]));assert len(parse_items(json.dumps(raw)+'\n'+json.dumps({**raw,'id':'other'})))==2
    assert content.import_items(imported)=={'inserted':1,'skipped':0}
    assert content.import_items(imported)=={'inserted':0,'skipped':1}
    assert content.item('own-test',1).publication_status=='draft'
    with pytest.raises(ContentError):content.transition('own-test',1,'publish')
    with pytest.raises(ContentError):content.import_items([BankItem.model_validate({**raw,'stem':'different'})])
    content.transition('own-test',1,'publish',reviewed=True,review_note='test fixture review')
    assert content.transition('own-test',1,'validate').validation_status=='human_reviewed'
    value=create(client,sid,count=1,bank_item_ids=['own-test'])
    saved=repo.get_practice_set_internal(value['id'],student_id=sid).model_dump(mode='json')
    content.import_items([BankItem.model_validate({**raw,'version':2,'stem':'new version'})])
    content.transition('own-test',1,'archive')
    assert repo.get_practice_set_internal(value['id'],student_id=sid).model_dump(mode='json')==saved
    assert 'own-test' not in [x.id for x in content.bank(publication_status='published')]

def test_admin_protected_import_and_publish(content_app):
    client,repo,content,sid,llm,_=content_app
    assert client.get('/api/v1/admin/question-bank').status_code==401
    authorized={'X-Admin-Token':'test-admin-token'}
    response=client.get('/api/v1/admin/question-bank',headers=authorized)
    assert response.status_code==200 and 'answer_key' in response.text
    item=response.json()['items'][0];item.update(id='api-import',family_id='api-import')
    result=client.post('/api/v1/admin/question-bank/import',headers=authorized,json={'items':[item]})
    assert result.json()=={'inserted':1,'skipped':0}
    assert client.post('/api/v1/admin/question-bank/api-import/1/validate',headers=authorized).json()['validation_status']=='needs_review'
    assert client.post('/api/v1/admin/question-bank/api-import/1/publish',headers=authorized,json={}).status_code==422
    preflight=client.options('/api/v1/knowledge/documents/test/topics',headers={
        'Origin':'http://localhost:3000','Access-Control-Request-Method':'PATCH',
        'Access-Control-Request-Headers':'content-type,x-admin-token'})
    assert preflight.status_code==200

def test_formal_family_exclusion_and_archiving(content_app):
    client,repo,content,sid,llm,settings=content_app
    practice=create(client,sid)
    submit(client,repo,sid,practice)
    item=content.bank(topic_id='la.vectors',usage='checkpoint',publication_status='published')[0]
    content.exposure(sid,'different-id',1,item.family_id,'example')
    response=client.post('/api/v1/assessments',headers=headers(sid),json={'topic_ids':['la.vectors'],'purpose':'checkpoint','question_count':5})
    assert response.status_code==422 and response.json()['error']['code']=='QUESTION_BANK_INSUFFICIENT'
    content.transition(item.id,item.version,'archive')
    assert item.id not in [q.id for q in content.bank(publication_status='published',usage='checkpoint')]
    assert not llm.calls

def test_untrusted_numeric_claim_stays_review_required_and_variants_saved(content_app):
    client,repo,content,sid,llm,_=content_app
    item=content.bank(topic_id='la.vectors',usage='practice',publication_status='published')[0]
    value=item.model_copy(update={'id':'external-numeric','family_id':'external-numeric','accepted_variants':['equivalent statement']})
    assert 'equivalent statement' in value.as_question('q',1).answer_key.accepted_variants
    content.import_items([value])
    assert content.transition(value.id,1,'validate').validation_status=='needs_review'
    with pytest.raises(ContentError):content.transition(value.id,1,'publish')

def test_single_canonical_scope_is_explicit(content_app):
    client,repo,content,sid,llm,_=content_app
    response=client.post('/api/v1/assessments',headers=headers(sid),json={'topic_ids':['la.vectors','la.dot_product']})
    assert response.status_code==422 and response.json()['error']['code']=='CONTENT_INVALID'

def test_generated_candidates_only_persist_real_snapshot_as_unreviewed_draft(content_app):
    from app.api.dependencies import practice_service_from_request
    from app.services.learning_grounding import GroundingResult
    client,repo,content,sid,llm,_=content_app
    source={'source_id':'S1','document_id':'real-fixture-document','source_file':'fixture.md',
        'content':'真实测试资料：一元预测 y = wx + b，变量替换后逐步计算。',
        'chunk_index':0,'source_course':'machine_learning','topic_id':'ml.linear_regression'}
    question=content.bank(topic_id='ml.linear_regression',usage='practice',publication_status='published')[0].as_question('q1',1)
    question.source_ids=['S1']
    class FakeGrounding:
        async def retrieve(self,**kwargs):
            return GroundingResult(query='query',candidates=[],source_snapshot=[source])
    class Generator:
        grounding=FakeGrounding()
        async def _generate(self,**kwargs):
            assert kwargs['source_snapshot']==[source]
            return SimpleNamespace(questions=[question])
    client.app.dependency_overrides[practice_service_from_request]=Generator
    response=client.post('/api/v1/admin/question-bank/generate-draft',headers={'X-Admin-Token':'test-admin-token'},
        json={'topic_id':'ml.linear_regression','count':1})
    assert response.status_code==201,response.text
    generated=[item for item in content.bank() if item.content_origin=='generated_draft']
    assert len(generated)==1
    item=generated[0]
    assert item.publication_status=='draft' and item.validation_status=='needs_review'
    assert item.source_snapshot[0].content==source['content']
    assert item.source_snapshot[0].document_id==source['document_id']
    with pytest.raises(ContentError):content.transition(item.id,item.version,'publish')
    assert item.id not in [x['id'] for x in client.get('/api/v1/question-bank').json()['items']]

def test_linear_regression_full_sample_with_local_math_verification(content_app):
    client,repo,content,sid,llm,_=content_app
    assert client.get('/api/v1/topics/ml.linear_regression/lesson').status_code==200
    examples=client.get('/api/v1/topics/ml.linear_regression/examples').json()['examples']
    assert len(examples)==2
    for value in examples:
        client.post('/api/v1/examples/'+value['id']+'/exposure',headers=headers(sid),json={'version':value['version']})
    # Preview/practice first, then complete missing skills: the gate must
    # recompute, rather than trapping a previously "practicing" student.
    practice=create(client,sid,topic='ml.linear_regression')
    result,_=submit(client,repo,sid,practice)
    assert result['total_score']==100
    for required in repo.get_unmet_prerequisites(sid,'ml.linear_regression'):
        response=client.post('/api/v1/prerequisite-tests',headers=headers(sid),json={'topic_id':'ml.linear_regression','prerequisite_topic_id':required['id'],'generation_mode':'bank'})
        assert response.status_code==201,response.text
        result,_=submit(client,repo,sid,response.json())
        assert result['total_score']==100
    checkpoint=create(client,sid,topic='ml.linear_regression',purpose='checkpoint',count=5)
    result,_=submit(client,repo,sid,checkpoint)
    assert result['gate']['passed'] and result['total_score']==100
    assert not llm.calls

@pytest.mark.parametrize('answer,expected',[('1.2e3 m',True),('1200',True),('NaN m',False),('Infinity',False),('__import__("os")',False),('1200 kg',False)])
def test_numeric_units_scientific_and_unsafe_input(answer,expected):
    assert score_numeric(answer,1200,0.01,['m'])==(100.0,True) if expected else score_numeric(answer,1200,0.01,['m'])[1] is False
