"""Blueprint selection, exposure eligibility and local prerequisite evidence.

Grading and Beta updates stay in EvaluationService; all assessments are saved in
the existing practice tables. No generated fallback is used by this service.
"""
from __future__ import annotations

import itertools
import json
import uuid
from collections import Counter
from app.core.exceptions import TopicLockedError, InvalidQuestionAnswerError, PracticeSetNotFoundError
from app.core.types import Course
from app.schemas.content import AssessmentBlueprint
from app.schemas.practice import PracticeSetInternal,SourceSnapshotItem
from app.services.content_repository import ContentRepository,ContentError,BankInsufficient

RULE_VERSION='content-v1'

def evidence_allowed(repository,assessment_id):
    with repository._connect() as c:
        row=c.execute('SELECT purpose FROM practice_sets WHERE id=?',(assessment_id,)).fetchone()
        if not row or row['purpose']=='legacy': return True
        if row['purpose']=='prerequisite_test': return False
        invalid=c.execute('SELECT COUNT(*) FROM practice_questions WHERE practice_set_id=? AND (used_hint=1 OR solution_revealed=1 OR evidence_eligible=0 OR attempt_count>0)',(assessment_id,)).fetchone()[0]
        invalid+=c.execute("SELECT COUNT(*) FROM practice_questions q JOIN practice_sets ps ON ps.id=q.practice_set_id WHERE ps.id=? AND EXISTS (SELECT 1 FROM content_exposures e WHERE e.student_id=ps.student_id AND e.family_id=q.family_id AND e.kind IN ('example','solution','hint','feedback') AND (e.assessment_id IS NULL OR e.assessment_id<>ps.id))",(assessment_id,)).fetchone()[0]
    return invalid==0

class AssessmentService:
    def __init__(self,repository,settings):
        self.repository=repository
        self.content=ContentRepository(repository)
        self.settings=settings

    def create(self,student_id,blueprint:AssessmentBlueprint,*,prerequisite_for=None):
        self.repository.require_student(student_id)
        if len(blueprint.topic_ids)!=1:
            raise ContentError('本阶段按单个 canonical 知识点组卷，避免将跨知识点成绩错误累计；请分别创建测试。')
        topics=[self.repository.require_topic(t) for t in blueprint.topic_ids]
        course=Course(topics[0]['course'])
        if any(t['course']!=course.value for t in topics):
            raise ContentError('一份小测只检测同一知识域的能力，请分次完成各数学前置。')
        if blueprint.purpose=='prerequisite_test' and not prerequisite_for:
            raise ContentError('请通过前置小测入口按已注册关系创建')
        if blueprint.purpose=='checkpoint':
            for topic in topics:
                unmet=self.repository.get_unmet_prerequisites(student_id,topic['id'])
                if unmet:
                    raise TopicLockedError('正式测试前请补充：'+'、'.join(t['title'] for t in unmet)+'；可先预览或做前置小测。')
                state=self.repository.get_topic_state(student_id,topic['id']) or {}
                from app.services.learning_service import LearningService
                gate=LearningService(settings=self.settings,repository=self.repository).evaluate_topic_gate(student_id,topic,state or None)
                if state.get('status') not in {'mastered','ready_for_checkpoint'} and gate.next_action!='take_checkpoint':
                    raise TopicLockedError('请先完成无提示练习，积累足够证据后再参加正式测试。')
        required_dimensions=list(topics[0]['required_dimensions'])
        with self.repository._connect() as c:
            c.execute('BEGIN IMMEDIATE')
            candidates=self.content.bank(course=course.value,publication_status='published',usage=blueprint.purpose)
            candidates=[q for q in candidates if set(q.topic_ids)&set(blueprint.topic_ids) and q.validation_status in {'deterministic_checks_passed','human_reviewed'}]
            if blueprint.allowed_question_types: candidates=[q for q in candidates if q.question_type in blueprint.allowed_question_types]
            if blueprint.bank_item_ids:
                candidates=[q for q in candidates if q.id in blueprint.bank_item_ids]
                if len(candidates)!=len(set(blueprint.bank_item_ids)) or blueprint.question_count!=len(candidates):
                    raise BankInsufficient('所选题目不是当前已发布练习题，请返回题库重新选择。')
            exposures=c.execute('SELECT content_id,family_id,kind FROM content_exposures WHERE student_id=?',(student_id,)).fetchall()
            seen={x['content_id'] for x in exposures}
            exposed_families={x['family_id'] for x in exposures}
            formal=blueprint.purpose!='practice'
            if formal:
                # Published teaching solutions are public. Never use their
                # families for formal evidence, even if exposure wasn't logged.
                teaching_families={r[0] for r in c.execute("SELECT family_id FROM worked_examples WHERE status='published'")}
                teaching_families.update(q.family_id for q in self.content.bank(publication_status='published',usage='practice'))
                candidates=[q for q in candidates if q.family_id not in teaching_families]
            if formal or (blueprint.exclude_seen and not blueprint.bank_item_ids): candidates=[q for q in candidates if q.id not in seen]
            if formal or (blueprint.exclude_exposed_families and not blueprint.bank_item_ids): candidates=[q for q in candidates if q.family_id not in exposed_families]
            selected=self._select(candidates,blueprint,required_dimensions)
            assessment_id=str(uuid.uuid4()); blueprint_id=str(uuid.uuid4())
            questions=[];sources={}
            for index,item in enumerate(selected,1):
                question=item.as_question(str(uuid.uuid4()),index)
                questions.append(question)
                if item.source_snapshot:
                    for s in item.source_snapshot:
                        # RAG-local S1 ids may overlap across bank items: namespace
                        # the saved ids without changing any actual source text.
                        sid=f'{item.id}:v{item.version}:{s.source_id}'
                        sources[sid]=s.model_copy(update={'source_id':sid})
                    question.source_ids=[f'{item.id}:v{item.version}:{s.source_id}' for s in item.source_snapshot]
                else:
                    sid=question.source_ids[0]
                    sources[sid]=SourceSnapshotItem(source_id=sid,document_id=item.id,
                        source_file='项目自编教学示例',title=item.title,chunk_index=0,
                        content=item.stem+'\n'+ '\n'.join(item.solution_steps),
                        source_course=course.value,topic_id=item.topic_ids[0])
            internal=PracticeSetInternal(practice_set_id=assessment_id,course=course,
                topic_id=topics[0]['id'],kind='checkpoint' if blueprint.purpose=='checkpoint' else 'practice',
                questions=questions,required_dimensions=required_dimensions,
                source_snapshot=list(sources.values()),assessment_purpose=blueprint.purpose)
            c.execute('INSERT INTO assessment_blueprints VALUES (?,?,?)',(blueprint_id,blueprint.model_dump_json(),self.repository._now()))
            self.repository.create_practice_set(student_id=student_id,practice_set=internal,_connection=c)
            c.execute('UPDATE practice_sets SET blueprint_id=?,purpose=?,prerequisite_for=?,rule_version=?,required_dimensions_snapshot_json=? WHERE id=?',
                (blueprint_id,blueprint.purpose,prerequisite_for,RULE_VERSION,json.dumps(required_dimensions),assessment_id))
            for question,item in zip(questions,selected,strict=True):
                c.execute('UPDATE practice_questions SET bank_item_id=?,item_version=?,family_id=?,topic_ids_json=?,competencies_json=?,solution_steps_json=?,evidence_eligible=? WHERE id=?',
                    (item.id,item.version,item.family_id,json.dumps(item.topic_ids),json.dumps(item.tested_competencies),json.dumps(item.solution_steps),int(item.family_id not in exposed_families),question.question_id))
                c.execute('INSERT INTO content_exposures(student_id,content_id,content_version,family_id,kind,assessment_id,created_at) VALUES (?,?,?,?,?,?,?)',
                    (student_id,item.id,item.version,item.family_id,'assigned',assessment_id,self.repository._now()))
        return self.get(student_id,assessment_id)

    def _select(self,candidates,bp,dimensions):
        # Bounded depth search checks actual coverage before persisting any set.
        if len(candidates)<bp.question_count: raise BankInsufficient()
        required=set(bp.required_competencies)
        count=0
        for chosen in itertools.combinations(candidates,bp.question_count):
            count+=1
            if count>100000: break
            if len({q.id for q in chosen})!=bp.question_count or len({q.family_id for q in chosen})!=bp.question_count: continue
            if not required.issubset({k for q in chosen for k in q.tested_competencies}): continue
            if not set(bp.topic_ids).issubset({t for q in chosen for t in q.topic_ids}): continue
            if bp.difficulty_distribution and dict(Counter(q.difficulty for q in chosen))!={k:v for k,v in bp.difficulty_distribution.items() if v}: continue
            if bp.purpose=='checkpoint':
                if len({q.question_type for q in chosen})<2 and not (len(required)==1 and len(bp.allowed_question_types)==1): continue
                if len({d for q in chosen for d in q.dimensions}&set(dimensions))<min(3,len(dimensions)): continue
                if not any(q.difficulty in {'medium','hard'} for q in chosen): continue
            return list(chosen)
        raise BankInsufficient()

    def get(self,student_id,assessment_id):
        self.repository.require_student(student_id)
        internal=self.repository.get_practice_set_internal(assessment_id,student_id=student_id)
        with self.repository._connect() as c:
            row=c.execute('SELECT purpose,status FROM practice_sets WHERE id=?',(assessment_id,)).fetchone()
            result=c.execute('SELECT response_json FROM practice_submissions WHERE practice_set_id=? AND student_id=? AND response_json IS NOT NULL ORDER BY created_at DESC LIMIT 1',(assessment_id,student_id)).fetchone()
        return {'id':assessment_id,'practice_set_id':assessment_id,'topic_id':internal.topic_id,
            'mode':internal.kind,'purpose':row['purpose'],'status':row['status'],
            'questions':[q.to_public().model_dump(mode='json') for q in internal.questions],
            'source_summary':[{'source_id':s.source_id,'source_file':s.source_file,'title':s.title} for s in internal.source_snapshot],
            'result':json.loads(result['response_json']) if result else None}

    def prerequisite_target(self,student_id,payload):
        self.repository.require_student(student_id)
        self.repository.require_topic(payload.topic_id)
        unmet=self.repository.get_unmet_prerequisites(student_id,payload.topic_id)
        target=next((p for p in unmet if not payload.prerequisite_topic_id or p['id']==payload.prerequisite_topic_id),None)
        if not target: raise ContentError('没有对应的未满足正式前置要求，请返回知识点查看最新状态。')
        competencies=target.get('required_competencies') or [target['id']+'.foundation']
        return target,competencies

    def prerequisite_test(self,student_id,payload):
        target,competencies=self.prerequisite_target(student_id,payload)
        return self.create(student_id,AssessmentBlueprint(topic_ids=[target['id']],purpose='prerequisite_test',question_count=payload.question_count,required_competencies=competencies),prerequisite_for=payload.topic_id)

    def help(self,student_id,assessment_id,payload,*,solution=False):
        self.repository.get_practice_set_internal(assessment_id,student_id=student_id)
        with self.repository._connect() as c:
            c.execute('BEGIN IMMEDIATE')
            mode=c.execute('SELECT purpose FROM practice_sets WHERE id=?',(assessment_id,)).fetchone()['purpose']
            if mode!='practice': raise ContentError('正式测试和前置小测提交前不提供提示或解法，请先独立作答。')
            if c.execute("SELECT 1 FROM practice_submissions WHERE practice_set_id=? AND evaluation_status='processing'",(assessment_id,)).fetchone():
                raise ContentError('正在评价，请等待结果后再查看解法。')
            row=c.execute('SELECT * FROM practice_questions WHERE id=? AND practice_set_id=?',(payload.question_id,assessment_id)).fetchone()
            if not row: raise InvalidQuestionAnswerError()
            if solution and not payload.attempt.strip(): raise ContentError('请先尝试作答，再查看完整解法。')
            field='solution_revealed' if solution else 'used_hint'
            c.execute(f'UPDATE practice_questions SET {field}=1,evidence_eligible=0 WHERE id=?',(payload.question_id,))
            c.execute('INSERT INTO content_exposures(student_id,content_id,content_version,family_id,kind,assessment_id,created_at) VALUES (?,?,?,?,?,?,?)',
                (student_id,row['bank_item_id'],row['item_version'],row['family_id'],'solution' if solution else 'hint',assessment_id,self.repository._now()))
        if solution:
            return {'solution_steps':json.loads(row['solution_steps_json']),'answer':json.loads(row['answer_key_json'])['canonical_answer']}
        return {'hint':'先列出已知量、目标量及公式的适用条件，再逐步代入；检查符号、维度与单位。本次使用提示将不计为独立掌握证据。'}

    def after_submit(self,student_id,assessment_id,response):
        with self.repository._connect() as c:
            row=c.execute('SELECT purpose,prerequisite_for,topic_id,rule_version FROM practice_sets WHERE id=? AND student_id=?',(assessment_id,student_id)).fetchone()
            if not row: raise PracticeSetNotFoundError()
            if row['purpose']!='prerequisite_test' or response.status!='evaluated' or response.total_score<80: return
            qs=c.execute('SELECT id,competencies_json,attempt_count,evidence_eligible,used_hint,solution_revealed FROM practice_questions WHERE practice_set_id=?',(assessment_id,)).fetchall()
            if any(q['attempt_count']!=1 or not q['evidence_eligible'] or q['used_hint'] or q['solution_revealed'] for q in qs): return
            scores={e.question_id:e for e in response.evaluations}
            competencies={k for q in qs for k in json.loads(q['competencies_json'])}
            for competency in competencies:
                covered=[scores[q['id']] for q in qs if competency in json.loads(q['competencies_json'])]
                if not covered or not all(e.evaluation_reliable for e in covered) or sum(e.score for e in covered)/len(covered)<80: continue
                c.execute('INSERT INTO prerequisite_verifications(student_id,dependent_topic_id,prerequisite_topic_id,competency,rule_version,assessment_id,verified_at) VALUES (?,?,?,?,?,?,?) ON CONFLICT(student_id,dependent_topic_id,prerequisite_topic_id,competency,rule_version) DO NOTHING',
                    (student_id,row['prerequisite_for'],row['topic_id'],competency,row['rule_version'],assessment_id,self.repository._now()))
