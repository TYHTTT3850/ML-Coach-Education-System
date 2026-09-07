from __future__ import annotations

import json
from typing import Any
from app.core.exceptions import AppError
from app.schemas.content import BankItem, Lesson, WorkedExample
from app.services.learning_repository import LearningRepository


class ContentError(AppError):
    status_code = 422
    code = 'CONTENT_INVALID'

class ContentNotReady(ContentError):
    status_code = 404
    code = 'CONTENT_NOT_READY'
    public_message = '此知识点内容待补充，可先阅读简介、补学前置或向课程助手提问。'

class BankInsufficient(ContentError):
    code = 'QUESTION_BANK_INSUFFICIENT'
    public_message = '题库中未见过的独立题目或能力覆盖不足，请先做练习或请管理员补充题库。'

class ContentRepository:
    def __init__(self, repository: LearningRepository):
        self.repository = repository

    def _connect(self):
        return self.repository._connect()

    @staticmethod
    def dump(value):
        return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'))

    def _validate_topic(self, item):
        from app.schemas.practice import _MACHINE_LEARNING_QUESTION_TYPES, _ADVANCED_MATHEMATICS_QUESTION_TYPES
        types=_MACHINE_LEARNING_QUESTION_TYPES | _ADVANCED_MATHEMATICS_QUESTION_TYPES
        if item.question_type not in types:
            raise ContentError('该题型不属于当前课程支持范围，请检查主课程与题型。')
        for tid in item.topic_ids:
            topic=self.repository.require_topic(tid)
            if topic['course'] != item.primary_course.value:
                raise ContentError('题目知识点必须属于声明的主课程；跨课蓝图应选择注册前置。')
        allowed=({'concept','mathematics','mechanism','engineering','application'} if item.primary_course.value=='machine_learning' else {'concept','theorem_condition','derivation','calculation','application'})
        if not set(item.dimensions).issubset(allowed):
            raise ContentError('能力维度不属于课程')

    def import_items(self, items: list[BankItem], *, trusted_seed=False):
        normalized=[]
        for item in items:
            self._validate_topic(item)
            if not trusted_seed:
                item=item.model_copy(update={'publication_status':'draft','validation_status':'needs_review'})
            normalized.append(item)
        inserted=skipped=0
        now=self.repository._now()
        with self._connect() as c:
            for item in normalized:
                payload=item.model_dump(mode='json',exclude={'created_at','updated_at'})
                old=c.execute('SELECT payload_json FROM question_bank_items WHERE id=? AND version=?',(item.id,item.version)).fetchone()
                if old:
                    previous=json.loads(old['payload_json'])
                    # Publication is separately managed; idempotent import never unpublishes.
                    for key in ('publication_status','validation_status'):
                        payload[key]=previous.get(key,payload[key])
                    if previous != payload:
                        raise ContentError('已有题目版本不可覆盖，请增加 version 后重新导入。')
                    skipped+=1
                    continue
                c.execute('INSERT INTO question_bank_items VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                    (item.id,item.version,item.primary_course.value,item.question_type,item.difficulty,item.family_id,item.publication_status,item.validation_status,self.dump(payload),now,now))
                c.executemany('INSERT INTO question_bank_topics VALUES (?,?,?)',[(item.id,item.version,t) for t in item.topic_ids])
                inserted+=1
        return {'inserted':inserted,'skipped':skipped}

    def seed(self, payload: dict[str,Any]):
        for model, table, key in ((Lesson,'lessons','lessons'),(WorkedExample,'worked_examples','examples')):
            with self._connect() as c:
                for raw in payload[key]:
                    value=model.model_validate(raw)
                    self.repository.require_topic(value.topic_id)
                    old=c.execute(f'SELECT payload_json FROM {table} WHERE id=? AND version=?',(value.id,value.version)).fetchone()
                    serialized=self.dump(value.model_dump(mode='json'))
                    if old:
                        if old['payload_json'] != serialized:
                            raise ContentError('已存在内容版本不允许原地修改，请增加 version。')
                        continue
                    if table=='lessons':
                        c.execute('INSERT INTO lessons VALUES (?,?,?,?,?,?)',(value.id,value.version,value.topic_id,value.status,serialized,self.repository._now()))
                    else:
                        c.execute('INSERT INTO worked_examples VALUES (?,?,?,?,?,?,?)',(value.id,value.version,value.topic_id,value.family_id,value.status,serialized,self.repository._now()))
        return self.import_items([BankItem.model_validate(x) for x in payload['questions']],trusted_seed=True)

    def lesson(self, topic_id):
        self.repository.require_topic(topic_id)
        with self._connect() as c:
            row=c.execute("SELECT payload_json FROM lessons WHERE topic_id=? AND status='published' ORDER BY version DESC LIMIT 1",(topic_id,)).fetchone()
        if not row: raise ContentNotReady()
        return Lesson.model_validate_json(row['payload_json'])

    def examples(self, topic_id):
        self.repository.require_topic(topic_id)
        with self._connect() as c:
            rows=c.execute("SELECT e.payload_json FROM worked_examples e WHERE e.topic_id=? AND e.status='published' AND e.version=(SELECT MAX(x.version) FROM worked_examples x WHERE x.id=e.id AND x.status='published') ORDER BY e.id",(topic_id,)).fetchall()
        return [WorkedExample.model_validate_json(row['payload_json']) for row in rows]

    def example(self, example_id,version=None):
        with self._connect() as c:
            sql="SELECT payload_json FROM worked_examples WHERE id=? AND status='published'"
            args=[example_id]
            if version is not None: sql+=' AND version=?'; args.append(version)
            row=c.execute(sql+' ORDER BY version DESC LIMIT 1',args).fetchone()
        if not row: raise ContentNotReady()
        return WorkedExample.model_validate_json(row['payload_json'])

    def bank(self, *,course=None,topic_id=None,question_type=None,difficulty=None,publication_status=None,q=None,usage=None,latest=True):
        sql='SELECT b.* FROM question_bank_items b WHERE 1=1';args=[]
        for col,val in (('primary_course',course),('question_type',question_type),('difficulty',difficulty),('publication_status',publication_status)):
            if val:
                sql+=f' AND b.{col}=?';args.append(val)
        if topic_id:
            sql+=' AND EXISTS(SELECT 1 FROM question_bank_topics t WHERE t.item_id=b.id AND t.item_version=b.version AND t.topic_id=?)';args.append(topic_id)
        if latest:
            # A newly drafted version must not remove the previous published version.
            status_clause=' AND x.publication_status=b.publication_status' if publication_status else ''
            sql+=f' AND b.version=(SELECT MAX(x.version) FROM question_bank_items x WHERE x.id=b.id{status_clause})'
        with self._connect() as c: rows=c.execute(sql+' ORDER BY b.id,b.version',args).fetchall()
        items=[self._item(row) for row in rows]
        return [v for v in items if (not usage or usage in v.usage) and (not q or q.casefold() in (v.title+' '+v.stem).casefold())]

    def _item(self,row):
        data=json.loads(row['payload_json'])
        data.update(publication_status=row['publication_status'],validation_status=row['validation_status'],created_at=row['created_at'],updated_at=row['updated_at'])
        return BankItem.model_validate(data)

    def item(self,item_id,version):
        with self._connect() as c: row=c.execute('SELECT * FROM question_bank_items WHERE id=? AND version=?',(item_id,version)).fetchone()
        if not row: raise ContentNotReady()
        return self._item(row)

    def transition(self,item_id,version,action,*,reviewed=False,review_note=''):
        item=self.item(item_id,version)
        if action=='validate':
            # Schema consistency is not a substantive review. Only the bundled
            # independently checked catalog can receive automatic validation.
            checked=self._trusted_check(item)
            validation=('human_reviewed' if item.validation_status=='human_reviewed' else
                'deterministic_checks_passed' if checked else 'needs_review')
            status=item.publication_status
        elif action=='publish':
            validation=item.validation_status
            if reviewed and review_note.strip(): validation='human_reviewed'
            elif not self._trusted_check(item) and validation!='human_reviewed':
                raise ContentError('内容仅完成结构校验。请检查答案、解析和来源后提交人工审核说明。')
            elif validation != 'human_reviewed': validation='deterministic_checks_passed'
            if item.content_origin=='generated_draft' and validation!='human_reviewed':
                raise ContentError('生成候选题需人工审核后发布')
            status='published'
        elif action=='archive': validation=item.validation_status;status='archived'
        else: raise ContentError('非法题目状态操作')
        with self._connect() as c:
            c.execute('UPDATE question_bank_items SET publication_status=?,validation_status=?,updated_at=? WHERE id=? AND version=?',(status,validation,self.repository._now(),item_id,version))
            c.execute('INSERT INTO content_review_events(item_id,item_version,action,review_note,created_at) VALUES (?,?,?,?,?)',
                (item_id,version,action,review_note if reviewed else '',self.repository._now()))
        return self.item(item_id,version)

    def _trusted_check(self,item):
        from app.content.seeds.core import build_content, verify_content
        if verify_content(): return False
        raw=next((x for x in build_content()['questions'] if x['id']==item.id and x['version']==item.version),None)
        if raw is None: return False
        original=BankItem.model_validate(raw)
        ignore={'created_at','updated_at','publication_status','validation_status'}
        return original.validation_status=='deterministic_checks_passed' and original.model_dump(exclude=ignore)==item.model_dump(exclude=ignore)

    def exposure(self,student_id,content_id,version,family_id,kind,assessment_id=None):
        with self._connect() as c:
            c.execute('INSERT INTO content_exposures(student_id,content_id,content_version,family_id,kind,assessment_id,created_at) VALUES (?,?,?,?,?,?,?)',(student_id,content_id,version,family_id,kind,assessment_id,self.repository._now()))

    def coverage(self):
        with self._connect() as c:
            topics=c.execute('SELECT id,title,course,required_dimensions_json FROM curriculum_topics WHERE active=1 ORDER BY course,stage,order_index').fetchall()
            ls=c.execute("SELECT topic_id,COUNT(DISTINCT id) n FROM lessons WHERE status='published' GROUP BY topic_id").fetchall()
            es=c.execute("SELECT topic_id,COUNT(DISTINCT id) n FROM worked_examples WHERE status='published' GROUP BY topic_id").fetchall()
            example_families={row['family_id'] for row in c.execute("SELECT DISTINCT family_id FROM worked_examples WHERE status='published'")}
        lc={r['topic_id']:r['n'] for r in ls}; ec={r['topic_id']:r['n'] for r in es}
        # Drafting/archiving a new version must not hide a still-published older
        # version from readiness. Publication and pending review are independent
        # inventories, just like the actual question selection endpoint.
        bank=self.bank(latest=False)
        published=self.bank(publication_status='published')
        eligible=[item for item in published if item.validation_status in {'deterministic_checks_passed','human_reviewed'}]
        # These are the same public-teaching exclusions used by formal selection.
        formal_exclusions=example_families | {item.family_id for item in published if 'practice' in item.usage}
        from app.schemas.content import AssessmentBlueprint
        from app.services.assessment_service import AssessmentService
        selector=AssessmentService(self.repository, None)
        out=[]
        for t in topics:
            items=[x for x in published if t['id'] in x.topic_ids]
            practice=[item for item in eligible if t['id'] in item.topic_ids and 'practice' in item.usage and item.family_id not in example_families]
            checkpoint=[item for item in eligible if t['id'] in item.topic_ids and 'checkpoint' in item.usage and item.family_id not in formal_exclusions]
            independent_practice_count=len({item.family_id for item in practice})
            independent_checkpoint_count=len({item.family_id for item in checkpoint})
            dimensions=set(json.loads(t['required_dimensions_json']))
            checkpoint_dimension_count=len({dimension for item in checkpoint for dimension in item.dimensions} & dimensions)
            checkpoint_type_count=len({item.question_type for item in checkpoint})
            checkpoint_ready=False
            if independent_checkpoint_count >= 5 and checkpoint_dimension_count >= 3 and checkpoint_type_count >= 2:
                try:
                    # Reuse the actual bounded blueprint selector; aggregate
                    # counts alone cannot prove one valid five-question set.
                    selector._select(checkpoint,AssessmentBlueprint(topic_ids=[t['id']],purpose='checkpoint',question_count=5),list(dimensions))
                    checkpoint_ready=True
                except BankInsufficient:
                    pass
            gaps=[]
            if lc.get(t['id'],0)<1: gaps.append('缺少已发布讲解')
            if ec.get(t['id'],0)<2: gaps.append('完整例题不足 2 道')
            if independent_practice_count<4: gaps.append('排除例题同源题后的独立练习不足 4 道')
            if not checkpoint_ready: gaps.append('独立正式题不能组成至少 5 题、3 个能力维度及 2 种题型的有效测试')
            out.append({'topic_id':t['id'],'title':t['title'],'owner_course':t['course'],
                'lesson_count':lc.get(t['id'],0),'example_count':ec.get(t['id'],0),
                'practice_count':sum('practice' in x.usage for x in items),
                'checkpoint_count':sum('checkpoint' in x.usage for x in items),
                'independent_practice_count':independent_practice_count,
                'independent_checkpoint_count':independent_checkpoint_count,
                'checkpoint_dimension_count':checkpoint_dimension_count,
                'checkpoint_question_type_count':checkpoint_type_count,
                'checkpoint_blueprint_ready':checkpoint_ready,
                'local_flow_ready':not gaps,
                'has_learning_content':bool(lc.get(t['id'],0) or ec.get(t['id'],0)),
                'readiness_gaps':gaps,
                'prerequisite_test_count':sum('prerequisite_test' in x.usage for x in items),
                'published_count':len(items),
                'needs_review_count':sum(t['id'] in x.topic_ids and x.validation_status=='needs_review' and x.publication_status!='archived' for x in bank)})
        byid={t['topic_id']:t for t in out}
        dependencies={topic['id']:[p for p in self.repository.list_prerequisites(topic['id']) if p.get('relation_type','required')=='required' and p.get('scope','topic')=='topic'] for topic in topics}
        memo={}
        def ready(tid,seen):
            if tid in seen: return False
            if tid in memo: return memo[tid]
            row=byid.get(tid,{})
            value=bool(row.get('local_flow_ready')) and all(ready(p['id'],seen|{tid}) for p in dependencies.get(tid,[]))
            memo[tid]=value
            return value
        for row in out:
            unmet=[p for p in dependencies[row['topic_id']] if not ready(p['id'],set())]
            row['prerequisites_ready']=not unmet
            row['full_flow_ready']=ready(row['topic_id'],set())
            if unmet:
                row['readiness_gaps'].append('前置内容待补充：'+'、'.join(p['title'] for p in unmet))
        return {'topics':out}
