"""Read-only source audit and recoverable SQLite backup outside backend/data.

The manifest contains counts/hashes only, never document text or credentials.
Run capture before an authorized migration, compare after it.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sqlite3
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'backend'/'data'
AUDIT=ROOT/'.runtime'/'content-upgrade-audit'
PROTECTED=('documents','students','student_topic_states','student_dimension_states',
    'practice_sets','practice_questions','practice_submissions','practice_attempts',
    'misconception_events','learning_sessions')

def digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,default=str).encode()).hexdigest()

def connect():
    value=sqlite3.connect((DATA/'documents.db').as_uri()+'?mode=ro',uri=True)
    value.row_factory=sqlite3.Row
    return value

def files():
    return {str(p.relative_to(DATA)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (DATA/'uploads').rglob('*') if p.is_file()}

def capture():
    AUDIT.mkdir(parents=True,exist_ok=True)
    manifest=AUDIT/'before.json'
    if manifest.exists():raise SystemExit('Baseline already exists; preserved without overwrite.')
    with connect() as c:
        values={}
        for table in PROTECTED:
            rows=c.execute(f'SELECT * FROM "{table}"').fetchall()
            columns=[r['name'] for r in c.execute(f'PRAGMA table_info("{table}")')]
            values[table]={'columns':columns,'rows':sorted(digest(dict(r)) for r in rows),'count':len(rows)}
        backup=AUDIT/'documents.before.sqlite'
        if backup.exists():raise SystemExit('Backup already exists; preserved without overwrite.')
        with sqlite3.connect(backup) as target:c.backup(target)
        result={'tables':values,'files':files(),'integrity_check':c.execute('PRAGMA integrity_check').fetchone()[0]}
    manifest.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'backup':str(backup),'integrity_check':result['integrity_check'],'counts':{t:v['count'] for t,v in values.items()}},ensure_ascii=False))

def compare():
    before=json.loads((AUDIT/'before.json').read_text(encoding='utf-8'))
    result={}
    with connect() as c:
        for table,value in before['tables'].items():
            columns=','.join('"'+col+'"' for col in value['columns'])
            current=[digest(dict(r)) for r in c.execute(f'SELECT {columns} FROM "{table}"')]
            result[table]={'original_count':value['count'],'current_count':len(current),'all_original_rows_unchanged':all(h in current for h in value['rows'])}
        integrity=c.execute('PRAGMA integrity_check').fetchone()[0]
        foreign=len(c.execute('PRAGMA foreign_key_check').fetchall())
    after_files=files()
    unchanged=all(after_files.get(p)==h for p,h in before['files'].items())
    output={'integrity_check':integrity,'foreign_key_violations':foreign,'tables':result,'original_uploads_unchanged':unchanged}
    (AUDIT/'after.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(output,ensure_ascii=False))
    return 0 if integrity=='ok' and foreign==0 and unchanged and all(r['all_original_rows_unchanged'] for r in result.values()) else 1

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['capture','compare'])
    args=parser.parse_args()
    raise SystemExit(capture() if args.action=='capture' else compare())
