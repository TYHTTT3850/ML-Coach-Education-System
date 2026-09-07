from __future__ import annotations
import argparse
import json
from pathlib import Path
from app.content.importing import parse_items
from app.content.seeds.core import build_content,verify_content
from app.schemas.content import BankItem,Lesson,WorkedExample

def main(argv=None):
    parser=argparse.ArgumentParser(description='校验课程结构与固定可信数值；不执行导入文件中的表达式或代码')
    parser.add_argument('file',type=Path,nargs='?')
    args=parser.parse_args(argv)
    if args.file:
        items=parse_items(args.file.read_text(encoding='utf-8-sig'))
        print(json.dumps({'structurally_valid':len(items),'validation_status':'needs_review','note':'外部题目仍需审核答案、解析、来源和 family'},ensure_ascii=False))
    else:
        payload=build_content()
        for key,model in [('lessons',Lesson),('examples',WorkedExample),('questions',BankItem)]:
            for value in payload[key]: model.model_validate(value)
        errors=verify_content()
        print(json.dumps({'counts':{k:len(v) for k,v in payload.items()},'errors':errors},ensure_ascii=False))
        if errors:return 1
    return 0

if __name__=='__main__': raise SystemExit(main())
