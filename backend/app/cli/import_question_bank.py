from __future__ import annotations
import argparse
import json
from pathlib import Path
from app.core.config import Settings
from app.content.importing import parse_items
from app.services.learning_repository import LearningRepository
from app.services.content_repository import ContentRepository

def main(argv=None):
    parser=argparse.ArgumentParser(description='导入 JSON/JSONL 为待审核 draft，已有版本不覆盖')
    parser.add_argument('file',type=Path);parser.add_argument('--database',type=Path)
    args=parser.parse_args(argv)
    values=parse_items(args.file.read_text(encoding='utf-8-sig'))
    repo=LearningRepository(args.database or Settings().sqlite_path);repo.initialize()
    print(json.dumps(ContentRepository(repo).import_items(values),ensure_ascii=False))
    return 0

if __name__=='__main__': raise SystemExit(main())
