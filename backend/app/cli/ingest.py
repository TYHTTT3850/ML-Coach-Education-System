from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.core.config import Settings
from app.core.logging import configure_logging
from app.core.types import Course
from app.rag.loaders import ALLOWED_EXTENSIONS
from app.rag.vector_store import ChromaVectorStore
from app.services.document_registry import DocumentRegistry
from app.services.embedding.factory import create_embedding_client
from app.services.knowledge_service import KnowledgeService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="批量导入 Course Coach 课程资料")
    parser.add_argument("--course", required=True, choices=[course.value for course in Course])
    parser.add_argument("--path", required=True, type=Path, help="单个文件或资料目录")
    parser.add_argument("--topic-id", action="append", default=[], help="canonical知识点ID，可重复；标签必须与资料课程一致")
    return parser


def discover_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(
            candidate
            for candidate in path.rglob("*")
            if candidate.is_file() and candidate.suffix.lower() in ALLOWED_EXTENSIONS
        )
    return []


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = Settings()
    configure_logging(
        settings.app_log_level,
        (
            settings.embedding_api_key.get_secret_value(),
            settings.llm_api_key.get_secret_value(),
            settings.admin_token.get_secret_value(),
        ),
    )
    input_path = args.path.resolve()
    if not input_path.exists():
        print(f"失败：路径不存在：{input_path}", file=sys.stderr)
        return 2
    files = discover_files(input_path)
    if not files:
        print("失败：没有找到支持的 .md、.txt、.pdf 或 .docx 文件", file=sys.stderr)
        return 2

    try:
        settings.ensure_data_directories()
        registry = DocumentRegistry(settings.sqlite_path)
        registry.initialize()
        vector_store = ChromaVectorStore(settings.chroma_path, settings.chroma_distance_metric)
        vector_store.initialize()
        embedding_client = create_embedding_client(settings)
        service = KnowledgeService(
            settings=settings,
            registry=registry,
            vector_store=vector_store,
            embedding_client=embedding_client,
        )
    except Exception as exc:
        print(f"失败：无法初始化导入服务：{exc}", file=sys.stderr)
        return 2

    course = Course(args.course)
    succeeded = skipped = failed = 0
    for file_path in files:
        try:
            result = service.ingest_path(course=course, path=file_path, topic_ids=args.topic_id)
            if result.duplicate:
                skipped += 1
                print(f"[跳过] {file_path}")
            else:
                succeeded += 1
                print(f"[成功] {file_path} ({result.document['chunk_count']} chunks)")
        except Exception as exc:
            failed += 1
            print(f"[失败] {file_path}: {exc}", file=sys.stderr)

    print(f"导入完成：成功 {succeeded}，跳过 {skipped}，失败 {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
