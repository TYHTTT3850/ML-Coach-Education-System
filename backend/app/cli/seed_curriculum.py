from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.core.config import Settings
from app.core.types import Course
from app.curriculum.seed import DEFAULT_SEED_PATHS, seed_curriculum
from app.services.learning_repository import LearningRepository


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="导入 Course Coach 默认课程知识点")
    parser.add_argument("--database", type=Path, help="覆盖配置中的 SQLite 路径")
    parser.add_argument(
        "--seed",
        action="append",
        type=Path,
        help="自定义 Seed JSON；可重复传入。省略时导入两门默认课程",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    path = args.database or Settings().sqlite_path
    repository = LearningRepository(path)
    try:
        repository.initialize()
        counts = seed_curriculum(repository, args.seed or DEFAULT_SEED_PATHS)
        if not args.seed:
            counts = {course.value: len(repository.list_topics(course)) for course in Course}
    except Exception as exc:
        print(f"Seed 导入失败：{exc}", file=sys.stderr)
        return 1
    summary = "，".join(f"{course} {count} 个" for course, count in counts.items())
    print(f"课程 Seed 导入完成：{summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
