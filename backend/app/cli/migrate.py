from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.core.config import Settings
from app.db.migrations import LATEST_SCHEMA_VERSION, migrate_database


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="升级 Course Coach SQLite 数据库")
    parser.add_argument("--database", type=Path, help="覆盖配置中的 SQLite 路径")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    path = args.database or Settings().sqlite_path
    try:
        applied = migrate_database(path)
    except Exception as exc:
        print(f"迁移失败：{exc}", file=sys.stderr)
        return 1
    if applied:
        print(f"迁移完成：已应用版本 {', '.join(map(str, applied))}")
    else:
        print(f"数据库已是最新版本：{LATEST_SCHEMA_VERSION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
