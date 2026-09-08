"""Initialize an empty PostgreSQL database with the current application schema."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.config import Settings
from app.db.database import PostgresDatabase


SCHEMA_FILE = Path(__file__).with_name("initialize_postgresql.sql")


def initialize_database(database_url: str) -> None:
    """Create all current application tables, indexes, functions, and triggers."""

    schema = SCHEMA_FILE.read_text(encoding="utf-8")
    with PostgresDatabase(database_url).connect() as connection:
        connection.execute_script(schema)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="初始化 Course Coach PostgreSQL 数据库")
    parser.add_argument("--database-url", help="覆盖配置中的 PostgreSQL DATABASE_URL")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        initialize_database(args.database_url or Settings().database_url)
    except Exception as exc:
        print(f"初始化失败：{exc}", file=sys.stderr)
        return 1
    print("PostgreSQL 数据库初始化完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())