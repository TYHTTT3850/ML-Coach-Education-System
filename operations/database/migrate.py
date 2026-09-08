"""One-time PostgreSQL schema initialization and migration entry point."""
from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.config import Settings
from app.db.database import PostgresDatabase


SQL_DIRECTORY = Path(__file__).with_name("sql")


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    sql_file: str | None


MIGRATIONS: Sequence[Migration] = (
    Migration(1, "learning_loop_schema", "001_learning_loop_schema.sql"),
    Migration(2, "learning_loop_v1_compatibility", None),
    Migration(3, "course_content_curriculum", "003_course_content_curriculum.sql"),
    Migration(4, "course_content_assessment", "004_course_content_assessment.sql"),
    Migration(5, "document_registry", "005_document_registry.sql"),
)
LATEST_SCHEMA_VERSION = MIGRATIONS[-1].version


def _read_sql(filename: str) -> str:
    return (SQL_DIRECTORY / filename).read_text(encoding="utf-8")


def migrate_database(database_url: str) -> list[int]:
    """Apply pending PostgreSQL migrations without rebuilding existing tables."""

    applied_now: list[int] = []
    with PostgresDatabase(database_url).connect() as connection:
        connection.execute_script(_read_sql("000_schema_migrations.sql"))
        applied = {
            int(row["version"])
            for row in connection.execute("SELECT version FROM schema_migrations")
        }
        for migration in MIGRATIONS:
            if migration.version in applied:
                continue
            try:
                if migration.sql_file is not None:
                    connection.execute_script(_read_sql(migration.sql_file))
                connection.execute(
                    "INSERT INTO schema_migrations(version, name) VALUES (?, ?)",
                    (migration.version, migration.name),
                )
            except Exception:
                connection.rollback()
                raise
            applied_now.append(migration.version)
    return applied_now


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="升级 Course Coach PostgreSQL 数据库")
    parser.add_argument("--database-url", help="覆盖配置中的 PostgreSQL DATABASE_URL")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        applied = migrate_database(args.database_url or Settings().database_url)
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
