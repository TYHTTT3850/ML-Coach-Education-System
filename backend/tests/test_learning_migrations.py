from __future__ import annotations

import sqlite3

from app.db.database import connect_sqlite
from app.db.migrations import LEARNING_SCHEMA_SQL, LATEST_SCHEMA_VERSION, migrate_database


def test_migration_preserves_existing_documents_and_is_idempotent(tmp_path) -> None:
    path = tmp_path / "existing.db"
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE documents (id TEXT PRIMARY KEY, course TEXT NOT NULL, title TEXT)"
        )
        connection.execute(
            "INSERT INTO documents(id, course, title) VALUES ('doc-1', 'machine_learning', '保留我')"
        )

    assert migrate_database(path) == list(range(1, LATEST_SCHEMA_VERSION + 1))
    assert migrate_database(path) == []

    with connect_sqlite(path) as connection:
        document = connection.execute(
            "SELECT id, course, title FROM documents WHERE id = 'doc-1'"
        ).fetchone()
        version = connection.execute(
            "SELECT MAX(version) AS version FROM schema_migrations"
        ).fetchone()["version"]
        tables = {
            row["name"]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        foreign_keys = connection.execute("PRAGMA foreign_keys").fetchone()[0]

    assert dict(document) == {
        "id": "doc-1",
        "course": "machine_learning",
        "title": "保留我",
    }
    assert version == LATEST_SCHEMA_VERSION
    assert foreign_keys == 1
    assert {
        "students",
        "curriculum_topics",
        "topic_prerequisites",
        "student_topic_states",
        "student_dimension_states",
        "practice_sets",
        "practice_questions",
        "practice_submissions",
        "practice_attempts",
        "misconception_events",
        "learning_sessions",
    } <= tables


def test_foreign_keys_are_enforced(tmp_path) -> None:
    path = tmp_path / "foreign-keys.db"
    migrate_database(path)
    with connect_sqlite(path) as connection:
        try:
            connection.execute(
                """
                INSERT INTO student_topic_states(student_id, topic_id, updated_at)
                VALUES ('missing-student', 'missing-topic', 'now')
                """
            )
        except sqlite3.IntegrityError:
            pass
        else:
            raise AssertionError("foreign key constraint was not enforced")


def test_early_v1_is_upgraded_without_rebuilding_database(tmp_path) -> None:
    path = tmp_path / "early-v1.db"
    early_v1 = (
        LEARNING_SCHEMA_SQL.replace(
            "    target_misconception_tags_json TEXT NOT NULL DEFAULT '[]',\n",
            "",
        )
        .replace(
            "    practice_set_id TEXT NOT NULL,\n",
            "    practice_set_id TEXT NOT NULL UNIQUE,\n",
            1,
        )
        .replace(
            "        evaluation_status IN ('processing', 'evaluated', 'evaluation_uncertain')\n",
            "        evaluation_status IN ('evaluated', 'evaluation_uncertain')\n",
        )
        .replace("    response_json TEXT,\n", "")
    )
    with connect_sqlite(path) as connection:
        connection.executescript(early_v1)
        connection.execute(
            "CREATE TABLE schema_migrations ("
            "version INTEGER PRIMARY KEY, name TEXT NOT NULL, "
            "applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
        )
        connection.execute(
            "INSERT INTO schema_migrations VALUES (1, 'learning_loop_schema', 'then')"
        )

    assert migrate_database(path) == list(range(2, LATEST_SCHEMA_VERSION + 1))

    with connect_sqlite(path) as connection:
        practice_columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(practice_sets)")
        }
        submission_columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(practice_submissions)")
        }
        submission_sql = connection.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'practice_submissions'"
        ).fetchone()["sql"]
        set_unique = [
            row["name"]
            for index in connection.execute("PRAGMA index_list(practice_submissions)")
            if index["unique"]
            for row in connection.execute(
                f"PRAGMA index_info('{index['name']}')"
            ).fetchall()
            if row["name"] == "practice_set_id"
        ]

    assert "target_misconception_tags_json" in practice_columns
    assert "response_json" in submission_columns
    assert "'processing'" in submission_sql
    assert set_unique == []
