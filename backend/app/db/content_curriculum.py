"""Bounded, transactional curriculum migration; preserve all existing rows and IDs."""
from __future__ import annotations

import re
import sqlite3

from app.core.types import Course


CURRICULUM_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS courses (
    id TEXT PRIMARY KEY, title TEXT NOT NULL, active INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS topic_placements (
    course_id TEXT NOT NULL REFERENCES courses(id),
    topic_id TEXT NOT NULL REFERENCES curriculum_topics(id),
    parent_group_id TEXT NOT NULL,
    group_title TEXT NOT NULL,
    order_index INTEGER NOT NULL,
    is_reference INTEGER NOT NULL DEFAULT 0 CHECK(is_reference IN (0,1)),
    PRIMARY KEY(course_id, topic_id)
);
CREATE TABLE IF NOT EXISTS prerequisite_verifications (
    student_id TEXT NOT NULL REFERENCES students(id),
    dependent_topic_id TEXT NOT NULL REFERENCES curriculum_topics(id),
    prerequisite_topic_id TEXT NOT NULL REFERENCES curriculum_topics(id),
    competency TEXT NOT NULL,
    rule_version TEXT NOT NULL,
    assessment_id TEXT NOT NULL REFERENCES practice_sets(id),
    verified_at TEXT NOT NULL,
    PRIMARY KEY(student_id, dependent_topic_id, prerequisite_topic_id, competency, rule_version)
);
"""


def upgrade_content_curriculum(connection: sqlite3.Connection) -> None:
    # SQLite cannot widen a CHECK in-place. Copy only the affected schema tables,
    # retain every column/row/index, and verify FKs before committing the swap.
    connection.commit()
    connection.execute("PRAGMA foreign_keys=OFF")
    try:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("DROP TRIGGER IF EXISTS trg_topic_prerequisites_same_course_insert")
        connection.execute("DROP TRIGGER IF EXISTS trg_topic_prerequisites_same_course_update")
        for table in ("curriculum_topics", "practice_sets", "learning_sessions"):
            row = connection.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
            if not row:
                continue
            original = row["sql"]
            old = "'machine_learning', 'advanced_mathematics'"
            if old not in original or "'linear_algebra'" in original:
                continue
            indexes = [r["sql"] for r in connection.execute(
                "SELECT sql FROM sqlite_master WHERE type='index' AND tbl_name=? AND sql IS NOT NULL", (table,))]
            replacement = f"{table}_content_v3"
            sql = re.sub(r"CREATE TABLE(?: IF NOT EXISTS)?\s+[\"`\[]?" + table + r"[\"`\]]?", "CREATE TABLE " + replacement, original, count=1, flags=re.I)
            sql = sql.replace(old, old + ", 'linear_algebra', 'probability_statistics'")
            connection.execute(sql)
            columns = ",".join('"' + r["name"] + '"' for r in connection.execute(f'PRAGMA table_info("{table}")'))
            connection.execute(f'INSERT INTO "{replacement}" ({columns}) SELECT {columns} FROM "{table}"')
            connection.execute(f'DROP TABLE "{table}"')
            connection.execute(f'ALTER TABLE "{replacement}" RENAME TO "{table}"')
            for index in indexes:
                connection.execute(index)
        connection.execute("""CREATE TABLE topic_prerequisites_content_v3 (
            topic_id TEXT NOT NULL REFERENCES curriculum_topics(id) ON DELETE CASCADE,
            prerequisite_topic_id TEXT NOT NULL REFERENCES curriculum_topics(id) ON DELETE RESTRICT,
            required_mastery REAL NOT NULL DEFAULT 0.8 CHECK(required_mastery BETWEEN 0 AND 1),
            relation_type TEXT NOT NULL DEFAULT 'required' CHECK(relation_type IN ('required','recommended','related')),
            reason TEXT NOT NULL DEFAULT '掌握这一基础有助于完成当前知识点的正式验收。',
            required_competencies_json TEXT NOT NULL DEFAULT '[]',
            scope TEXT NOT NULL DEFAULT 'topic' CHECK(scope IN ('topic','lesson_section')),
            section_id TEXT NOT NULL DEFAULT '',
            rule_version TEXT NOT NULL DEFAULT 'content-v1',
            PRIMARY KEY(topic_id,prerequisite_topic_id,relation_type,scope,section_id),
            CHECK(topic_id<>prerequisite_topic_id))""")
        connection.execute("INSERT INTO topic_prerequisites_content_v3(topic_id,prerequisite_topic_id,required_mastery) SELECT topic_id,prerequisite_topic_id,required_mastery FROM topic_prerequisites")
        connection.execute("DROP TABLE topic_prerequisites")
        connection.execute("ALTER TABLE topic_prerequisites_content_v3 RENAME TO topic_prerequisites")
        connection.execute("CREATE INDEX idx_topic_prerequisites_prerequisite ON topic_prerequisites(prerequisite_topic_id)")
        for statement in CURRICULUM_SCHEMA_SQL.split(";"):
            if statement.strip():
                connection.execute(statement)
        for course in Course:
            connection.execute("INSERT INTO courses(id,title) VALUES (?,?) ON CONFLICT(id) DO NOTHING", (course.value, course.display_name))
        for operation in ("INSERT", "UPDATE"):
            connection.execute(f"""CREATE TRIGGER trg_curriculum_registered_course_{operation.lower()}
                BEFORE {operation} ON curriculum_topics
                WHEN NOT EXISTS(SELECT 1 FROM courses WHERE id=NEW.course AND active=1)
                BEGIN SELECT RAISE(ABORT,'unregistered_course'); END""")
            connection.execute(f"""CREATE TRIGGER trg_required_cycle_{operation.lower()}
                BEFORE {operation} ON topic_prerequisites WHEN NEW.relation_type='required'
                BEGIN
                SELECT RAISE(ABORT,'required_prerequisite_cycle') WHERE EXISTS(
                    WITH RECURSIVE ancestors(id) AS (
                      SELECT NEW.prerequisite_topic_id UNION
                      SELECT p.prerequisite_topic_id FROM topic_prerequisites p JOIN ancestors a ON p.topic_id=a.id
                      WHERE p.relation_type='required'
                    ) SELECT 1 FROM ancestors WHERE id=NEW.topic_id
                ); END""")
        violations = connection.execute("PRAGMA foreign_key_check").fetchall()
        if violations:
            raise sqlite3.IntegrityError(f"curriculum migration has {len(violations)} foreign-key violations")
        connection.execute("INSERT INTO schema_migrations(version,name) VALUES (3,'course_content_curriculum')")
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.execute("PRAGMA foreign_keys=ON")
