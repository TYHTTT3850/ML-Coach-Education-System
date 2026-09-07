from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from app.db.database import SQLiteDatabase
from app.db.content_curriculum import upgrade_content_curriculum
from app.db.content_schema import CONTENT_SCHEMA_SQL


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    sql: str


LEARNING_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS students (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    last_active_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS curriculum_topics (
    id TEXT PRIMARY KEY,
    course TEXT NOT NULL CHECK (course IN ('machine_learning', 'advanced_mathematics')),
    slug TEXT NOT NULL,
    title TEXT NOT NULL,
    stage INTEGER NOT NULL CHECK (stage > 0),
    stage_title TEXT NOT NULL,
    order_index INTEGER NOT NULL CHECK (order_index > 0),
    description TEXT NOT NULL,
    learning_objectives_json TEXT NOT NULL,
    aliases_json TEXT NOT NULL,
    retrieval_keywords_json TEXT NOT NULL,
    difficulty TEXT NOT NULL CHECK (difficulty IN ('easy', 'medium', 'hard')),
    required_dimensions_json TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0, 1)),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(course, slug)
);

CREATE TABLE IF NOT EXISTS topic_prerequisites (
    topic_id TEXT NOT NULL,
    prerequisite_topic_id TEXT NOT NULL,
    required_mastery REAL NOT NULL DEFAULT 0.8
        CHECK (required_mastery >= 0 AND required_mastery <= 1),
    PRIMARY KEY (topic_id, prerequisite_topic_id),
    CHECK (topic_id <> prerequisite_topic_id),
    FOREIGN KEY (topic_id) REFERENCES curriculum_topics(id) ON DELETE CASCADE,
    FOREIGN KEY (prerequisite_topic_id) REFERENCES curriculum_topics(id) ON DELETE RESTRICT
);

CREATE TRIGGER IF NOT EXISTS trg_topic_prerequisites_same_course_insert
BEFORE INSERT ON topic_prerequisites
WHEN (
    SELECT course FROM curriculum_topics WHERE id = NEW.topic_id
) <> (
    SELECT course FROM curriculum_topics WHERE id = NEW.prerequisite_topic_id
)
BEGIN
    SELECT RAISE(ABORT, 'cross_course_prerequisite');
END;

CREATE TRIGGER IF NOT EXISTS trg_topic_prerequisites_same_course_update
BEFORE UPDATE ON topic_prerequisites
WHEN (
    SELECT course FROM curriculum_topics WHERE id = NEW.topic_id
) <> (
    SELECT course FROM curriculum_topics WHERE id = NEW.prerequisite_topic_id
)
BEGIN
    SELECT RAISE(ABORT, 'cross_course_prerequisite');
END;

CREATE TABLE IF NOT EXISTS student_topic_states (
    student_id TEXT NOT NULL,
    topic_id TEXT NOT NULL,
    alpha REAL NOT NULL DEFAULT 1 CHECK (alpha > 0),
    beta REAL NOT NULL DEFAULT 1 CHECK (beta > 0),
    mastery REAL NOT NULL DEFAULT 0.5 CHECK (mastery >= 0 AND mastery <= 1),
    confidence REAL NOT NULL DEFAULT 0 CHECK (confidence >= 0 AND confidence <= 1),
    evidence_count INTEGER NOT NULL DEFAULT 0 CHECK (evidence_count >= 0),
    attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
    practice_set_count INTEGER NOT NULL DEFAULT 0 CHECK (practice_set_count >= 0),
    checkpoint_count INTEGER NOT NULL DEFAULT 0 CHECK (checkpoint_count >= 0),
    latest_score REAL,
    best_score REAL,
    latest_checkpoint_score REAL,
    status TEXT NOT NULL DEFAULT 'not_started' CHECK (
        status IN (
            'locked', 'not_started', 'learning', 'practicing',
            'needs_remediation', 'ready_for_checkpoint', 'mastered'
        )
    ),
    started_at TEXT,
    last_practiced_at TEXT,
    mastered_at TEXT,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (student_id, topic_id),
    FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE,
    FOREIGN KEY (topic_id) REFERENCES curriculum_topics(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS student_dimension_states (
    student_id TEXT NOT NULL,
    topic_id TEXT NOT NULL,
    dimension TEXT NOT NULL,
    alpha REAL NOT NULL DEFAULT 1 CHECK (alpha > 0),
    beta REAL NOT NULL DEFAULT 1 CHECK (beta > 0),
    mastery REAL NOT NULL DEFAULT 0.5 CHECK (mastery >= 0 AND mastery <= 1),
    confidence REAL NOT NULL DEFAULT 0 CHECK (confidence >= 0 AND confidence <= 1),
    evidence_count INTEGER NOT NULL DEFAULT 0 CHECK (evidence_count >= 0),
    updated_at TEXT NOT NULL,
    PRIMARY KEY (student_id, topic_id, dimension),
    FOREIGN KEY (student_id, topic_id)
        REFERENCES student_topic_states(student_id, topic_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS practice_sets (
    id TEXT PRIMARY KEY,
    student_id TEXT NOT NULL,
    topic_id TEXT NOT NULL,
    course TEXT NOT NULL CHECK (course IN ('machine_learning', 'advanced_mathematics')),
    mode TEXT NOT NULL CHECK (mode IN ('practice', 'checkpoint')),
    status TEXT NOT NULL CHECK (
        status IN ('generated', 'submitted', 'evaluation_uncertain')
    ),
    requested_difficulty TEXT NOT NULL CHECK (
        requested_difficulty IN ('adaptive', 'easy', 'medium', 'hard')
    ),
    source_snapshot_json TEXT NOT NULL,
    target_misconception_tags_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    submitted_at TEXT,
    FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE,
    FOREIGN KEY (topic_id) REFERENCES curriculum_topics(id) ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS practice_questions (
    id TEXT PRIMARY KEY,
    practice_set_id TEXT NOT NULL,
    position INTEGER NOT NULL CHECK (position > 0),
    question_type TEXT NOT NULL,
    difficulty TEXT NOT NULL CHECK (difficulty IN ('easy', 'medium', 'hard')),
    question TEXT NOT NULL,
    options_json TEXT,
    dimensions_json TEXT NOT NULL,
    answer_key_json TEXT NOT NULL,
    rubric_json TEXT NOT NULL,
    source_ids_json TEXT NOT NULL,
    question_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (practice_set_id, position),
    FOREIGN KEY (practice_set_id) REFERENCES practice_sets(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS practice_submissions (
    id TEXT PRIMARY KEY,
    practice_set_id TEXT NOT NULL,
    student_id TEXT NOT NULL,
    client_submission_id TEXT NOT NULL UNIQUE,
    total_score REAL NOT NULL CHECK (total_score >= 0 AND total_score <= 100),
    evaluation_status TEXT NOT NULL CHECK (
        evaluation_status IN ('processing', 'evaluated', 'evaluation_uncertain')
    ),
    mastery_before REAL CHECK (mastery_before >= 0 AND mastery_before <= 1),
    mastery_after REAL CHECK (mastery_after >= 0 AND mastery_after <= 1),
    gate_result_json TEXT NOT NULL,
    response_json TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (practice_set_id) REFERENCES practice_sets(id) ON DELETE CASCADE,
    FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS practice_attempts (
    id TEXT PRIMARY KEY,
    submission_id TEXT NOT NULL,
    question_id TEXT NOT NULL,
    student_id TEXT NOT NULL,
    answer_text TEXT NOT NULL,
    score REAL NOT NULL CHECK (score >= 0 AND score <= 100),
    verdict TEXT NOT NULL,
    dimension_scores_json TEXT NOT NULL,
    correct_points_json TEXT NOT NULL,
    missing_points_json TEXT NOT NULL,
    misconception_tags_json TEXT NOT NULL,
    critical_misconception INTEGER NOT NULL DEFAULT 0 CHECK (critical_misconception IN (0, 1)),
    feedback TEXT NOT NULL,
    reference_answer TEXT NOT NULL,
    evidence_source_ids_json TEXT NOT NULL,
    evaluator_confidence REAL NOT NULL CHECK (evaluator_confidence >= 0 AND evaluator_confidence <= 1),
    evaluation_reliable INTEGER NOT NULL CHECK (evaluation_reliable IN (0, 1)),
    created_at TEXT NOT NULL,
    UNIQUE (submission_id, question_id),
    FOREIGN KEY (submission_id) REFERENCES practice_submissions(id) ON DELETE CASCADE,
    FOREIGN KEY (question_id) REFERENCES practice_questions(id) ON DELETE RESTRICT,
    FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS misconception_events (
    id TEXT PRIMARY KEY,
    student_id TEXT NOT NULL,
    topic_id TEXT NOT NULL,
    attempt_id TEXT NOT NULL,
    tag TEXT NOT NULL,
    critical INTEGER NOT NULL DEFAULT 0 CHECK (critical IN (0, 1)),
    evidence TEXT NOT NULL,
    created_at TEXT NOT NULL,
    resolved_at TEXT,
    FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE,
    FOREIGN KEY (topic_id) REFERENCES curriculum_topics(id) ON DELETE CASCADE,
    FOREIGN KEY (attempt_id) REFERENCES practice_attempts(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS learning_sessions (
    id TEXT PRIMARY KEY,
    student_id TEXT NOT NULL,
    course TEXT NOT NULL CHECK (course IN ('machine_learning', 'advanced_mathematics')),
    topic_id TEXT,
    state TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE,
    FOREIGN KEY (topic_id) REFERENCES curriculum_topics(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_curriculum_topics_course_stage_order
    ON curriculum_topics(course, active, stage, order_index);
CREATE INDEX IF NOT EXISTS idx_topic_prerequisites_prerequisite
    ON topic_prerequisites(prerequisite_topic_id);
CREATE INDEX IF NOT EXISTS idx_student_topic_states_topic_status
    ON student_topic_states(topic_id, status);
CREATE INDEX IF NOT EXISTS idx_student_topic_states_student_status
    ON student_topic_states(student_id, status);
CREATE INDEX IF NOT EXISTS idx_student_dimension_states_topic
    ON student_dimension_states(topic_id, dimension);
CREATE INDEX IF NOT EXISTS idx_practice_sets_student_topic_created
    ON practice_sets(student_id, topic_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_practice_sets_status
    ON practice_sets(status);
CREATE INDEX IF NOT EXISTS idx_practice_questions_hash
    ON practice_questions(question_hash);
CREATE INDEX IF NOT EXISTS idx_practice_submissions_student_created
    ON practice_submissions(student_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_practice_attempts_student_created
    ON practice_attempts(student_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_misconception_events_student_topic_unresolved
    ON misconception_events(student_id, topic_id, resolved_at);
CREATE INDEX IF NOT EXISTS idx_learning_sessions_student_course_updated
    ON learning_sessions(student_id, course, updated_at DESC);
"""


MIGRATIONS: Sequence[Migration] = (
    Migration(1, "learning_loop_schema", LEARNING_SCHEMA_SQL),
    # Version 1 was developed incrementally before Phase 2 shipped.  Keep a
    # Python compatibility step so a database that already recorded the early
    # v1 can gain the final idempotency/remediation columns and constraints
    # without being deleted or recreated by the operator.
    Migration(2, "learning_loop_v1_compatibility", ""),
    Migration(3, "course_content_curriculum", ""),
    Migration(4, "course_content_assessment", CONTENT_SCHEMA_SQL),
)
LATEST_SCHEMA_VERSION = MIGRATIONS[-1].version


def _table_columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {
        str(row["name"])
        for row in connection.execute(f'PRAGMA table_info("{table}")').fetchall()
    }


def _has_single_column_unique_index(
    connection: sqlite3.Connection,
    table: str,
    column: str,
) -> bool:
    for index in connection.execute(f'PRAGMA index_list("{table}")').fetchall():
        if not bool(index["unique"]):
            continue
        index_name = str(index["name"]).replace('"', '""')
        columns = [
            str(row["name"])
            for row in connection.execute(
                f'PRAGMA index_info("{index_name}")'
            ).fetchall()
        ]
        if columns == [column]:
            return True
    return False


def _upgrade_learning_v1_compatibility(connection: sqlite3.Connection) -> None:
    """Finish early Phase 2 v1 databases while preserving every stored row."""

    # Re-run the current CREATE IF NOT EXISTS statements first.  This is a
    # no-op for a complete database and fills in any table/index omitted by an
    # early development build that nevertheless recorded migration 1.
    connection.executescript(LEARNING_SCHEMA_SQL)

    practice_set_columns = _table_columns(connection, "practice_sets")
    if "target_misconception_tags_json" not in practice_set_columns:
        connection.execute(
            "ALTER TABLE practice_sets ADD COLUMN "
            "target_misconception_tags_json TEXT NOT NULL DEFAULT '[]'"
        )

    submission_columns = _table_columns(connection, "practice_submissions")
    submission_sql_row = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
        ("practice_submissions",),
    ).fetchone()
    submission_sql = str(submission_sql_row["sql"] or "") if submission_sql_row else ""
    has_obsolete_set_uniqueness = _has_single_column_unique_index(
        connection,
        "practice_submissions",
        "practice_set_id",
    )
    needs_processing_status = "'processing'" not in submission_sql

    if not needs_processing_status and not has_obsolete_set_uniqueness:
        if "response_json" not in submission_columns:
            connection.execute(
                "ALTER TABLE practice_submissions ADD COLUMN response_json TEXT"
            )
        return

    response_expression = (
        "response_json" if "response_json" in submission_columns else "NULL"
    )
    # Rebuild only this table to widen evaluation_status and remove the early
    # UNIQUE(practice_set_id).  Foreign keys are disabled only for this bounded
    # transaction; the final foreign_key_check fails closed if data is invalid.
    connection.commit()
    connection.execute("PRAGMA foreign_keys = OFF")
    try:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            """
            CREATE TABLE practice_submissions_v2 (
                id TEXT PRIMARY KEY,
                practice_set_id TEXT NOT NULL,
                student_id TEXT NOT NULL,
                client_submission_id TEXT NOT NULL UNIQUE,
                total_score REAL NOT NULL CHECK (total_score >= 0 AND total_score <= 100),
                evaluation_status TEXT NOT NULL CHECK (
                    evaluation_status IN ('processing', 'evaluated', 'evaluation_uncertain')
                ),
                mastery_before REAL CHECK (mastery_before >= 0 AND mastery_before <= 1),
                mastery_after REAL CHECK (mastery_after >= 0 AND mastery_after <= 1),
                gate_result_json TEXT NOT NULL,
                response_json TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (practice_set_id) REFERENCES practice_sets(id) ON DELETE CASCADE,
                FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE
            )
            """
        )
        connection.execute(
            f"""
            INSERT INTO practice_submissions_v2 (
                id, practice_set_id, student_id, client_submission_id,
                total_score, evaluation_status, mastery_before, mastery_after,
                gate_result_json, response_json, created_at
            )
            SELECT id, practice_set_id, student_id, client_submission_id,
                   total_score, evaluation_status, mastery_before, mastery_after,
                   gate_result_json, {response_expression}, created_at
            FROM practice_submissions
            """
        )
        connection.execute("DROP TABLE practice_submissions")
        connection.execute(
            "ALTER TABLE practice_submissions_v2 RENAME TO practice_submissions"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_practice_submissions_student_created "
            "ON practice_submissions(student_id, created_at DESC)"
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.execute("PRAGMA foreign_keys = ON")

    violations = connection.execute("PRAGMA foreign_key_check").fetchall()
    if violations:
        raise sqlite3.IntegrityError(
            f"learning migration produced {len(violations)} foreign-key violation(s)"
        )


def migrate_database(path: str | Path) -> list[int]:
    """Apply pending migrations without rebuilding or deleting existing tables."""

    database = SQLiteDatabase(path)
    database.ensure_parent()
    applied_now: list[int] = []
    with database.connect() as connection:
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        applied = {
            int(row["version"])
            for row in connection.execute("SELECT version FROM schema_migrations")
        }
        for migration in MIGRATIONS:
            if migration.version in applied:
                continue
            try:
                if migration.version == 2:
                    _upgrade_learning_v1_compatibility(connection)
                elif migration.version == 3:
                    upgrade_content_curriculum(connection)
                elif migration.version == 4:
                    # executescript otherwise commits each preceding change before
                    # running SQL. Its explicit BEGIN keeps every ALTER and the
                    # version marker below in one rollback-capable transaction.
                    connection.commit()
                    connection.executescript("BEGIN IMMEDIATE;\n" + migration.sql)
                else:
                    connection.executescript(migration.sql)
                if migration.version != 3:  # v3 records its marker before restoring FKs.
                    connection.execute(
                        "INSERT INTO schema_migrations(version, name) VALUES (?, ?)",
                        (migration.version, migration.name),
                    )
                if migration.version == 4:
                    connection.commit()
            except sqlite3.DatabaseError:
                connection.rollback()
                raise
            applied_now.append(migration.version)
    return applied_now
