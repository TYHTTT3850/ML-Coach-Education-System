"""Additive content/assessment storage; existing attempts remain the authority."""

CONTENT_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS lessons (
 id TEXT NOT NULL, version INTEGER NOT NULL CHECK(version>0),
 topic_id TEXT NOT NULL REFERENCES curriculum_topics(id),
 status TEXT NOT NULL CHECK(status IN ('draft','published','archived')),
 payload_json TEXT NOT NULL, created_at TEXT NOT NULL,
 PRIMARY KEY(id,version)
);
CREATE TABLE IF NOT EXISTS worked_examples (
 id TEXT NOT NULL, version INTEGER NOT NULL CHECK(version>0),
 topic_id TEXT NOT NULL REFERENCES curriculum_topics(id), family_id TEXT NOT NULL,
 status TEXT NOT NULL CHECK(status IN ('draft','published','archived')),
 payload_json TEXT NOT NULL, created_at TEXT NOT NULL,
 PRIMARY KEY(id,version)
);
CREATE TABLE IF NOT EXISTS question_bank_items (
 id TEXT NOT NULL, version INTEGER NOT NULL CHECK(version>0),
 primary_course TEXT NOT NULL REFERENCES courses(id),
 question_type TEXT NOT NULL, difficulty TEXT NOT NULL,
 family_id TEXT NOT NULL, publication_status TEXT NOT NULL,
 validation_status TEXT NOT NULL, payload_json TEXT NOT NULL,
 created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
 PRIMARY KEY(id,version)
);
CREATE TABLE IF NOT EXISTS question_bank_topics (
 item_id TEXT NOT NULL, item_version INTEGER NOT NULL,
 topic_id TEXT NOT NULL REFERENCES curriculum_topics(id),
 PRIMARY KEY(item_id,item_version,topic_id),
 FOREIGN KEY(item_id,item_version) REFERENCES question_bank_items(id,version)
);
CREATE INDEX IF NOT EXISTS idx_bank_filters ON question_bank_items(primary_course,publication_status,question_type,difficulty);
CREATE TABLE IF NOT EXISTS content_review_events (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 item_id TEXT NOT NULL, item_version INTEGER NOT NULL,
 action TEXT NOT NULL, review_note TEXT NOT NULL, created_at TEXT NOT NULL,
 FOREIGN KEY(item_id,item_version) REFERENCES question_bank_items(id,version)
);
CREATE TABLE IF NOT EXISTS assessment_blueprints (
 id TEXT PRIMARY KEY, payload_json TEXT NOT NULL, created_at TEXT NOT NULL
);
ALTER TABLE practice_sets ADD COLUMN blueprint_id TEXT REFERENCES assessment_blueprints(id);
ALTER TABLE practice_sets ADD COLUMN purpose TEXT NOT NULL DEFAULT 'legacy';
ALTER TABLE practice_sets ADD COLUMN prerequisite_for TEXT REFERENCES curriculum_topics(id);
ALTER TABLE practice_sets ADD COLUMN rule_version TEXT NOT NULL DEFAULT 'legacy';
ALTER TABLE practice_sets ADD COLUMN required_dimensions_snapshot_json TEXT;
ALTER TABLE practice_questions ADD COLUMN bank_item_id TEXT;
ALTER TABLE practice_questions ADD COLUMN item_version INTEGER;
ALTER TABLE practice_questions ADD COLUMN family_id TEXT;
ALTER TABLE practice_questions ADD COLUMN topic_ids_json TEXT NOT NULL DEFAULT '[]';
ALTER TABLE practice_questions ADD COLUMN competencies_json TEXT NOT NULL DEFAULT '[]';
ALTER TABLE practice_questions ADD COLUMN solution_steps_json TEXT NOT NULL DEFAULT '[]';
ALTER TABLE practice_questions ADD COLUMN used_hint INTEGER NOT NULL DEFAULT 0;
ALTER TABLE practice_questions ADD COLUMN solution_revealed INTEGER NOT NULL DEFAULT 0;
ALTER TABLE practice_questions ADD COLUMN evidence_eligible INTEGER NOT NULL DEFAULT 1;
ALTER TABLE practice_questions ADD COLUMN attempt_count INTEGER NOT NULL DEFAULT 0;
CREATE TABLE IF NOT EXISTS content_exposures (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 student_id TEXT NOT NULL REFERENCES students(id),
 content_id TEXT NOT NULL, content_version INTEGER NOT NULL,
 family_id TEXT NOT NULL, kind TEXT NOT NULL,
 assessment_id TEXT, created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_exposure_student_family ON content_exposures(student_id,family_id,kind);
"""
