CREATE TABLE IF NOT EXISTS courses (
    id TEXT PRIMARY KEY, title TEXT NOT NULL, active INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0, 1))
);

INSERT INTO courses(id, title) VALUES
    ('machine_learning', '机器学习'),
    ('advanced_mathematics', '高等数学'),
    ('linear_algebra', '线性代数'),
    ('probability_statistics', '概率论与统计基础')
ON CONFLICT (id) DO NOTHING;

CREATE TABLE IF NOT EXISTS topic_placements (
    course_id TEXT NOT NULL REFERENCES courses(id), topic_id TEXT NOT NULL REFERENCES curriculum_topics(id),
    parent_group_id TEXT NOT NULL, group_title TEXT NOT NULL, order_index INTEGER NOT NULL,
    is_reference INTEGER NOT NULL DEFAULT 0 CHECK (is_reference IN (0, 1)), PRIMARY KEY(course_id, topic_id)
);

CREATE TABLE IF NOT EXISTS prerequisite_verifications (
    student_id TEXT NOT NULL REFERENCES students(id), dependent_topic_id TEXT NOT NULL REFERENCES curriculum_topics(id),
    prerequisite_topic_id TEXT NOT NULL REFERENCES curriculum_topics(id), competency TEXT NOT NULL,
    rule_version TEXT NOT NULL, assessment_id TEXT NOT NULL REFERENCES practice_sets(id), verified_at TEXT NOT NULL,
    PRIMARY KEY(student_id, dependent_topic_id, prerequisite_topic_id, competency, rule_version)
);

CREATE OR REPLACE FUNCTION enforce_registered_course() RETURNS trigger AS $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM courses WHERE id = NEW.course AND active = 1) THEN
        RAISE EXCEPTION 'unregistered_course';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_curriculum_registered_course ON curriculum_topics;
CREATE TRIGGER trg_curriculum_registered_course BEFORE INSERT OR UPDATE ON curriculum_topics
FOR EACH ROW EXECUTE FUNCTION enforce_registered_course();

CREATE OR REPLACE FUNCTION prevent_required_prerequisite_cycle() RETURNS trigger AS $$
BEGIN
    IF NEW.relation_type = 'required' AND EXISTS (
        WITH RECURSIVE ancestors(id) AS (
            SELECT NEW.prerequisite_topic_id
            UNION
            SELECT prerequisite_topic_id FROM topic_prerequisites p JOIN ancestors a ON p.topic_id = a.id
            WHERE p.relation_type = 'required'
        ) SELECT 1 FROM ancestors WHERE id = NEW.topic_id
    ) THEN
        RAISE EXCEPTION 'required_prerequisite_cycle';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_required_cycle ON topic_prerequisites;
CREATE TRIGGER trg_required_cycle BEFORE INSERT OR UPDATE ON topic_prerequisites
FOR EACH ROW EXECUTE FUNCTION prevent_required_prerequisite_cycle();
