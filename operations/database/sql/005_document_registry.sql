CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    course TEXT NOT NULL,
    original_filename TEXT NOT NULL,
    stored_filename TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    file_size INTEGER NOT NULL,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL,
    error_message TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(course, sha256)
);
CREATE INDEX IF NOT EXISTS idx_documents_course ON documents(course);
CREATE INDEX IF NOT EXISTS idx_documents_updated_at ON documents(updated_at);
