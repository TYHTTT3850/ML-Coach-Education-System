"""Shared SQLite connection and schema migration helpers."""

from app.db.database import SQLiteDatabase, connect_sqlite
from app.db.migrations import LATEST_SCHEMA_VERSION, migrate_database

__all__ = [
    "LATEST_SCHEMA_VERSION",
    "SQLiteDatabase",
    "connect_sqlite",
    "migrate_database",
]
