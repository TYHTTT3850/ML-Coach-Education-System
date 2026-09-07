from __future__ import annotations

import sqlite3
from pathlib import Path


class SQLiteConnection(sqlite3.Connection):
    """Transaction context manager that also releases the file handle on exit."""

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        try:
            return bool(super().__exit__(exc_type, exc_value, traceback))
        finally:
            self.close()


def connect_sqlite(path: str | Path) -> sqlite3.Connection:
    """Open the application's SQLite database with the same safety settings everywhere."""

    connection = sqlite3.connect(Path(path), timeout=15, factory=SQLiteConnection)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 15000")
    return connection


class SQLiteDatabase:
    """A lightweight connection factory shared by repositories and migrations."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def connect(self) -> sqlite3.Connection:
        return connect_sqlite(self.path)

    def ensure_parent(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
