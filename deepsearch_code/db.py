import os

import aiosqlite

from . import settings  # Use relative import within the package

# --- Database Setup SQL (copied from previous main.py) ---
# SQL command to create the main table
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS repo_index (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    repo_name TEXT NOT NULL,
    item_type TEXT NOT NULL CHECK(item_type IN ('code', 'issue', 'pr', 'discussion', 'release')),
    item_id TEXT NOT NULL, -- File path for code, number/id string for others
    url TEXT UNIQUE NOT NULL,
    title_or_path TEXT,
    author TEXT,
    status TEXT,
    labels TEXT, -- Store as JSON string
    created_at TEXT, -- ISO 8601 format
    updated_at TEXT, -- ISO 8601 format
    -- Store the *path* to the materialized file instead of the content directly
    materialized_path TEXT UNIQUE NOT NULL,
    content TEXT -- Keep content field for FTS, populated during indexing
);
"""

# SQL command to create the FTS5 virtual table
# Indexing content, title_or_path, item_id
CREATE_FTS_TABLE_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS repo_index_fts USING fts5(
    content,
    title_or_path,
    item_id,
    content='repo_index',
    content_rowid='id',
    tokenize='unicode61 remove_diacritics 1'
);
"""
# Triggers and standard indexes remain the same...
CREATE_TRIGGERS_SQL = [
    """
    CREATE TRIGGER IF NOT EXISTS repo_index_ai AFTER INSERT ON repo_index BEGIN
        INSERT INTO repo_index_fts (rowid, content, title_or_path, item_id)
        VALUES (new.id, new.content, new.title_or_path, new.item_id);
    END;
    """,
    """
    CREATE TRIGGER IF NOT EXISTS repo_index_ad AFTER DELETE ON repo_index BEGIN
        DELETE FROM repo_index_fts WHERE rowid=old.id;
    END;
    """,
    """
    CREATE TRIGGER IF NOT EXISTS repo_index_au AFTER UPDATE ON repo_index BEGIN
        UPDATE repo_index_fts SET
            content=new.content,
            title_or_path=new.title_or_path,
            item_id=new.item_id
        WHERE rowid=old.id;
    END;
    """,
]
CREATE_INDEXES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_repo_name ON repo_index(repo_name);",
    "CREATE INDEX IF NOT EXISTS idx_item_type ON repo_index(item_type);",
    "CREATE INDEX IF NOT EXISTS idx_item_id ON repo_index(item_id);",
    "CREATE INDEX IF NOT EXISTS idx_url ON repo_index(url);",
    "CREATE INDEX IF NOT EXISTS idx_materialized_path ON repo_index(materialized_path);",  # Index the new path field
]
# --- Database Setup SQL End ---


async def initialize_database(db_path: str = settings.DB_FILE, overwrite: bool = False):
    """Initializes the SQLite database asynchronously."""
    if overwrite and os.path.exists(db_path):
        print(f"Overwriting existing database: {db_path}")
        try:
            os.remove(db_path)
        except OSError as e:
            print(f"Error removing existing database file {db_path}: {e}")
            return False

    print(f"Initializing database at: {db_path}")
    conn = None
    try:
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)

        # Use aiosqlite.connect
        conn = await aiosqlite.connect(db_path)
        await conn.execute("PRAGMA journal_mode=WAL;")

        print("Creating main table...")
        await conn.execute(CREATE_TABLE_SQL)

        print("Creating FTS5 table...")
        await conn.execute(CREATE_FTS_TABLE_SQL)

        print("Creating FTS sync triggers...")
        for trigger_sql in CREATE_TRIGGERS_SQL:
            await conn.execute(trigger_sql)

        print("Creating standard indexes...")
        for index_sql in CREATE_INDEXES_SQL:
            await conn.execute(index_sql)

        await conn.commit()
        print("Database initialized successfully.")
        return True

    except aiosqlite.Error as e:
        print(f"Database error during initialization: {e}")
        if conn:
            await conn.rollback()
        return False
    finally:
        if conn:
            await conn.close()


async def get_db_connection(db_path: str = settings.DB_FILE) -> aiosqlite.Connection:
    """Establishes an async connection to the SQLite database."""
    try:
        conn = await aiosqlite.connect(db_path)
        await conn.execute("PRAGMA foreign_keys = ON;")
        # Set row factory for easy dict access if needed later
        # conn.row_factory = aiosqlite.Row
        return conn
    except aiosqlite.Error as e:
        print(f"Error connecting to database at {db_path}: {e}")
        raise  # Re-raise error


# Add async insertion/query functions here later...
