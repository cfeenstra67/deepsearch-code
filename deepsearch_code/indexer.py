import os
import time
from pathlib import Path

import aiofiles
import aiosqlite

from . import settings


async def index_materialized_code(
    repo_name: str, materialized_code_dir: str, conn: aiosqlite.Connection
):
    """Indexes code files previously materialized on the filesystem."""
    print(f"Starting indexing of code in: {materialized_code_dir}")
    base_path = Path(materialized_code_dir)
    items_to_insert = []
    skipped_count = 0
    processed_count = 0

    # Walk the directory
    # Using os.walk which is synchronous, wrap in asyncio.to_thread if perf critical
    # For typical repo sizes, direct walk might be fine.
    for root, _, files in os.walk(materialized_code_dir):
        for filename in files:
            file_path_abs = Path(root) / filename
            # Calculate path relative to the code dir for item_id
            try:
                file_path_rel = file_path_abs.relative_to(base_path).as_posix()
            except ValueError:
                print(f"Warning: Could not make path relative: {file_path_abs}")
                skipped_count += 1
                continue

            # Basic check (redundant with materialize filtering, but safe)
            _, ext = os.path.splitext(filename)
            if ext.lower() in settings.CODE_IGNORE_EXTENSIONS:
                skipped_count += 1
                continue

            # TODO: Add URL construction - needs repo_name and branch/commit info
            # Placeholder URL for now
            file_url = f"https://github.com/{repo_name}/blob/main/{file_path_rel}"  # Needs actual branch/commit

            try:
                # Get file stats (synchronous)
                stat_result = file_path_abs.stat()
                file_size = stat_result.st_size
                updated_at = stat_result.st_mtime  # Use modification time

                if file_size > settings.MAX_INDEXING_FILE_SIZE_BYTES:
                    # print(f"Skipping large file for indexing: {file_path_rel} ({file_size} bytes)")
                    skipped_count += 1
                    continue

                # Read file content asynchronously
                try:
                    async with aiofiles.open(
                        file_path_abs, mode="r", encoding="utf-8", errors="replace"
                    ) as f:
                        content = await f.read()
                except OSError as read_err:
                    print(f"Error reading file {file_path_abs}: {read_err}")
                    skipped_count += 1
                    continue

                items_to_insert.append(
                    (
                        repo_name,
                        "code",
                        file_path_rel,  # Item ID is relative path
                        file_url,
                        file_path_rel,  # title_or_path is also relative path
                        None,  # author
                        None,  # status
                        None,  # labels (as JSON null)
                        None,  # created_at (can maybe get from git later)
                        time.strftime(
                            "%Y-%m-%dT%H:%M:%SZ", time.gmtime(updated_at)
                        ),  # updated_at
                        file_path_abs.as_posix(),  # materialized_path
                        content,  # Actual content for FTS
                    )
                )
                processed_count += 1

                # Insert in batches
                if len(items_to_insert) >= 100:
                    await _insert_batch(conn, items_to_insert)
                    print(f"Indexed {processed_count} files...")
                    items_to_insert = []

            except OSError as e:
                print(f"Error processing file {file_path_abs}: {e}")
                skipped_count += 1
            except Exception as e:
                print(f"Unexpected error processing file {file_path_abs}: {e}")
                skipped_count += 1

    # Insert any remaining items
    if items_to_insert:
        await _insert_batch(conn, items_to_insert)

    print(
        f"Finished indexing code. Processed: {processed_count}, Skipped: {skipped_count}"
    )


async def _insert_batch(conn: aiosqlite.Connection, items: list):
    """Helper to insert a batch of items into the database."""
    sql = """
        INSERT INTO repo_index (
            repo_name, item_type, item_id, url, title_or_path,
            author, status, labels, created_at, updated_at,
            materialized_path, content
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(url) DO UPDATE SET -- Use URL as conflict target for now
            updated_at = excluded.updated_at,
            content = excluded.content,
            title_or_path = excluded.title_or_path,
            materialized_path = excluded.materialized_path
            -- Add other fields if they can change
    """
    try:
        await conn.executemany(sql, items)
        await conn.commit()
    except aiosqlite.Error as e:
        print(f"Database error during batch insert: {e}")
        await conn.rollback()
        # Decide if we should raise or just log


# Add indexing for issues, PRs etc. later
# async def index_materialized_issues(...)
