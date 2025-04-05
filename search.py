import argparse
import asyncio
import os
import time

import aiosqlite

from deepsearch_code import db, github_client, indexer, materialize, settings


async def run_search_tool(repo_name: str):
    """Main async function to materialize, index, and query."""
    print(f"--- Starting DeepSearch Code Indexing for {repo_name} ---")

    # --- 1. Initialize DB ---
    db_initialized = await db.initialize_database(overwrite=True)
    if not db_initialized:
        print("Failed to initialize database. Exiting.")
        return

    # --- 2. Materialize Code ---
    try:
        # Ensure base materialized dir exists
        os.makedirs(settings.MATERIALIZED_DATA_DIR, exist_ok=True)
        print("\n--- Materializing Code ---")
        # Pass the base directory, materialize creates the repo-specific subdir
        materialized_code_path = await materialize.materialize_code_from_tarball(
            repo_name, settings.MATERIALIZED_DATA_DIR
        )
        print(f"Code materialized at: {materialized_code_path}")
    except Exception as e:
        print(f"Fatal error during code materialization: {e}")
        return  # Stop if materialization fails

    # --- Add materialization for issues, PRs etc. here later ---

    # --- 3. Index Materialized Data ---
    db_conn = None
    try:
        db_conn = await db.get_db_connection()
        print("\n--- Indexing Code ---")
        await indexer.index_materialized_code(
            repo_name, materialized_code_path, db_conn
        )

        # --- Add indexing for issues, PRs etc. here later ---

        print("\n--- Indexing Complete ---")

    except Exception as e:
        print(f"Fatal error during indexing: {e}")
        return  # Stop if indexing fails
    finally:
        if db_conn:
            await db_conn.close()

    # --- 4. Interactive Search Loop ---
    print("\n--- Enter Search Queries (Ctrl+C to exit) ---")
    search_conn = None
    try:
        search_conn = await db.get_db_connection()
        while True:
            try:
                query = input("fts> ")
                if not query.strip():
                    continue

                start_time = time.time()
                cursor = await search_conn.execute(
                    """
                        SELECT id, item_type, title_or_path, rank
                        FROM repo_index_fts
                        WHERE repo_index_fts MATCH ?
                        ORDER BY rank -- FTS5 rank
                        LIMIT 10;
                    """,
                    (query,),
                )
                results = list(await cursor.fetchall())
                await cursor.close()
                end_time = time.time()

                print(
                    f"\nFound {len(results)} results in {end_time - start_time:.3f}s:"
                )
                if results:
                    for row_id, item_type, title_or_path, rank in results:
                        # rank is lower for better matches in FTS5
                        print(
                            f"  - Rank: {rank:.4f} | Type: {item_type:<10} | Path/Title: {title_or_path} (ID: {row_id})"
                        )
                else:
                    print("  No matches found.")
                print("-" * 20)

            except EOFError:  # Handle Ctrl+D
                break
            except KeyboardInterrupt:  # Handle Ctrl+C
                break
            except aiosqlite.Error as e:
                print(f"Database search error: {e}")
            except Exception as e:
                print(f"Error during query input: {e}")

    finally:
        if search_conn:
            await search_conn.close()
        print("\nExiting search tool.")


async def main():
    parser = argparse.ArgumentParser(
        description="Materialize, index, and search a GitHub repository."
    )
    parser.add_argument("repo_name", help="Repository name in 'owner/repo' format.")
    args = parser.parse_args()

    try:
        await run_search_tool(args.repo_name)
    finally:
        # Ensure the global session is closed on exit
        await github_client.close_session()


if __name__ == "__main__":
    asyncio.run(main())
