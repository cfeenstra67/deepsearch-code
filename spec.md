# Specification: GitHub Repo DeepSearch Tool (V1)

## 1. Goal

To create a tool that answers complex, cross-cutting questions about a specific GitHub repository by applying an iterative search-read-reason process inspired by the DeepSearch methodology. The tool will leverage a local index of repository data and interact with an LLM using a tool-based approach to explore this data and synthesize answers. V1 focuses on simplicity and core functionality, laying the groundwork for more advanced capabilities.

## 2. Data Sources & Ingestion

The tool will ingest data from a target GitHub repository:

*   **Code:** Full content of files (filtered to likely source/text files, excluding binaries, large assets, `.git` dir). Filter criteria TBD (e.g., by extension, size limit).
*   **Issues, PRs, Discussions:** Title, body, and all comments. This data will be **formatted into readable Markdown**, including key inline metadata like authors and timestamps for context, while avoiding excessive clutter.
*   **Releases:** Tag name, title, release notes (body - formatted as Markdown), publication timestamp, URL.
*   **Metadata:** URL, author, labels, status, timestamps will be included where available.
*   **Deferred (V1):** Detailed commit history, Git graph navigation, structured code analysis (CodeQL).

**Ingestion Strategy (V1):**
*   An ingestion script will pull data using the GitHub API.
*   **For V1, the script will perform a full repository scan and rebuild the index from scratch on each run.** Authentication and rate limiting must be handled.
*   _Future:_ Incremental updates, ability to load/reuse pre-built indexes.

## 3. Indexing Backend

*   **Technology:** SQLite.
*   **Schema:** A single table (`repo_index`) storing all ingested items.
*   **Full-Text Search:** SQLite's FTS5 extension enabled on the formatted `content` field.
*   **Table Structure (Revised Draft):**
    *   `id`: INTEGER PRIMARY KEY
    *   `repo_name`: TEXT (e.g., "pulumi/pulumi")
    *   `item_type`: TEXT ('code', 'issue', 'pr', 'discussion', 'release')
    *   `item_id`: TEXT (e.g., file path, issue/pr number, discussion number, release tag)
    *   `url`: TEXT
    *   `title_or_path`: TEXT
    *   `author`: TEXT (Primary author/owner, if applicable)
    *   `status`: TEXT
    *   `labels`: TEXT (JSON array of labels)
    *   `created_at`: TEXT (ISO 8601)
    *   `updated_at`: TEXT (ISO 8601)
    *   `content`: TEXT (Full, formatted Markdown content)
    *   _Note:_ FTS5 indexes `content`. Other fields like `item_type`, `title_or_path` should have standard indexes for filtering.

## 4. LLM Interaction Model (Tool Use)

The LLM interacts with the system by requesting specific "tools" to be called. The LLM response should specify the tool and its parameters, preceded by optional free-form observations.

*   **LLM Response Format (Conceptual):**
    ```json
    {
      "observations": "Based on the previous search results, it seems the core logic might be in `provider_sdk.py`. I need to read that file to confirm.",
      "tool_call": {
        "name": "read",
        "arguments": {
          "item_db_id": 123
        }
      }
    }
    // Or for the final answer:
    {
      "observations": "I have gathered enough information from the code and the related issue discussion.",
      "tool_call": {
        "name": "answer",
        "arguments": {
          "text": "Resource providers work by implementing the `ResourceProvider` interface defined in..."
        }
      }
    }
    ```

*   **Available Tools:**
    *   `search(keywords: string, item_type: Optional[list[str]] = None, path_glob: Optional[str] = None)`:
        *   Executes an FTS5 query on `content` using `keywords`.
        *   Optionally filters results to match specific `item_type`(s) (e.g., `['issue', 'pr']`).
        *   Optionally filters `code` items by `path_glob` (e.g., `src/pulumi/**.py`). SQLite's `GLOB` operator can be used on the `item_id` field for this.
        *   Returns top N (e.g., 5) results, including: `id`, `item_type`, `title_or_path`, URL, and a **precise match location indicator** (e.g., line number or character offset) along with a relevant FTS5 snippet.
    *   `read(item_db_id: int, location: Optional[int] = None)`:
        *   Retrieves the full `content` for the item with database `id`.
        *   Presents an initial **view window** (e.g., +/- 50 lines or ~2000 chars).
        *   If `location` (line number/offset from search result) is provided, the initial window is centered there. Otherwise, it starts at the beginning.
        *   The presented window includes markers (e.g., `[...more above...]`, `[...more below...]`) indicating available content outside the window. Stores the current window bounds and full content reference internally.
    *   `scroll(direction: 'up' | 'down')`:
        *   Requires a previous `read` action on the same item.
        *   Shifts the view window (e.g., by 80% of window size, maintaining overlap) in the specified direction.
        *   Returns the new window content. Updates internal window bounds.
    *   `search_within_item(keywords: string)`:
        *   Requires a previous `read` action on the same item.
        *   Searches for `keywords` *only within the full content* of the currently "read" item.
        *   Returns a list of locations (line numbers/offsets) where the keywords are found within that specific item. It might optionally update the view window to the first new match.
    *   `answer(text: string)`:
        *   The LLM provides the final answer. This terminates the loop.

## 5. Core Loop Logic

1.  **User Query:** Start with the user's question.
2.  **Initial Prompt:** Present the query, tool descriptions, and goal to the LLM.
3.  **LLM Turn:**
    *   LLM generates a response containing optional `observations` and a `tool_call`.
4.  **System Execution:**
    *   Parse the `tool_call`.
    *   If `answer`, return `text` to the user and terminate.
    *   If any other tool, execute it using the provided arguments.
    *   Prepare the result (search results list, content window, scroll status, search locations) to be presented back to the LLM.
5.  **Next Prompt:** Construct the next prompt including the history (previous tool calls and results, summarized if necessary) and the result of the latest tool execution. Go back to step 3.
6.  **Context Management (V1):**
    *   Maintain a **linear conversation history** (User Query -> LLM Response -> Tool Result -> LLM Response -> ...).
    *   Rely on the LLM's ability to use the history and its `observations` field.
    *   **No explicit summarization or complex memory management in V1.** We accept the limitation imposed by the LLM's context window size.
    *   _Future:_ Implement a coordinator/sub-agent architecture where sub-agents handle parts of the problem with fresh contexts, managed by a coordinator that synthesizes results, effectively extending context recursively.
7.  **Stopping Conditions:**
    *   LLM calls the `answer` tool.
    *   A predefined **global token limit** for the entire interaction is reached. If hit, the system should terminate and report that the limit was reached, possibly providing a partial answer if the LLM's last `observations` are useful.

## 6. Implementation Details

*   **Language:** Python
*   **Libraries:** `openai`, `sqlite3` / `sqlite-utils`, GitHub API library (`ghapi`, `PyGithub`), `python-dotenv`.
*   **Approach:** Direct implementation using tool-calling capabilities of the chosen LLM API.

## 7. Target Queries (Examples - unchanged)

*   "How do resource providers work in the Pulumi SDK? Focus on the Python SDK." (Uses `path_glob`)
*   "Where is the main authentication logic handled?" (Broad keyword search)
*   "Summarize the arguments in PR #123." (Filters by `item_type` and `item_id`, then reads/scrolls)
*   "Find discussions related to 'state locking'." (Filters by `item_type`, uses keywords)

This specification should provide a solid foundation for building the first version. Let me know if you'd like any adjustments! 