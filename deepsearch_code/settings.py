# Configuration settings for the DeepSearch Code tool

import os

from dotenv import load_dotenv

load_dotenv()

# --- GitHub ---
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    # Allow running without a token for public repos, but rate limits will be stricter
    print(
        "Warning: GITHUB_TOKEN environment variable not set. Using anonymous access (strict rate limits apply)."
    )
GITHUB_API_URL = "https://api.github.com"
# Increase timeout for potentially large downloads
REQUESTS_TIMEOUT = 60  # Timeout for HTTP requests in seconds

# --- Data Storage ---
DB_FILE = "repo_index.db"
# Base directory to store downloaded/formatted data
MATERIALIZED_DATA_DIR = "materialized_data"

# --- Indexing ---
# Files/directories to ignore during code indexing
CODE_IGNORE_DIRS = {
    ".git",
    ".github",
    "__pycache__",
    "node_modules",
    "vendor",
    "build",
    "dist",
}
CODE_IGNORE_EXTENSIONS = {
    # Add common non-text file extensions
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".tiff",
    ".ico",
    ".svg",
    ".pdf",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
    ".zip",
    ".tar",
    ".gz",
    ".rar",
    ".7z",
    ".bz2",
    ".mp3",
    ".wav",
    ".ogg",
    ".mp4",
    ".avi",
    ".mov",
    ".wmv",
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".class",
    ".jar",
    ".pyc",
    ".lock",
    ".bin",
    ".obj",
    ".o",
    ".a",
    ".lib",
    ".iso",
    ".dmg",
    ".ttf",
    ".woff",
    ".woff2",
    ".eot",  # Fonts
}
# Max size for files *read into memory* for indexing, tarball download handles large repo size.
MAX_INDEXING_FILE_SIZE_BYTES = 5 * 1024 * 1024

# --- API Fetching ---
PAGE_SIZE = 100  # Number of items to fetch per page for lists (issues, PRs, etc.)
