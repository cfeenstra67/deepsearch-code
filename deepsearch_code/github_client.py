import asyncio
import os
import time
from typing import Optional

import aiofiles
import aiohttp

from . import settings

# --- Global Session ---
# Create the session contextually in functions needing it or manage globally if preferred
_SESSION: Optional[aiohttp.ClientSession] = None


async def get_session() -> aiohttp.ClientSession:
    """Creates or returns a global aiohttp ClientSession."""
    global _SESSION
    if _SESSION is None or _SESSION.closed:
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if settings.GITHUB_TOKEN:
            headers["Authorization"] = f"Bearer {settings.GITHUB_TOKEN}"

        timeout = aiohttp.ClientTimeout(total=settings.REQUESTS_TIMEOUT)
        _SESSION = aiohttp.ClientSession(headers=headers, timeout=timeout)
    return _SESSION


async def close_session():
    """Closes the global session."""
    global _SESSION
    if _SESSION:
        await _SESSION.close()
        _SESSION = None


# --- Request Helper ---
async def _make_request(method: str, url: str, **kwargs) -> aiohttp.ClientResponse:
    """Makes an HTTP request using aiohttp, handling rate limits and errors."""
    session = await get_session()
    while True:
        try:
            response = await session.request(method, url, **kwargs)

            # Handle rate limiting
            if (
                response.status == 403
                and "X-RateLimit-Remaining" in response.headers
                and int(response.headers["X-RateLimit-Remaining"]) == 0
            ):
                reset_time = int(
                    response.headers.get("X-RateLimit-Reset", time.time() + 60)
                )
                wait_time = max(1, reset_time - time.time()) + 2  # Add buffer
                print(f"Rate limit hit. Waiting for {wait_time:.0f} seconds...")
                await response.release()  # Release connection before sleeping
                await asyncio.sleep(wait_time)
                continue  # Retry the request

            # Raise exceptions for other client/server errors (4xx/5xx)
            # raise_for_status() will release the response payload
            response.raise_for_status()
            return response  # Return the response object directly

        except aiohttp.ClientError as e:
            print(f"HTTP Error during {method} request to {url}: {e}")
            raise  # Re-raise client errors
        except asyncio.TimeoutError:
            print(f"Request timed out for {method} {url}")
            raise  # Re-raise timeout


# --- Core API Functions ---


async def get_repo_default_branch(repo_name: str) -> str:
    """Gets the default branch name for a repository."""
    url = f"{settings.GITHUB_API_URL}/repos/{repo_name}"
    async with await _make_request("GET", url) as response:
        data = await response.json()
        if "default_branch" not in data:
            raise ValueError(
                f"Could not determine default branch for {repo_name}. Response: {data}"
            )
        return data["default_branch"]


async def download_repo_tarball(repo_name: str, ref: str, output_path: str):
    """Downloads the repository source code as a tarball."""
    url = f"{settings.GITHUB_API_URL}/repos/{repo_name}/tarball/{ref}"
    print(f"Attempting to download tarball from {url} to {output_path}")

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        # Use stream=True equivalent by not reading the body immediately
        async with await _make_request("GET", url, allow_redirects=True) as response:
            if response.status == 200:
                async with aiofiles.open(output_path, mode="wb") as f:
                    downloaded_size = 0
                    start_time = time.time()
                    async for chunk in response.content.iter_chunked(
                        8192
                    ):  # Read in chunks
                        await f.write(chunk)
                        downloaded_size += len(chunk)
                        # Optional: Add progress indicator here if needed
                    end_time = time.time()
                    duration = end_time - start_time
                    print(
                        f"Downloaded {downloaded_size / (1024 * 1024):.2f} MB in {duration:.2f}s to {output_path}"
                    )
            else:
                # Try to read error message if download fails
                error_body = await response.text()
                raise aiohttp.ClientResponseError(
                    response.request_info,
                    response.history,
                    status=response.status,
                    message=f"Failed to download tarball. Status: {response.status}. Body: {error_body[:500]}",  # Truncate long errors
                    headers=response.headers,
                )
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
        print(f"Failed to download or save tarball for {repo_name}@{ref}: {e}")
        # Clean up partial download if it exists
        if os.path.exists(output_path):
            os.remove(output_path)
        raise  # Re-raise the exception
