import contextlib

import aiofiles
import aiohttp

from . import settings


@contextlib.asynccontextmanager
async def github_client_session():
    """Creates or returns a global aiohttp ClientSession."""
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if settings.GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {settings.GITHUB_TOKEN}"

    timeout = aiohttp.ClientTimeout(total=settings.REQUESTS_TIMEOUT)
    async with aiohttp.ClientSession(
        headers=headers, timeout=timeout, base_url=settings.GITHUB_API_URL
    ) as session:
        yield session


async def get_repo_default_branch(
    session: aiohttp.ClientSession, repo_name: str
) -> str:
    """Gets the default branch name for a repository."""
    async with await session.get(f"/repos/{repo_name}") as response:
        response.raise_for_status()
        data = await response.json()
        if "default_branch" not in data:
            raise ValueError(
                f"Could not determine default branch for {repo_name}. Response: {data}"
            )
        return data["default_branch"]


async def download_repo_tarball(
    session: aiohttp.ClientSession, repo_name: str, ref: str, output_path: str
):
    """Downloads the repository source code as a tarball."""
    url = f"/repos/{repo_name}/tarball/{ref}"

    async with await session.get(url, allow_redirects=True) as response:
        response.raise_for_status()

        async with aiofiles.open(output_path, mode="wb+") as f:
            async for chunk in response.content.iter_chunked(8192):
                await f.write(chunk)
