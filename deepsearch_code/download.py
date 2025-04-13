import asyncio
import tempfile

from deepsearch_code.github_client import (
    download_repo_tarball,
    get_repo_default_branch,
    github_client_session,
)


async def extract_repo(tar_path: str, dest: str) -> None:
    proc = await asyncio.create_subprocess_exec(
        "tar", "-xzf", tar_path, "--strip=1", cwd=dest
    )
    stdout, stderr = await proc.communicate()
    exit_code = await proc.wait()
    if exit_code != 0:
        output = b"\n".join([stdout, stderr]).decode()
        raise RuntimeError(f"failed to extract archive; output: {output}")


async def download_repo(name: str, path: str) -> None:
    async with github_client_session() as session:
        ref = await get_repo_default_branch(session, name)

        with tempfile.NamedTemporaryFile(suffix=".tar.gz") as ntf:
            await download_repo_tarball(session, name, ref, ntf.name)
            await extract_repo(ntf.name, path)
