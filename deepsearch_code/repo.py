import asyncio
import contextlib
import logging
import os
import re
import shlex
import shutil
from typing import Any

import aiohttp

from deepsearch_code import core, github_client, shell
from deepsearch_code.constants import DATA_DIR

REPOS_DIR = os.path.join(DATA_DIR, "repos")

LOGGER = logging.getLogger(__name__)


async def run_command(args: list[str], **kws) -> str:
    proc = await asyncio.create_subprocess_exec(
        args[0],
        *args[1:],
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        **kws,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode == 0:
        return stdout.decode().strip()

    raise RuntimeError(
        f"Command '{shlex.join(args)}' failed:"
        f"\nStdout:\n{stdout.decode()}\nStderr:\n{stderr.decode()}"
    )


async def current_branch(repo_dir: str) -> str:
    return await run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_dir)


def pull_request_filename(pr):
    pr_title = pr["title"]
    pr_num = pr["number"]
    created_at = pr["created_at"]
    created_date = created_at.split("T")[0]

    sluggified_name = re.sub(r"[^0-9a-zA-Z]", "-", pr_title.lower())
    sluggified_name = re.sub(r"-+", "-", sluggified_name)
    sluggified_name = sluggified_name[:100]

    return f"{pr_num}-{created_date}-{sluggified_name}.md"


def release_filename(release):
    tag_name = release["tag_name"]
    created_at = release["created_at"]
    created_date = created_at.split("T")[0]
    
    sluggified_name = re.sub(r"[^0-9a-zA-Z]", "-", tag_name.lower())
    sluggified_name = re.sub(r"-+", "-", sluggified_name)
    sluggified_name = sluggified_name[:100]
    
    return f"{created_date}-{sluggified_name}.md"


def issue_filename(issue):
    issue_title = issue["title"]
    issue_num = issue["number"]
    created_at = issue["created_at"]
    created_date = created_at.split("T")[0]

    sluggified_name = re.sub(r"[^0-9a-zA-Z]", "-", issue_title.lower())
    sluggified_name = re.sub(r"-+", "-", sluggified_name)
    sluggified_name = sluggified_name[:100]

    return f"{issue_num}-{created_date}-{sluggified_name}.md"


async def format_pull_request(
    pr, repo: "Repository", semaphore: asyncio.Semaphore
) -> str:
    session = repo.client_session
    if session is None:
        raise RuntimeError("repo not initialized")

    pr_num = pr["number"]
    pr_title = pr["title"]
    pr_body = pr["body"]
    pr_status = pr["state"]
    pr_labels = [label["name"] for label in pr["labels"]]
    created_at = pr["created_at"]
    closed_at = pr.get("closed_at")
    assignees = [assignee["login"] for assignee in pr["assignees"]]
    head_sha = pr["head"]["sha"]
    base_sha = pr["base"]["sha"]
    repo_name = pr["base"]["repo"]["full_name"]

    async with semaphore:
        comments = []
        async for comment in github_client.list_pull_request_comments(
            session, repo_name, pr_num
        ):
            comments.append(comment)

        diff = await run_command(["git", "diff", base_sha, head_sha], cwd=repo.path)

    lines = [
        f"# {pr_title} (#{pr_num})",
        pr_body,
        "## Metadata",
        f"Status: {pr_status}",
    ]
    if pr_labels:
        lines.append(f"Labels: {', '.join(pr_labels)}")
    lines.append(
        f"Created at: {created_at}",
    )
    if closed_at:
        lines.append(f"Closed at: {closed_at}")
    lines.extend([f"Assignees: {', '.join(assignees)}", f"Head SHA: {head_sha}", ""])

    if comments:
        lines.extend(["", "## Comments", ""])
        for idx, comment in enumerate(comments):
            lines.extend(
                [
                    f"### Comment {idx + 1}From: {comment['user']['login']}",
                    f"Date: {comment['created_at']}",
                    "",
                    comment["body"],
                    "",
                ]
            )

    lines.extend(
        [
            "## Diff",
            diff,
        ]
    )

    return "\n".join(lines).strip()


async def format_release(release) -> str:
    tag_name = release["tag_name"]
    release_name = release["name"] or tag_name
    release_body = release["body"] or ""
    draft = release["draft"]
    prerelease = release["prerelease"]
    created_at = release["created_at"]
    published_at = release.get("published_at")
    author = release["author"]["login"] if release.get("author") else "Unknown"
    
    lines = [
        f"# {release_name} ({tag_name})",
        release_body,
        "## Metadata",
        f"Tag: {tag_name}",
        f"Author: {author}",
        f"Draft: {draft}",
        f"Prerelease: {prerelease}",
        f"Created at: {created_at}",
    ]
    
    if published_at:
        lines.append(f"Published at: {published_at}")
    
    return "\n".join(lines).strip()


async def format_issue(issue, repo: "Repository", semaphore: asyncio.Semaphore) -> str:
    session = repo.client_session
    if session is None:
        raise RuntimeError("repo not initialized")

    issue_num = issue["number"]
    issue_title = issue["title"]
    issue_body = issue["body"] or ""
    issue_state = issue["state"]
    issue_labels = [label["name"] for label in issue["labels"]]
    created_at = issue["created_at"]
    closed_at = issue.get("closed_at")
    assignees = [assignee["login"] for assignee in issue["assignees"]]
    milestone = issue.get("milestone", {}).get("title") if issue.get("milestone") else None
    author = issue["user"]["login"]
    repo_name = f"{repo.owner}/{repo.repo}"

    async with semaphore:
        comments = []
        async for comment in github_client.list_pull_request_comments(
            session, repo_name, issue_num
        ):
            comments.append(comment)

    lines = [
        f"# {issue_title} (#{issue_num})",
        issue_body,
        "## Metadata",
        f"State: {issue_state}",
        f"Author: {author}",
    ]
    
    if issue_labels:
        lines.append(f"Labels: {', '.join(issue_labels)}")
    
    lines.append(f"Created at: {created_at}")
    
    if closed_at:
        lines.append(f"Closed at: {closed_at}")
    
    if milestone:
        lines.append(f"Milestone: {milestone}")
    
    lines.append(f"Assignees: {', '.join(assignees)}")

    if comments:
        lines.extend(["", "## Comments", ""])
        for idx, comment in enumerate(comments):
            lines.extend(
                [
                    f"### Comment {idx + 1}",
                    f"From: {comment['user']['login']}",
                    f"Date: {comment['created_at']}",
                    "",
                    comment["body"],
                    "",
                ]
            )

    return "\n".join(lines).strip()


class Repository:
    def __init__(self, owner: str, repo: str, path: str) -> None:
        self.owner = owner
        self.repo = repo
        self.path = path
        self.branch: str | None = None
        self.initial_branch: str | None = None
        self.client_session_ctx: Any = None
        self.client_session: aiohttp.ClientSession | None = None

    def meta_description(self) -> str:
        rel_path = os.path.relpath(self._meta_dir(), self.path)
        return f"""
Additional metadata has been added about the repository within the {rel_path} sub-directory as plain text files, so you can read and search against them (or exclude them from searches). The additional information provided there includes:
- {rel_path}/branches.txt - The output of "git branch", showing all branches in the repository.
- {rel_path}/commits.txt - The output of "git log", showing details about all commits in the current branch.
- {rel_path}/tags.txt - The output of "git tag", showing the names of all of the individual tags in the repository.
- {rel_path}/pulls/*.md - Information about all pull requests in the repository, formatted as markdown.
""".strip()

    async def __aenter__(self) -> "Repository":
        self.client_session_ctx = github_client.github_client_session()
        self.client_session = await self.client_session_ctx.__aenter__()
        await self.load()
        return self

    async def __aexit__(self, exc_type, exc_value, exc_tb) -> None:
        return await self.client_session_ctx.__aexit__(exc_type, exc_value, exc_tb)

    async def load(self) -> None:
        if self.initial_branch is None:
            self.initial_branch = await current_branch(self.path)
        if self.branch is None:
            self.branch = self.initial_branch
        await self.write()

    async def reset(self) -> None:
        await self.load()
        if self.initial_branch and self.branch != self.initial_branch:
            await self.checkout(self.initial_branch)

    async def checkout(self, ref: str) -> None:
        await run_command(["git", "checkout", ref], cwd=self.path)
        self.branch = await current_branch(self.path)
        await self.write(overwrite=True)

    async def write(self, overwrite: bool = False) -> None:
        await self.write_branches(overwrite=overwrite)
        await self.write_commits(overwrite=overwrite)
        await self.write_tags(overwrite=overwrite)
        await self.write_pull_requests(overwrite=overwrite)

    def _meta_dir(self) -> str:
        meta_dir = os.path.join(self.path, "REPO-METADATA")
        if not os.path.exists(meta_dir):
            os.makedirs(meta_dir)
        return meta_dir

    async def write_branches(self, overwrite: bool = False) -> None:
        branches_path = os.path.join(self._meta_dir(), "branches.txt")
        if os.path.exists(branches_path) and not overwrite:
            return

        branches = await run_command(["git", "branch"], cwd=self.path)
        with open(branches_path, "w+") as f:
            f.write(branches)

    async def write_commits(self, overwrite: bool = False) -> None:
        commits_path = os.path.join(self._meta_dir(), "commits.txt")
        if os.path.exists(commits_path) and not overwrite:
            return

        commits = await run_command(["git", "log"], cwd=self.path)
        with open(commits_path, "w+") as f:
            f.write(commits)

    async def write_tags(self, overwrite: bool = False) -> None:
        tags_path = os.path.join(self._meta_dir(), "tags.txt")
        if os.path.exists(tags_path) and not overwrite:
            return

        tags = await run_command(["git", "tag"], cwd=self.path)
        with open(tags_path, "w+") as f:
            f.write(tags)

    async def write_pull_requests(self, overwrite: bool = False) -> None:
        if self.client_session is None:
            raise RuntimeError("Repository must be used as an async context manager")

        pulls_path = os.path.join(self._meta_dir(), "pulls")
        if not os.path.exists(pulls_path):
            os.makedirs(pulls_path)

        semaphore = asyncio.Semaphore(10)

        names: set[str] = set()
        async for pull in github_client.list_pull_requests(
            self.client_session, f"{self.owner}/{self.repo}"
        ):
            file_name = pull_request_filename(pull)
            file_path = os.path.join(pulls_path, file_name)
            names.add(file_name)
            if os.path.exists(file_path) and not overwrite:
                continue

            file_body = await format_pull_request(pull, self, semaphore)
            with open(file_path, "w+") as f:
                f.write(file_body)

            LOGGER.debug("Wrote PR #%s '%s'", pull["number"], pull["title"])

        to_remove = set(os.listdir(pulls_path)) - names
        for file in to_remove:
            os.remove(os.path.join(pulls_path, file))

    def checkout_tool(self) -> core.Tool:
        @core.tool
        async def git_checkout(ref: str) -> str:
            """
            Check out the given ref in the current git repository. You'll pass a `ref` to check out.
            """
            await self.checkout(ref)
            return f"Checked out {ref}"

        return git_checkout

    def diff_tool(self, shell_obj: shell.Shell) -> core.Tool:
        @core.tool
        async def git_diff(cmd_args: list[str]):
            """
            Access to the `git diff` command line tool for viewing the underlying diff hunks of commits in the repository. You'll pass command line arguments (AFTER the git diff part of the command) as an array. The typical form of the command is `git diff ref1 ref2`, where you provide ref1 and ref2 and references to commits within the repository. You can also do something like `git diff ref1 ref2 ./some/path` to inspect the diff within a specific file or directory. You can use ref~N and ref^N to reference commits before and after refs respectively.
            Example 1, views the diff of the last 3 commits in master: ["master~3", "master"]
            Example 2, views the diff of a specific commit and a certain path: ["7da83d0~1", "7da83d0", "README.md"]
            """
            git_args = ["diff", *cmd_args]
            return await shell.run_tool_command("git", git_args, shell_obj)

        return git_diff


@contextlib.asynccontextmanager
async def download_github_repo(
    name: str,
    dest_dir: str | None = None,
    overwrite: bool = False,
    delete_on_failure: bool = True,
):
    owner, repo = name.split("/", 1)
    if dest_dir is None:
        repo_dir = os.path.join(REPOS_DIR, owner, repo)

    if os.path.exists(repo_dir) and not overwrite:
        LOGGER.debug("Path already exists: %s, skipping download", repo_dir)
        async with Repository(owner, repo, repo_dir) as repo_obj:
            yield repo_obj
        return

    if os.path.exists(repo_dir):
        shutil.rmtree(repo_dir)

    os.makedirs(repo_dir)
    try:
        LOGGER.info("Cloning %s/%s", owner, repo)
        await run_command(
            ["git", "clone", "-q", f"git@github.com:{owner}/{repo}", repo_dir]
        )
    except Exception:
        if delete_on_failure:
            shutil.rmtree(repo_dir)
        raise

    async with Repository(owner, repo, repo_dir) as repo_obj:
        yield repo_obj
