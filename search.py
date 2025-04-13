#!/usr/bin/env -S uv run python
import os
import shutil

import click

from deepsearch_code import core, oracles, search_tools
from deepsearch_code.download import download_repo
from deepsearch_code.openrouter import get_openrouter_client
from deepsearch_code.utils import async_command, setup_logging

REPOS_DIR = ".repos"


@click.group()
def cli():
    pass


class ToolStats:
    def __init__(self) -> None:
        self.tool_calls: dict[str, int] = {}
        core.agent_created.connect(self.agent_created)

    def agent_created(self, agent):
        self.bind(agent)

    async def tool_called(self, sender, tool, args, response):
        self.tool_calls.setdefault(tool.name(), 0)
        self.tool_calls[tool.name()] += 1

    def bind(self, agent: core.Agent) -> None:
        core.tool_called.connect(self.tool_called, sender=agent)


def search_agent(
    repo: str, oracle: core.Oracle, questions: search_tools.ResearchQuestions
) -> core.Agent:
    conversation = core.Conversation(oracle)

    prompt = core.StringPrompt(
        f"""
You are a seasoned researcher who thinks deeply, synthesizes large amounts of information and produces comprehensive and precise yet concise responses. Responses should be a long as needed to answer the user's question and no more.

You'll be researching the following repository: {repo}. In order to answer questions about it, you'll have the ability to read files and search files within the repository, and you'll have to piece together your answer based on the content of the repository. Even if you think you already know the answer, you should find code that confirms your priors.

In your response, please include key links to files within the repository as markdown links, using relative paths from the repository root, optionally with line numbers or ranges in the hash parameter. [for example](README.md#L112-L145)
""".strip()
    )

    repo_path = os.path.join(REPOS_DIR, repo)

    shell = search_tools.Shell(repo_path)
    ripgrep = search_tools.ripgrep_tool(shell)
    find = search_tools.find_tool(shell)
    tree = search_tools.tree_tool(shell)
    read_file = search_tools.read_file_tool(shell)

    return core.BasicAgent(
        conversation,
        prompt=core.Prompts([prompt, questions.prompt()]),
        tools=[find, tree, ripgrep, read_file],
    )


def manager_agent(
    repo: str,
    oracle: core.Oracle,
    questions: search_tools.ResearchQuestions,
    search_agent: core.Agent,
) -> core.Agent:
    conversation = core.Conversation(oracle)

    prompt = core.StringPrompt(
        f"""
You are a shrewd and prescient manager who manages a team of researchers to give your client comprehensive research reports for their quesions about a specific github repo. Right now you're researching the {repo} repository. Your ultimate responsibility is to ensure the quality of the output that goes back to the client. You will have the ability to ask questions that your team of researchers will attempt to answer for you. As you get more information, you should update your expectation of what the output will look like and ask more questions to fill out gaps in the information you have. In the end, you will also be the one that writes the final report that goes out to the client. Stay on top of these guys--don't just accept it if they tell you something isn't possible, if they can't clearly articulate why they likely need to dig deeper. You're the great orchestrator in this process, and again, you're accountable for the final output. Make sure to be precise in the questions you ask your team--they don't have as much context as you.

In your response, please include key links to files within the repository as markdown links, using relative paths from the repository root, optionally with line numbers or ranges in the hash parameter. [for example](README.md#L112-L145)
""".strip()
    )

    expected_output = search_tools.ExpectedOutput()

    return core.BasicAgent(
        conversation,
        prompt=core.Prompts([prompt, expected_output.prompt(), questions.prompt()]),
        tools=[expected_output.update_tool(), questions.research_tool(search_agent)],
    )


@async_command(cli)
@click.argument("repo")
@click.argument("question")
@click.option("--download", is_flag=True)
@click.option("--repl", is_flag=True)
@click.option("-o", "--output", default=None)
@click.option("-m", "--model", default="google/gemini-2.5-pro-preview-03-25")
async def search(
    repo: str, question: str, download: bool, repl: bool, output: str | None, model: str
) -> None:
    setup_logging()

    repo_path = os.path.join(REPOS_DIR, repo)
    if not os.path.isdir(repo_path) or download:
        if os.path.isdir(repo_path):
            shutil.rmtree(repo_path)
        os.makedirs(repo_path)
        await download_repo(repo, repo_path)
        print("Downloaded", repo)
    else:
        print("Already downloaded", repo)

    client = get_openrouter_client()
    tracker = oracles.UsageTracker()
    oracle: core.Oracle
    if repl:
        oracle = oracles.ReplOracle()
    else:
        oracle = oracles.LLMOracle(
            model=model,
            client=client,
            tracker=tracker,
        )

    stats = ToolStats()

    questions = search_tools.ResearchQuestions()

    search = search_agent(repo, oracle, questions)

    manager = manager_agent(repo, oracle, questions, search)

    response = await manager.run(question)

    print("USAGE", tracker.total())
    print("TOOL CALLS", stats.tool_calls)
    print("RESPONSE", response.answer)

    if output:
        with open(output, "w+") as f:
            f.write(response.answer)


if __name__ == "__main__":
    cli()
