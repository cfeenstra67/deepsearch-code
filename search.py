#!/usr/bin/env -S uv run python
import json
import os
import shutil
import time

import click
from pydantic import BaseModel

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
    conversation: core.Conversation,
    questions: search_tools.ResearchQuestions,
    search_agent: core.Agent,
) -> core.Agent:
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


class ResearchConversation(BaseModel):
    repo: str
    question: str
    answer: str
    research_questions: dict[str, str]
    manager_messages: list[core.Message]


def readable_time(seconds: float) -> str:
    hour = 3600
    day = 24 * hour

    current = seconds
    parts: list[str] = []
    for period, letter in [(day, "d"), (hour, "h"), (1, "s")]:
        whole_values = current // period
        if whole_values < 1:
            continue
        current -= whole_values * period
        parts.append(f"{whole_values}{letter}")
    
    return "".join(parts)


@async_command(cli)
@click.argument("repo")
@click.argument("question")
@click.option("--download", is_flag=True)
@click.option("--repl", is_flag=True)
@click.option("-i", "--input", default=None)
@click.option("-o", "--output", default=None)
@click.option("--researcher-model", default="google/gemini-2.0-flash-001")
@click.option("--manager-model", default="google/gemini-2.5-pro-preview-03-25")
async def search(
    repo: str,
    question: str,
    download: bool,
    repl: bool,
    output: str | None,
    input: str | None,
    researcher_model: str,
    manager_model: str,
) -> None:
    setup_logging()

    existing: ResearchConversation | None = None
    if input:
        with open(input) as f:
            json_data = json.load(f)
        existing = ResearchConversation.model_validate(json_data)
        if existing.repo != repo:
            raise click.Abort(
                f"The conversation at {input} is about {existing.repo}, not {repo}"
            )

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
    manager_oracle: core.Oracle
    researcher_oracle: core.Oracle
    if repl:
        manager_oracle = oracles.ReplOracle()
        researcher_oracle = oracles.ReplOracle()
    else:
        manager_oracle = oracles.LLMOracle(
            model=manager_model,
            client=client,
            tracker=tracker,
        )
        researcher_oracle = oracles.LLMOracle(
            model=researcher_model,
            client=client,
            tracker=tracker,
        )

    stats = ToolStats()

    questions = search_tools.ResearchQuestions(
        existing.research_questions if existing is not None else None
    )

    search = search_agent(repo, researcher_oracle, questions)

    manager_convo = core.Conversation(
        manager_oracle, existing.manager_messages if existing is not None else None
    )

    manager = manager_agent(repo, manager_convo, questions, search)

    start = time.time()

    response = await manager.run(question)

    elapsed = time.time() - start

    print(response.answer)

    print()
    print(f"Elapsed: ", readable_time(elapsed))
    print("LLM Usage:")
    llms = sorted(tracker.models.items(), key=lambda x: x[1]["total"], reverse=True)
    for llm, usage in llms:
        print(
            f"- {llm}: {usage['input']} input tokens ({usage['cached_input']} "
            f"cached), output {usage['output']}"
            
        )

    print()
    print("Tool calls:")
    tool_calls = sorted(stats.tool_calls.items(), key=lambda x: x[1], reverse=True)
    for tool_name, count in tool_calls:
        print(f"- {tool_name}: {count}x")

    if not output:
        return

    result = ResearchConversation(
        repo=repo,
        question=question,
        answer=response.answer,
        research_questions=questions.questions,
        manager_messages=manager_convo.messages,
    )

    with open(output, "w+") as f:
        data = result.model_dump(mode="json")
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    cli()
