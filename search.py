#!/usr/bin/env -S uv run python
import difflib
import os
import shutil

import click

from deepsearch_code import core, search_tools
from deepsearch_code.click import async_command
from deepsearch_code.download import download_repo
from deepsearch_code.logging import setup_logging
from deepsearch_code.openrouter import get_openrouter_client

REPOS_DIR = ".repos"


@click.group()
def cli():
    pass


class AgentLogger:
    def __init__(self) -> None:
        self.tool_calls: dict[str, int] = {}

    def agent_created(self, agent):
        print("Agent created", agent)
        self.bind(agent)

    async def agent_responded(self, sender, raw, tool, args):
        print("Agent response:\n", raw.content)

    async def tool_called(self, sender, tool, args, response):
        print(
            f"Tool response for {tool.name()} with {args!r}:\n{tool.format(response)}"
        )
        self.tool_calls.setdefault(tool.name(), 0)
        self.tool_calls[tool.name()] += 1

    def bind(self, agent: core.Agent) -> None:
        core.agent_responded.connect(self.agent_responded, sender=agent)
        core.tool_called.connect(self.tool_called, sender=agent)

    def bind_all(self) -> None:
        core.agent_created.connect(self.agent_created)


class ExpectedOutput(core.Plugin):
    def __init__(self) -> None:
        self.expected_output: str | None = None

    def prompt(self) -> core.Prompt:
        def get_message():
            lines = ["## Expected Research Output Structure", ""]
            if self.expected_output is None:
                lines.append(
                    "You haven't set an expectation of how the output should be structured yet. Use the update_expected_output tool to do so."
                )
            else:
                lines.append(self.expected_output)
            return "\n".join(lines)

        return core.FunctionPrompt(get_message)

    def tools(self) -> list[core.Tool]:
        @core.tool
        def update_expected_output(new_expected_output: str) -> str:
            """
            Update the expected outpu structure to match your expectations based on the information now available to you. This will replace the existing expectations, so take care to avoid regressions.
            """
            if self.expected_output is None:
                response = "The expected output structure has been set"
            else:
                diff = difflib.ndiff(
                    self.expected_output.splitlines(keepends=True),
                    new_expected_output.splitlines(keepends=True),
                )
                diff_txt = "".join(diff)
                response = f"The expected output structure has been updated. Diff from previous:\n{diff_txt or '<none>'}"

            self.expected_output = new_expected_output

            return response

        return [update_expected_output]

    def clone(self) -> "ExpectedOutput":
        return ExpectedOutput()


class ResearchTeam(core.Plugin):
    def __init__(self, search_agent: core.Agent) -> None:
        self.search_agent = search_agent
        self.questions: dict[str, str] = {}

    def prompt(self) -> core.Prompt:
        def get_message():
            lines = ["## Research Questions", ""]
            if not self.questions:
                lines.append("You haven't posed any questions for your team yet")
            else:
                for question, answer in self.questions.items():
                    lines.append(f"### {question}")
                    lines.append(answer)
                    lines.append("")

            return "\n".join(lines)

        return core.FunctionPrompt(get_message)

    def tools(self) -> list[core.Tool]:
        @core.tool
        async def research_question(question: str):
            """
            Ask a question for your team of researchers to try to answer from the repository
            """
            agent = self.search_agent.clone()

            response = await agent.run(question)

            @core.tool
            async def request_changes(changes: str):
                """
                Request changes to the previous response from research_question
                """
                response = await agent.run(
                    f"The following changes were requested:\n{changes}"
                )
                return response.answer, [request_changes]

            self.questions[question] = response.answer

            return f"The team responded with:\n{response.answer}", [request_changes]

        return [research_question]

    def clone(self) -> "ResearchTeam":
        return ResearchTeam(self.search_agent)


def search_agent(repo: str, oracle: core.Oracle) -> core.Agent:
    conversation = core.Conversation(oracle)

    prompt = core.StringPrompt(
        f"""
You are a seasoned researcher who thinks deeply, synthesizes large amounts of information and produces comprehensive and precise yet concise responses. Responses should be a long as needed to answer the user's question and no more.

You'll be researching the following repository: {repo}. In order to answer questions about it, you'll have the ability to read files and search files within the repository, and you'll have to piece together your answer based on the content of the repository. Even if you think you already know the answer, you should find code that confirms your intuitions.

In your response, please include key links to files within the repository as markdown links, using relative paths from the repository root, optionally with line numbers or ranges in the hash parameter. [for example](README.md#L112-L145)
""".strip()
    )

    repo_path = os.path.join(REPOS_DIR, repo)

    shell = search_tools.Shell(repo_path)
    ripgrep = search_tools.ripgrep_tool(shell)
    find = search_tools.find_tool(shell)
    read_file = search_tools.read_file_tool(shell)

    return core.BasicAgent(
        conversation, prompt=prompt, tools=[find, ripgrep, read_file]
    )


def manager_agent(
    repo: str, oracle: core.Oracle, search_agent: core.Agent
) -> core.Agent:
    conversation = core.Conversation(oracle)

    prompt = core.StringPrompt(
        f"""
You are a shrewd and prescient manager who manages a team of researchers to give your client comprehensive research reports for their quesions about a specific github repo. Right now you're researching the {repo} repository. Your ultimate responsibility is to ensure the quality of the output that goes back to the client. You will have the ability to ask questions that your team of researchers will attempt to answer for you. As you get more information, you should update your expectation of what the output will look like and ask more questions to fill out gaps in the information you have. In the end, you will also be the one that writes the final report that goes out to the client. Stay on top of these guys--don't just accept it if they tell you something isn't possible, if they can't clearly articulate why they likely need to dig deeper. You're the great orchestrator in this process, and again, you're accountable for the final output. Make sure to be precise in the questions you ask your team--they don't have as much context as you.

In your response, please include key links to files within the repository as markdown links, using relative paths from the repository root, optionally with line numbers or ranges in the hash parameter. [for example](README.md#L112-L145)
""".strip()
    )

    expected_output = ExpectedOutput()
    team = ResearchTeam(search_agent)

    return core.BasicAgent(conversation, prompt=prompt, plugins=[expected_output, team])


@async_command(cli)
@click.argument("repo")
@click.argument("question")
@click.option("--download", is_flag=True)
@click.option("--repl", is_flag=True)
@click.option("-o", "--output", default=None)
@click.option("-m", "--model", default="deepseek/deepseek-r1")
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
    tracker = core.UsageTracker()
    oracle: core.Oracle
    if repl:
        oracle = core.ReplOracle()
    else:
        oracle = core.LLMOracle(
            model=model,
            client=client,
            tracker=tracker,
        )

    logger = AgentLogger()
    logger.bind_all()

    search = search_agent(repo, oracle)

    manager = manager_agent(repo, oracle, search)

    response = await manager.run(question)

    print("USAGE", tracker.total())
    print("TOOL CALLS", logger.tool_calls)
    print("RESPONSE", response.answer)

    if output:
        with open(output, "w+") as f:
            f.write(response.answer)


if __name__ == "__main__":
    cli()
