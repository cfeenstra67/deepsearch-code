import difflib
import glob
import os

from pydantic import BaseModel, Field

from deepsearch_code import core
from deepsearch_code.shell import Shell, run_tool_command


def ripgrep_tool(shell: Shell) -> core.Tool:
    @core.tool
    async def rg(cmd_args: list[str]):
        """
        The ripgrep (called rg) search tool for searching the content of files. You'll pass command line arguments as an array. The search query is provided as the first argument, and a directory or file can be provided as the second argument to limit the search to that. A few of the key options to be aware of are:
        - -e PATTERN, --regexp=PATTERN - A pattern to search for. This option can be provided multiple times, where all patterns given are searched. Lines matching at least one of the provided patterns are printed. This flag can also be used when searching for patterns that start with a dash.
        - -l, --files-with-matches - Print only the paths with at least one match and suppress match contents. This overrides --files-without-match.
        - --files-without-match - Print the paths that contain zero matches and suppress match contents. This overrides -l/--files-with-matches.
        - -g GLOB, --glob=GLOB - Include or exclude files and directories for searching that match the given glob. This always overrides any other ignore logic. Multiple glob flags may be used. Globbing rules match .gitignore globs. Precede a glob with a ! to exclude it. If multiple globs match a file or directory, the glob given later in the command line takes precedence.
        Example 1, searches for 'python' in the 'src' directory: ["python", "src"]
        Example 2, does the same but limits it to python files: ["python", "src", "--glob=*.py"]
        """
        rg_args = ["--heading", "--line-number", *cmd_args]
        return await run_tool_command("rg", rg_args, shell)

    return rg


def find_tool(shell: Shell) -> core.Tool:
    @core.tool
    async def find(cmd_args: list[str]) -> tuple[str, list[core.Tool]]:
        """
        The `find` command line tool for finding files. You'll pass the command line arguments as an array. The first argument must always be first. Some of the key arguments be aware of are:
        - -name GLOB - search for a file name with glob syntax
        - -type d or -type f - filter to just directories or just files respectively.
        - -maxdepth DEPTH - limit results to those within a provided depth
        Example: [".", "-name", "*.py", "-type", "f"]
        """
        return await run_tool_command("find", cmd_args, shell)

    return find


def tree_tool(shell: Shell) -> core.Tool:
    @core.tool
    async def tree(cmd_args: list[str]) -> tuple[str, list[core.Tool]]:
        """
        The `tree` command line tool for viewing directory structure. No arguments are required, in which case you'll receive the entire directory structure of the repository. You can pass a path to see the directory structure of that path. Key options:
        - -L level - Max display depth of the directory tree.
        - -a - All files are printed.  By default tree does not print hidden files (those beginning with a dot `.')
        Example: ["src", "-L", "2"]
        """
        return await run_tool_command("tree", cmd_args, shell)

    return tree


def awk_tool(shell: Shell) -> core.Tool:
    @core.tool
    async def awk(cmd_args: list[str]) -> tuple[str, list[core.Tool]]:
        """
        awk is a small but expressive data-driven language that reads one line at a time from standard input or a list of files, splits each line into fields (default: whitespace, or any regular expression you give with -F), then runs every line through a set of pattern > action rules-printing lines that match, transforming fields, accumulating statistics, or driving shell pipelines all in a single pass. You'll pass the command line arguments as an array. The first argument must always be first. Some of the key arguments be aware of are:
        - -F fs Set the input field separator FS to the regular-expression _fs_
        - -v var=val Assign shell vars before the program starts (e.g., -v total=0)
        - 'prog' Inline program in single quotes, e.g. 'NF>2 {print $1,$NF}'
        - -f file Load the program from a file (may repeat)
        - Pattern syntax /regex/, relational ($3>100), or range /start/,/end/
        - Actions Anything inside { }, defaults to {print $0} if omitted
        - Field/record vars $0 (whole line), $1...$NF (fields), NR/FNR (line numbers), NF (field count)
        - Special blocks BEGIN { ... } (runs once before input) and END { ... } (after all input)
        - Handy statements print, printf, next (skip to next record), exit, split(), gsub()
        Example 1, Print the second column of a CSV file: ["-F,", "{ print $2 }", "data.csv"]
        Example 2, Sum the first column and report the average: ["-F,", "{ sum+=$1 } END { print "total", sum, \"avg\", sum/NR }", "**/*.txt"]
        """
        cmd_args = list(cmd_args)
        for idx, arg in cmd_args:
            if not arg.startswith("*"):
                continue
            del cmd_args[idx]
            expanded = glob.glob(arg, shell.cwd)
            for item in expanded:
                cmd_args.insert(idx, item)

        return await run_tool_command("awk", cmd_args, shell)

    return awk


class ReadFileArgs(BaseModel):
    path: str = Field(description="Relative path to a file")
    line_number: int | None = Field(
        default=None,
        description="Optional line number to start on. You'll be able to scroll up and down as needed to view more of the file.",
    )


def read_file_tool(shell: Shell) -> core.Tool:
    def read_file(path: str, line_number: int | None) -> tuple[str, list[core.Tool]]:
        """
        Read the contents of a specific file. If the file is large, you'll be able to scroll up and down as needed to see the relevant content.
        """
        full_path = os.path.join(shell.cwd, path)
        try:
            with open(full_path) as f:
                content = f.read()

                view = shell.view(content)

                if line_number is not None:
                    view.scroll_to(line_number)

                return view.output()
        except FileNotFoundError:
            return f"File not found: {path}", []

    return core.tool(read_file, input_schema=ReadFileArgs)


class ExpectedOutput:
    def __init__(self, expected_output: str | None = None) -> None:
        self.expected_output = expected_output

    def prompt(self) -> core.Prompt:
        def get_message():
            lines = ["<expected-research-output-structure>"]
            if self.expected_output is None:
                lines.append(
                    "You haven't set an expectation of how the output should be structured yet. Use the update_expected_output_structure tool to do so."
                )
            else:
                lines.append(self.expected_output)
            lines.append("</expected-research-output-structure>")
            return "\n".join(lines)

        return core.FunctionPrompt(get_message)

    def update_tool(self) -> core.Tool:
        @core.tool
        def update_expected_output_structure(new_expected_output: str) -> str:
            """
            Update the expected output structure to match your expectations based on the information now available to you. This will replace the existing expectations, so take care to avoid regressions.
            """
            if self.expected_output is None:
                response = "The expected output structure has been set"
            else:
                diff = difflib.unified_diff(
                    self.expected_output.splitlines(keepends=True),
                    new_expected_output.splitlines(keepends=True),
                )
                diff_txt = "".join(diff)
                response = f"The expected output structure has been updated. Diff from previous:\n{diff_txt or '<none>'}"

            self.expected_output = new_expected_output

            return response

        return update_expected_output_structure


class ResearchQuestions:
    def __init__(self, questions: dict[str, str] | None = None) -> None:
        if questions is None:
            questions = {}
        self.questions = questions

    def prompt(self) -> core.Prompt:
        def get_message():
            lines = ["<past-research-questions>"]
            if not self.questions:
                lines.append("No research questions have been answered yet")
            else:
                for question, answer in self.questions.items():
                    lines.append("<question>")
                    lines.append(question)
                    lines.append("<answer>")
                    lines.append(answer)
                    lines.append("</answer>")
                    lines.append("</question>")

            lines.append("</past-research-questions>")

            return "\n".join(lines)

        return core.FunctionPrompt(get_message)

    def research_tool(self, search_agent: core.Agent) -> core.Tool:
        @core.tool
        async def research_question(question: str):
            """
            Ask a question for your team of researchers to try to answer from the repository
            """
            agent = search_agent.clone()

            response = await agent.run(question)

            @core.tool
            async def request_changes(changes: str):
                """
                Request changes to the previous response from research_question
                """
                response = await agent.run(
                    f"The following changes were requested:\n{changes}"
                )
                return response, [request_changes]

            self.questions[question] = response

            return f"The team responded with:\n{response}", [request_changes]

        return research_question
