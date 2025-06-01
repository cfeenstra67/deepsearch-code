import asyncio
import os
import tempfile

from deepsearch_code import core


class ScrollableString:
    def __init__(
        self, value: str, line_limit: int, scroll_cushion: int | None = None
    ) -> None:
        if scroll_cushion is None:
            scroll_cushion = max(1, int(line_limit / 10))
        self.value = value
        self.lines = value.splitlines()
        self.line_limit = line_limit
        self.line_offset = 0
        self.scroll_cushion = scroll_cushion

    def scroll_to(self, line: int) -> tuple[str, list[core.Tool]]:
        min_offset = 0
        max_offset = max(len(self.lines) - self.line_limit, 0)

        expected_offset = line - (self.line_limit // 2)
        new_offset = max(min_offset, min(max_offset, expected_offset))

        self.line_offset = new_offset

        return self.output()

    def scroll_down(self) -> tuple[str, list[core.Tool]]:
        """
        Scroll the displayed content down
        """
        max_offset = max(len(self.lines) - self.line_limit, 0)
        if self.line_offset >= max_offset:
            return self.output()

        new_offset = self.line_offset + self.line_limit - self.scroll_cushion
        self.line_offset = min(new_offset, max_offset)

        return self.output()

    def scroll_up(self) -> tuple[str, list[core.Tool]]:
        """
        Scroll the displayed content up
        """
        min_offset = 0
        if self.line_offset <= min_offset:
            return self.output()

        new_offset = self.line_offset - self.line_limit + self.scroll_cushion
        self.line_offset = max(new_offset, min_offset)

        return self.output()

    def output(self) -> tuple[str, list[core.Tool]]:
        if self.value == "":
            return "<no content>", []

        if len(self.lines) <= self.line_limit:
            return self.value, []

        tools: list[core.Tool] = []

        out_lines: list[str] = []
        if self.line_offset > 0:
            out_lines.append(
                f"--- {self.line_offset} line(s) hidden, scroll up to view ---"
            )
            tools.append(core.tool(self.scroll_up))

        out_lines.extend(
            self.lines[self.line_offset : self.line_offset + self.line_limit]
        )

        end_offset = len(self.lines) - (self.line_offset + self.line_limit)
        if end_offset > 0:
            out_lines.append(
                f"--- {end_offset} line(s) hidden, scroll down to view ---"
            )
            tools.append(core.tool(self.scroll_down))

        return "\n".join(out_lines), tools


class Shell:
    def __init__(self, cwd: str = ".", line_limit: int = 500) -> None:
        self.cwd = cwd
        self.line_limit = line_limit

    def view(self, value: str) -> ScrollableString:
        return ScrollableString(value, self.line_limit)


async def run_tool_command(
    cmd: str, args: list[str], shell: Shell
) -> tuple[str, list[core.Tool]]:
    with tempfile.TemporaryFile() as ntf, open(os.devnull, "ab") as devnull:
        process = await asyncio.create_subprocess_exec(
            cmd, *args, stdout=ntf, stderr=ntf, stdin=devnull, cwd=shell.cwd
        )

        await process.wait()

        ntf.flush()
        ntf.seek(0)
        result = ntf.read().decode()

        return shell.view(result).output()
