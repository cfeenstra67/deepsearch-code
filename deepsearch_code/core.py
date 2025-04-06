import abc
import asyncio
import dataclasses as dc
import inspect
import json
import os
import shlex
import tempfile
import textwrap
from typing import Any, Awaitable, Callable, Generic, Literal, Type, TypeVar

from pydantic import BaseModel, TypeAdapter, create_model

ToolInput = TypeVar("ToolInput", bound=BaseModel)

ToolOutput = TypeVar("ToolOutput")


class Tool(Generic[ToolInput, ToolOutput], abc.ABC):
    @abc.abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def description(self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def args(self) -> Type[ToolInput]:
        raise NotImplementedError

    @abc.abstractmethod
    async def call(
        self, args: ToolInput, conversation: "Conversation"
    ) -> tuple[ToolOutput, list["Tool"]]:
        raise NotImplementedError

    @abc.abstractmethod
    def format(self, output: ToolOutput) -> str:
        raise NotImplementedError


@dc.dataclass(frozen=True)
class FunctionTool(Tool[ToolInput, ToolOutput]):
    func: Callable[
        [ToolInput, "Conversation"], Awaitable[tuple[ToolOutput, list[Tool]]]
    ]
    tool_name: str
    tool_description: str
    tool_args: Type[ToolInput]
    tool_format: Callable[[ToolOutput], str]

    def name(self) -> str:
        return self.tool_name

    def description(self) -> str:
        return self.tool_description

    def args(self) -> Type[ToolInput]:
        return self.tool_args

    async def call(
        self, args: ToolInput, conversation: "Conversation"
    ) -> tuple[ToolOutput, list["Tool"]]:
        return await self.func(args, conversation)

    def format(self, output: ToolOutput) -> str:
        return self.tool_format(output)


def tool(
    func: Callable[..., ToolOutput | Awaitable[ToolOutput]],
    *,
    name: str | None = None,
    description: str | None = None,
    format: Callable[[ToolOutput], str] | None = None,
) -> FunctionTool:
    if name is None:
        name = func.__name__
    if description is None:
        description = textwrap.dedent(func.__doc__ or "").strip()
    if format is None:

        def format(x):
            if isinstance(x, str):
                return x
            return json.dumps(x, indent=2)

    input_fields = {}
    pass_conversation = False

    sig = inspect.signature(func)

    for param_name, param_val in sig.parameters.items():
        if param_val.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            raise ValueError("Cannot use variable args w/ tool()")
        if param_name == "conversation":
            pass_conversation = True
        else:
            input_fields[param_name] = param_val.annotation

    input_schema = create_model(f"{name}InputSchema", **input_fields)

    async def wrapper(input, conversation):
        kws = input.model_dump()
        if pass_conversation:
            kws["conversation"] = conversation
        result = func(**kws)
        if inspect.isawaitable(result):
            result = await result
        if not isinstance(result, tuple):
            return result, []
        return result

    return FunctionTool(
        wrapper,
        tool_name=name,
        tool_description=description,
        tool_args=input_schema,
        tool_format=format,
    )


@dc.dataclass(frozen=True)
class Message:
    role: Literal["user", "assistant", "system"]
    content: str


class Oracle(abc.ABC):
    @abc.abstractmethod
    async def ask(
        self, messages: list[Message], tools: list[Tool]
    ) -> tuple[str, Tool, Any]:
        raise NotImplementedError


class Conversation:
    def __init__(
        self,
        oracle: Oracle,
        messages: list[Message] | None = None,
    ) -> None:
        if messages is None:
            messages = []
        self.oracle = oracle
        self.messages = messages

    def new(self) -> "Conversation":
        return Conversation(self.oracle)

    def fork(self) -> "Conversation":
        return Conversation(self.oracle, self.messages)

    async def ask(self, message: Message, tools: list[Tool]) -> tuple[Tool, Any]:
        all_messages = self.messages + [message]
        raw, tool, args = await self.oracle.ask(all_messages, tools)
        self.messages.append(message)
        self.messages.append(Message(role="assistant", content=raw))

        return tool, args


class ReplOracle(Oracle):
    async def ask(
        self, messages: list[Message], tools: list[Tool]
    ) -> tuple[str, Tool, Any]:
        for message in messages:
            print(f"Role: {message.role}")
            print(f"Content: {message.content}")
            print()

        tools_by_name = {tool.name(): tool for tool in tools}
        print("Available tools")
        for available_tool in tools:
            print(f"{available_tool.name()}: {available_tool.description()}")
        print()

        tool: Tool | None = None
        while tool is None:
            tool_name = input("Input tool name: ")
            if tool_name in tools_by_name:
                tool = tools_by_name[tool_name]
            else:
                print(f"Invalid tool: {repr(tool_name)}. Try again")

        input_schema = tool.args()

        fields = {}
        for field, value in input_schema.model_fields.items():
            raw = input(f"{field}: ")
            fields[field] = TypeAdapter(value.annotation).validate_strings(raw)

        input_obj = input_schema.model_validate(fields)

        print("==========================================")

        raw = f"Called {tool_name} with {json.dumps(fields)}"

        return raw, tool, input_obj


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

    def scroll_down(self) -> tuple[str, list[Tool]]:
        """
        Scroll the displayed content down
        """
        max_offset = max(len(self.lines) - self.line_limit, 0)
        if self.line_offset >= max_offset:
            return self.output()

        new_offset = self.line_offset + self.line_limit - self.scroll_cushion
        self.line_offset = min(new_offset, max_offset)

        return self.output()

    def scroll_up(self) -> tuple[str, list[Tool]]:
        """
        Scroll the displayed content up
        """
        min_offset = 0
        if self.line_offset <= min_offset:
            return self.output()

        new_offset = self.line_offset - self.line_limit + self.scroll_cushion
        self.line_offset = max(new_offset, min_offset)

        return self.output()

    def output(self) -> tuple[str, list[Tool]]:
        if len(self.lines) <= self.line_limit:
            return self.value, []

        tools: list[Tool] = []

        out_lines: list[str] = []
        if self.line_offset > 0:
            out_lines.append(
                f"--- {self.line_offset} line(s) hidden, scroll up to view ---"
            )
            tools.append(tool(self.scroll_up))

        out_lines.extend(
            self.lines[self.line_offset : self.line_offset + self.line_limit]
        )

        end_offset = len(self.lines) - (self.line_offset + self.line_limit)
        if end_offset > 0:
            out_lines.append(
                f"--- {end_offset} line(s) hidden, scroll down to view ---"
            )
            tools.append(tool(self.scroll_down))

        return "\n".join(out_lines), tools


class RipGrepArgs(BaseModel):
    args: str


class RipGrep(Tool[RipGrepArgs, str]):
    def __init__(self, line_limit: int = 100) -> None:
        self.line_limit = line_limit

    def name(self) -> str:
        return "ripgrep"

    def description(self) -> str:
        return "Search tool"

    def args(self) -> Type[RipGrepArgs]:
        return RipGrepArgs

    async def call(
        self, args: RipGrepArgs, conversation: Conversation | None = None
    ) -> tuple[str, list[Tool]]:
        with tempfile.TemporaryFile() as ntf, open(os.devnull, "ab") as devnull:
            rg_args = ["--heading", "--line-number", *shlex.split(args.args)]

            process = await asyncio.create_subprocess_exec(
                "rg", *rg_args, stdout=ntf, stderr=ntf, stdin=devnull
            )

            await process.wait()

            ntf.flush()
            ntf.seek(0)
            result = ntf.read().decode()

            view = ScrollableString(result, self.line_limit)

            return view.output()

    def format(self, output: str) -> str:
        return output


class ReadFileArgs(BaseModel):
    path: str


class ReadFile(Tool[ReadFileArgs, str]):
    def __init__(self, line_limit: int = 100) -> None:
        self.line_limit = line_limit

    def name(self) -> str:
        return "read_file"

    def description(self) -> str:
        return "Read the contents of a specific file"

    def args(self) -> Type[ReadFileArgs]:
        return ReadFileArgs

    async def call(
        self, args: ReadFileArgs, conversation: Conversation | None = None
    ) -> tuple[str, list[Tool]]:
        with open(args.path) as f:
            content = f.read()

            view = ScrollableString(content, self.line_limit)

            return view.output()

    def format(self, output: str) -> str:
        return output


AgentOutput = TypeVar("AgentOutput", bound=BaseModel)


class AgentRespond(Tool[AgentOutput, str]):
    def __init__(self, response_schema: Type[AgentOutput]) -> None:
        self.response_schema = response_schema
        self.response: AgentOutput | None = None

    def name(self) -> str:
        return "respond"

    def description(self) -> str:
        return "Confidently provide a response to your task"

    def args(self) -> Type[AgentOutput]:
        return self.response_schema

    async def call(
        self, args: AgentOutput, conversation: Conversation | None = None
    ) -> tuple[str, list[Tool]]:
        self.response = args
        return "Your response has been recorded. You're done!", []

    def format(self, output: str) -> str:
        return output


class Agent(Generic[AgentOutput], abc.ABC):
    @abc.abstractmethod
    async def run(self, conversation: Conversation, question: str) -> AgentOutput:
        raise NotImplementedError

    @abc.abstractmethod
    def tool(
        self,
        name: str,
        description: str,
        question: Callable[[str], str] | None = None,
        conversation: Conversation | None = None,
        fork: bool = False,
    ) -> Tool:
        raise NotImplementedError


class AgentInput(BaseModel):
    question: str


class AgentTool(Tool[AgentInput, AgentOutput]):
    def __init__(
        self,
        name: str,
        description: str,
        agent: "Agent[AgentOutput]",
        fork: bool = False,
        question: Callable[[str], str] | None = None,
        conversation: Conversation | None = None,
    ) -> None:
        if question is None:

            def question(x: str) -> str:
                return x

        self._name = name
        self._description = description
        self.agent = agent
        self.fork = fork
        self.question = question
        self.conversation = conversation

    def name(self) -> str:
        return self._name

    def description(self) -> str:
        return self._description

    def args(self) -> Type[AgentInput]:
        return AgentInput

    async def call(
        self, args: AgentInput, conversation: Conversation
    ) -> tuple[AgentOutput, list[Tool]]:
        question = self.question(args.question)

        convo = self.conversation
        if convo is None:
            convo = conversation.fork() if self.fork else conversation.new()

        tools: list[Tool] = []
        if not self.fork:
            tool = self.agent.tool(
                name="request_changes",
                description=f"Request changes to the previous response from {self._name}",
                question=lambda x: f"The following changes have been requested:\n{x}",
                conversation=convo,
            )
            tools.append(tool)

        return await self.agent.run(convo, question), tools

    def format(self, output: AgentOutput) -> str:
        if isinstance(output, AgentResponse):
            return output.answer
        return json.dumps(output.model_dump(mode="json"), indent=2)


class AgentResponse(BaseModel):
    answer: str


class BasicAgent(Agent[AgentOutput]):
    def __init__(
        self,
        tools: list[Tool],
        response_schema: Type[AgentOutput] = AgentResponse,  # type: ignore
    ) -> None:
        self.response_schema = response_schema
        self.tools = tools

    async def run(self, conversation: Conversation, question: str) -> AgentOutput:
        respond = AgentRespond(self.response_schema)
        all_tools = [*self.tools, respond]

        message = Message("user", question)
        message_tools = all_tools

        while respond.response is None:
            tool, args = await conversation.ask(message, message_tools)
            response, follow_up_tools = await tool.call(args, conversation)
            string_value = tool.format(response)
            message = Message("user", string_value)
            message_tools = all_tools + follow_up_tools

        return respond.response

    def tool(
        self,
        name: str,
        description: str,
        question: Callable[[str], str] | None = None,
        conversation: Conversation | None = None,
        fork: bool = False,
    ) -> Tool:
        return AgentTool(
            name=name,
            description=description,
            agent=self,
            question=question,
            conversation=conversation,
            fork=fork,
        )
