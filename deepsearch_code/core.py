import abc
import asyncio
import dataclasses as dc
import json
import shlex
import tempfile
from typing import Any, Generic, Literal, Type, TypeVar

from pydantic import BaseModel, TypeAdapter

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
    async def call(self, args: ToolInput) -> ToolOutput:
        raise NotImplementedError

    @abc.abstractmethod
    def format(self, output: ToolOutput) -> str:
        raise NotImplementedError


@dc.dataclass(frozen=True)
class Message:
    role: Literal["user", "assistant", "system"]
    content: str


class Oracle(abc.ABC):
    @abc.abstractmethod
    async def ask(self, messages: list[Message], tools: list[Tool]) -> tuple[Tool, Any]:
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
        tool, args = await self.oracle.ask(all_messages, tools)
        self.messages.append(message)
        self.messages.append(
            Message(
                role="assistant",
                content=json.dumps(
                    {"tool": tool.name(), "args": args.model_dump(mode="json")}
                ),
            )
        )

        return tool, args


class ReplOracle(Oracle):
    async def ask(self, messages: list[Message], tools: list[Tool]) -> tuple[Tool, Any]:
        for message in messages:
            print(f"Role: {message.role}")
            print(f"Content: {message.content}")
            print()

        tools_by_name = {tool.name(): tool for tool in tools}
        print("Available tools")
        for tool in tools:
            print(f"{tool.name()}: {tool.description()}")
        print()

        tool_name = input("Input tool name: ")

        tool = tools_by_name[tool_name]

        input_schema = tool.args()

        fields = {}
        for field, value in input_schema.model_fields.items():
            raw = input(f"{field}: ")
            fields[field] = TypeAdapter(value.annotation).validate_strings(raw)

        input_obj = input_schema.model_validate(fields)

        return tool, input_obj


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

    async def call(self, args: RipGrepArgs) -> str:
        with tempfile.TemporaryFile() as ntf:
            rg_args = ["--heading", "--line-number", *shlex.split(args.args)]

            process = await asyncio.create_subprocess_exec(
                "rg", *rg_args, stdout=ntf, stderr=ntf
            )

            await process.wait()

            ntf.flush()
            ntf.seek(0)
            result = ntf.read().decode()

            lines = result.split("\n")
            if len(lines) <= self.line_limit:
                return result

            truncated = lines[: self.line_limit]
            remaining = len(lines) - self.line_limit

            truncated.append(
                f"--- {remaining} line(s) truncated, run a more specific search to view more results ---"
            )

            return "\n".join(truncated)

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

    async def call(self, args: AgentOutput) -> str:
        self.response = args
        return "Your response has been recorded. You're done!"

    def format(self, output: str) -> str:
        return output


class AgentResponse(BaseModel):
    answer: str


class Agent(Generic[AgentOutput]):
    def __init__(
        self,
        conversation: Conversation,
        tools: list[Tool],
        response_schema: Type[AgentOutput] = AgentResponse,  # type: ignore
    ) -> None:
        self.response_schema = response_schema
        self.tools = tools
        self.conversation = conversation

    async def run(self, question: str) -> AgentOutput:
        respond = AgentRespond(self.response_schema)
        all_tools = [*self.tools, respond]

        message = Message("user", question)

        while respond.response is None:
            tool, args = await self.conversation.ask(message, all_tools)
            response = await tool.call(args)
            string_value = tool.format(response)
            message = Message("user", string_value)

        return respond.response


class AgentInput(BaseModel):
    question: str


class AgentTool(Tool[ToolInput, AgentOutput]):
    def __init__(
        self,
        name: str,
        description: str,
        agent: Agent[AgentOutput],
        input_schema: Type[ToolInput] = AgentInput,  # type: ignore
    ) -> None:
        self._name = name
        self._description = description
        self.agent = agent
        self.input_schema = input_schema

    def name(self) -> str:
        return self._name

    def description(self) -> str:
        return self._description

    def args(self) -> Type[ToolInput]:
        return self.input_schema

    async def call(self, args: ToolInput) -> AgentOutput:
        question = f"You are provided with the following input:\n{json.dumps(args.model_dump(mode='json'), indent=2)}"

        return await self.agent.run(question)

    def format(self, output: AgentOutput) -> str:
        return json.dumps(output.model_dump(mode="json"), indent=2)
