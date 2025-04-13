import abc
import dataclasses as dc
import inspect
import json
import logging
import textwrap
from typing import (
    Any,
    Awaitable,
    Callable,
    Coroutine,
    Generic,
    Literal,
    Type,
    TypeVar,
)

from blinker import signal
from openai.types.chat import ChatCompletionMessageToolCall
from pydantic import BaseModel, create_model

LOGGER = logging.getLogger(__name__)

ToolInput = TypeVar("ToolInput", bound=BaseModel)

ToolOutput = TypeVar("ToolOutput")

agent_created = signal("agent-created")

agent_responded = signal("agent-responded")

tool_called = signal("tool-called")


class Tool(Generic[ToolInput, ToolOutput], abc.ABC):
    def enabled(self) -> bool:
        return True

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
    async def call(self, args: ToolInput) -> tuple[ToolOutput, list["Tool"]]:
        raise NotImplementedError

    @abc.abstractmethod
    def format(self, output: ToolOutput) -> str:
        raise NotImplementedError


@dc.dataclass(frozen=True)
class FunctionTool(Tool[ToolInput, ToolOutput]):
    func: Callable[[ToolInput], Awaitable[tuple[ToolOutput, list[Tool]]]]
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

    async def call(self, args: ToolInput) -> tuple[ToolOutput, list["Tool"]]:
        return await self.func(args)

    def format(self, output: ToolOutput) -> str:
        return self.tool_format(output)


def tool(
    func: Callable[..., ToolOutput | Coroutine[ToolOutput, Any, Any]],
    *,
    name: str | None = None,
    description: str | None = None,
    format: Callable[[ToolOutput], str] | None = None,
    input_schema: Type[BaseModel] | None = None,
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

    if input_schema is None:
        input_fields = {}

        sig = inspect.signature(func)

        for param_name, param_val in sig.parameters.items():
            if param_val.kind in {
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            }:
                raise ValueError("Cannot use variable args w/ tool()")
            input_fields[param_name] = param_val.annotation

        input_schema = create_model(f"{name}InputSchema", **input_fields)

    async def wrapper(input):
        result = func(**input.model_dump())
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


class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
    tool_calls: list[ChatCompletionMessageToolCall] = dc.field(default_factory=list)


class Prompt(abc.ABC):
    @abc.abstractmethod
    def system_message(self) -> str | None:
        raise NotImplementedError


@dc.dataclass(frozen=True)
class StringPrompt(Prompt):
    content: str
    role: Literal["system"] = "system"

    def system_message(self) -> str | None:
        if self.role == "system":
            return self.content
        return None


@dc.dataclass(frozen=True)
class FunctionPrompt(Prompt):
    content: Callable[[], str]
    role: Literal["system"] = "system"

    def system_message(self) -> str | None:
        if self.role == "system":
            return self.content()
        return None


@dc.dataclass(frozen=True)
class Prompts(Prompt):
    prompts: list[Prompt]
    delimiter: str = "\n\n"

    def system_message(self) -> str | None:
        messages: list[str] = []
        for prompt in self.prompts:
            system = prompt.system_message()
            if system is not None:
                messages.append(system)

        if not messages:
            return None

        return self.delimiter.join(messages)


class Oracle(abc.ABC):
    @abc.abstractmethod
    async def ask(
        self, messages: list[Message], tools: list[Tool]
    ) -> tuple[Message, Tool, Any]:
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
        return Conversation(self.oracle, list(self.messages))

    async def ask(
        self, message: Message, tools: list[Tool], prompt: Prompt
    ) -> tuple[Message, Tool, Any]:
        all_messages = self.messages + [message]

        system_message = prompt.system_message()
        if system_message is not None:
            msg = Message(role="system", content=system_message)
            all_messages.insert(0, msg)

        raw, tool, args = await self.oracle.ask(all_messages, tools)
        self.messages.append(message)
        self.messages.append(raw)

        return raw, tool, args


class AgentResponse(BaseModel):
    answer: str


AgentOutput = TypeVar("AgentOutput", bound=BaseModel, default=AgentResponse)


class AgentRespond(Tool[AgentOutput, str]):
    def __init__(
        self,
        response_schema: Type[AgentOutput],
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        if name is None:
            name = "respond"
        if description is None:
            description = "Confidently provide a response to your task"

        self.response_schema = response_schema
        self.response: AgentOutput | None = None
        self._name = name
        self._description = description

    def name(self) -> str:
        return self._name

    def description(self) -> str:
        return self._description

    def args(self) -> Type[AgentOutput]:
        return self.response_schema

    async def call(self, args: AgentOutput) -> tuple[str, list[Tool]]:
        self.response = args
        return "Your response has been recorded. You're done!", []

    def format(self, output: str) -> str:
        return output


class Plugin(abc.ABC):
    def prompt(self) -> Prompt | None:
        return None

    def tools(self) -> list[Tool]:
        return []

    @abc.abstractmethod
    def clone(self) -> "Plugin":
        raise NotImplementedError


class Agent(abc.ABC):
    conversation: Conversation

    @abc.abstractmethod
    def clone(self, fork: bool = False) -> "Agent":
        raise NotImplementedError

    @abc.abstractmethod
    async def run(
        self,
        question: str,
        *,
        response_name: str | None = None,
        response_description: str | None = None,
        response_schema: Type[AgentOutput] = AgentResponse,  # type: ignore
        tools: list[Tool] | None = None,
        plugins: list[Plugin] | None = None,
    ) -> AgentOutput:
        raise NotImplementedError


class AgentInput(BaseModel):
    question: str


class AgentTool(Tool[AgentInput, AgentOutput]):
    def __init__(
        self,
        name: str,
        description: str,
        agent: Agent,
        fork: bool = False,
        question: Callable[[str], str] | None = None,
        tools: list[Callable[[Agent], Tool]] | None = None,
    ) -> None:
        if tools is None:
            tools = []

        if question is None:

            def question(x: str) -> str:
                return x

        self._name = name
        self._description = description
        self.agent = agent
        self.fork = fork
        self.question = question
        self.tools = tools

    def name(self) -> str:
        return self._name

    def description(self) -> str:
        return self._description

    def args(self) -> Type[AgentInput]:
        return AgentInput

    async def call(self, args: AgentInput) -> tuple[AgentOutput, list[Tool]]:
        question = self.question(args.question)

        agent = self.agent.clone(fork=self.fork)

        response: AgentOutput = await agent.run(question)
        tools = [tool(agent) for tool in self.tools]

        return response, tools

    def format(self, output: AgentOutput) -> str:
        if isinstance(output, AgentResponse):
            return output.answer
        return json.dumps(output.model_dump(mode="json"), indent=2)


def agent_tool(
    agent: Agent,
    name: str,
    description: str,
    question: Callable[[str], str] | None = None,
    fork: bool = False,
) -> Tool[AgentInput, AgentOutput]:
    tools: list[Callable[[Agent], Tool]] = []
    if not fork:

        def request_changes(agent: Agent) -> Tool:
            return AgentTool(
                agent=agent,
                name="request_changes",
                description=f"Request changes to the previous response from {name}",
                question=lambda x: f"The following changes have been requested:\n{x}",
            )

        tools.append(request_changes)

    return AgentTool(
        name=name,
        description=description,
        agent=agent,
        question=question,
        fork=fork,
        tools=tools,
    )


class BasicAgent(Agent):
    def __init__(
        self,
        conversation: Conversation,
        tools: list[Tool] | None = None,
        prompt: Prompt = StringPrompt("You are a helpful assistant"),
        plugins: list[Plugin] | None = None,
    ) -> None:
        if tools is None:
            tools = []
        if plugins is None:
            plugins = []

        self.conversation = conversation
        self.tools = tools
        self.prompt = prompt
        self.plugins = plugins
        agent_created.send(self)

    def clone(self, fork: bool = False) -> Agent:
        if fork:
            conversation = self.conversation.fork()
            plugins = list(self.plugins)
        else:
            conversation = self.conversation.new()
            plugins = [p.clone() for p in self.plugins]
        return BasicAgent(conversation, self.tools, self.prompt, plugins)

    async def run(
        self,
        question: str,
        *,
        response_schema: Type[AgentOutput] = AgentResponse,  # type: ignore
        response_name: str | None = None,
        response_description: str | None = None,
        tools: list[Tool] | None = None,
        plugins: list[Plugin] | None = None,
    ) -> AgentOutput:
        if tools is None:
            tools = []

        respond = AgentRespond(
            response_schema, name=response_name, description=response_description
        )

        all_plugins = self.plugins + (plugins or [])
        plugin_tools = [tool for p in all_plugins for tool in p.tools()]
        all_tools = self.tools + plugin_tools + tools + [respond]

        prompts = [self.prompt]
        for plugin in all_plugins:
            plugin_prompt = plugin.prompt()
            if plugin_prompt is None:
                continue
            prompts.append(plugin_prompt)

        prompt = Prompts(prompts)

        message = Message(role="user", content=question)
        message_tools = all_tools

        while respond.response is None:
            candidates = [tool for tool in message_tools if tool.enabled()]
            if not candidates:
                raise NoToolsAvailable

            raw, tool, args = await self.conversation.ask(message, candidates, prompt)

            tool_name = tool.name()
            LOGGER.info(
                "Response from agent to '%s': %s(%r)", question, tool_name, args
            )
            LOGGER.debug("Raw: %s", raw.content)

            await agent_responded.send_async(self, raw=raw, tool=tool, args=args)

            response, follow_up_tools = await tool.call(args)

            await tool_called.send_async(self, tool=tool, args=args, response=response)

            string_value = tool.format(response)

            if not string_value.strip():
                string_value = repr(string_value)

            LOGGER.info("Tool response from %s: %s", tool_name, string_value)

            message = Message(role="user", content=string_value)
            message_tools = all_tools + follow_up_tools

        return respond.response


class NoToolsAvailable(Exception):
    pass
