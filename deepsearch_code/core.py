import abc
import asyncio
import dataclasses as dc
import inspect
import json
import os
import shlex
import tempfile
import textwrap
from typing import Any, Awaitable, Callable, Generic, Literal, Type, TypeVar, cast

import openai
from litellm import model_cost
from openai.types.chat import ChatCompletionMessageToolCall, ChatCompletionToolParam
from pydantic import BaseModel, TypeAdapter, ValidationError, create_model

ToolInput = TypeVar("ToolInput", bound=BaseModel)

ToolOutput = TypeVar("ToolOutput")


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
    ) -> tuple[Tool, Any]:
        all_messages = self.messages + [message]

        system_message = prompt.system_message()
        if system_message is not None:
            msg = Message(role="system", content=system_message)
            all_messages.insert(0, msg)

        raw, tool, args = await self.oracle.ask(all_messages, tools)
        self.messages.append(message)
        self.messages.append(raw)

        return tool, args


class ReplOracle(Oracle):
    async def ask(
        self, messages: list[Message], tools: list[Tool]
    ) -> tuple[Message, Tool, Any]:
        for message in messages:
            print(f"Role: {message.role}")
            print(f"Content: {message.content}")
            print()

        tools_by_name = {tool.name(): tool for tool in tools}
        tool: Tool | None = None
        if len(tools_by_name) == 1:
            tool_name, tool = next(iter(tools_by_name.items()))
        else:
            print("Available tools")
            for available_tool in tools:
                print(f"{available_tool.name()}: {available_tool.description()}")
            print()

            while tool is None:
                tool_name = input("Input tool name: ")
                if tool_name in tools_by_name:
                    tool = tools_by_name[tool_name]
                else:
                    print(f"Invalid tool: {repr(tool_name)}. Try again")

        input_schema = tool.args()

        fields = {}
        for field, value in input_schema.model_fields.items():
            while True:
                raw = input(f"{field}: ")
                try:
                    fields[field] = TypeAdapter(value.annotation).validate_python(raw)
                    break
                except ValidationError:
                    print("Invalid response. Try again")

        input_obj = input_schema.model_validate(fields)

        print("==========================================")

        raw = f"Called {tool_name} with {json.dumps(fields)}"

        return Message(role="user", content=raw), tool, input_obj


def compute_model_cost(
    model: str, input: int, output: int, cached_input: int = 0
) -> float | None:
    cost_data = None
    if f"openrouter/{model}" in model_cost:
        cost_data = model_cost[f"openrouter/{model}"]
    elif model in model_cost:
        cost_data = model_cost[model]
    elif "/" in model:
        provider, model_name = model.split("/", 1)
        if model_name in model_cost and model_cost[model_name][
            "litellm_provider"
        ].startswith(provider):
            cost_data = model_cost[model_name]

    if cost_data is None:
        return None

    if cost_data.get("cache_read_input_token_cost") is not None:
        non_cached_prompt = input - cached_input
        prompt_cost = cost_data["input_cost_per_token"] * non_cached_prompt
        prompt_cost += cost_data["cache_read_input_token_cost"] * cached_input
    else:
        prompt_cost = cost_data["input_cost_per_token"] * input

    completion_cost = cost_data["output_cost_per_token"] * output

    return prompt_cost + completion_cost


def compute_cost(usage: dict[str, dict[str, Any]]) -> float | None:
    items = []
    for model, tokens in usage.items():
        result = compute_model_cost(
            model, tokens["input"], tokens["output"], tokens["cached_input"]
        )
        if result is not None:
            items.append(result)

    if not items:
        return None

    return sum(items)


class UsageTracker:
    def __init__(self) -> None:
        self.models: dict[str, dict[str, Any]] = {}

    def log(
        self, model: str, total: int, input: int, output: int, cached_input: int = 0
    ) -> None:
        self.models.setdefault(
            model, {"count": 0, "total": 0, "input": 0, "output": 0, "cached_input": 0}
        )
        self.models[model]["count"] += 1
        self.models[model]["total"] += total
        self.models[model]["input"] += input
        self.models[model]["output"] += output
        self.models[model]["cached_input"] += cached_input

    def cost(self) -> float | None:
        return compute_cost(self.models)


class LLMToolUseOracle(Oracle):
    def __init__(
        self,
        model: str,
        client: openai.AsyncOpenAI,
        tracker: UsageTracker | None = None,
        **kwargs,
    ) -> None:
        if tracker is None:
            tracker = UsageTracker()
        self.model = model
        self.client = client
        self.tracker = tracker
        self.kwargs = kwargs

    async def ask(
        self, messages: list[Message], tools: list[Tool]
    ) -> tuple[Message, Tool, Any]:
        model_tools: list[ChatCompletionToolParam] = []
        tools_by_name: dict[str, Tool] = {}
        for tool in tools:
            name = tool.name()
            tools_by_name[name] = tool
            model_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": tool.description(),
                        "parameters": tool.args().model_json_schema(),
                    },
                }
            )

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=cast(
                Any, [{"role": msg.role, "content": msg.content} for msg in messages]
            ),
            tools=model_tools,
            tool_choice="required",
            **cast(Any, self.kwargs),
        )

        if response.usage is not None:
            cached_input = 0
            if response.usage.prompt_tokens_details:
                cached_input = response.usage.prompt_tokens_details.cached_tokens or 0

            self.tracker.log(
                self.model,
                input=response.usage.prompt_tokens,
                output=response.usage.completion_tokens,
                total=response.usage.total_tokens,
                cached_input=cached_input,
            )

        raw = response.choices[0].message.content
        tool_call = response.choices[0].message.tool_calls[0]

        tool_name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)

        tool_obj = tools_by_name[tool_name]

        input_obj = tool_obj.args().model_validate(args)

        msg = Message(
            role="assistant",
            content=raw,
            tool_calls=response.choices[0].message.tool_calls,
        )

        return msg, tool_obj, input_obj


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

    async def call(self, args: RipGrepArgs) -> tuple[str, list[Tool]]:
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

    async def call(self, args: ReadFileArgs) -> tuple[str, list[Tool]]:
        with open(args.path) as f:
            content = f.read()

            view = ScrollableString(content, self.line_limit)

            return view.output()

    def format(self, output: str) -> str:
        return output


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


class Agent(abc.ABC):
    conversation: Conversation

    @abc.abstractmethod
    def clone(self, conversation: Conversation) -> "Agent":
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

        conversation = self.agent.conversation
        conversation = conversation.fork() if self.fork else conversation.new()
        agent = self.agent.clone(conversation)

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
        response_schema: Type[AgentOutput] = AgentResponse,  # type: ignore
    ) -> None:
        if tools is None:
            tools = []

        self.conversation = conversation
        self.tools = tools
        self.prompt = prompt
        self.response_schema = response_schema

    def clone(self, conversation: Conversation) -> Agent:
        return BasicAgent(conversation, self.tools, self.prompt, self.response_schema)

    async def run(
        self,
        question: str,
        *,
        response_schema: Type[AgentOutput] = AgentResponse,  # type: ignore
        response_name: str | None = None,
        response_description: str | None = None,
        tools: list[Tool] | None = None,
    ) -> AgentOutput:
        if tools is None:
            tools = []

        respond = AgentRespond(
            response_schema, name=response_name, description=response_description
        )
        all_tools = self.tools + tools + [respond]

        message = Message(role="user", content=question)
        message_tools = all_tools

        while respond.response is None:
            candidates = [tool for tool in message_tools if tool.enabled()]
            if not candidates:
                raise NoToolsAvailable

            tool, args = await self.conversation.ask(message, candidates, self.prompt)
            response, follow_up_tools = await tool.call(args)
            string_value = tool.format(response)
            message = Message(role="user", content=string_value)
            message_tools = all_tools + follow_up_tools

        return respond.response


class NoToolsAvailable(Exception):
    pass
