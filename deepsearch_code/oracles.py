import json
import logging
import re
from typing import (
    Any,
    cast,
)

import openai
from litellm import model_cost
from openai.lib._pydantic import to_strict_json_schema
from openai.types.chat import ChatCompletionToolParam
from pydantic import TypeAdapter, ValidationError

from deepsearch_code import utils
from deepsearch_code.core import Message, Oracle, Tool

LOGGER = logging.getLogger(__name__)


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
                raw: Any = input(f"{field}: ")
                if raw == "":
                    raw = None
                else:
                    try:
                        raw = json.loads(raw)
                    except json.JSONDecodeError:
                        pass

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
        self.models: dict[str, dict[str, int]] = {}
        self.default_usage = {
            "count": 0,
            "total": 0,
            "input": 0,
            "output": 0,
            "cached_input": 0,
        }

    def log(
        self, model: str, total: int, input: int, output: int, cached_input: int = 0
    ) -> None:
        self.models.setdefault(model, self.default_usage.copy())
        self.models[model]["count"] += 1
        self.models[model]["total"] += total
        self.models[model]["input"] += input
        self.models[model]["output"] += output
        self.models[model]["cached_input"] += cached_input

    def get(self, model: str) -> dict[str, int]:
        return self.models.get(model, self.default_usage.copy())

    def total(self) -> dict[str, int]:
        usage = self.default_usage.copy()
        for _, counts in self.models.items():
            for key, val in counts.items():
                usage[key] += val
        return usage

    def cost(self) -> float | None:
        return compute_cost(self.models)


class LLMToolUseOracle(Oracle):
    def __init__(
        self,
        model: str,
        client: openai.AsyncOpenAI,
        tracker: UsageTracker | None = None,
        allowed_attempts: int = 3,
        **kwargs,
    ) -> None:
        if tracker is None:
            tracker = UsageTracker()
        self.model = model
        self.client = client
        self.tracker = tracker
        self.allowed_attempts = allowed_attempts
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

        failures = 0
        use_messages = [{"role": msg.role, "content": msg.content} for msg in messages]
        while failures < self.allowed_attempts:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=cast(Any, use_messages),
                tools=model_tools,
                tool_choice="required",
                **cast(Any, self.kwargs),
            )

            if response.usage is not None:
                cached_input = 0
                if response.usage.prompt_tokens_details:
                    cached_input = (
                        response.usage.prompt_tokens_details.cached_tokens or 0
                    )

                self.tracker.log(
                    self.model,
                    input=response.usage.prompt_tokens,
                    output=response.usage.completion_tokens,
                    total=response.usage.total_tokens,
                    cached_input=cached_input,
                )

            raw = response.choices[0].message.content
            if not response.choices[0].message.tool_calls:
                print(
                    f"No tool call in response from {self.model}; model responded with: {raw}"
                )
                failures += 1
                remaining = self.allowed_attempts - failures
                use_messages.append(
                    {
                        "role": "user",
                        "content": f"Invalid response; your respond must contain a tool call. {remaining} attempt(s) remaining",
                    }
                )
                continue

            tool_call = response.choices[0].message.tool_calls[0]
            break

        if failures >= self.allowed_attempts:
            raise ValueError(
                f"Model {self.model} was unable to give a response after {self.allowed_attempts} attempt(s)"
            )

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


class LLMStructuredOutputsOracle(Oracle):
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
        tools_by_name: dict[str, Tool] = {}

        schemas = []
        for tool in tools:
            tools_by_name[tool.name()] = tool
            schemas.append(
                {
                    "type": "object",
                    "description": tool.description(),
                    "properties": {
                        "name": {"type": "string", "const": tool.name()},
                        "args": to_strict_json_schema(tool.args()),
                    },
                    "required": ["name", "args"],
                    "additionalProperties": False,
                }
            )

        response_schema = {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "The reasoning for your response",
                },
                "tool": {"anyOf": schemas},
            },
            "required": ["reasoning", "tool"],
            "additionalProperties": False,
        }

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=cast(
                Any, [{"role": msg.role, "content": msg.content} for msg in messages]
            ),
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "ToolCall",
                    "strict": True,
                    "schema": response_schema,
                },
            },
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

        if not response.choices:
            raise RuntimeError(f"No choices in response: {response!r}")

        raw = response.choices[0].message.content
        try:
            response_data = json.loads(raw)["tool"]
        except json.JSONDecodeError as err:
            raise RuntimeError(
                f"Invalid JSON in structured output response: {raw}"
            ) from err

        tool_name = response_data["name"]
        args = response_data["args"]

        tool_obj = tools_by_name[tool_name]

        input_obj = tool_obj.args().model_validate(args)

        msg = Message(
            role="assistant",
            content=raw,
        )

        return msg, tool_obj, input_obj


class LLMOracle(Oracle):
    def __init__(
        self,
        model: str,
        client: openai.AsyncOpenAI,
        allowed_attempts: int = 3,
        tracker: UsageTracker | None = None,
        **kwargs,
    ) -> None:
        if tracker is None:
            tracker = UsageTracker()
        self.model = model
        self.client = client
        self.tracker = tracker
        self.allowed_attempts = allowed_attempts
        self.kwargs = kwargs

    def get_system_prompt(self, tools: list[Tool]) -> str:
        tool_strs = []
        for tool in tools:
            name = tool.name()
            schema = utils.get_resolved_pydantic_schema(tool.args())
            lines = [
                f"**{name}**:- {tool.description()}",
                f"- args schema: {json.dumps(schema)}",
            ]
            tool_strs.append("\n".join(lines))

        tools_str = "\n\n".join(tool_strs)

        return f"""
<response-instructions>
First respond by thinking deeply about your next move, making observations, and spelling out your reasoning for your final decision. Then you must provide the tool that you'd like to call next to continue. The tools available to you are as follows:
{tools_str}

Each time you send a response, you must choose exactly one tool to call to continue. Your tool call should always be within your message and formatted as an XML tag as follows. The body of the `tool` tag MUST be provided a JSON object, and it must match the JSON schema provided in the corresponding tool's description above:
<tool name="example_tool_name">
{{"some_field": "some_value"}}
</tool>
</response-instructions>
""".strip()

    async def ask(
        self, messages: list[Message], tools: list[Tool]
    ) -> tuple[Message, Tool, Any]:
        tools_by_name: dict[str, Tool] = {tool.name(): tool for tool in tools}

        tools_prompt = self.get_system_prompt(tools)

        message_dicts = [{"role": msg.role, "content": msg.content} for msg in messages]

        message_dicts[0]["content"] += f"\n{tools_prompt}"

        failures = 0
        last_response = None
        while failures < self.allowed_attempts:
            if failures:
                LOGGER.info(
                    "Attempting to get a response, attempt #%d; last response: %s",
                    failures + 1,
                    last_response,
                )

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=cast(Any, message_dicts),
                **cast(Any, self.kwargs),
            )

            if response.usage is not None:
                cached_input = 0
                if response.usage.prompt_tokens_details:
                    cached_input = (
                        response.usage.prompt_tokens_details.cached_tokens or 0
                    )

                self.tracker.log(
                    self.model,
                    input=response.usage.prompt_tokens,
                    output=response.usage.completion_tokens,
                    total=response.usage.total_tokens,
                    cached_input=cached_input,
                )

            if not response.choices:
                raise RuntimeError(f"No choices in response: {response!r}")

            message_dicts.append(response.choices[0].message)
            raw = response.choices[0].message.content
            last_response = raw

            tool_call = re.search(
                r"<tool\s+name=(\".+\")>(.+?)</tool>", raw, re.I | re.S
            )
            if not tool_call:
                LOGGER.warn("No tool call in response: %s", raw)
                failures += 1
                message_dicts.append(
                    {
                        "role": "user",
                        "content": f"""
You didn't provide a valid tool call in your response. Remember, your response must include exactly one tool call formatted as an XML tag as follows:
<tool name="example_tool_name">
{{"some_field": "some_value"}}
</tool>
You have {self.allowed_attempts - failures} attempt(s) remaining.
""".strip(),
                    }
                )
                continue

            tool_name: str | None = None
            tool_args: Any = None
            error: str | None = None
            step = "getting name"
            try:
                tool_name = json.loads(tool_call.group(1))
                step = "getting args"
                args_txt = tool_call.group(2).strip()
                assert args_txt, (
                    f"No args object was provided in the body of the <tool> tag for {tool_name}"
                )
                step = "decoding args"
                tool_args = json.loads(args_txt)
            except (AssertionError, json.JSONDecodeError) as err:
                error = f"Error parsing your tool call while {step}: {str(err)}"

            if error:
                LOGGER.warn(
                    "Error extracting tool call from %s: %s", tool_call.group(), error
                )
                failures += 1
                message_dicts.append(
                    {
                        "role": "user",
                        "content": f"""
Your tool call was not valid.
Error: {error}
Please try again.
You have {self.allowed_attempts - failures} attempt(s) remaining.
""".strip(),
                    }
                )
                continue

            if tool_name not in tools_by_name:
                LOGGER.warn(
                    "The model attempted to call an invalid tool: %s (available: %s)",
                    tool_name,
                    ", ".join(tools_by_name),
                )
                failures += 1
                available_tools = ", ".join(list(tools_by_name))
                message_dicts.append(
                    {
                        "role": "user",
                        "content": f"""
You've attempted to call a tool that does not exist: {tool_name}
As a reminder, the following tools are avilable: {available_tools}
Please try again.
You have {self.allowed_attempts - failures} attempt(s) remaining.
""".strip(),
                    }
                )
                continue

            tool = tools_by_name[tool_name]

            tool_input: Any = None
            try:
                tool_input = TypeAdapter(tool.args()).validate_python(tool_args)
            except ValidationError as err:
                error = json.dumps(err.errors())

            if error:
                LOGGER.warn(
                    "The tool call args did not match the expected schema; args: %s, error: %s",
                    tool_args,
                    error,
                )
                failures += 1
                available_tools = ", ".join(list(tools_by_name))
                message_dicts.append(
                    {
                        "role": "user",
                        "content": f"""
You've attempted to call a tool that does not exist: {tool_name}
As a reminder, the following tools are avilable: {available_tools}
Please try again.
You have {self.allowed_attempts - failures} attempt(s) remaining.
""".strip(),
                    }
                )
                continue

            LOGGER.info(
                "Extracted tool call %s(%s) from message after %d attempt(s)",
                tool_name,
                tool_input,
                failures + 1,
            )

            return Message(role="assistant", content=raw), tool, tool_input

        last_error = message_dicts[-1]["content"]
        raise RuntimeError(
            f"{self.model} was unable to produce a valid tool call after "
            f"{self.allowed_attempts} attempt(s). Last response: {last_response}\nLast error: {last_error}"
        )
