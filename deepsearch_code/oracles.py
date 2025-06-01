import json
import logging
import re
from typing import (
    Any,
    cast,
)

import openai
from litellm import model_cost
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
            print(format_tools(tools))
            print()

            while tool is None:
                tool_name = input("Input tool name: ")
                if tool_name in tools_by_name:
                    tool = tools_by_name[tool_name]
                else:
                    print(f"Invalid tool: {repr(tool_name)}. Try again")

        input_schema = tool.schema()
        if input_schema is None:
            input_obj = input("body: ")
        else:
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
                        fields[field] = TypeAdapter(value.annotation).validate_python(
                            raw
                        )
                        break
                    except ValidationError:
                        print("Invalid response. Try again")

            input_obj = input_schema.model_validate(fields)

        print("==========================================")

        raw = f"Called {tool_name} with {input_obj!r}"

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


def format_tools(tools: list[Tool]) -> str:
    tool_strs = []
    for tool in tools:
        name = tool.name()
        schema = tool.schema()
        if schema is None:
            schema_str = "raw text"
        else:
            json_schema = utils.get_resolved_pydantic_schema(schema)
            schema_str = f"JSON object with schema: {json.dumps(json_schema)}"
        lines = [
            f"**{name}**:- {tool.description()}",
            f"- body format: {schema_str}",
        ]
        tool_strs.append("\n".join(lines))

    return "\n\n".join(tool_strs)


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
        tools_str = format_tools(tools)

        return f"""
<response-format-instructions>
First respond by thinking deeply about your next move, making observations, and spelling out your reasoning for your final decision. Then you must provide the tool that you'd like to call next to continue. The tools available to you are as follows:
{tools_str}

Each time you send a response, you must choose exactly one tool to call to continue. Your tool call should always be within your message and formatted as an XML tag as follows. There must be a `name` attribute being the tool name to call. The body of the `tool` tag depends on the description of the tool above; it will either be raw text or a JSON object matching the schema provided with the tool description above where applicable. Examples of valid tool calls:
<tool name="example_tool_name">
This is the response body. I'm calling example_tool_name with the response body
</tool>
This is a tool call which requires JSON body:
<tool name="example_tool_name_with_json_body">
{{"some_field": "some_value"}}
</tool>
</response-format-instructions>
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
                r"<tool\s+name=(\".+?\")>(.+?)</tool>", raw, re.I | re.S
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
            # tool_args: Any = None
            error: str | None = None
            body_text: str | None = None
            step = "getting tool name"
            try:
                tool_name = json.loads(tool_call.group(1))
                step = "getting tool body"
                body_text = tool_call.group(2).strip()
                assert body_text, (
                    f"No args object was provided in the body of the <tool> tag for {tool_name}"
                )
                # step = "decoding args"
                # tool_args = json.loads(args_txt)
            except AssertionError as err:
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
                    "The model attempted to call an invalid tool: %r (available: %s)",
                    tool_name,
                    ", ".join(map(repr, tools_by_name)),
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

            tool_schema = tool.schema()
            if tool_schema is None:
                tool_input = body_text
            else:
                try:
                    tool_args = json.loads(cast(str, body_text))
                    tool_input = TypeAdapter(tool_schema).validate_python(tool_args)
                except json.JSONDecodeError as err:
                    error = f"Error while decoding JSON body: {err}"
                except ValidationError as err:
                    error = json.dumps(err.errors())

            if error:
                LOGGER.warn(
                    "The tool call args did not match the expected schema; args: %s, error: %s",
                    body_text,
                    error,
                )
                failures += 1
                available_tools = ", ".join(list(tools_by_name))
                message_dicts.append(
                    {
                        "role": "user",
                        "content": f"""
Your the body for your {tool_name} tool is not structured correctly.
Error: {error}
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
