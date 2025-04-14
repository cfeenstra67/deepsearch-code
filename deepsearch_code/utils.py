import asyncio
import os
from functools import wraps
from logging.config import dictConfig
from typing import Any, Callable, Type

import click
from pydantic import BaseModel


def resolve_pydantic_schema(
    schema: dict[str, Any], definitions: dict[str, Any]
) -> dict[str, Any]:
    if "$ref" in schema:
        name = schema["$ref"].rsplit("/", 1)[-1]
        return resolve_pydantic_schema(definitions[name], definitions)

    for key in ["anyOf", "allOf", "oneOf"]:
        if key in schema:
            kws = {
                key: [
                    resolve_pydantic_schema(subschema, definitions)
                    for subschema in schema[key]
                ]
            }
            return dict(schema, **kws)

    if "type" not in schema and "enum" in schema:
        schema = dict(schema, type="string")

    if "type" not in schema:
        schema = dict(schema, type="object")

    if schema["type"] == "array":
        return dict(schema, items=resolve_pydantic_schema(schema["items"], definitions))

    if schema["type"] == "object":
        properties = schema.get("properties", {})
        out = {}
        for key, value in properties.items():
            out[key] = resolve_pydantic_schema(value, definitions)

        return dict(schema, properties=out)

    return schema


def trim_pydantic_schema(schema: dict[str, Any]) -> dict[str, Any]:
    schema_copy = schema.copy()
    schema_copy.pop("title", None)

    if schema_copy.get("enum") and schema_copy.get("description") == "An enumeration.":
        schema_copy.pop("description", None)

    if schema_copy.get("type") == "object":
        properties = schema_copy.get("properties", {})
        out = {}
        for key, value in properties.items():
            out[key] = trim_pydantic_schema(value)
        schema_copy["properties"] = out
        return schema_copy

    if schema_copy.get("type") == "array":
        schema_copy["items"] = trim_pydantic_schema(schema_copy["items"])
        return schema_copy

    if schema_copy.get("anyOf") and len(schema_copy["anyOf"]) == 2:
        null_items = [item for item in schema_copy["anyOf"] if item == {"type": "null"}]
        other_items = [
            item for item in schema_copy["anyOf"] if item != {"type": "null"}
        ]
        if len(null_items) == 1:
            return trim_pydantic_schema(dict(other_items[0], nullable=True))

    return schema_copy


def get_resolved_pydantic_schema(model: Type[BaseModel]) -> dict[str, Any]:
    schema = model.model_json_schema(mode="serialization").copy()
    definitions = schema.pop("$defs", {})
    resolved = resolve_pydantic_schema(schema, definitions)
    return trim_pydantic_schema(resolved)


def async_command(
    group: click.Group, **kws
) -> Callable[[Callable[..., Any]], click.Command]:
    def dec(f):
        @group.command(**kws)
        @wraps(f)
        def wrapper(*args, **kwargs):
            return asyncio.run(f(*args, **kwargs))

        return wrapper

    return dec


def setup_logging(level: str | None = None) -> None:
    if level is None:
        level = os.getenv("LOG_LEVEL", "ERROR")
    dictConfig(
        {
            "version": 1,
            "formatters": {
                "main": {"format": "%(asctime)s - %(name)s [%(levelname)s] %(message)s"}
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "main",
                },
            },
            "loggers": {
                "deepsearch_code": {
                    "level": level.upper(),
                    "handlers": ["console"],
                    "propagate": False,
                },
                "__main__": {
                    "level": level.upper(),
                    "handlers": ["console"],
                    "propagate": False,
                },
            },
        }
    )
