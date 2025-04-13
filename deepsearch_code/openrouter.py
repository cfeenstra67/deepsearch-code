import os

import openai


def get_openrouter_client() -> openai.AsyncOpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key is None:
        raise ValueError("OPENROUTER_API_KEY must be set")

    return openai.AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1", api_key=api_key, timeout=300.0
    )
