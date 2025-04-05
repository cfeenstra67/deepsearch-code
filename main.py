import asyncio

from pydantic import BaseModel

from deepsearch_code import core


class QuestionResponse(BaseModel):
    answer: str


async def main():
    oracle = core.ReplOracle()

    conversation = core.Conversation(oracle)

    ripgrep = core.RipGrep(10)

    agent = core.Agent(conversation, [ripgrep])

    response = await agent.run("How's it working?")

    print("RESPONSE", response.answer)


if __name__ == "__main__":
    asyncio.run(main())
