import asyncio

from pydantic import BaseModel

from deepsearch_code import core


class QuestionResponse(BaseModel):
    answer: str


async def main():
    oracle = core.ReplOracle()

    conversation = core.Conversation(oracle)

    ripgrep = core.RipGrep(10)
    read_file = core.ReadFile(10)

    agent = core.BasicAgent([ripgrep, read_file])

    tool = agent.tool(
        "best_agent", "blah", lambda x: f"Message from your supervisor:\n{x}"
    )

    agent2 = core.BasicAgent([tool])

    response = await agent2.run(conversation, "How's it working?")

    print("RESPONSE", response.answer)


if __name__ == "__main__":
    asyncio.run(main())
