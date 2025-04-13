import asyncio

from deepsearch_code import core


async def main():
    oracle = core.ReplOracle()

    conversation = core.Conversation(oracle)

    ripgrep = core.RipGrep(10)
    read_file = core.ReadFile(10)

    agent = core.BasicAgent(
        core.StringPrompt("You are an asshole"), [ripgrep, read_file]
    )

    tool = agent.tool(
        "best_agent", "blah", lambda x: f"Message from your supervisor:\n{x}"
    )

    agent2 = core.BasicAgent(
        core.StringPrompt("You are supervising an asshole"), [tool]
    )

    response = await agent2.run(conversation, "How's it working?")

    print("RESPONSE", response.answer)


if __name__ == "__main__":
    asyncio.run(main())
