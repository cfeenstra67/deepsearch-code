import asyncio
import itertools
import json
import os

import openai
from dotenv import load_dotenv
from pydantic import BaseModel

from deepsearch_code import core


class TicTacToeMove(BaseModel):
    row: int
    col: int


class TicTacToe:
    def __init__(self, size: int = 3) -> None:
        self.board = [[" " for _ in range(size)] for _ in range(size)]
        self.winner: str | None = None
        self.size = size

    def make_move(self, char: str, row: int, col: int) -> None:
        self.board[row][col] = char

        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for x, y in directions:
            r, c = row, col
            count = 0

            while (
                0 <= r < self.size and 0 <= c < self.size and self.board[r][c] == char
            ):
                count += 1
                r += x
                c += y

            r, c = row - x, col - y
            while (
                0 <= r < self.size and 0 <= c < self.size and self.board[r][c] == char
            ):
                count += 1
                r -= x
                c -= y

            if count == self.size:
                self.winner = char
                break

    def board_is_full(self) -> bool:
        return all(all(x != " " for x in row) for row in self.board)

    def print_board(self) -> str:
        out = []
        for idx, row in enumerate(self.board):
            out.append(f"{idx} {'|'.join(row)}")

        row_length = len(out[0]) - 2
        separator = "  " + "-" * row_length

        final = f"\n{separator}\n".join(out)
        footer = " ".join(map(str, range(self.size)))
        return f"{final}\n  {footer}"

    async def run(
        self,
        player1: core.Agent,
        player2: core.Agent,
        max_failures: int = 3,
    ) -> int | None:
        markers = ["X", "O"]
        agents = list(zip(markers, [player1, player2]))

        turns = itertools.cycle(agents)

        run_kwargs = {
            "response_name": "play",
            "response_description": "Make your next move",
        }

        while self.winner is None and not self.board_is_full():
            char, agent = next(turns)

            def forfeit():
                """
                Forfeit the game and be forever shamed
                """
                raise Forfeit

            response = await agent.run(
                f"You are playing '{char}'\n"
                f"Current board:\n{self.print_board()}\n\n"
                "What's your next move?",
                response_schema=TicTacToeMove,
                tools=[core.tool(forfeit)],
                **run_kwargs,
            )

            failures = 0
            while True:
                validation_errors: list[str] = []
                if not (
                    0 <= response.row < self.size and 0 <= response.col < self.size
                ):
                    validation_errors.append("cell does not exist")
                elif self.board[response.row][response.col] != " ":
                    validation_errors.append("that cell is already taken")

                if not validation_errors:
                    self.make_move(char, response.row, response.col)
                    break

                failures += 1
                if failures >= max_failures:
                    raise Forfeit

                remaining_attempts = max_failures - failures
                response = await agent.run(
                    f"Invalid move: {', '.join(validation_errors)}. "
                    f"Please try again ({remaining_attempts} attempt(s) remaining)",
                    response_schema=TicTacToeMove,
                    tools=[core.tool(forfeit)],
                    **run_kwargs,
                )

        if self.winner is not None:
            return markers.index(self.winner) + 1
        return None


class Forfeit(Exception):
    pass


async def main():
    load_dotenv()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key is None:
        raise ValueError("OPENROUTER_API_KEY must be set")

    client = openai.AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1", api_key=api_key
    )

    player = core.BasicAgent(core.Conversation(core.ReplOracle()))

    llm = core.LLMToolUseOracle(model="openai/gpt-4o-mini", client=client)
    llm_convo = core.Conversation(llm)

    ai = core.BasicAgent(
        llm_convo,
        prompt=core.StringPrompt(
            "You are playing a game of tic tac toe, and your life is on the line!!! First one to get three in a row up, down or diagonally wins. May the odds be even in your favor!"
        ),
    )

    game = TicTacToe()

    try:
        winner = await game.run(player, ai)
    finally:
        convo = [msg.model_dump(mode="json") for msg in llm_convo.messages]
        with open("conversation.json", "w+") as f:
            json.dump(convo, f, indent=2)

    print("Final:")
    print(game.print_board())

    if winner is None:
        print("It's a tie!")
    else:
        print("The winner is player", winner)


if __name__ == "__main__":
    asyncio.run(main())
