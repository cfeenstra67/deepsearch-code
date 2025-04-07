#!/usr/bin/env -S uv run python
import argparse
import asyncio
import contextlib
import dataclasses as dc
import json
import os
import statistics
import time
import traceback
from typing import Any, Literal, cast

import openai
from dotenv import load_dotenv
from pydantic import BaseModel
from tqdm.auto import tqdm

from deepsearch_code import core


class TicTacToeMove(BaseModel):
    row: int
    col: int


class TicTacToe:
    def __init__(self, size: int = 3) -> None:
        self.board = [[" " for _ in range(size)] for _ in range(size)]
        self.winner: int | None = None
        self.size = size
        self.markers = ["X", "O"]
        self.turn = 1
        self.num_turns = 0

    def is_empty(self, row: int, col: int) -> bool:
        return self.board[row][col] == " "

    def get_space(self, row: int, col: int) -> int | None:
        if self.is_empty(row, col):
            return None
        return self.markers.index(self.board[row][col]) + 1

    def check_winner(self) -> int | None:
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        filled = 0
        for row in range(self.size):
            for col in range(self.size):
                for x, y in directions:
                    player = self.get_space(row, col)
                    if player is None:
                        continue

                    filled += 1

                    r, c = row, col
                    count = 0

                    while (
                        0 <= r < self.size
                        and 0 <= c < self.size
                        and self.get_space(r, c) == player
                    ):
                        count += 1
                        r += x
                        c += y

                    r, c = row - x, col - y
                    while (
                        0 <= r < self.size
                        and 0 <= c < self.size
                        and self.get_space(r, c) == player
                    ):
                        r -= x
                        c -= y

                    if count == self.size:
                        self.winner = player
                        break

        return self.winner

    @contextlib.contextmanager
    def test(self, player: int, row: int, col: int):
        prev = self.board[row][col]
        prev_winner = self.winner

        self.board[row][col] = self.player_marker(player)
        try:
            yield
        finally:
            self.board[row][col] = prev
            self.winner = prev_winner

    def other_player(self, player: int) -> int:
        return (player % 2) + 1

    def take_turn(self, row: int, col: int) -> None:
        self.board[row][col] = self.player_marker(self.turn)
        self.check_winner()
        self.turn = self.other_player(self.turn)
        self.num_turns += 1

    def is_full(self) -> bool:
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

    def player_marker(self, player: int) -> str:
        return self.markers[player - 1]

    async def run(
        self,
        player1: core.Agent,
        player2: core.Agent,
        max_failures: int = 3,
    ) -> int | None:
        agents = [player1, player2]

        run_kwargs = {
            "response_name": "play",
            "response_description": "Make your next move",
        }

        while self.winner is None and not self.is_full():
            player_num = self.turn
            agent = agents[player_num - 1]
            char = self.player_marker(player_num)

            def forfeit(reason: str):
                """
                Forfeit the game and be forever shamed
                """
                raise Forfeit(player_num, reason)

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
                    validation_errors.append(
                        f"space w/ row {response.row}, col {response.col} does not exist"
                    )
                elif not self.is_empty(response.row, response.col):
                    taken_by = self.get_space(response.row, response.col)
                    validation_errors.append(
                        f"that space is already taken by {taken_by}"
                    )

                if not validation_errors:
                    self.take_turn(response.row, response.col)
                    break

                failures += 1
                if failures >= max_failures:
                    raise Forfeit(
                        player_num,
                        f"Did not provide a response with {max_failures} attempt(s)",
                    )

                remaining_attempts = max_failures - failures
                response = await agent.run(
                    f"Invalid move: {', '.join(validation_errors)}. "
                    f"Please try again ({remaining_attempts} attempt(s) remaining). "
                    f"If you do not provide a valid response you ",
                    response_schema=TicTacToeMove,
                    tools=[core.tool(forfeit)],
                    **run_kwargs,
                )

        return self.winner


class TicTacToeAIOracle(core.Oracle):
    def __init__(self, game: TicTacToe, player: int) -> None:
        self.game = game
        self.player = player

    def minimax(self, depth: int, maximizing: bool) -> float:
        winner = self.game.check_winner()
        if winner is not None or self.game.is_full():
            if winner is None:
                return 0
            if winner == self.player:
                return 1
            return -1

        best_score = -float("inf") if maximizing else float("inf")
        for row in range(self.game.size):
            for col in range(self.game.size):
                if not self.game.is_empty(row, col):
                    continue
                dest = (
                    self.player if maximizing else self.game.other_player(self.player)
                )
                with self.game.test(dest, row, col):
                    score = self.minimax(depth + 1, not maximizing)
                best_score = (
                    max(best_score, score) if maximizing else min(best_score, score)
                )

        return best_score

    def best_next_move(self) -> tuple[int, int]:
        best_score = -float("inf")
        best_move = None
        for row in range(self.game.size):
            for col in range(self.game.size):
                if not self.game.is_empty(row, col):
                    continue
                with self.game.test(self.player, row, col):
                    score = self.minimax(0, False)

                if score > best_score:
                    best_score = score
                    best_move = row, col

        if best_move is None:
            raise RuntimeError("No move found. This is unexpected")

        return best_move

    async def ask(
        self, messages: list[core.Message], tools: list[core.Tool]
    ) -> tuple[core.Message, core.Tool, Any]:
        play = [t for t in tools if t.name() == "play"][0]

        row, col = self.best_next_move()
        response = TicTacToeMove(row=row, col=col)

        return core.Message(role="user", content=""), play, response


class Forfeit(Exception):
    def __init__(self, player: int, reason: str) -> None:
        self.player = player
        self.reason = reason
        super().__init__(f"Player {player} forfeited because: {reason}")


GameResultType = Literal["win", "loss", "tie", "forfeit", "error"]


@dc.dataclass(frozen=True)
class GameResult:
    type: GameResultType
    model: str
    game: TicTacToe
    cost: float | None = None
    forfeit_reason: str | None = None


async def play_game(
    game: TicTacToe,
    model: str,
    client: openai.AsyncOpenAI,
    semaphore: asyncio.Semaphore | None = None,
) -> GameResult:
    simple_ai_oracle = TicTacToeAIOracle(game, 2)
    simple_ai = core.BasicAgent(core.Conversation(simple_ai_oracle))

    tracker = core.UsageTracker()
    llm = core.LLMToolUseOracle(model=model, client=client, tracker=tracker)
    llm_convo = core.Conversation(llm)

    ai = core.BasicAgent(
        llm_convo,
        prompt=core.StringPrompt(
            f"""
You are playing a competitive game of tic-tac-toe on a {game.size}x{game.size} board. If you win this game, it'll win you the world championship and cement you as the great of all time.

You know the rules, of course. Each turn you'll select a move and you'll place X or O at the corresponding spot on the board. First one to get {game.size} in a row up, down, or diagonally wins. Good luck!
""".strip()
        ),
    )

    # # Random first move
    # row, col = random.randint(0, game.size - 1), random.randint(0, game.size - 1)
    # game.take_turn(row, col)

    result: GameResultType
    forfeit_reason: str | None = None
    try:
        async with contextlib.AsyncExitStack() as stack:
            if semaphore is not None:
                await stack.enter_async_context(semaphore)
            winner = await game.run(ai, simple_ai)
        if winner is None:
            result = "tie"
        elif winner == 1:
            result = "win"
        else:
            result = "loss"

    except Forfeit as err:
        if err.player == 2:
            raise RuntimeError("Simple AI code forfeited. This is unexpected") from err

        result = "forfeit"
        forfeit_reason = err.reason

    return GameResult(
        type=result,
        model=model,
        game=game,
        cost=tracker.cost(),
        forfeit_reason=forfeit_reason,
    )


async def play_game_interactive(
    model: str,
    size: int,
    client: openai.AsyncOpenAI,
):
    game = TicTacToe(size)
    result = await play_game(game, model, client)

    print(f"Final (${result.cost:.3f}, {result.game.num_turns} turns):")
    print(result.game.print_board())

    if result.type == "forfeit":
        print("The LLM forfeited because:", result.forfeit_reason)
    elif result.type == "win":
        print("The LLM won!")
    elif result.type == "loss":
        print("The LLM lost!")
    else:
        print("It's a tie!")


class ModelMetrics(BaseModel):
    model: str
    results: dict[GameResultType, int]
    cost: float
    avg_time: float
    avg_turns: float
    forfeit_reasons: list[str]


def compute_metrics(games: list[tuple[GameResult, float]]) -> list[ModelMetrics]:
    games_by_model: dict[str, list[tuple[GameResult, float]]] = {}
    for game, game_time in games:
        games_by_model.setdefault(game.model, []).append((game, game_time))

    out: list[ModelMetrics] = []
    for model, model_games in games_by_model.items():
        count_by_status: dict[GameResultType, int] = {}
        forfeit_reasons: list[str] = []
        for game, _ in model_games:
            count_by_status.setdefault(game.type, 0)
            count_by_status[game.type] += 1
            if game.forfeit_reason:
                forfeit_reasons.append(game.forfeit_reason)

        cost = sum(game.cost for game, _ in model_games if game.cost is not None)

        avg_time = statistics.mean(t for _, t in model_games)

        avg_turns = statistics.mean(game.game.num_turns for game, _ in model_games)

        out.append(
            ModelMetrics(
                model=model,
                results=count_by_status,
                cost=cost,
                avg_time=avg_time,
                avg_turns=avg_turns,
                forfeit_reasons=forfeit_reasons,
            )
        )

    return out


def print_metrics(metrics: list[ModelMetrics]) -> None:
    for model_metrics in metrics:
        statuses = ", ".join(
            f"{status}: {model_metrics.results[cast(GameResultType, status)]}"
            for status in ["win", "loss", "tie", "forfeit", "error"]
            if status in model_metrics.results
        )

        print(
            f"{model_metrics.model}: {statuses} (${model_metrics.cost:.3f}, "
            f"avg {model_metrics.avg_time:.2f}s {model_metrics.avg_turns:.1f} turns)"
        )
        if model_metrics.forfeit_reasons:
            unique_reasons = set(model_metrics.forfeit_reasons)
            print(f"forfitted because: {'; '.join(unique_reasons)}")


async def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser("tictactoe.py")

    parser.add_argument("model", help="Model to play against", nargs="+")
    parser.add_argument("-n", "--num", type=int, default=10)
    parser.add_argument("-c", "--concurrency", type=int, default=None)
    parser.add_argument("-o", "--output", default=None, help="File to write output to")

    args = parser.parse_args()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key is None:
        raise ValueError("OPENROUTER_API_KEY must be set")

    client = openai.AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1", api_key=api_key
    )

    tasks: dict[asyncio.Task[GameResult], tuple[TicTacToe, str, int, float]] = {}
    size = 3
    semaphore: asyncio.Semaphore | None = None
    if args.concurrency is not None:
        semaphore = asyncio.Semaphore(args.concurrency)
    for model in args.model:
        for i in range(args.num):
            game = TicTacToe(size)
            task = asyncio.create_task(play_game(game, model, client, semaphore))
            tasks[task] = game, model, i, time.time()

    progress = tqdm(total=len(tasks))

    pending = set(tasks)
    games: list[tuple[GameResult, float]] = []
    while pending:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            progress.update(1)
            game, model, idx, start = tasks[task]
            try:
                games.append((task.result(), time.time() - start))
            except Exception:
                print(f"Error w/ model {model}, game {idx}:")
                traceback.print_exc()
                result = GameResult(type="error", model=model, game=game)
                games.append((result, time.time() - start))

    metrics = compute_metrics(games)
    print_metrics(metrics)

    total_cost = sum(game.cost for game, _ in games if game.cost is not None)
    print()
    print(f"Total cost: ${total_cost:.3f}")

    if args.output:
        data = {
            "total_cost": total_cost,
            "models": [m.model_dump(mode="json") for m in metrics],
        }
        with open(args.output, "w+") as f:
            json.dump(data, f, indent=2)


if __name__ == "__main__":
    asyncio.run(main())
