#!/usr/bin/env -S uv run python
import asyncio
import contextlib
import dataclasses as dc
import json
import os
import random
import statistics
import time
import traceback
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Literal, cast

import click
import openai
from pydantic import BaseModel
from tqdm.auto import tqdm

from deepsearch_code import core


class TicTacToeMove(BaseModel):
    row: int
    column: int


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

    def remaining_spaces(self) -> set[tuple[int, int]]:
        return {
            (row, col)
            for row in range(self.size)
            for col in range(self.size)
            if self.board[row][col] == " "
        }

    def is_full(self) -> bool:
        return len(self.remaining_spaces()) == 0

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

    def construct_question(self, player: int, messages: list[str]) -> str:
        char = self.player_marker(player)
        prompt = f"""
You are placing '{char}'
Current board:
{self.print_board()}

What's your next move?
""".strip()

        if messages:
            prompt = "\n\n".join(["\n".join(messages), prompt])

        return prompt

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

        flash_messages: dict[int, list[str]] = {}

        for player in [1, 2]:
            wins = "even" if player == 1 else "odd"
            symbol = self.player_marker(player)
            flash_messages[player] = [
                f"You are Player {player}; you'll be playing '{symbol}'. A 12-sided dice roll will be used to determine who goes first. If it's {wins}, you'll go first."
            ]

        dice_roll = random.randint(1, 12)
        dice_winner = 1 if dice_roll % 2 == 0 else 2
        for msgs in flash_messages.values():
            msgs.append(
                f"The dice roll is {dice_roll}. Player {dice_winner} will go first."
            )

        self.turn = dice_winner

        while self.winner is None and not self.is_full():
            remaining = self.remaining_spaces()
            if len(remaining) == 1:
                row, col = remaining.pop()
                self.take_turn(row, col)
                continue

            player_num = self.turn
            agent = agents[player_num - 1]

            def forfeit(reason: str):
                """
                Forfeit the game and be forever shamed
                """
                raise Forfeit(player_num, reason)

            msgs = flash_messages.get(player_num, [])
            prompt = self.construct_question(player_num, msgs)
            msgs.clear()

            response = await agent.run(
                prompt,
                response_schema=TicTacToeMove,
                tools=[core.tool(forfeit)],
                **run_kwargs,
            )

            failures = 0
            while True:
                validation_errors: list[str] = []
                if not (
                    0 <= response.row < self.size and 0 <= response.column < self.size
                ):
                    validation_errors.append(
                        f"space w/ row {response.row}, col {response.column} does not exist"
                    )
                elif not self.is_empty(response.row, response.column):
                    validation_errors.append(
                        f"that space is already taken by {self.board[response.row][response.column]}"
                    )

                if not validation_errors:
                    self.take_turn(response.row, response.column)
                    for player, msgs in flash_messages.items():
                        if player == player_num:
                            continue
                        msgs.append(
                            f"Player {player_num} plays ({response.row}, {response.column})"
                        )
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
        response = TicTacToeMove(row=row, column=col)

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
    game: TicTacToe
    elapsed: float | None = None
    forfeit_reason: str | None = None


async def play_game(
    game: TicTacToe,
    oracle: core.Oracle,
    semaphore: asyncio.Semaphore | None = None,
) -> GameResult:
    simple_ai_oracle = TicTacToeAIOracle(game, 2)
    simple_ai = core.BasicAgent(core.Conversation(simple_ai_oracle))

    llm_convo = core.Conversation(oracle)

    ai = core.BasicAgent(
        llm_convo,
        prompt=core.StringPrompt(
            f"""
You are playing a competitive game of tic-tac-toe on a {game.size}x{game.size} board.

Each turn you'll select a move and you'll place X or O at the corresponding spot on the board. Your move will consist of a row and a column. First one to get {game.size} in a straight line up, down, or diagonally wins. Good luck!
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
            before = time.perf_counter()
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

    after = time.perf_counter()

    return GameResult(
        type=result, game=game, forfeit_reason=forfeit_reason, elapsed=after - before
    )


class ModelInfo(BaseModel):
    model: str
    usage: dict[str, Any] | None
    cost: float | None


class ModelMetrics(BaseModel):
    model: str
    results: dict[GameResultType, int]
    cost: float
    usage: dict[str, Any]
    avg_time: float
    avg_turns: float
    forfeit_reasons: list[str]
    final_boards: list[str]


def compute_metrics(games: list[tuple[GameResult, ModelInfo]]) -> list[ModelMetrics]:
    games_by_model: dict[str, list[tuple[GameResult, ModelInfo]]] = {}
    for game, model in games:
        games_by_model.setdefault(model.model, []).append((game, model))

    out: list[ModelMetrics] = []
    for model_name, model_games in games_by_model.items():
        count_by_status: dict[GameResultType, int] = {}
        forfeit_reasons: list[str] = []
        final_boards: list[str] = []
        for game, _ in model_games:
            final_boards.append(game.game.print_board())
            count_by_status.setdefault(game.type, 0)
            count_by_status[game.type] += 1
            if game.forfeit_reason:
                forfeit_reasons.append(game.forfeit_reason)

        cost = sum(model.cost for _, model in model_games if model.cost is not None)

        usage = {"input": 0, "output": 0, "cached_input": 0}
        for _, model in model_games:
            if model.usage is None:
                continue
            for key in list(usage):
                usage[key] += model.usage[key]

        elapsed = [game.elapsed for game, _ in model_games if game.elapsed is not None]
        avg_time = statistics.mean(elapsed) if elapsed else -1

        avg_turns = statistics.mean(game.game.num_turns for game, _ in model_games)

        out.append(
            ModelMetrics(
                model=model_name,
                results=count_by_status,
                cost=cost,
                usage=usage,
                avg_time=avg_time,
                avg_turns=avg_turns,
                forfeit_reasons=forfeit_reasons,
                final_boards=final_boards,
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

        input = model_metrics.usage["input"]
        output = model_metrics.usage["output"]
        cached = model_metrics.usage["cached_input"]
        token_state = f"{input:,} input ({cached:,} cached), {output:,} output"

        print(
            f"{model_metrics.model}: {statuses} (${model_metrics.cost:.3f}, {token_state}, "
            f"avg {model_metrics.avg_time:.2f}s, {model_metrics.avg_turns:.1f} turns)"
        )
        if model_metrics.forfeit_reasons:
            unique_reasons = set(model_metrics.forfeit_reasons)
            print(f"forfitted because: {'; '.join(unique_reasons)}")


@click.group()
def cli():
    pass


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


@async_command(cli)
@click.argument("model", nargs=-1)
@click.option("-n", "--num", type=int, default=20)
@click.option("-c", "--concurrency", type=int, default=None)
@click.option("-w", "--write", default=None, help="File to write output to")
@click.option("-a", "--append", default=None, help="File to append output to")
@click.option(
    "-m",
    "--method",
    type=click.Choice(["structured_outputs", "tools"]),
    default="structured_outputs",
)
async def benchmark(
    model: str,
    num: int,
    concurrency: int | None,
    write: str | None,
    append: str | None,
    method: str,
) -> None:
    if write and append:
        raise click.BadArgumentUsage("--write and --append cannot be used together")

    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key is None:
        raise ValueError("OPENROUTER_API_KEY must be set")

    client = openai.AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1", api_key=api_key
    )

    def create_oracle(tracker: core.UsageTracker) -> core.Oracle:
        if method == "structured_outputs":
            return core.LLMStructuredOutputsOracle(
                model=model, client=client, tracker=tracker
            )
        if method == "tools":
            return core.LLMToolUseOracle(model=model, client=client, tracker=tracker)
        raise ValueError(f"Invalid method: '{method}'")

    tasks: dict[
        asyncio.Task[GameResult], tuple[TicTacToe, str, int, core.UsageTracker]
    ] = {}
    size = 3
    semaphore: asyncio.Semaphore | None = None
    if concurrency is not None:
        semaphore = asyncio.Semaphore(concurrency)
    for model in model:
        for i in range(num):
            tracker = core.UsageTracker()
            llm = create_oracle(tracker)

            game = TicTacToe(size)
            task = asyncio.create_task(play_game(game, llm, semaphore))
            tasks[task] = game, model, i, tracker

    progress = tqdm(total=len(tasks))

    pending = set(tasks)
    games: list[tuple[GameResult, ModelInfo]] = []
    while pending:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            progress.update(1)
            game, model, idx, tracker = tasks[task]
            info = ModelInfo(
                model=model, usage=tracker.models.get(model), cost=tracker.cost()
            )
            try:
                games.append((task.result(), info))
            except Exception:
                print(f"Error w/ model {model}, game {idx}:")
                traceback.print_exc()
                result = GameResult(type="error", game=game)
                games.append((result, info))

    metrics = compute_metrics(games)
    print_metrics(metrics)

    total_cost = sum(model.cost for _, model in games if model.cost is not None)
    print()
    print(f"Total cost: ${total_cost:.3f}")

    if write or append:
        mode = "w+" if write else "a+"

        timestamp = int(datetime.now().timestamp())

        with open(cast(str, write or append), mode) as f:
            for metric in metrics:
                body = metric.model_dump(mode="json")
                body["timestamp"] = timestamp
                body["method"] = method
                f.write(json.dumps(body) + "\n")


@async_command(cli)
async def play():
    size = 3
    game = TicTacToe(size)

    oracle = core.ReplOracle()

    result = await play_game(game, oracle)

    if result.type == "forfeit":
        print("You forfeited because:", result.forfeit_reason)
    elif result.type == "win":
        print("You won!")
    elif result.type == "loss":
        print("You lost!")
    else:
        print("It's a tie!")


if __name__ == "__main__":
    cli()
