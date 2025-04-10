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
from pydantic import BaseModel, Field
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
        verbose: bool = False,
        roll_die: bool = True,
    ):
        agents = [player1, player2]

        run_kwargs = {
            "response_name": "play",
            "response_description": "Make your next move",
        }

        flash_messages: dict[int, list[str]] = {}

        def flash(
            msg: str | Callable[[int], str],
            *,
            exclude: list[int] | None = None,
            include: list[int] | None = None,
        ) -> None:
            if verbose and isinstance(msg, str):
                print(msg)
            for player in [1, 2]:
                if exclude is not None and player in exclude:
                    continue
                if include is not None and player not in include:
                    continue
                player_msg = msg
                if callable(player_msg):
                    player_msg = player_msg(player)
                flash_messages.setdefault(player, []).append(player_msg)

        def initial(player: int) -> str:
            symbol = self.player_marker(player)
            return f"You are Player {player}; you'll be playing '{symbol}'."

        flash(initial)

        if roll_die:
            flash(
                "A 12-sided die roll will be used to determine who goes first. If it's even, Player 1 will go first. If it's odd, Player 2 will go first."
            )
            die_roll = random.randint(1, 12)
            die_winner = 1 if die_roll % 2 == 0 else 2
            flash(f"The die roll is {die_roll}. Player {die_winner} will go first.")
            self.turn = die_winner
        else:
            flash("Player 1 will go first.")

        while self.winner is None and not self.is_full():
            remaining = self.remaining_spaces()
            if len(remaining) == 1:
                row, col = remaining.pop()
                self.take_turn(row, col)
                continue

            yield self.turn
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
                    if verbose:
                        char = self.player_marker(player_num)
                        print(
                            f"Player {player_num} places {char} at ({response.row}, {response.column})"
                        )
                    flash(
                        f"Player {player_num} plays ({response.row}, {response.column})",
                        exclude=[player_num],
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


@dc.dataclass(frozen=True)
class GameResult:
    game: TicTacToe
    winner: int | None
    finished: bool
    error_player: int | None
    forfeit_player: int | None
    forfeit_reason: str | None
    elapsed: float | None = None

    def player_result(self, player: int) -> "PlayerGameResult":
        other_error = self.error_player == self.game.other_player(player)
        other_forfeit_reason = (
            self.forfeit_reason
            if self.forfeit_player == self.game.other_player(player)
            else None
        )
        forfeit_reason = self.forfeit_reason if self.forfeit_player == player else None
        type: PlayerResultType
        if self.winner == player:
            type = "win"
        elif self.winner is not None and self.winner != player:
            type = "loss"
        elif other_error or other_forfeit_reason:
            type = "win"
        elif self.finished and self.winner is None:
            type = "tie"
        elif forfeit_reason:
            type = "forfeit"
        else:
            type = "error"

        return PlayerGameResult(
            type=type,
            game=self.game,
            forfeit_reason=forfeit_reason,
            elapsed=self.elapsed,
        )


PlayerResultType = Literal["win", "loss", "tie", "forfeit", "error"]


@dc.dataclass(frozen=True)
class PlayerGameResult:
    type: PlayerResultType
    game: TicTacToe
    forfeit_reason: str | None = None
    elapsed: float | None = None


async def play_game(
    game: TicTacToe,
    player1: core.Oracle,
    player2: core.Oracle,
    semaphore: asyncio.Semaphore | None = None,
    verbose: bool = False,
) -> GameResult:
    prompt = core.StringPrompt(
        f"""
You are playing a competitive game of tic-tac-toe on a {game.size}x{game.size} board.

Each turn you'll select a move and you'll place X or O at the corresponding spot on the board. Your move will consist of a row and a column. First one to get {game.size} in a straight line up, down, or diagonally wins. Good luck!
""".strip()
    )

    player1_agent = core.BasicAgent(
        core.Conversation(player1),
        prompt=prompt,
    )
    player2_agent = core.BasicAgent(core.Conversation(player2), prompt=prompt)

    current_turn = game.turn
    error_player: int | None = None
    forfeit_player: int | None = None
    forfeit_reason: str | None = None
    finished = False
    try:
        async with contextlib.AsyncExitStack() as stack:
            if semaphore is not None:
                await stack.enter_async_context(semaphore)
            before = time.perf_counter()
            async for turn in game.run(player1_agent, player2_agent, verbose=verbose):
                current_turn = turn
        finished = True
    except Forfeit as err:
        forfeit_player = err.player
        forfeit_reason = err.reason
    except Exception:
        print(f"Error from Player {current_turn}")
        traceback.print_exc()
        error_player = current_turn

    after = time.perf_counter()

    return GameResult(
        game=game,
        winner=game.winner,
        finished=finished,
        error_player=error_player,
        forfeit_player=forfeit_player,
        forfeit_reason=forfeit_reason,
        elapsed=after - before,
    )


class ModelInfo(BaseModel):
    model: str
    method: str
    usage: dict[str, Any] | None
    cost: float | None


class OracleMetrics(BaseModel):
    name: str
    meta: dict[str, Any] = Field(default_factory=dict)
    results: dict[PlayerResultType, int]
    cost: float
    usage: dict[str, Any]
    avg_time: float
    avg_turns: float
    forfeit_reasons: list[str]
    final_boards: list[str]


def compute_metrics(
    games: list[tuple[PlayerGameResult, "NamedOracle"]],
) -> list[OracleMetrics]:
    games_by_model: dict[str, list[tuple[PlayerGameResult, NamedOracle]]] = {}
    for game, model in games:
        games_by_model.setdefault(model.name, []).append((game, model))

    out: list[OracleMetrics] = []
    for model_name, model_games in games_by_model.items():
        count_by_status: dict[PlayerResultType, int] = {}
        forfeit_reasons: list[str] = []
        final_boards: list[str] = []
        for game, _ in model_games:
            final_boards.append(game.game.print_board())
            count_by_status.setdefault(game.type, 0)
            count_by_status[game.type] += 1
            if game.forfeit_reason:
                forfeit_reasons.append(game.forfeit_reason)

        cost = sum(
            cast(float, model.tracker.cost())
            for _, model in model_games
            if model.tracker.cost() is not None
        )

        usage = {"input": 0, "output": 0, "cached_input": 0}
        for _, model in model_games:
            llm_name = model.meta.get("model")
            if llm_name is None:
                continue
            if model.tracker.models.get(llm_name) is None:
                continue
            for key in list(usage):
                usage[key] += model.tracker.models[llm_name][key]

        elapsed = [game.elapsed for game, _ in model_games if game.elapsed is not None]
        avg_time = statistics.mean(elapsed) if elapsed else -1

        avg_turns = statistics.mean(game.game.num_turns for game, _ in model_games)

        out.append(
            OracleMetrics(
                name=model_name,
                meta=model_games[0][1].meta,
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


def print_metrics(metrics: list[OracleMetrics]) -> None:
    for model_metrics in metrics:
        statuses = ", ".join(
            f"{status}: {model_metrics.results[cast(PlayerResultType, status)]}"
            for status in ["win", "loss", "tie", "forfeit", "error"]
            if status in model_metrics.results
        )

        input = model_metrics.usage["input"]
        output = model_metrics.usage["output"]
        cached = model_metrics.usage["cached_input"]
        token_state = f"{input:,} input ({cached:,} cached), {output:,} output"

        print(
            f"{model_metrics.name}: {statuses} (${model_metrics.cost:.3f}, {token_state}, "
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


def parse_model_name(model_name: str) -> tuple[str, str]:
    parts = model_name.rsplit("::", 1)
    if len(parts) == 1:
        method = "text"
    else:
        model_name, method = parts

    return model_name, method


@dc.dataclass(frozen=True)
class NamedOracle:
    name: str
    oracle: core.Oracle
    tracker: core.UsageTracker
    meta: dict[str, Any] = Field(default_factory=dict)


def create_llm_oracle(
    model_name: str, method: str, client: openai.AsyncOpenAI, tracker: core.UsageTracker
) -> core.Oracle:
    if method == "text":
        return core.LLMOracle(model=model_name, client=client, tracker=tracker)
    if method == "structured_outputs":
        return core.LLMStructuredOutputsOracle(
            model=model_name, client=client, tracker=tracker
        )
    if method == "tools":
        return core.LLMToolUseOracle(model=model_name, client=client, tracker=tracker)
    raise ValueError(f"Invalid method: '{method}'")


def create_oracle(
    name: str,
    client: openai.AsyncOpenAI,
    game: TicTacToe,
    player: int,
    allow_manual: bool = False,
    allow_minmax: bool = True,
) -> NamedOracle:
    tracker = core.UsageTracker()

    if name == "me" and allow_manual:
        return NamedOracle(name=name, oracle=core.ReplOracle(), tracker=tracker)
    if name == "minmax" and allow_minmax:
        return NamedOracle(
            name=name, oracle=TicTacToeAIOracle(game, player), tracker=tracker
        )
    if "/" not in name:
        raise ValueError(
            f"{name} is not a valid model name; should be "
            f"<provider>/<model> (or `me` or `minmax` for non-LLM player(s))"
        )
    model, method = parse_model_name(name)
    llm_oracle = create_llm_oracle(model, method, client, tracker)

    return NamedOracle(
        name=name,
        oracle=llm_oracle,
        meta={"model": model, "method": method},
        tracker=tracker,
    )


async def play_games(
    games: list[tuple[TicTacToe, NamedOracle, NamedOracle]],
    concurrency: int | None = None,
    show_progress: bool = True,
) -> list[tuple[GameResult, NamedOracle, NamedOracle]]:
    semaphore: asyncio.Semaphore | None = None
    if concurrency is not None:
        semaphore = asyncio.Semaphore(concurrency)

    tasks: dict[asyncio.Task[GameResult], tuple[NamedOracle, NamedOracle]] = {}
    for game, oracle1, oracle2 in games:
        task = asyncio.create_task(
            play_game(game, oracle1.oracle, oracle2.oracle, semaphore)
        )
        tasks[task] = oracle1, oracle2

    progress = tqdm(total=len(tasks)) if show_progress else None

    pending = set(tasks)
    out_games: list[tuple[GameResult, NamedOracle, NamedOracle]] = []
    while pending:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            if progress is not None:
                progress.update(1)
            model1, model2 = tasks[task]
            out_games.append((task.result(), model1, model2))

    return out_games


def get_openai_client() -> openai.AsyncOpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key is None:
        raise ValueError("OPENROUTER_API_KEY must be set")

    return openai.AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


@async_command(cli)
@click.argument("model", nargs=-1)
@click.option("-n", "--num", type=int, default=20)
@click.option("-c", "--concurrency", type=int, default=None)
@click.option("-w", "--write", default=None, help="File to write output to")
@click.option("-a", "--append", default=None, help="File to append output to")
async def bench(
    model: list[str],
    num: int,
    concurrency: int | None,
    write: str | None,
    append: str | None,
) -> None:
    if write and append:
        raise click.BadArgumentUsage("--write and --append cannot be used together")

    client = get_openai_client()

    games: list[tuple[TicTacToe, NamedOracle, NamedOracle]] = []
    size = 3
    for model_name in model:
        for _ in range(num):
            game = TicTacToe(size)
            llm = create_oracle(model_name, client, game, 1, allow_minmax=False)
            minmax = create_oracle("minmax", client, game, 2)

            games.append((game, llm, minmax))

    results = await play_games(games, concurrency=concurrency)
    llm_results = [(game.player_result(1), model) for game, model, _ in results]

    metrics = compute_metrics(llm_results)
    print_metrics(metrics)

    total_cost = sum(
        cast(float, model.tracker.cost())
        for _, model, _ in games
        if model.tracker.cost() is not None
    )
    print()
    print(f"Total cost: ${total_cost:.3f}")

    if write or append:
        mode = "w+" if write else "a+"

        timestamp = int(datetime.now().timestamp())

        with open(cast(str, write or append), mode) as f:
            for metric in metrics:
                body = metric.model_dump(mode="json")
                body["timestamp"] = timestamp
                f.write(json.dumps(body) + "\n")


@async_command(cli)
@click.argument("player1")
@click.argument("player2")
async def play(player1: str, player2: str) -> None:
    size = 3
    game = TicTacToe(size)

    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key is None:
        raise ValueError("OPENROUTER_API_KEY must be set")

    client = openai.AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1", api_key=api_key
    )

    player1_oracle = create_oracle(player1, client, game, 1)
    player2_oracle = create_oracle(player2, client, game, 2)

    result = await play_game(
        game, player1_oracle.oracle, player2_oracle.oracle, verbose=True
    )

    if result.winner is not None:
        print(f"Player {result.winner} wins!")
    elif result.finished:
        print("It's a tie!")
    elif result.error_player:
        print(f"Player {result.error_player} encountered an error")
    elif result.forfeit_player:
        print(
            f"Player {result.forfeit_player} forfeited because: {result.forfeit_reason}"
        )
    else:
        print("The game exited for an unknown reason. This is unexpected")


@async_command(cli)
@click.argument("model", nargs=-1)
@click.argument("-o", "--output")
@click.argument("-c", "--concurrency", type=int, default=None)
async def tournament(model: list[str], output: str | None) -> None:
    pass


if __name__ == "__main__":
    cli()
