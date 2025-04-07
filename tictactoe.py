import asyncio
import itertools

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
    ) -> int | None:
        markers = ["X", "O"]
        agents = list(zip(markers, [player1, player2]))

        turns = itertools.cycle(agents)

        while self.winner is None and not self.board_is_full():
            char, agent = next(turns)
            response = await agent.run(
                f"You are playing '{char}'\n"
                f"Current board:\n{self.print_board()}\n\n"
                "What's your next move?",
                response_schema=TicTacToeMove,
            )

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

                response = await agent.run(
                    f"Invalid move: {', '.join(validation_errors)}. Please try again", response_schema=TicTacToeMove
                )

        if self.winner is not None:
            return markers.index(self.winner) + 1
        return None


async def main():
    oracle = core.ReplOracle()

    player1 = core.BasicAgent(core.Conversation(oracle), [])

    player2 = core.BasicAgent(core.Conversation(oracle), [])

    game = TicTacToe()

    winner = await game.run(player1, player2)

    print("Final:")
    print(game.print_board())

    if winner is None:
        print("It's a tie!")
    else:
        print("The winner is player", winner)


if __name__ == "__main__":
    asyncio.run(main())
