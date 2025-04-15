
SRC=deepsearch_code

fmt:
	uv run ruff format $(SRC)
	uv run ruff check --fix $(SRC)

check:
	uv run ruff format --check $(SRC)
	uv run ruff check $(SRC)
	uv run mypy $(SRC)
