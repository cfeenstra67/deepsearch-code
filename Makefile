fmt:
	uv run ruff format deepsearch_code
	uv run ruff check --fix deepsearch_code

check:
	uv run ruff format --check deepsearch_code
	uv run ruff check deepsearch_code
	uv run mypy deepsearch_code
