fmt:
	uv run ruff format deepsearch_code *.py
	uv run ruff check --fix deepsearch_code *.py

check:
	uv run ruff format --check deepsearch_code *.py
	uv run ruff check deepsearch_code *.py
	uv run mypy deepsearch_code *.py
