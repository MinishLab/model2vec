clean:


venv:
	uv venv

install:
	uv sync --all-extras
	uv run pre-commit install

install-base:
	uv sync --extra dev

fix:
	uv run pre-commit run --all-files

test:
	uv run pytest --cov=korok --cov-report=term-missing
