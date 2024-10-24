clean:


venv:
	uv venv

install:
	uv sync --all-extras
	uv run pre-commit install

install-no-pre-commit:
	uv pip install ".[dev,distill]"

install-base:
	uv sync --extra dev

fix:
	uv run pre-commit run --all-files

test:
	uv run pytest --cov=model2vec --cov-report=term-missing
