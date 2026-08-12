VERBOSITY=

venv:
	uv venv

install:
	uv sync --all-extras
	uv run pre-commit install

install-no-pre-commit:
	uv pip install ".[dev,distill,train,onnx,quantization,integration,tests]"

install-base:
	uv sync --extra dev

fix:
	uv run pre-commit run --all-files

test:
	uv run pytest --cov=model2vec --cov-report=term-missing --ignore=tests/integration $(VERBOSITY)

test-verbose:
	make test VERBOSITY="-vvv"

test-integration:
	uv run pytest tests/integration $(VERBOSITY)

test-integration-update:
	uv run python -m tests.integration.update_distill_baseline
