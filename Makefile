.PHONY: help install install-no-pre-commit install-base test test-integration test-integration-update test-integration-pretrained-update lint typecheck fix pre-commit

help:
	@echo "Available targets:"
	@echo "  install                             Install all dependencies and pre-commit hooks"
	@echo "  install-no-pre-commit               Install all dependencies without pre-commit hooks"
	@echo "  install-base                        Install only the dev dependencies"
	@echo "  test                                Run unit tests with coverage (excludes integration tests)"
	@echo "  test-integration                    Run integration tests"
	@echo "  test-integration-update             Regenerate the distill integration baseline"
	@echo "  test-integration-pretrained-update  Regenerate the pretrained integration baseline"
	@echo "  lint                                Run ruff and pydoclint"
	@echo "  typecheck                           Run mypy"
	@echo "  fix                                 Auto-fix lint issues and format code"
	@echo "  pre-commit                          Run all pre-commit hooks"

install: install-no-pre-commit
	uv run pre-commit install

install-no-pre-commit:
	uv sync --all-extras

install-base:
	uv sync --extra dev

lint:
	uv run ruff check model2vec/ tests/
	uv run pydoclint model2vec/

typecheck:
	uv run mypy model2vec/

fix:
	uv run ruff check --fix model2vec/ tests/
	uv run ruff format model2vec/ tests/

pre-commit:
	uv run pre-commit run --all-files

test:
	uv run pytest --cov=model2vec --cov-report=term-missing --ignore=tests/integration $(VERBOSITY)

test-integration:
	uv run pytest tests/integration $(VERBOSITY)

test-integration-update:
	uv run python -m tests.integration.update_distill_baseline

test-integration-pretrained-update:
	uv run python -m tests.integration.update_pretrained_baseline
