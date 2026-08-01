VERBOSITY=

venv:
	uv venv

install:
	uv sync --all-extras
	uv run pre-commit install

install-no-pre-commit:
	uv pip install ".[dev,distill,inference,train,onnx,quantization]"

install-base:
	uv sync --extra dev

fix:
	uv run pre-commit run --all-files

test:
	uv run pytest --cov=model2vec --cov-report=term-missing --ignore=tests/integration $(VERBOSITY)

test-verbose:
	make test VERBOSITY="-vvv"

# Downloads a real model, distills several variants from it, and compares the
# result (vocab size, token order, embedding rank, semantic sanity, etc) against
# the stored JSON baseline in tests/integration/data/distill_baseline.json.
# Not run as part of `make test`: it needs network access and is much slower.
test-integration:
	uv run pytest tests/integration $(VERBOSITY)

# Regenerates the JSON baseline used above. Only run this deliberately after a
# change that intentionally alters distillation output, then review the diff of
# tests/integration/data/distill_baseline.json before committing it.
test-integration-update:
	uv run python -m tests.integration.update_distill_baseline
