VENV_PATH = ./venv
PYTHON_VERSION = $(shell cat .python-version)
SYSTEM_PYTHON = python$(PYTHON_VERSION)
PIP_CMD = $(VENV_PATH)/bin/pip
PYTHON_CMD = $(VENV_PATH)/bin/python

OS = $(shell uname -s)
ifeq ($(OS),Darwin)
  OS = macos
else ifeq ($(OS),Linux)
  OS = linux
else
  $(error Unknown operating system)
endif

clean: ## Clean up
	rm -rf $(VENV_PATH) .pytest_cache .mypy_cache .coverage .tox .eggs build dist *.egg-info

venv:  ## Create venv
	test -d $(VENV_PATH) || $(SYSTEM_PYTHON) -m venv $(VENV_PATH) && $(PIP_CMD) install --upgrade pip uv

install: venv ## Install dependencies
	$(PYTHON_CMD) -m uv pip install -r requirements/requirements-$(OS).txt
	$(PYTHON_CMD) -m uv pip install --no-deps -e .
	$(VENV_PATH)/bin/pre-commit install

test:  ## Run pytest
	$(VENV_PATH)/bin/pytest .

fix:  ## Run pre-commit
	$(VENV_PATH)/bin/pre-commit run --all-files

compile:  ## Compile requirements.txt from pyproject.toml for macos and linux
	$(VENV_PATH)/bin/uv pip compile pyproject.toml --upgrade --python-platform macos --output-file=requirements/requirements-macos.txt --all-extras
	$(VENV_PATH)/bin/uv pip compile pyproject.toml --upgrade --python-platform linux --output-file=requirements/requirements-linux.txt --all-extras
