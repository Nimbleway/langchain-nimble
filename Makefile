.PHONY: all format lint test tests integration_test integration_tests test_watch check_imports help

# Default target executed when no arguments are given to make.
all: help

# Freeze lockfile for reproducibility
.EXPORT_ALL_VARIABLES:
UV_FROZEN = true

# Define a variable for the test file path.
TEST_FILE ?= tests/unit_tests/
integration_test integration_tests: TEST_FILE = tests/integration_tests/

# unit tests are run with the --disable-socket flag to prevent network calls
test tests:
	uv run --group test pytest -vvv --disable-socket --allow-unix-socket $(TEST_FILE)

test_watch:
	uv run --group test ptw --snapshot-update --now . -- -vv $(TEST_FILE)

# integration tests are run without the --disable-socket flag to allow network calls
integration_test integration_tests:
	uv run --group test --group test_integration pytest -n=auto -vvv $(TEST_FILE)

######################
# LINTING AND FORMATTING
######################

# Define a variable for Python and notebook files.
PYTHON_FILES=.
MYPY_CACHE=.mypy_cache
lint format: PYTHON_FILES=.
lint_package: PYTHON_FILES=langchain_nimble
lint_tests: PYTHON_FILES=tests
lint_tests: MYPY_CACHE=.mypy_cache_test

lint lint_package lint_tests:
	[ "$(PYTHON_FILES)" = "" ] || uv run --all-groups ruff check $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || uv run --all-groups ruff format $(PYTHON_FILES) --diff
	[ "$(PYTHON_FILES)" = "" ] || mkdir -p $(MYPY_CACHE) && uv run --all-groups mypy $(PYTHON_FILES) --cache-dir $(MYPY_CACHE)

format:
	[ "$(PYTHON_FILES)" = "" ] || uv run --all-groups ruff format $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || uv run --all-groups ruff check --select I --fix $(PYTHON_FILES)

check_imports: $(shell find langchain_nimble -name '*.py')
	uv run python ./scripts/check_imports.py $^

######################
# HELP
######################

help:
	@echo '----'
	@echo 'Development Commands:'
	@echo '  make test                  - run unit tests (socket disabled)'
	@echo '  make test_watch            - run tests in watch mode'
	@echo '  make integration_tests     - run integration tests (requires NIMBLE_API_KEY)'
	@echo ''
	@echo 'Code Quality:'
	@echo '  make lint                  - run linters (ruff, mypy)'
	@echo '  make lint_package          - lint only langchain_nimble/'
	@echo '  make lint_tests            - lint only tests/'
	@echo '  make format                - auto-format code'
	@echo '  make check_imports         - verify all imports work'
	@echo ''
	@echo 'Variables:'
	@echo '  TEST_FILE=<path>           - run specific test file'
