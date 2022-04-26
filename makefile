.PHONY: help install-deps unit-test format

.DEFAULT: help
help:
	@echo "install-deps: Installs all dependencies necessary for this project."
	@echo "unit-tests: Executes the unit tests."
	@echo "format: Formats all files in a common format."
	@echo "lint: Check for errors, bugs, stylistic errors and suspicious constructs."

install-deps: requirements.txt
	@pip install --upgrade -r requirements.txt

unit-tests:
	@pytest --cov=framework test.py

format:
	@autoflake --remove-all-unused-imports -ir .
	@isort . --overwrite-in-place
	@black . --line-length=160

lint:
	@flake8 --max-line-length=160 .
