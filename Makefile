.PHONY: install test lint format clean help

help:
	@echo "Available targets:"
	@echo "  install    - Install package in dev mode"
	@echo "  test       - Run tests"
	@echo "  lint       - Run ruff linter"
	@echo "  format     - Format code with black"
	@echo "  clean      - Remove build artifacts"

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

lint:
	ruff check piu_annotate/

format:
	black piu_annotate/

clean:
	rm -rf build/ dist/ *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache