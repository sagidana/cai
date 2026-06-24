.PHONY: install dev test clean

install:
	pip install .

dev:
	pip install -e .[dev]

test:
	pytest -q

clean:
	rm -rf build dist .eggs *.egg-info src/*.egg-info
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
	find . -type d -name '__pycache__' -exec rm -rf {} +
	find . -type f -name '*.py[cod]' -delete
