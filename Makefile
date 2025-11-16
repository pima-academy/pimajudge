.PHONY: test setup

setup:
	uv sync

test: setup
	uv run -m pytest
