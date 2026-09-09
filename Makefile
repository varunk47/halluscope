UV ?= uv

.PHONY: install test lint fmt reproduce serve ui

install:
	$(UV) sync --extra dev --extra interp

test:
	$(UV) run pytest -m "not gpu and not network"

test-gpu:
	$(UV) run pytest

lint:
	$(UV) run ruff check . && $(UV) run ruff format --check .

fmt:
	$(UV) run ruff check --fix . && $(UV) run ruff format .

reproduce:
	$(UV) run halluscope probe --model qwen --target gap
	$(UV) run halluscope probe --model qwen --target will_assume
	$(UV) run halluscope report

serve:
	$(UV) run halluscope serve --reload

ui:
	cd ui && npm run dev
