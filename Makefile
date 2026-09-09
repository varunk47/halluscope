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

# Full pipeline on the primary model. Steps are idempotent and resume from caches.
ITEMS ?= data/augmented/items.jsonl
MODEL ?= qwen

data:
	$(UV) run halluscope build-data --paraphrases 3 --out $(ITEMS)

capture:
	$(UV) run halluscope capture --model $(MODEL) --items $(ITEMS)

probes:
	$(UV) run halluscope probe --model $(MODEL) --items $(ITEMS) --target gap --pooling last
	$(UV) run halluscope probe --model $(MODEL) --items $(ITEMS) --target gap --pooling mean_user

behavior:
	$(UV) run halluscope behavior --model $(MODEL) --items $(ITEMS)
	$(UV) run halluscope probe --model $(MODEL) --items $(ITEMS) --target will_assume --pooling last
	$(UV) run halluscope probe --model $(MODEL) --items $(ITEMS) --target will_ask --pooling last

uq:
	$(UV) run halluscope uq --model $(MODEL) --items $(ITEMS) --split test

loop:
	$(UV) run halluscope loop --model $(MODEL) --items $(ITEMS) --condition off
	$(UV) run halluscope loop --model $(MODEL) --items $(ITEMS) --condition gate
	$(UV) run halluscope loop --model $(MODEL) --items $(ITEMS) --condition always
	$(UV) run halluscope loop --model $(MODEL) --items $(ITEMS) --condition prompt

reproduce: probes behavior uq loop
	$(UV) run halluscope report

serve:
	$(UV) run halluscope serve --reload

ui:
	cd ui && npm run dev
