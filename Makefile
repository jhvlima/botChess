PYTHON ?= .venv/bin/python
PIP ?= .venv/bin/pip
PLAYER ?=
RAW_DIR ?= partidas
DATA_DIR ?= data/processed
ARTIFACTS_DIR ?= artifacts
REPORTS_DIR ?= reports
DEMO_MODEL ?= $(ARTIFACTS_DIR)/demo_model.joblib
MODEL ?= $(ARTIFACTS_DIR)/final_model.joblib

.PHONY: install demo-model demo prepare validate train test check pipeline play

install:
	python3 -m venv .venv
	$(PIP) install -e .

demo-model:
	$(PYTHON) -B -m botchess.demo --output $(DEMO_MODEL)

demo: demo-model
	$(PYTHON) -B -m botchess.web --model $(DEMO_MODEL)

prepare:
	@test -n "$(PLAYER)" || (echo "Informe PLAYER, por exemplo: make prepare PLAYER=MeuUsuario" >&2; exit 2)
	$(PYTHON) -B -m botchess.data --raw-dir $(RAW_DIR) --output-dir $(DATA_DIR) --username $(PLAYER)

validate:
	@test -n "$(PLAYER)" || (echo "Informe PLAYER, por exemplo: make validate PLAYER=MeuUsuario" >&2; exit 2)
	$(PYTHON) -B -m botchess.validation --data-dir $(DATA_DIR) --raw-dir $(RAW_DIR) --username $(PLAYER)

train:
	$(PYTHON) -B -m botchess.training --data-dir $(DATA_DIR) --artifacts-dir $(ARTIFACTS_DIR) --reports-dir $(REPORTS_DIR) --max-negatives 8 --seed 42

test:
	$(PYTHON) -B -m unittest discover -v

check: test
	$(PYTHON) -m compileall -q src tests
	node --check src/botchess/static/app.js
	$(PIP) check

play:
	$(PYTHON) -B -m botchess.web --model $(MODEL)

pipeline: prepare validate train test
