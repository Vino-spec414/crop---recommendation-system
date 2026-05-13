# ============================================================
# Makefile — Crop Recommendation System
# Convenience shortcuts for the full development workflow.
# ============================================================
# Usage:
#   make setup        → create venv and install dependencies
#   make eda          → run EDA and generate charts
#   make train        → train all models and save the best one
#   make tune         → run hyperparameter tuning
#   make explain      → run SHAP explainability analysis
#   make test         → run all 103 unit + integration tests
#   make coverage     → run tests with coverage report
#   make run          → start the Flask development server
#   make docker-build → build the Docker image
#   make docker-run   → run the containerised app
#   make clean        → remove generated artefacts
# ============================================================

.PHONY: setup eda train tune explain test coverage run \
        docker-build docker-run docker-dev clean help

PYTHON   = python3
VENV     = venv
PIP      = $(VENV)/bin/pip
PYTEST   = $(PYTHON) -m pytest
FLASK    = $(PYTHON) app/app.py
IMAGE    = crop-recommendation
PORT     = 5000

# ── Environment ───────────────────────────────────────────────

setup:
	@echo "→ Creating virtual environment..."
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r requirements.txt
	@echo "✓ Setup complete. Activate: source $(VENV)/bin/activate"

# ── ML Pipeline ───────────────────────────────────────────────

eda:
	@echo "→ Running EDA..."
	$(PYTHON) src/eda.py

train:
	@echo "→ Training models..."
	$(PYTHON) src/train_model.py

tune:
	@echo "→ Hyperparameter tuning..."
	$(PYTHON) src/tune_model.py

explain:
	@echo "→ SHAP explainability analysis..."
	$(PYTHON) src/explain_model.py

pipeline: eda train tune explain
	@echo "✓ Full pipeline complete."

# ── Testing ───────────────────────────────────────────────────

test:
	@echo "→ Running test suite..."
	$(PYTEST) tests/ -v

coverage:
	@echo "→ Running tests with coverage..."
	$(PYTEST) tests/ --cov=src --cov-report=term-missing --cov-report=html
	@echo "✓ Coverage report: htmlcov/index.html"

test-preprocess:
	$(PYTEST) tests/test_data_preprocessing.py -v

test-predict:
	$(PYTEST) tests/test_predict.py -v

test-api:
	$(PYTEST) tests/test_api.py -v

# ── Application ───────────────────────────────────────────────

run:
	@echo "→ Starting Flask development server on http://localhost:$(PORT)"
	PYTHONPATH=src $(FLASK)

# ── Docker ────────────────────────────────────────────────────

docker-build:
	@echo "→ Building Docker image: $(IMAGE)"
	docker build -t $(IMAGE) .

docker-run: docker-build
	@echo "→ Running container on http://localhost:$(PORT)"
	docker run --rm -p $(PORT):$(PORT) --name $(IMAGE)-container $(IMAGE)

docker-dev:
	@echo "→ Starting development container with hot-reload..."
	docker compose --profile dev up

docker-stop:
	docker compose down

# ── Utilities ─────────────────────────────────────────────────

clean:
	@echo "→ Removing generated artefacts..."
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true
	find . -name ".coverage" -delete 2>/dev/null; true
	rm -rf htmlcov/ .pytest_cache/
	@echo "✓ Clean complete. Models and data preserved."

clean-models:
	@echo "→ Removing trained model artefacts (run 'make train' to regenerate)..."
	rm -f models/*.pkl models/*.png models/*.csv
	@echo "✓ Models removed."

help:
	@echo ""
	@echo "Crop Recommendation System — Available Commands"
	@echo "================================================"
	@echo "  make setup        Create virtual environment + install deps"
	@echo "  make eda          Run EDA + generate charts"
	@echo "  make train        Train and save best ML model"
	@echo "  make tune         Hyperparameter optimisation"
	@echo "  make explain      SHAP feature importance analysis"
	@echo "  make pipeline     Run eda → train → tune → explain"
	@echo "  make test         Run all 103 unit + integration tests"
	@echo "  make coverage     Tests + HTML coverage report"
	@echo "  make run          Start Flask dev server (localhost:5000)"
	@echo "  make docker-build Build Docker image"
	@echo "  make docker-run   Build + run containerised app"
	@echo "  make docker-dev   Hot-reload dev container"
	@echo "  make clean        Remove cache and temp files"
	@echo ""
