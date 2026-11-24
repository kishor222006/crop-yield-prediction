.PHONY: help install data train app test lint format check clean docker-build docker-test

PYTHON ?= python3
VENV   ?= .venv
BIN    := $(VENV)/bin

help:  ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "} {printf "  %-14s %s\n", $$1, $$2}'

install:  ## Create the virtual environment and install all dependencies
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -e ".[dev]"

data:  ## Generate the synthetic demo dataset
	$(BIN)/python scripts/generate_sample_data.py

train:  ## Train the models and save the best one to models/saved/
	$(BIN)/python scripts/train_model.py

app:  ## Start the website at http://localhost:8501
	$(BIN)/streamlit run app/Home.py

test:  ## Run the test suite
	$(BIN)/pytest

lint:  ## Check code style and common errors
	$(BIN)/ruff check src tests scripts app

format:  ## Fix auto-fixable style issues
	$(BIN)/ruff check --fix src tests scripts app

check: lint test  ## Run lint and tests

clean:  ## Remove caches and build artefacts (keeps data and trained models)
	find . -type d \( -name __pycache__ -o -name .pytest_cache -o -name .ruff_cache -o -name "*.egg-info" \) -prune -exec rm -rf {} +

docker-build:  ## Build the Docker image
	docker build -t weather-impact .

docker-test:  ## Run the test suite inside Docker
	docker run --rm weather-impact
