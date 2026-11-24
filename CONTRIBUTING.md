# Contributing

Thank you for helping improve the Weather Impact Analysis project.

## Getting started

```bash
git clone <repository-url> weather-analysis
cd weather-analysis
python3 -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

Requires Python 3.10 or newer.

## Before you open a pull request

Run all checks. Each must pass.

```bash
make test     # or: pytest
make lint     # or: ruff check src tests scripts
```

## Branches and commits

- Create a branch from `main` for each change, for example `feature/risk-thresholds` or `fix/cleaning-median`.
- Write commit messages in the imperative mood, for example "Add humidity validation to what-if simulator".
- Keep each pull request focused on one change.

## Code standards

- Put all reusable logic in `src/weather_impact/`. Scripts in `scripts/` should only parse arguments and call the package.
- Keep the core free of UI code. Streamlit, FastAPI, and mobile code belong in their own folders and import from the core.
- Use type hints on public functions and docstrings on modules and public functions.
- Raise the custom exceptions in `weather_impact/utils/exceptions.py`. Do not raise bare `Exception`.
- Put every tunable constant in `config/settings.py` or `config/crops.py`, not inside functions.
- Line length is 110 characters. `ruff` enforces formatting and import order.

## Tests

- Every bug fix needs a test that fails before the fix and passes after it.
- Add unit tests in `tests/unit/` and end-to-end tests in `tests/integration/`.
- Tests must be reproducible. Use fixed random seeds.
- Keep the suite fast. Use `fast=True` when training models in tests.

## Data and model files

- Do not commit real or personal data. Put raw data in `data/raw/`, which Git ignores.
- Do not commit trained models from `models/saved/`. Regenerate them with `scripts/train_model.py`.
- The bundled dataset is synthetic. Label any new dataset clearly as synthetic or real.

## Agronomic values

Crop thresholds in `config/crops.py` affect farmer-facing advice. Changes to them need a source or review from an agronomist, and the source should be noted in the pull request.

## Reporting issues

Include the steps to reproduce, the expected result, the actual result, your Python version, and your operating system.
