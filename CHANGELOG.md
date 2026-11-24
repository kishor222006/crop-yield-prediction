# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Planned
- Streamlit website (`app/`)
- SQLite storage for saved What-If scenarios
- REST API (`api/`) and mobile app (`mobile/`)

## [0.1.0] - 2026-10-08

### Added
- ML core package `weather_impact` with configuration, data loading, validation, and cleaning.
- Synthetic crop and weather dataset generator, with reproducible seed.
- Yield prediction with Random Forest and XGBoost, selected by R².
- Evaluation metrics: R², MAE, RMSE, MAPE, WAPE, and accuracy (100 × (1 − WAPE)).
- Model registry to save and load trained models with JSON metadata.
- Environmental Risk Score (0–100) with Low, Moderate, High, and Severe levels.
- Plain-language mitigation recommendations.
- What-If scenario simulator for temperature, rainfall, and humidity.
- Mitigation resource optimiser (fractional knapsack).
- SHAP explanations with one-hot encoded features grouped back to their source.
- Command-line tools: `generate_sample_data.py` and `train_model.py`.
- Unit and integration tests (62 tests).
