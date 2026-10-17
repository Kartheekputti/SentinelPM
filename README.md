# SentinelPM

Industrial predictive maintenance and explainable machine learning platform.

SentinelPM predicts industrial machine failures from sensor telemetry, with a
modular architecture designed for reproducible training, explainable
predictions, a REST API, monitoring and a web dashboard.

> **Data source:** the AI4I 2020 Predictive Maintenance dataset (UCI ML
> Repository) is **simulated data**, not real factory telemetry. It is useful
> for prototyping and method validation, not for drawing operational
> conclusions about real plants.

## Status

| Module | Description | Status |
| ------ | ----------- | ------ |
| M01 | Project foundation (config, logging, tooling) | done |
| M02 | Data ingestion and validation | done |
| M03 | Exploratory analysis and feature engineering | done |
| M04 | Model training pipeline | done |
| — | Part 2 (SHAP, registry, API, dashboard, Docker) | not started |

## Tech stack

Python 3.11, pandas, NumPy, scikit-learn, XGBoost, Pandera, Pydantic, pytest,
Ruff. Part 2 will add MLflow, SHAP, FastAPI, Streamlit, Docker and GitHub
Actions.

## Setup

```powershell
# Create and activate a virtual environment (Python 3.11+)
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1

# Install the project in editable mode with dev tooling
pip install -e ".[dev]"
```

Configuration is optional: every setting has a default. To customise, copy
`.env.example` to `.env`, or point `SENTINELPM_CONFIG_PATH` at a YAML file.
All settings are resolved relative to the project root — no absolute paths
are hard-coded.

## Dataset

Download the AI4I 2020 dataset and place `ai4i2020.csv` in `data/raw/`:

- UCI: <https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset>
- Direct: <https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip>

A helper script is provided:

```powershell
python scripts/download_dataset.py
```

Raw data is never modified in place and is excluded from Git.

## Common commands

```powershell
# Run the test suite
pytest

# Run tests with coverage
pytest --cov=app --cov-report=term-missing

# Lint and format checks
ruff check .
ruff format --check .

# Validate the dataset
python -m scripts.validate_dataset

# Generate the EDA report
python -m scripts.run_eda

# Train models
python -m scripts.train
```

## Configuration reference

| Variable | Default | Purpose |
| -------- | ------- | ------- |
| `SENTINELPM_ENVIRONMENT` | `dev` | `dev`, `test` or `prod` |
| `SENTINELPM_LOG_LEVEL` | `INFO` | Logging verbosity |
| `SENTINELPM_RANDOM_SEED` | `42` | Global reproducibility seed |
| `SENTINELPM_DATA_DIR` | `data` | Dataset root directory |
| `SENTINELPM_ARTIFACTS_DIR` | `artifacts` | Model/evaluation artifact root |
| `SENTINELPM_CONFIG_PATH` | — | Optional YAML override file |

## Repository layout

```
app/
  core/        configuration and logging
  data/        ingestion and validation
  features/    feature engineering transformers
  training/    training pipeline and evaluation
configs/       experiment and pipeline configuration
data/raw/      source dataset (not committed)
artifacts/     trained models and reports (not committed)
scripts/       runnable entry points
tests/         unit and integration tests
docs/          design notes and reports
```
