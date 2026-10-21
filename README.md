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
| M05 | Model explainability & SHAP reports | done |
| M06 | Model registry & inference engine | done |
| M07 | FastAPI prediction service | done |
| M08 | Drift monitoring & alert manager | done |
| — | Part 3 (Streamlit dashboard, Docker, CI/CD) | not started |

## Tech stack

Python 3.11, pandas, NumPy, scikit-learn, XGBoost, Pandera, Pydantic, SHAP,
MLflow, FastAPI, Uvicorn, SciPy, pytest, Ruff. Part 3 will add Streamlit,
Docker and GitHub Actions.

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
# Run the complete test suite (220+ tests)
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

# Train models (Random Forest, XGBoost, Baseline)
python -m scripts.train

# Generate SHAP explainability report
python -m scripts.run_explainability

# Run Kolmogorov-Smirnov & PSI drift checks
python -m scripts.check_drift

# Launch the FastAPI prediction service
python -m scripts.run_api
```

## API quickstart

Once the service is started with `python -m scripts.run_api`, interactive docs
are available at `http://127.0.0.1:8000/docs`.

### Health & Readiness

```powershell
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/ready
```

### Single Prediction

```powershell
curl -X POST http://127.0.0.1:8000/predict `
  -H "Content-Type: application/json" `
  -d '{
    "Type": "M",
    "Air_temperature_K": 298.2,
    "Process_temperature_K": 308.7,
    "Rotational_speed_rpm": 1408,
    "Torque_Nm": 46.3,
    "Tool_wear_min": 3
  }'
```

### Explanation Attribution

```powershell
curl -X POST http://127.0.0.1:8000/explain `
  -H "Content-Type: application/json" `
  -d '{
    "record": {
      "Type": "L",
      "Air_temperature_K": 302.1,
      "Process_temperature_K": 311.0,
      "Rotational_speed_rpm": 1300,
      "Torque_Nm": 65.0,
      "Tool_wear_min": 220
    },
    "top_k": 5
  }'
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
| `SENTINELPM_API_PORT` | `8000` | REST API port |
| `SENTINELPM_ACTIVE_MODEL` | `random_forest` | Active model for inference |
| `SENTINELPM_DRIFT_KS_ALPHA` | `0.05` | KS-test alpha for drift detection |
| `SENTINELPM_DRIFT_PSI_THRESHOLD` | `0.25` | PSI threshold for drift detection |
| `SENTINELPM_ALERT_LATENCY_P95_MS` | `250.0` | Latency SLA alert threshold |
| `SENTINELPM_ALERT_ERROR_RATE_THRESHOLD` | `0.05` | Error rate alert threshold |

## Repository layout

```
app/
  api/            FastAPI application, routes, schemas, and middleware
  core/           configuration, exceptions, and logging
  data/           ingestion and Pandera validation
  explainability/ SHAP TreeExplainer, narrative generation, causality disclaimer
  features/       feature engineering transformers
  inference/      deterministic scoring engine, risk tiers, payload validation
  monitoring/     in-memory tracker, KS-test/PSI drift detector, alert manager
  registry/       sandboxed artifact loading, MLflow experiment tracking
  training/       training pipeline and cross-validation
configs/          experiment, training, and monitoring configurations
data/raw/         source dataset (not committed)
artifacts/        trained models, MLflow runs, and evaluation reports
docs/             API, explainability, monitoring, and registry documentation
scripts/          runnable entry points (training, API, drift, explainability)
tests/            comprehensive unit and integration test suite
```
