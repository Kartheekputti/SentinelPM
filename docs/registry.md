# Model Registry and Experiment Tracking

## Overview
SentinelPM uses MLflow for experiment tracking and provides a sandboxed, secure Model Registry Manager to load approved model pipelines and metadata.

---

## MLflow Integration
During training runs, `MLflowTracker`:
1. Creates or connects to the configured experiment (`sentinelpm`).
2. Logs training and test evaluation metrics (PR-AUC, ROC-AUC, F1, precision, recall, confusion matrix counts).
3. Logs hyperparameter search configurations and decision thresholds.
4. Serializes the complete fitted scikit-learn pipeline (coupling preprocessing and estimator) using cloudpickle format.

### Configuration
```yaml
# Environment variables or .env
SENTINELPM_MLFLOW_TRACKING_URI=mlruns
SENTINELPM_MLFLOW_EXPERIMENT_NAME=sentinelpm
```

---

## Approved Model Artifact Resolution

### Security Sandboxing
`ModelRegistryManager` enforces strict path isolation:
- All requested model names are resolved strictly relative to `artifacts/models/`.
- Path traversal sequences (`../`, absolute outside paths) are immediately rejected with `ModelRegistryError`.

### Model Metadata (`metadata.json`)
The registry maintains a verified metadata profile:
```json
{
  "best_model": "random_forest",
  "primary_metric": "average_precision",
  "random_seed": 42,
  "feature_names": [
    "air_temperature_k",
    "process_temperature_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
    "temperature_difference_k",
    "rotational_power_w",
    "type_h",
    "type_l",
    "type_m"
  ],
  "models": [
    {
      "name": "random_forest",
      "threshold": 0.31,
      "cv_score": 0.874,
      "artifact": "artifacts/models/random_forest.joblib"
    }
  ]
}
```

### Loading an Approved Model
```python
from app.registry.manager import ModelRegistryManager

registry = ModelRegistryManager()
pipeline, metadata = registry.load_approved_model()
print(f"Loaded {metadata.model_name} (threshold={metadata.threshold})")
```
