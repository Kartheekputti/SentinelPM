# Data dictionary and feature reference

Source: **AI4I 2020 Predictive Maintenance** dataset (UCI ML Repository).
The dataset is **simulated**, not real factory telemetry.

## Raw columns

| Column | Type | Unit | Description |
| ------ | ---- | ---- | ----------- |
| `UDI` | int | — | Row identifier (excluded from features) |
| `Product ID` | string | — | Product identifier (excluded from features) |
| `Type` | category | — | Product quality variant (`L`, `M`, `H`) |
| `Air temperature [K]` | float | K | Ambient air temperature |
| `Process temperature [K]` | float | K | Process temperature |
| `Rotational speed [rpm]` | int | rpm | Spindle rotational speed |
| `Torque [Nm]` | float | N·m | Applied torque |
| `Tool wear [min]` | int | min | Accumulated tool wear |
| `Machine failure` | int | — | **Target**: 1 if a failure occurred |
| `TWF`, `HDF`, `PWF`, `OSF`, `RNF` | int | — | Individual failure-mode indicators |

## Leakage policy

The failure-mode indicators `TWF`, `HDF`, `PWF`, `OSF`, `RNF` are **derived from
the target** and are therefore **excluded from all model inputs**, as are the
identifier columns and the target itself. Only `RAW_FEATURE_COLUMNS` may enter
the feature matrix:

```
Air temperature [K], Process temperature [K], Rotational speed [rpm],
Torque [Nm], Tool wear [min], Type
```

## Engineered features

| Feature | Unit | Formula | Rationale |
| ------- | ---- | ------- | --------- |
| `temperature_difference [K]` | K | `Process temperature [K] - Air temperature [K]` | Heat-dissipation failures depend on the process-to-air gap |
| `rotational_power [W]` | W | `Torque [Nm] * (2*pi*Rotational speed [rpm] / 60)` | Power failures depend on combined load, not a single sensor |

## Feature matrix layout

The preprocessing pipeline emits columns in this stable order:

1. Numeric block (raw sensors, then engineered features):
   `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`,
   `Torque [Nm]`, `Tool wear [min]`, `temperature_difference [K]`,
   `rotational_power [W]`
2. One-hot encoded `Type` columns (sorted): `Type_H`, `Type_L`, `Type_M`

The final pipeline step renames these display names to model-safe identifiers
(XGBoost forbids `[`, `]` and `<` in feature names). The matrix therefore uses:

```
air_temperature_k, process_temperature_k, rotational_speed_rpm, torque_nm,
tool_wear_min, temperature_difference_k, rotational_power_w, type_h, type_l, type_m
```

The human-readable names remain the source of truth for documentation and EDA
output; the mapping lives in `app/features/definitions.py`.

Unseen `Type` values at inference are ignored (encoded as all zeros) rather than
raising an error.

## Reproducibility and leakage rules

- Transformations are deterministic and stateless with respect to the target.
- Transformations are fitted **only** on training data.
- The pipeline emits named pandas columns so training and inference matrices
  share identical ordering.
- PCA is intentionally **not** applied unless an experiment demonstrates a
  benefit.
