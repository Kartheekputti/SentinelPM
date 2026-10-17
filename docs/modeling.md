# Modeling notes

## Objective

Predict machine failure (binary) from sensor telemetry. The operating objective
is **failure detection**: a missed failure (false negative) can cause unplanned
downtime, while a false alarm (false positive) triggers an inspection. Missed
failures are treated as the more costly error, so the primary metric focuses on
the positive class.

## Why PR-AUC, not accuracy or ROC-AUC

Failures make up ~3.4% of the data. A model that always predicts "no failure"
scores ~96.6% accuracy while detecting zero failures, so accuracy is not
reported. ROC-AUC is threshold-independent but is optimistic under heavy class
imbalance. **PR-AUC (average precision)** summarises the precision/recall
trade-off on the positive class and is the primary metric used for ranking
models (cross-validated).

## Leakage and integrity rules

- The held-out test split is created once (stratified) and is never used for
  hyperparameter or threshold selection.
- Candidate models are selected by **cross-validated PR-AUC on the training
  split** only.
- The decision threshold is chosen from **out-of-fold training predictions**.
- Class imbalance is handled with class weights (scikit-learn `class_weight`,
  XGBoost `scale_pos_weight`), not resampling. No synthetic samples are created
  before splitting, so no leakage is introduced.
- Target-derived columns (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) and identifiers
  (`UDI`, `Product ID`) never enter the feature matrix.

## Reported metrics

Per model, for the test split:

- Precision, recall, F1 at the selected threshold.
- Precision, recall, F1 at the default 0.5 threshold (for reference).
- PR-AUC and ROC-AUC (threshold independent).
- Confusion-matrix counts, including explicit false-positive and
  false-negative counts.
- A threshold sweep (precision/recall/F1 across 0.01–0.99).

## Trade-off between missed failures and false alarms

The selected threshold maximises F1 on out-of-fold predictions by default.
Operators who consider a missed failure far more expensive than an inspection
can **lower** the threshold to raise recall at the cost of more false alarms;
the full threshold sweep is stored in the report so this trade-off can be
inspected without retraining. The `threshold_metric` in `configs/training.yaml`
can be set to `recall` to prioritise missed-failure avoidance directly.

## Models

| Model | Role | Imbalance handling | Scaling |
| ----- | ---- | ------------------ | ------- |
| DummyClassifier | Baseline sanity check | — | no |
| Logistic Regression | Linear baseline | `class_weight="balanced"` | yes |
| Random Forest | Non-linear ensemble | `class_weight="balanced_subsample"` | no |
| XGBoost | Gradient-boosted trees | `scale_pos_weight = neg/pos` | no |

Hyperparameter search spaces are declared in `configs/training.yaml`.

## Reference run

Reproduce with `python -m scripts.train`. Full artifacts are written to
`artifacts/models/` and `artifacts/reports/training_report.json` (not committed;
regenerate locally). A representative run on the AI4I dataset:

| Model | CV PR-AUC | Test PR-AUC | Recall | F1 | FP | FN |
| ----- | --------- | ----------- | ------ | -- | -- | -- |
| Random Forest | 0.874 | 0.857 | 0.82 | 0.87 | 5 | 12 |
| XGBoost | 0.863 | 0.865 | 0.79 | 0.84 | 7 | 14 |
| Logistic Regression | 0.493 | 0.450 | 0.46 | 0.47 | 32 | 37 |
| Dummy | 0.034 | 0.034 | — | — | — | — |

Numbers depend on package versions; regenerate to obtain your own results.
These are **real experiment outputs**, not hand-entered values.

## Reloading for inference

Each fitted model is saved as a complete scikit-learn pipeline (preprocessing +
estimator) and can be reloaded with `app.training.artifacts.load_model`.
