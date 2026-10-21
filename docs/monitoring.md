# Operational Monitoring and Drift Detection

## Architecture Overview
SentinelPM provides production telemetry tracking, non-parametric statistical data drift detection, prediction distribution monitoring, and an alert manager with cooldown deduplication.

### Critical Distinction
SentinelPM strictly separates three independent telemetry concepts:
1. **Input Feature Data Drift**: Shifts in the statistical distribution of operating sensor measurements.
2. **Prediction Distribution Shift**: Divergence in model-predicted failure probabilities and positive rates over a rolling window.
3. **True Accuracy Degradation**: Verified loss of predictive performance (precision, recall, PR-AUC). This strictly requires ground-truth operational outcomes (actual physical failures or maintenance logs).

> **NOTE:** Unlabeled telemetry cannot confirm predictive accuracy degradation. Data drift alerts prompt investigation and model auditing, not unverified automatic retraining.

---

## Statistical Drift Methodology

### Continuous Sensor Features: Two-Sample Kolmogorov-Smirnov Test
For continuous measurements (`Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`):
- SentinelPM executes the two-sided Kolmogorov-Smirnov test (`scipy.stats.ks_2samp`) comparing the empirical cumulative distribution function (ECDF) of the reference baseline $F_{ref}(x)$ to the recent telemetry window $F_{cur}(x)$:
  $$D = \sup_x |F_{ref}(x) - F_{cur}(x)|$$
- A feature is flagged as drifted when the p-value falls below the configured threshold:
  $$p < \alpha \quad (\text{default } \alpha = 0.05)$$

### Categorical Descriptors: Population Stability Index (PSI)
For categorical product variants (`Type: L, M, H`):
- Population Stability Index measures shifts in proportional category distribution:
  $$\text{PSI} = \sum_{k \in \{L, M, H\}} (P_{cur}(k) - P_{ref}(k)) \times \ln\left(\frac{P_{cur}(k)}{P_{ref}(k)}\right)$$
- An index value $\text{PSI} \ge 0.20$ triggers a categorical distribution drift flag.

---

## Alerting and Cooldown Deduplication
To prevent alert storms during ongoing process shifts:
- Each alert condition tracks its last dispatch timestamp.
- Repeated triggers within `alert_cooldown_seconds` (default: 300 seconds) are suppressed.
- Dispatched alerts are logged and retained in an in-memory bounded ring buffer accessible via `GET /monitoring/summary`.

### Trigger Conditions
- **High Error Rate (`high_error_rate`)**: Error rate $> 5\%$ over a minimum window of 20 requests (Severity: `critical`).
- **High Latency (`high_latency`)**: 95th percentile latency $> 250\text{ms}$ (Severity: `warning`).
- **Data Drift (`data_drift_detected`)**: $\ge 1$ sensor features statistically drifted ($p < 0.05$) (Severity: `warning`).
- **Prediction Shift (`prediction_distribution_shift`)**: Rolling failure prediction rate diverges by $> 10$ percentage points from baseline (Severity: `warning`).

---

## Offline Drift Evaluation
To evaluate drift against any dataset:
```powershell
python -m scripts.check_drift --dataset data/raw/ai4i2020.csv --output artifacts/reports/drift_report.json
```
