# SentinelPM REST API Documentation

The SentinelPM backend is powered by FastAPI, providing industrial predictive maintenance inference, risk scoring, explainable machine learning with SHAP, and statistical drift monitoring.

## Base URL
Default local endpoint: `http://127.0.0.1:8000`
Interactive documentation: `http://127.0.0.1:8000/docs` (OpenAPI Swagger UI) and `/redoc`.

---

## Endpoints

### 1. Health Check
`GET /health`

Confirms that the application process is running and responding.

**Response (200 OK):**
```json
{
  "status": "healthy",
  "service": "sentinelpm-api",
  "timestamp": "2026-10-21T12:00:00Z"
}
```

### 2. Readiness Probe
`GET /ready`

Reports whether the approved model pipeline, explainer, and dependencies are loaded and ready to serve inference. Returns HTTP 503 if unavailable.

**Response (200 OK):**
```json
{
  "status": "ready",
  "model_loaded": true,
  "model_name": "random_forest",
  "model_version": "1.0.0"
}
```

### 3. Model Information
`GET /model/info`

Returns non-sensitive model metadata, training evaluation scores, decision threshold, and risk tier boundaries.

**Response (200 OK):**
```json
{
  "model_name": "random_forest",
  "model_version": "1.0.0",
  "primary_metric": "average_precision",
  "cv_score": 0.874,
  "decision_threshold": 0.31,
  "risk_threshold_low": 0.2,
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
  "created_at": "2026-10-21T11:00:00Z"
}
```

### 4. Single Prediction
`POST /predict`

Accepts one machine sensor observation, validates ranges, strictly rejects target labels, and outputs prediction with probability and risk tier.

**Headers:**
- `Content-Type: application/json`
- `X-Request-ID: <optional-correlation-uuid>`

**Request Body:**
```json
{
  "Air temperature [K]": 298.1,
  "Process temperature [K]": 308.6,
  "Rotational speed [rpm]": 1551.0,
  "Torque [Nm]": 42.8,
  "Tool wear [min]": 10.0,
  "Type": "M"
}
```
*(snake_case aliases such as `air_temperature_k` are also supported).*

**Response (200 OK):**
```json
{
  "prediction_id": "7b0b2e8a-85d7-400d-9b16-efc4b5d55bfb",
  "model_version": "1.0.0",
  "model_name": "random_forest",
  "predicted_failure": false,
  "failure_probability": 0.04,
  "risk_level": "low",
  "threshold": 0.31,
  "timestamp": "2026-10-21T12:00:00Z"
}
```

### 5. Batch Predictions
`POST /predict/batch`

Executes high-throughput scoring on a collection of sensor records with configurable batch limit (`max_batch_size=500`).

**Request Body:**
```json
{
  "records": [
    {
      "Air temperature [K]": 298.1,
      "Process temperature [K]": 308.6,
      "Rotational speed [rpm]": 1551.0,
      "Torque [Nm]": 42.8,
      "Tool wear [min]": 10.0,
      "Type": "M"
    }
  ]
}
```
*(A direct JSON array `[{...}]` is also accepted).*

**Response (200 OK):**
```json
{
  "count": 1,
  "model_version": "1.0.0",
  "predictions": [
    {
      "prediction_id": "9a5efd24-3420-4107-b2eb-d16c9ebf40f0",
      "model_version": "1.0.0",
      "model_name": "random_forest",
      "predicted_failure": false,
      "failure_probability": 0.04,
      "risk_level": "low",
      "threshold": 0.31,
      "timestamp": "2026-10-21T12:00:00Z"
    }
  ]
}
```

### 6. Prediction Explanation (SHAP)
`POST /explain`

Generates individual SHAP attribution breakdown and narrative explanation for an observation.

**Request Body:**
```json
{
  "record": {
    "Air temperature [K]": 298.1,
    "Process temperature [K]": 308.6,
    "Rotational speed [rpm]": 1551.0,
    "Torque [Nm]": 42.8,
    "Tool wear [min]": 10.0,
    "Type": "M"
  },
  "top_n": 3
}
```

**Response (200 OK):**
```json
{
  "prediction_id": "example-uuid",
  "model_name": "random_forest",
  "model_version": "1.0.0",
  "predicted_probability": 0.04,
  "base_value": 0.0339,
  "contributions": [
    {
      "feature_name": "rotational_power_w",
      "display_name": "rotational_power [W]",
      "feature_value": 6954.2,
      "contribution": -0.012,
      "direction": "decreases_risk"
    }
  ],
  "top_risk_drivers": [],
  "top_risk_mitigators": [],
  "summary_text": "Predicted machine failure probability is 4.0%...",
  "causality_disclaimer": "SHAP values describe statistical model attribution and feature contributions to the model's decision; they do not prove physical or mechanical causation."
}
```

### 7. Monitoring Summary
`GET /monitoring/summary`

Returns rolling request counts, error rate, latency percentiles (min, max, mean, p50, p95, p99), drift analysis, and active alerts.

---

## Error Envelope
All error responses return a consistent error envelope:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Request schema validation failed.",
    "details": [...],
    "request_id": "0e31b451-7dce-4468-8e0d-79bd600e8262"
  }
}
```

### Error Codes
- `VALIDATION_ERROR` (422): Schema or bound mismatch, or prohibited target indicator present.
- `INFERENCE_ERROR` (400): Batch size exceeded, corrupted payload, or scoring fault.
- `EXPLAINABILITY_ERROR` (400): Unsupported estimator or attribution calculation failure.
- `MODEL_UNAVAILABLE` (503): Model artifact not yet loaded or missing.
- `INTERNAL_SERVER_ERROR` (500): Unhandled exception (stack trace concealed, request ID recorded).
