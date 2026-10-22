"""HTTP client for communicating with the SentinelPM FastAPI backend."""

from __future__ import annotations

import logging
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import httpx

from app.core.exceptions import ApiClientError

logger = logging.getLogger(__name__)


class ApiClient:
    """Resilient HTTP client for FastAPI REST endpoints."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        timeout: float = 10.0,
        transport: httpx.BaseTransport | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.transport = transport
        self._custom_client = client

    @contextmanager
    def _client(self, timeout: float | None = None) -> Generator[httpx.Client, None, None]:
        if self._custom_client is not None:
            yield self._custom_client
        else:
            with httpx.Client(timeout=timeout or self.timeout, transport=self.transport) as c:
                yield c

    def _url(self, path: str) -> str:
        return f"{self.base_url}/{path.lstrip('/')}"

    def is_available(self) -> bool:
        """Quickly check if the API server is reachable."""
        try:
            with self._client(timeout=2.0) as client:
                resp = client.get(self._url("/health"))
                return resp.status_code == 200
        except Exception:
            return False

    def check_health(self) -> dict[str, Any]:
        """Fetch API service health probe."""
        try:
            with self._client() as client:
                resp = client.get(self._url("/health"))
                resp.raise_for_status()
                return resp.json()
        except httpx.HTTPStatusError as exc:
            msg = f"API health check failed with status {exc.response.status_code}"
            logger.error(msg)
            raise ApiClientError(msg) from exc
        except Exception as exc:
            msg = f"Cannot connect to API at {self.base_url}: {exc}"
            logger.error(msg)
            raise ApiClientError(msg) from exc

    def check_readiness(self) -> dict[str, Any]:
        """Fetch API model readiness probe."""
        try:
            with self._client() as client:
                resp = client.get(self._url("/ready"))
                resp.raise_for_status()
                return resp.json()
        except httpx.HTTPStatusError as exc:
            msg = f"API readiness check failed with status {exc.response.status_code}"
            logger.error(msg)
            raise ApiClientError(msg) from exc
        except Exception as exc:
            msg = f"Cannot reach readiness probe at {self.base_url}: {exc}"
            logger.error(msg)
            raise ApiClientError(msg) from exc

    def get_model_info(self) -> dict[str, Any]:
        """Fetch model metadata, thresholds and feature specifications."""
        try:
            with self._client() as client:
                resp = client.get(self._url("/model/info"))
                resp.raise_for_status()
                return resp.json()
        except httpx.HTTPStatusError as exc:
            msg = f"Failed to retrieve model info: {exc.response.text}"
            logger.error(msg)
            raise ApiClientError(msg) from exc
        except Exception as exc:
            msg = f"API connection error retrieving model info: {exc}"
            logger.error(msg)
            raise ApiClientError(msg) from exc

    def predict(self, record: dict[str, Any]) -> dict[str, Any]:
        """Submit a single sensor record for failure prediction."""
        try:
            with self._client() as client:
                resp = client.post(self._url("/predict"), json=record)
                if resp.status_code == 422:
                    error_data = resp.json()
                    detail = error_data.get("error", {}).get("message", resp.text)
                    raise ApiClientError(f"Validation error: {detail}")
                resp.raise_for_status()
                return resp.json()
        except ApiClientError:
            raise
        except httpx.HTTPStatusError as exc:
            msg = f"Prediction failed with status {exc.response.status_code}: {exc.response.text}"
            logger.error(msg)
            raise ApiClientError(msg) from exc
        except Exception as exc:
            msg = f"Failed to connect to prediction endpoint: {exc}"
            logger.error(msg)
            raise ApiClientError(msg) from exc

    def predict_batch(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        """Submit a batch of sensor records for failure prediction."""
        try:
            with self._client(timeout=max(self.timeout, 30.0)) as client:
                resp = client.post(self._url("/predict/batch"), json={"records": records})
                if resp.status_code == 422:
                    error_data = resp.json()
                    detail = error_data.get("error", {}).get("message", resp.text)
                    raise ApiClientError(f"Batch validation error: {detail}")
                resp.raise_for_status()
                return resp.json()
        except ApiClientError:
            raise
        except httpx.HTTPStatusError as exc:
            msg = f"Batch prediction failed with status {exc.response.status_code}: {exc.response.text}"
            logger.error(msg)
            raise ApiClientError(msg) from exc
        except Exception as exc:
            msg = f"Failed to connect to batch prediction endpoint: {exc}"
            logger.error(msg)
            raise ApiClientError(msg) from exc

    def explain(self, record: dict[str, Any], top_n: int = 3) -> dict[str, Any]:
        """Fetch SHAP explanation and narrative for a sensor record."""
        try:
            with self._client() as client:
                resp = client.post(
                    self._url("/explain"),
                    json={"record": record, "top_n": top_n},
                )
                if resp.status_code == 422:
                    error_data = resp.json()
                    detail = error_data.get("error", {}).get("message", resp.text)
                    raise ApiClientError(f"Explanation validation error: {detail}")
                resp.raise_for_status()
                return resp.json()
        except ApiClientError:
            raise
        except httpx.HTTPStatusError as exc:
            msg = f"Explanation failed with status {exc.response.status_code}: {exc.response.text}"
            logger.error(msg)
            raise ApiClientError(msg) from exc
        except Exception as exc:
            msg = f"Failed to connect to explain endpoint: {exc}"
            logger.error(msg)
            raise ApiClientError(msg) from exc

    def get_monitoring_summary(self) -> dict[str, Any]:
        """Fetch real-time operational telemetry, latency, drift, and alerts."""
        try:
            with self._client() as client:
                resp = client.get(self._url("/monitoring/summary"))
                resp.raise_for_status()
                return resp.json()
        except httpx.HTTPStatusError as exc:
            msg = f"Monitoring summary request failed with status {exc.response.status_code}"
            logger.error(msg)
            raise ApiClientError(msg) from exc
        except Exception as exc:
            msg = f"Failed to connect to monitoring endpoint: {exc}"
            logger.error(msg)
            raise ApiClientError(msg) from exc
