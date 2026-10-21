"""FastAPI application factory and middleware configuration for SentinelPM."""

from __future__ import annotations

import contextlib
import time
import uuid
from collections.abc import AsyncIterator

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from app.api.dependencies import (
    get_drift_detector,
    get_inference_service,
    get_model_explainer,
    get_monitoring_tracker,
)
from app.api.routes import router
from app.api.schemas import ErrorDetail, ErrorResponse
from app.core.config import Settings, get_settings
from app.core.exceptions import ExplainabilityError, InferenceError, ModelRegistryError
from app.core.logging import get_logger

logger = get_logger(__name__)


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifecycle manager to initialize and warm up model services."""
    logger.info("Initializing SentinelPM API services...")

    # Warm up inference service, explainer, and drift detector
    try:
        service = get_inference_service()
        logger.info(
            "Approved model loaded: %s (v%s)",
            service.metadata.model_name,
            service.metadata.model_version,
        )
        get_model_explainer()
        get_drift_detector()
        logger.info("SentinelPM model and explainability services initialized successfully.")
    except Exception as exc:
        logger.warning(
            "Model services could not be pre-loaded at startup: %s. "
            "Endpoints requiring the model will return 503 until resolved.",
            exc,
        )

    yield
    logger.info("Shutting down SentinelPM API.")


class RequestTracingMiddleware(BaseHTTPMiddleware):
    """Middleware for request-id propagation, latency measurement and operational metrics."""

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        # Resolve or generate request ID
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id

        tracker = get_monitoring_tracker()
        start_time = time.perf_counter()

        try:
            response = await call_next(request)
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            response.headers["X-Request-ID"] = request_id

            # Record operational telemetry
            tracker.record_request(
                endpoint=request.url.path,
                status_code=response.status_code,
                latency_ms=duration_ms,
            )
            return response
        except Exception as exc:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            tracker.record_request(
                endpoint=request.url.path,
                status_code=500,
                latency_ms=duration_ms,
            )
            tracker.record_error(
                endpoint=request.url.path,
                status_code=500,
                error_message=str(exc),
            )
            raise


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        settings: Optional Settings override.

    Returns:
        Configured :class:`FastAPI` instance.
    """
    active_settings = settings or get_settings()

    app = FastAPI(
        title="SentinelPM Predictive Maintenance API",
        description=(
            "Industrial predictive maintenance, explainable machine learning (SHAP), "
            "and statistical drift monitoring platform."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )
    app.state.settings = active_settings

    # Add CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add tracing and metrics middleware
    app.add_middleware(RequestTracingMiddleware)

    # Exception Handlers
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        req_id = getattr(request.state, "request_id", None)
        tracker = get_monitoring_tracker()
        tracker.record_input_quality_issue("invalid_types")

        clean_errors = []
        for err in exc.errors():
            clean_err = dict(err)
            if "ctx" in clean_err and isinstance(clean_err["ctx"], dict):
                clean_err["ctx"] = {k: str(v) for k, v in clean_err["ctx"].items()}
            clean_errors.append(clean_err)

        error_response = ErrorResponse(
            error=ErrorDetail(
                code="VALIDATION_ERROR",
                message="Request schema validation failed.",
                details=clean_errors,
                request_id=req_id,
            )
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=error_response.model_dump(),
        )

    @app.exception_handler(InferenceError)
    async def inference_error_handler(
        request: Request,
        exc: InferenceError,
    ) -> JSONResponse:
        req_id = getattr(request.state, "request_id", None)
        tracker = get_monitoring_tracker()
        tracker.record_input_quality_issue("inference_failure")

        error_response = ErrorResponse(
            error=ErrorDetail(
                code="INFERENCE_ERROR",
                message=str(exc),
                request_id=req_id,
            )
        )
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=error_response.model_dump(),
        )

    @app.exception_handler(ExplainabilityError)
    async def explainability_error_handler(
        request: Request,
        exc: ExplainabilityError,
    ) -> JSONResponse:
        req_id = getattr(request.state, "request_id", None)
        error_response = ErrorResponse(
            error=ErrorDetail(
                code="EXPLAINABILITY_ERROR",
                message=str(exc),
                request_id=req_id,
            )
        )
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=error_response.model_dump(),
        )

    @app.exception_handler(ModelRegistryError)
    async def model_registry_error_handler(
        request: Request,
        exc: ModelRegistryError,
    ) -> JSONResponse:
        req_id = getattr(request.state, "request_id", None)
        error_response = ErrorResponse(
            error=ErrorDetail(
                code="MODEL_UNAVAILABLE",
                message=str(exc),
                request_id=req_id,
            )
        )
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=error_response.model_dump(),
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(
        request: Request,
        exc: HTTPException,
    ) -> JSONResponse:
        req_id = getattr(request.state, "request_id", None)
        msg = exc.detail if isinstance(exc.detail, str) else "HTTP request error."
        details = exc.detail if isinstance(exc.detail, dict) else None

        error_response = ErrorResponse(
            error=ErrorDetail(
                code="HTTP_ERROR",
                message=msg,
                details=details,
                request_id=req_id,
            )
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=error_response.model_dump(),
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        req_id = getattr(request.state, "request_id", None)
        logger.exception("Unhandled server exception: %s", exc)

        error_response = ErrorResponse(
            error=ErrorDetail(
                code="INTERNAL_SERVER_ERROR",
                message="An unexpected internal server error occurred.",
                request_id=req_id,
            )
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_response.model_dump(),
        )

    # Include routes
    app.include_router(router)

    return app
