# SentinelPM — Production Container Image
FROM python:3.11-slim AS base

# System configuration
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    SENTINELPM_ENVIRONMENT=prod

WORKDIR /app

# Install system dependencies (curl for container healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create unprivileged application user
RUN groupadd -g 1000 appgroup && \
    useradd -u 1000 -g appgroup -s /bin/bash -m appuser

# Install Python dependencies
COPY requirements.txt pyproject.toml /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code and configurations
COPY app/ /app/app/
COPY configs/ /app/configs/
COPY scripts/ /app/scripts/
COPY artifacts/ /app/artifacts/
COPY README.md /app/

# Set ownership to unprivileged user
RUN chown -R appuser:appgroup /app

USER appuser

# Expose API and Dashboard ports
EXPOSE 8000 8501

# Healthcheck for API service
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command: launch FastAPI server
CMD ["uvicorn", "app.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
