# syntax=docker/dockerfile:1
# ── Stage 1: build dependencies ──────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

COPY requirements.txt .

RUN pip install --no-cache-dir --prefix=/install \
        -r requirements.txt

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

# libgomp нужна LightGBM, libgomp1 — её runtime-зависимость
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code and model artifacts
COPY src/ ./src/
COPY models/ ./models/

# Ownership
RUN chown -R appuser:appuser /app
USER appuser

ENV MODEL_PATH=models/final_model.pkl \
    PREPROCESSOR_PATH=models/preprocessor.pkl \
    THRESHOLD_PATH=models/threshold.json \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    LOG_LEVEL=INFO

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["python", "-m", "uvicorn", "src.service.main:app", \
     "--host", "0.0.0.0", "--port", "8000"]
