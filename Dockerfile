# ---- Hewa: Nairobi air-quality anomaly service ------------------------------------
# One container serves the landing page (/), the interactive analyzer (/app),
# the REST API (/predict, /predict/file, /api/...) and the API docs (/docs).
# Works locally (docker compose up) and on Render, which sets $PORT.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    AIRQ_ROOT=/app \
    PORT=8000 \
    OMP_NUM_THREADS=1 \
    MALLOC_ARENA_MAX=2

# libgomp: OpenMP runtime for xgboost / scikit-learn
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps .

COPY configs ./configs
COPY api ./api
COPY web ./web
COPY models ./models
COPY reports/metrics.json ./reports/metrics.json
COPY data/sample ./data/sample

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD curl -fs "http://localhost:${PORT}/health" || exit 1
# Shell form so ${PORT} is expanded (Render injects it; locally it defaults to 8000)
CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT} --workers 1 --proxy-headers --forwarded-allow-ips="*"
