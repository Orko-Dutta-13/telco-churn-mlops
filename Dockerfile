# ─────────────────────────────────────────────────────────────────
# Telco Churn Prediction API — Docker image
# Base: official Python 3.11 slim image (small, production-safe)
# ─────────────────────────────────────────────────────────────────

FROM python:3.11-slim


# ── System dependencies ───────────────────────────────────────────
# libgomp1 is required by LightGBM and XGBoost for parallel trees.
# Without it, the model import crashes silently inside the container.

RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


# ── Set working directory ─────────────────────────────────────────
# All subsequent COPY and CMD commands are relative to /app.

WORKDIR /app


# ── Install Python dependencies ───────────────────────────────────
# Copy requirements FIRST so Docker caches this layer.
# If only app code changes, this expensive step is skipped.

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt


# ── Copy application code ─────────────────────────────────────────

COPY src/   ./src/
COPY api/   ./api/
COPY model/ ./model/


# ── Python path ───────────────────────────────────────────────────
# Tell Python where to find preprocess.py when app.py imports it.

ENV PYTHONPATH="/app/src:/app"


# ── Port ──────────────────────────────────────────────────────────
# Document which port the API listens on.
# The -p flag in docker run maps this to your local machine.

EXPOSE 8000


# ── Start the API ─────────────────────────────────────────────────
# uvicorn is the production ASGI server for FastAPI.
# --host 0.0.0.0 is required inside Docker — without it,
# the server binds to localhost inside the container only
# and is unreachable from outside.

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]