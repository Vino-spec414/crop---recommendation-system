# ============================================================
# Dockerfile — Crop Recommendation System
# ============================================================
# Multi-stage build:
#   Stage 1 (builder): install Python deps into a clean layer
#   Stage 2 (runtime): copy only what the app needs — no build tools
#
# WHY MULTI-STAGE:
#   The builder stage installs compilers and headers needed by
#   packages like scipy/shap.  The final image only copies the
#   compiled site-packages, keeping the runtime image small.
#
# Build:  docker build -t crop-recommendation .
# Run:    docker run -p 5000:5000 crop-recommendation
# ============================================================

# ── Stage 1: builder ─────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /install

# Install build tools needed for scipy / scikit-learn compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install into an isolated prefix so we can copy cleanly
RUN pip install --upgrade pip \
 && pip install --prefix=/install/deps --no-cache-dir -r requirements.txt


# ── Stage 2: runtime ─────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Non-root user for security best practice
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy compiled dependencies from builder
COPY --from=builder /install/deps /usr/local

# Copy application source
COPY src/       ./src/
COPY app/       ./app/
COPY models/    ./models/
COPY data/      ./data/

# Set PYTHONPATH so src/ modules are importable
ENV PYTHONPATH=/app/src
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Flask production config
ENV FLASK_APP=app/app.py
ENV FLASK_ENV=production

# Expose the API port
EXPOSE 5000

# Switch to non-root user
USER appuser

# Health check — Docker will restart the container if this fails
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')"

# Start with Gunicorn (production WSGI server, not Flask dev server)
# 2 workers × 2 threads = handles moderate concurrent load
CMD ["python3", "-m", "gunicorn", \
     "--bind", "0.0.0.0:5000", \
     "--workers", "2", \
     "--threads", "2", \
     "--timeout", "60", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "app.app:app"]
