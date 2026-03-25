# ==============================================================================
# STRAT-004 Mean Reversion — Dockerfile
# ==============================================================================
FROM python:3.11-slim-bookworm

# Working directory
WORKDIR /app

# Install Python dependencies
COPY strat_004_mean_reversion/requirements.txt .
RUN pip install --upgrade pip && \
    pip install --only-binary=:all: numpy pandas scipy scikit-learn 2>/dev/null || true && \
    pip install --no-cache-dir -r requirements.txt

# Copy shared library
COPY shared/ /app/shared/

# Copy strategy code
COPY strat_004_mean_reversion/src/ /app/src/
COPY strat_004_mean_reversion/templates/ /app/templates/
COPY strat_004_mean_reversion/config.yaml /app/config.yaml

# Create data directories
RUN mkdir -p /app/data/state /app/data/logs /app/data/shared

# Expose dashboard port
EXPOSE 8084

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8084/health')" || exit 1

# Environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV CONFIG_PATH=/app/config.yaml

# Run
CMD ["python", "-m", "src.main"]
