FROM python:3.11-slim-bookworm

# Set working directory
WORKDIR /app

# Copy requirements first for Docker layer caching
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --only-binary=:all: numpy pandas scipy scikit-learn 2>/dev/null || true && \
    pip install --no-cache-dir -r requirements.txt

# Copy shared library
COPY shared/ /app/shared/

# Copy strategy source and templates
COPY strat_009_signal_dca/src/ /app/src/
COPY strat_009_signal_dca/templates/ /app/templates/
COPY strat_009_signal_dca/config.yaml /app/config.yaml

# Create data directories
RUN mkdir -p /app/data/logs /app/data/state

# Expose dashboard port
EXPOSE 8089

# Health check
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://localhost:8089/health'); r.raise_for_status()" || exit 1

# Environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV CONFIG_PATH=/app/config.yaml

# Entry point
CMD ["python", "-m", "src.main"]
