FROM python:3.11-slim-bookworm

WORKDIR /app

# Python deps
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --only-binary=:all: numpy pandas scipy scikit-learn 2>/dev/null || true && \
    pip install --no-cache-dir -r requirements.txt

# Copy shared library
COPY shared/ /app/shared/

# Copy strategy code
COPY strat_005_grid_trading/ /app/strat_005_grid_trading/

# Create data directories
RUN mkdir -p /app/data/state /app/data/logs /app/data/shared

WORKDIR /app/strat_005_grid_trading

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://localhost:8085/health'); r.raise_for_status()" || exit 1

ENV PYTHONPATH=/app
ENV CONFIG_PATH=config.yaml

CMD ["python", "-m", "src.main"]
