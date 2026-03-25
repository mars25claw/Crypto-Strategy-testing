FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared library
COPY ../shared /app/shared

# Copy strategy code
COPY . /app/strat_010_ml_onchain

WORKDIR /app/strat_010_ml_onchain

# Create data directories
RUN mkdir -p data/logs data/state data/features data/onchain_cache models

# Environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

EXPOSE 8090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://localhost:8090/health'); r.raise_for_status()" || exit 1

CMD ["python", "-m", "src.main"]
