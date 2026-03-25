FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy shared library
COPY shared/ /app/shared/

# Copy strategy files
COPY strat_008_options_volatility/ /app/strat_008_options_volatility/

# Install Python dependencies
COPY strat_008_options_volatility/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Set working directory to strategy
WORKDIR /app/strat_008_options_volatility

# Create data directories
RUN mkdir -p data/logs data/state

# Expose dashboard port
EXPOSE 8088

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://localhost:8088/health'); r.raise_for_status()" || exit 1

# Entry point
CMD ["python", "-m", "src.main"]
