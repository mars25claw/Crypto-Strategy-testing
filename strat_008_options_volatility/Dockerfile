FROM python:3.11-slim-bookworm

WORKDIR /app

# Copy shared library
COPY shared/ /app/shared/

# Copy strategy files
COPY strat_008_options_volatility/ /app/strat_008_options_volatility/

# Install Python dependencies — use binary wheels for heavy packages
COPY strat_008_options_volatility/requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && \
    pip install --only-binary=:all: numpy pandas scipy scikit-learn 2>/dev/null || true && \
    pip install --no-cache-dir --no-build-isolation -r requirements.txt

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
