FROM python:3.11-slim-bookworm

WORKDIR /app

# Copy shared library first (changes less frequently)
COPY shared/ /app/shared/

# Copy strategy requirements and install
COPY strat_006_market_making/requirements.txt /app/strat_006_market_making/
RUN pip install --upgrade pip && \
    pip install --only-binary=:all: numpy pandas scipy scikit-learn 2>/dev/null || true && \
    pip install --no-cache-dir -r /app/strat_006_market_making/requirements.txt

# Copy strategy code
COPY strat_006_market_making/ /app/strat_006_market_making/

WORKDIR /app/strat_006_market_making

# Create data directories
RUN mkdir -p data/state data/logs

# Expose dashboard port
EXPOSE 8086

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://localhost:8086/health'); r.raise_for_status()" || exit 1

ENV PYTHONPATH=/app

# Run the bot
CMD ["python", "-m", "src.main"]
