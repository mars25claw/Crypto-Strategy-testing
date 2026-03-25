FROM python:3.11-slim-bookworm

LABEL maintainer="quant-desk"
LABEL strategy="STRAT-003-STAT-ARB"

WORKDIR /app

# Install Python deps first (layer caching) — use binary wheels for heavy packages
COPY strat_003_stat_arb_pairs/requirements.txt .
RUN pip install --upgrade pip && \
    pip install --only-binary=:all: numpy pandas scipy scikit-learn 2>/dev/null || true && \
    pip install --no-cache-dir --no-build-isolation -r requirements.txt

# Copy shared library
COPY shared/ /app/shared/

# Copy strategy code
COPY strat_003_stat_arb_pairs/ /app/strat_003_stat_arb_pairs/

WORKDIR /app/strat_003_stat_arb_pairs

# Create data directories
RUN mkdir -p data/state data/logs

# Expose dashboard port
EXPOSE 8083

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8083/health')" || exit 1

# Run the bot
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
CMD ["python", "-m", "src.main"]
