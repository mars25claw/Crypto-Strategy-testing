FROM python:3.11-slim-bookworm

# Prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Copy shared library first (for layer caching)
COPY shared/ /app/shared/

# Copy strategy requirements and install
COPY strat_001_trend_following/requirements.txt /app/strat_001/requirements.txt
RUN pip install --upgrade pip && \
    pip install --only-binary=:all: numpy pandas scipy scikit-learn 2>/dev/null || true && \
    pip install --no-cache-dir -r /app/strat_001/requirements.txt

# Copy strategy code
COPY strat_001_trend_following/ /app/strat_001/

# Create data directories
RUN mkdir -p /app/strat_001/data/state \
             /app/strat_001/data/logs

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8081/api/health')" || exit 1

EXPOSE 8081

WORKDIR /app/strat_001

# Entry point
CMD ["python", "-m", "src.main"]
