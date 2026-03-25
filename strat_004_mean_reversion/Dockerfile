# ==============================================================================
# STRAT-004 Mean Reversion — Dockerfile
# ==============================================================================
FROM python:3.11-slim

# System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

# Working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared library
COPY ../shared /app/shared

# Copy strategy code
COPY src/ /app/src/
COPY templates/ /app/templates/
COPY config.yaml /app/config.yaml

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
