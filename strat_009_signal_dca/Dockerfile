FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first for Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared library
COPY ../shared /app/shared

# Copy strategy source and templates
COPY src/ /app/src/
COPY templates/ /app/templates/
COPY config.yaml /app/config.yaml

# Create data directories
RUN mkdir -p /app/data/logs /app/data/state

# Expose dashboard port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://localhost:8080/health'); r.raise_for_status()" || exit 1

# Environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV CONFIG_PATH=/app/config.yaml

# Entry point
CMD ["python", "-m", "src.main"]
