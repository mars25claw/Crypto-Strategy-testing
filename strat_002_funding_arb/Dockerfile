FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared library
COPY ../shared /app/shared

# Copy strategy code
COPY . /app/strat_002_funding_arb

# Create data directories
RUN mkdir -p /app/data/state /app/data/logs

# Set Python path
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8082

CMD ["python", "-m", "strat_002_funding_arb.src.main"]
