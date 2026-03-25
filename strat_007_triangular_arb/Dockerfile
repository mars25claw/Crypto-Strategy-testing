FROM python:3.11-slim-bookworm

WORKDIR /app

# Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --only-binary=:all: numpy pandas scipy scikit-learn 2>/dev/null || true && \
    pip install --no-cache-dir -r requirements.txt

# Copy shared library
COPY shared/ /app/shared/

# Copy strategy code
COPY strat_007_triangular_arb/ /app/strat_007_triangular_arb/

# Create data directories
RUN mkdir -p /app/data/state /app/data/logs

# Set Python path
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8087

CMD ["python", "-m", "strat_007_triangular_arb.src.main"]
