FROM python:3.11-slim

WORKDIR /app

# Install libpq-dev for the PostgreSQL client connection (MLflow backend)
RUN apt-get update && apt-get install -y \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
COPY setup.py .

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -e .

COPY src/ ./src/

ENV PYTHONUNBUFFERED=1

# Expose FastAPI port
EXPOSE 8000

# Entrypoint for the Kubernetes Deployment
ENTRYPOINT ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]