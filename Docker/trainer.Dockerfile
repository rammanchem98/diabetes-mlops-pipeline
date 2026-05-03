# Use Python 3.11 as requested
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for PostgreSQL and build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and setup.py to leverage Docker cache
COPY requirements.txt .
COPY setup.py .

# Install dependencies and the project in editable mode
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -e .

# Copy source code
COPY src/ ./src/

# Environment variable to ensure logs are visible in K8s immediately
ENV PYTHONUNBUFFERED=1

# Entrypoint for the Kubernetes Job
ENTRYPOINT ["python", "-m", "src.trainer"]