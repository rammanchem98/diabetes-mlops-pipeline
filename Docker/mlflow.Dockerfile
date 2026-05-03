FROM ghcr.io/mlflow/mlflow:v2.16.0

# Install the PostgreSQL driver
RUN pip install psycopg2-binary