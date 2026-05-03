import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # MLflow Configuration
    MLFLOW_TRACKING_URI: str = "http://mlflow-service:5000"
    MLFLOW_EXPERIMENT_NAME: str = "Diabetes_Regression_Prod"

    # Database Configuration (for MLflow backend)
    DATABASE_URL: str = "postgresql://user:password@postgres-service:5432/mlflow_db"

    # Storage Configuration
    # This path will point to your PersistentVolumeClaim mount
    ARTIFACT_PATH: str = "/mnt/shared/artifacts"
    MODEL_NAME: str = "DiabetesNN"

    # App Configuration
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


# Global settings instance
settings = Settings()