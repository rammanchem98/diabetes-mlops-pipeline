import os
import pickle
import torch
import mlflow.pytorch
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel
from src import  settings

# Global variables for the model and scaler
model = None
scaler = None


class ModelInputs(BaseModel):
    age: float  # Changed to float for calculations
    sex: float  # Changed to float for calculations
    bmi: float
    bp: float
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, scaler
    try:
        # Load scaler from shared artifacts volume
        scaler_path = os.path.join(settings.ARTIFACT_PATH, "scaler.pkl")
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

        # Load model from MLflow tracking server
        model_uri = f"models:/{settings.MODEL_NAME}/latest"
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        model = mlflow.pytorch.load_model(model_uri)
        model.eval()

        print("Successfully loaded model and scaler.")
        yield  # Application starts here

    except Exception as e:
        print(f"Failed to load model and scaler. Error: {e}")
        raise e

    finally:
        # --- CLEANUP SECTION ---
        print("Shutting down: Clearing model from RAM")
        if 'model' in globals():
            del model
        if 'scaler' in globals():
            del scaler
        # If DB like SQLAlchemy or Motor we should use await database.disconnect() to close the DB

app = FastAPI(title="Diabetes Regressor Model", lifespan=lifespan)


@app.get("/health")
def health_check():
    # This endpoint powers your Kubernetes Readiness Probe[cite: 1]
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model or Scaler not loaded")
    return {"status": "OK"}


@app.post("/predict")
async def predict(request: ModelInputs):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not ready")

    try:
        # 1. Convert Pydantic object to list of values
        data_list = [list(request.model_dump().values())]

        # 2. Scale the data
        scaled_data = scaler.transform(data_list)

        # 3. Convert to Tensor
        input_tensor = torch.tensor(scaled_data, dtype=torch.float32)

        # 4. Inference
        with torch.no_grad():
            prediction = model(input_tensor)

        return {"prediction": float(prediction.item())}  # convert tensor to float which is helpful for api

    except Exception as e:
        # Use 400 for input errors, 500 for server/logic errors
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")