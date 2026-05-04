import os
import pickle
import torch
import mlflow.pytorch
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field, field_validator
from src import  settings

# Global variables for the model and scaler
model = None
scaler = None


class ModelInputs(BaseModel):
    # Use Field to enforce the -0.2 to 0.2 scaled range we discussed
    age: float = Field(..., ge=0, le=120, description="Age should be between 0 and 120")
    sex: float = Field(...,description="0 for Male and 1 for Female")
    bmi: float = Field(..., ge=10, le=60, description="Actual BMI")
    bp: float = Field(...,ge=50,le=200, description="Blood Pressure")
    s1: float = Field(...)
    s2: float = Field(...)
    s3: float = Field(...)
    s4: float = Field(...)
    s5: float = Field(...)
    s6: float = Field(...)

    # # The custom validator for "unusual" values
    # @field_validator('bmi')
    # @classmethod
    # def validate_bmi_extremes(cls, v: float) -> float:
    #     if abs(v) > 0.15:
    #         print(f"Warning: Unusual BMI value detected: {v}")
    #     return v


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
        print(f"Failed to load model or scaler. Error: {e}")
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
    return {"status": "OK, Model and Scaler Loaded"}


@app.post("/predict")
async def predict(request: ModelInputs):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model or Scaler not  ready")

    try:
        # 1. Convert Pydantic object to list of values
        data_list = [list(request.model_dump().values())]

        # 2. CONVERSION STEP: This uses the Mean/SD from your training data
        # This turns "45 years" into "-0.0234" (scaled)
        scaled_data = scaler.transform(data_list)

        # 3. Convert to Tensor
        input_tensor = torch.tensor(scaled_data, dtype=torch.float32)

        # 4. Inference
        with torch.no_grad():
            prediction = model(input_tensor)

        if prediction < 100:
            risk_level = "Good (Low Risk)"
            recommendation = "Maintain your current healthy lifestyle."
        elif 100 <= prediction <= 200:
            risk_level = "Average (Moderate Risk)"
            recommendation = "Consider a follow-up consultation and diet review."
        else:
            risk_level = "High Risk"
            recommendation = "Consult a healthcare professional immediately."

        return {
            "Prediction": float(prediction.item()),
            "Level of Risk": risk_level,
            "Recommendation": recommendation
        }

    except Exception as e:
        # Use 400 for input errors, 500 for server/logic errors
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")