import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch
from src.config import settings
from sklearn.preprocessing import StandardScaler
from src.data_utils import get_data_splits
from src.model import DiabetesRegressorModel


def run_training():
    # 1. Initialize MLflow
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)

    # 2. Get 60/20/20 Split DataFrames
    train_df, val_df, test_df = get_data_splits()

    # 3. Separate Features and Target
    X_train_raw = train_df.drop('target', axis=1).values
    y_train = train_df['target'].values.reshape(-1, 1)

    X_val_raw = val_df.drop('target', axis=1).values
    y_val = val_df['target'].values.reshape(-1, 1)

    X_test_raw = test_df.drop('target', axis=1).values
    y_test = test_df['target'].values.reshape(-1, 1)

    # 4. Feature Scaling (Fit only on Training set)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)

    # 5. Save Scaler to Shared Volume (for API consumption)
    os.makedirs(settings.ARTIFACT_PATH, exist_ok=True)
    scaler_file_path = os.path.join(settings.ARTIFACT_PATH, "scaler.pkl")
    with open(scaler_file_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {scaler_file_path}")

    # 6. Tensor Conversion
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

    # 7. Model Training
    model = DiabetesRegressorModel(input_dim=X_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    with mlflow.start_run():
        # Log training parameters
        mlflow.log_param("split_ratio", "60/20/20")
        mlflow.log_param("learning_rate", 0.01)
        mlflow.log_param("model_architecture", str(model))
        mlflow.log_param("batch_size", "Full Batch")

        epochs = 150
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()

            # Periodic Validation
            if epoch % 10 == 0:
                model.eval()
                with torch.no_grad():
                    val_preds = model(X_val_tensor)
                    val_loss = criterion(val_preds, y_val_tensor)

                mlflow.log_metric("train_mse", loss.item(), step=epoch)
                mlflow.log_metric("val_mse", val_loss.item(), step=epoch)

        # 8. Final Test Evaluation
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            test_preds = model(X_test_tensor)
            test_mse = criterion(test_preds, y_test_tensor)

        mlflow.log_metric("test_mse", test_mse.item())

        # 9. Log and Register Model
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
            registered_model_name=settings.MODEL_NAME
        )
        print(f"Training finished. Test MSE: {test_mse.item()}")


if __name__ == "__main__":
    run_training()