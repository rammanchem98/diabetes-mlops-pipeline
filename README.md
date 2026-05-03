# Diabetes Progression Predictor (MLOps)

A modular machine learning project that uses a Neural Network to predict diabetes progression based on physiological factors.

## 🛠 Tech Stack
* **ML:** PyTorch, Scikit-learn
* **API:** FastAPI, Pydantic
* **Orchestration:** Kubernetes (Minikube)
* **Tracking:** MLflow
* **CI/CD:** GitHub Actions & Docker Hub

## 🚀 How to Run
1. **Start Infrastructure:**
   `kubectl apply -f ./Kubernetes/postgres-deploy.yaml`
   `kubectl apply -f ./Kubernetes/mlflow-deploy.yaml`
2. **Train the Model:**
   `kubectl apply -f ./Kubernetes/trainer-job.yaml`
3. **Deploy API:**
   `kubectl apply -f ./Kubernetes/api-deploy.yaml`

## 📊 Dataset
Uses the Scikit-learn Diabetes dataset. Features are mean-centered and scaled.