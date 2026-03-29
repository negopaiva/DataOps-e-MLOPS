import os
import json
import mlflow
import uvicorn
import numpy as np
from pydantic import BaseModel
from fastapi import FastAPI
from contextlib import asynccontextmanager


# ==============================
# SCHEMA
# ==============================
class FetalHealthData(BaseModel):
    accelerations: float
    fetal_movement: float
    uterine_contractions: float
    severe_decelerations: float


# ==============================
# LOAD MODEL
# ==============================
def load_model():
    print("🔄 Loading model from MLflow...")

    MLFLOW_TRACKING_URI = 'https://dagshub.com/negopaiva/DataOps-e-MLOPS.mlflow'
    MLFLOW_TRACKING_USERNAME = 'negopaiva'
    MLFLOW_TRACKING_PASSWORD = 'ff454bf4d79befe1d31ad7698d21053f6f0bb922'
    os.environ['MLFLOW_TRACKING_USERNAME'] = MLFLOW_TRACKING_USERNAME
    os.environ['MLFLOW_TRACKING_PASSWORD'] = MLFLOW_TRACKING_PASSWORD
    print("🔐 MLflow authentication configured.")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    print(f"📍 MLflow tracking URI set to: {MLFLOW_TRACKING_URI}")
    client = mlflow.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    print("📊 MLflow client initialized.")
    registered_model = client.get_registered_model('fetal_health')
    print(f"📈 Registered model 'fetal_health' found with {len(registered_model.latest_versions)} versions.")
    run_id = registered_model.latest_versions[-1].run_id
    print(f"🔍 Latest model version run ID: {run_id}")
    logged_model = f'runs:/{run_id}/model'
    loaded_model = mlflow.pyfunc.load_model(logged_model)

    print("✅ Model loaded!")
    return loaded_model


# ==============================
# LIFESPAN (NOVO PADRÃO)
# ==============================
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting API...")

    app.state.model = load_model()

    yield  # aplicação roda aqui

    print("🛑 Shutting down API...")


# ==============================
# APP
# ==============================
app = FastAPI(
    title="Fetal Health API",
    openapi_tags=[
        {"name": "Health", "description": "Get API health"},
        {"name": "Prediction", "description": "Model prediction"},
    ],
)   

@app.on_event(event_type="startup")
def startup_event():
    print("🚀 Starting API...")
    global loaded_model
    loaded_model = load_model()
    print("✅ Model loaded!")

# ==============================
# HEALTH
# ==============================
@app.get(path="/", tags=["Health"])
def api_health():
    return {"status": "healthy"}


# ==============================
# PREDICT
# ==============================
@app.post(path="/predict", tags=["Prediction"])

def api_predict(request: FetalHealthData):
    global loaded_model
     
    data = np.array([
        request.accelerations,
        request.fetal_movement,
        request.uterine_contractions,
        request.severe_decelerations,
    ], dtype=np.float32).reshape(1, -1)
    print(f"📊 Data received for prediction: {data}")
    prediction = loaded_model.predict(data)
    print(f"🔍 Prediction made: {prediction}")

    

    return {"prediction": str(np.argmax(prediction[0]))}