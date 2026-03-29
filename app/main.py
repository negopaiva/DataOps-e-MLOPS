import os
import json
import mlflow
import uvicorn
import numpy as np
from pydantic import BaseModel
from fastapi import FastAPI


class FetalHealthData(BaseModel):
    accelerations: float
    fetal_movement: float
    uterine_contractions: float
    severe_decelerations: float


app = FastAPI(title="Fetal Health API",
              
              openapi_tags=[{
                  "name": "Health",
                    "description": "Get api health"
                },
                            {
                                "name": "Prediction",
                             "description": "Model prediction"
                             }
                             ]
            )


def load_model():
    print("Reading model...")
    MLFLOW_TRACKING_URI = 'https://dagshub.com/negopaiva/DataOps-e-MLOPS.mlflow'
    MLFLOW_TRACKING_USERNAME = 'negopaiva'
    MLFLOW_TRACKING_PASSWORD = 'd7da0c3c6de5b2c1455749b3a2794b713c2a36a5'
    os.environ['MLFLOW_TRACKING_USERNAME'] = MLFLOW_TRACKING_USERNAME
    os.environ['MLFLOW_TRACKING_PASSWORD'] = MLFLOW_TRACKING_PASSWORD
    print("Configuring mlflow...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    print("Creating client...")
    client = mlflow.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    print("Getting registered model...")
    registered_model = client.get_registered_model('fetal_health')
    print("read model...")
    run_id = registered_model.latest_versions[-1].run_id
    logged_model = f'runs:/{run_id}/model'
    loaded_model = mlflow.pyfunc.load_model(logged_model)
    print(loaded_model)
    return loaded_model

@app.get(path='/', tags=['Health'])

def api_health():
    return {"status": "healthy"}

@app.post(path='/predict', tags=['Prediction'])


def api_predict(request: FetalHealthData):
    loaded_model =load_model()
    
    received_data = np.array([
        request.accelerations,
        request.fetal_movement,
        request.uterine_contractions,
        request.severe_decelerations,
    ]).reshape(1, -1)

    print(loaded_model.predict(received_data))

    return {"prediction": "normal"}