
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sklearn.datasets import load_iris

# Load trained model
MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "iris_model.pkl"
)

model = joblib.load(MODEL_PATH)

# Get Iris species names
iris = load_iris()
species_names = iris.target_names.tolist()

# Create FastAPI application
app = FastAPI(
    title="Iris Flower Classification API",
    description="Predict Iris flower species using Random Forest",
    version="1.0.0"
)


# Input data structure
class IrisInput(BaseModel):
    sepal_length: float = Field(gt=0)
    sepal_width: float = Field(gt=0)
    petal_length: float = Field(gt=0)
    petal_width: float = Field(gt=0)


# Home endpoint
@app.get("/")
def home():
    return {
        "message": "Welcome to Iris Flower Classification API",
        "status": "API is running"
    }


# Prediction endpoint
@app.post("/predict")
def predict(data: IrisInput):

    input_data = pd.DataFrame(
        [[
            data.sepal_length,
            data.sepal_width,
            data.petal_length,
            data.petal_width
        ]],
        columns=[
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)"
        ]
    )

    prediction = int(model.predict(input_data)[0])
    species = species_names[prediction]

    return {
        "predicted_class": prediction,
        "predicted_species": species
    }
