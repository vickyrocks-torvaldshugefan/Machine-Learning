
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# Load the trained model
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "model.pkl"

model_artifact = joblib.load(MODEL_PATH)

model = model_artifact["model"]
categorical = model_artifact["categorical_features"]
numerical = model_artifact["numerical_features"]
categorical_categories = model_artifact["categorical_categories"]

feature_order = model.feature_name_


# Create FastAPI application
app = FastAPI(title="Online Shoppers Purchase Prediction")


# Input format
class ShopperData(BaseModel):
    administrative: int
    administrative_duration: float
    informational: int
    informational_duration: float
    productrelated: int
    productrelated_duration: float
    bouncerates: float
    exitrates: float
    pagevalues: float
    specialday: float

    month: str
    operatingsystems: int
    browser: int
    region: int
    traffictype: int
    visitortype: str
    weekend: bool


@app.get("/")
def home():
    return {"message": "Online Shoppers Prediction API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(data: ShopperData):

    input_df = pd.DataFrame([data.model_dump()])

    # Validate categorical values
    for col in categorical:
        if input_df[col].iloc[0] not in categorical_categories[col]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid value for {col}"
            )

        input_df[col] = pd.Categorical(
            input_df[col],
            categories=categorical_categories[col]
        )

    # Keep the same feature order used during training
    input_df = input_df[feature_order]

    probability = float(model.predict_proba(input_df)[0, 1])

    prediction = probability >= 0.5

    return {
        "revenue_probability": probability,
        "revenue_prediction": bool(prediction)
    }
