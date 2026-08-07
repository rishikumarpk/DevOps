import json
import os

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")

app = FastAPI(
    title="Boston Housing Price Prediction API",
    description="Predicts median home value (in $1000s) from housing features.",
    version="1.0.0"
)

model = joblib.load(os.path.join(MODEL_DIR, "model.joblib"))

with open(os.path.join(MODEL_DIR, "feature_order.json")) as f:
    FEATURE_ORDER = json.load(f)


class PredictionRequest(BaseModel):
    features: list[float] = Field(
        ...,
        description=f"List of {len(FEATURE_ORDER)} feature values in order: {FEATURE_ORDER}"
    )


class PredictionResponse(BaseModel):
    predicted_value: float
    unit: str = "$1000s"


@app.get("/")
def home():
    return {
        "message": "Boston Housing Price Prediction API",
        "expected_features": FEATURE_ORDER,
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if len(request.features) != len(FEATURE_ORDER):
        raise HTTPException(
            status_code=400,
            detail=f"Expected {len(FEATURE_ORDER)} features ({FEATURE_ORDER}), "
                   f"got {len(request.features)}."
        )

    prediction = model.predict([request.features])[0]

    return PredictionResponse(predicted_value=round(float(prediction), 2))
