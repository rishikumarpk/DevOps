"""
Trains the champion regression model (XGBoost, tuned) on the Boston Housing
dataset and saves it as a model artifact used by the FastAPI app.

Run this once locally before building the Docker image:
    python app/train.py
"""
import json
import pandas as pd
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "boston.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")

def main():
    df = pd.read_csv(DATA_PATH)

    feature_columns = [c for c in df.columns if c != "medv"]
    X = df[feature_columns]
    y = df["medv"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        random_state=42
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    r2 = float(r2_score(y_test, preds))
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test R2  : {r2:.4f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODEL_DIR, "model.joblib"))

    with open(os.path.join(MODEL_DIR, "feature_order.json"), "w") as f:
        json.dump(feature_columns, f, indent=2)

    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump({"rmse": rmse, "r2": r2}, f, indent=2)

    print("Saved model.joblib, feature_order.json, metrics.json to /model")

if __name__ == "__main__":
    main()
