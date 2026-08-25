# api/app.py

import os
import sys
import json

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from mangum import Mangum


# ================================================================
# PATHS
# ================================================================

SRC_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "src"
)

sys.path.append(SRC_DIR)

MODEL_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "model"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "xgboost_model.json"
)

FEATURE_NAMES_PATH = os.path.join(
    MODEL_DIR,
    "feature_names.json"
)


# ================================================================
# LOAD ONLY SMALL FEATURE-NAMES FILE
# ================================================================

print("Starting Telco Churn API...")

with open(FEATURE_NAMES_PATH, "r") as f:
    FEATURE_NAMES = json.load(f)

print(f"Feature names loaded: {len(FEATURE_NAMES)}")


# ================================================================
# LAZY MODEL VARIABLE
# ================================================================

model = None


def get_model():
    """
    Import and load XGBoost only when /predict is called.
    """

    global model

    if model is None:

        print("Loading XGBoost model...")

        import xgboost as xgb

        model = xgb.XGBClassifier()

        model.load_model(
            MODEL_PATH
        )

        print("XGBoost model loaded.")

    return model


# ================================================================
# FASTAPI APPLICATION
# ================================================================

app = FastAPI(
    title="Telco Churn Prediction API",
    description="Predict telecom customer churn.",
    version="1.0.0"
)


# ================================================================
# INPUT SCHEMA
# ================================================================

class CustomerInput(BaseModel):

    gender: str = Field(..., example="Male")
    SeniorCitizen: int = Field(..., example=0)
    Partner: str = Field(..., example="Yes")
    Dependents: str = Field(..., example="No")
    tenure: int = Field(..., example=4)

    PhoneService: str = Field(..., example="Yes")
    MultipleLines: str = Field(..., example="No")

    InternetService: str = Field(
        ...,
        example="Fiber optic"
    )

    OnlineSecurity: str = Field(..., example="No")
    OnlineBackup: str = Field(..., example="No")
    DeviceProtection: str = Field(..., example="No")
    TechSupport: str = Field(..., example="No")

    StreamingTV: str = Field(..., example="No")
    StreamingMovies: str = Field(..., example="No")

    Contract: str = Field(
        ...,
        example="Month-to-month"
    )

    PaperlessBilling: str = Field(
        ...,
        example="Yes"
    )

    PaymentMethod: str = Field(
        ...,
        example="Electronic check"
    )

    MonthlyCharges: float = Field(
        ...,
        example=91.20
    )

    TotalCharges: float = Field(
        ...,
        example=364.80
    )


# ================================================================
# OUTPUT SCHEMA
# ================================================================

class PredictionOutput(BaseModel):

    churn_probability: float
    risk_tier: str
    recommended_action: str
    model_version: str = "1.0.0"


# ================================================================
# PREPROCESSING — LAZY IMPORT
# ================================================================

def preprocess_for_api(data: dict):
    """
    pandas and preprocess.py are imported only when prediction
    is actually requested.

    They are NOT imported during Lambda cold startup.
    """

    import pandas as pd

    from preprocess import (
        encode_data,
        engineer_features
    )

    df = pd.DataFrame([data])

    df = encode_data(df)

    df = engineer_features(df)

    if "Churn" in df.columns:
        df = df.drop(
            columns=["Churn"]
        )

    df = df.reindex(
        columns=FEATURE_NAMES,
        fill_value=0
    )

    return df


# ================================================================
# RISK TIER
# ================================================================

def assign_risk_tier(probability: float):

    if probability >= 0.65:

        return (
            "High Risk",
            "Priority call within 48 hours — "
            "offer annual plan upgrade"
        )

    elif probability >= 0.40:

        return (
            "Medium Risk",
            "Email campaign + follow-up call "
            "within 2 weeks"
        )

    else:

        return (
            "Low Risk",
            "Automated monthly engagement — "
            "no immediate action needed"
        )


# ================================================================
# ROOT
# ================================================================

@app.get("/")
def root():

    return {
        "message":
            "Telco Churn Prediction API is running.",
        "health": "/health",
        "predict": "/predict"
    }


# ================================================================
# HEALTH CHECK
# ================================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_file_available":
            os.path.exists(MODEL_PATH),
        "model_loaded":
            model is not None,
        "n_features":
            len(FEATURE_NAMES)
    }


# ================================================================
# PREDICTION
# ================================================================

@app.post(
    "/predict",
    response_model=PredictionOutput
)
def predict(customer: CustomerInput):

    try:

        # Load model only now
        current_model = get_model()

        customer_dict = (
            customer.model_dump()
        )

        # Load pandas/preprocessing only now
        X = preprocess_for_api(
            customer_dict
        )

        churn_prob = float(
            current_model.predict_proba(X)[0][1]
        )

        churn_prob = round(
            churn_prob,
            4
        )

        risk_tier, action = assign_risk_tier(
            churn_prob
        )

        return PredictionOutput(
            churn_probability=churn_prob,
            risk_tier=risk_tier,
            recommended_action=action
        )

    except Exception as e:

        print(
            f"Prediction error: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


# ================================================================
# AWS LAMBDA HANDLER
# ================================================================

handler = Mangum(
    app,
    lifespan="off",
    api_gateway_base_path="/prod"
)