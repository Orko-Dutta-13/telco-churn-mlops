# src/preprocess.py
# ─────────────────────────────────────────────────────────────────
# All data cleaning and feature engineering in one place.
# Both the training script and the API import from here,
# so training and serving always process data identically.
# ─────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np


def load_data(filepath: str) -> pd.DataFrame:
    """Load raw CSV and return a DataFrame."""
    df = pd.read_csv(filepath)
    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix known data quality issues in the IBM Telco dataset.
    TotalCharges has hidden blank strings that look like valid entries.
    """
    df = df.copy()

    # Drop customerID — it is an identifier, not a feature
    df.drop(columns=["customerID"], inplace=True)

    # TotalCharges is stored as a string with hidden blank values
    # Convert to numeric and drop the ~11 rows that are blank
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(subset=["TotalCharges"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Convert target column: Yes → 1, No → 0
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    print(f"After cleaning: {df.shape[0]} rows remaining")
    return df


def encode_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical features.
    Binary columns → 0/1 directly.
    Multi-category columns → one-hot encoded.
    """
    df = df.copy()

    # Binary columns: replace Yes/No and Female/Male with 1/0
    binary_cols = [
        "Partner", "Dependents", "PhoneService", "PaperlessBilling",
        "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    for col in binary_cols:
        df[col] = df[col].map({"Yes": 1, "No": 0,
                                "No phone service": 0,
                                "No internet service": 0})

    df["gender"] = (df["gender"] == "Female").astype(int)
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)

    # Multi-category columns → one-hot encoding
    multi_cols = ["InternetService", "Contract", "PaymentMethod"]
    df = pd.get_dummies(df, columns=multi_cols, drop_first=False)

    # Convert any boolean columns from get_dummies to int
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create 5 new features that improve model performance.
    These were validated in the original project.
    """
    df = df.copy()

    # 1. tenure_bucket: group customers by loyalty stage
    def assign_tenure_bucket(tenure):
        if tenure <= 12:
            return 0   # Early — highest churn risk
        elif tenure <= 36:
            return 1   # Mid
        else:
            return 2   # Loyal — lowest churn risk

    df["tenure_bucket"] = df["tenure"].apply(assign_tenure_bucket)

    # 2. service_count: how many add-on services does this customer use?
    #    More services = more embedded = less likely to churn
    service_cols = [
        "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    df["service_count"] = df[service_cols].sum(axis=1)

    # 3. charges_per_tenure: monthly spend normalised by tenure
    #    High spend in early months = high risk signal
    df["charges_per_tenure"] = (
        df["MonthlyCharges"] / (df["tenure"] + 1)
    ).round(4)

    # 4. high_value_flag: customer pays above median monthly charge
    median_charge = df["MonthlyCharges"].median()
    df["high_value_flag"] = (df["MonthlyCharges"] > median_charge).astype(int)

    # 5. high_risk_flag: month-to-month + fiber optic + early tenure
    #    The three strongest churn predictors combined
    month_to_month = (
        df.get("Contract_Month-to-month", pd.Series(0, index=df.index))
    )
    fiber_optic = (
        df.get("InternetService_Fiber optic", pd.Series(0, index=df.index))
    )
    df["high_risk_flag"] = (
        (month_to_month == 1) &
        (fiber_optic == 1) &
        (df["tenure_bucket"] == 0)
    ).astype(int)

    print(f"Feature engineering complete. Total features: {df.shape[1]}")
    return df


def get_features_and_target(df: pd.DataFrame):
    """Split into feature matrix X and target vector y."""
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    return X, y