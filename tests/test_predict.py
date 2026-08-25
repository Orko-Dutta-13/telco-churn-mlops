# tests/test_predict.py
# ─────────────────────────────────────────────────────────────────
# Unit tests for the Telco Churn preprocessing pipeline.
#
# GitHub Actions will run these tests before deployment.
# If a test fails, deployment should stop.
# ─────────────────────────────────────────────────────────────────

import os
import sys

import pandas as pd
import pytest


# ================================================================
# ADD src/ TO PYTHON PATH
# ================================================================

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "src"
    )
)


from preprocess import (
    clean_data,
    encode_data,
    engineer_features,
    get_features_and_target,
)


# ================================================================
# FIXTURE
# ================================================================

@pytest.fixture
def sample_raw_row():
    """
    A single raw Telco customer before preprocessing.
    """

    return pd.DataFrame([{
        "customerID":       "TEST-001",
        "gender":           "Male",
        "SeniorCitizen":    0,
        "Partner":          "Yes",
        "Dependents":       "No",
        "tenure":           4,
        "PhoneService":     "Yes",
        "MultipleLines":    "No",
        "InternetService":  "Fiber optic",
        "OnlineSecurity":   "No",
        "OnlineBackup":     "No",
        "DeviceProtection": "No",
        "TechSupport":      "No",
        "StreamingTV":      "No",
        "StreamingMovies":  "No",
        "Contract":         "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod":    "Electronic check",
        "MonthlyCharges":   91.20,
        "TotalCharges":     "364.80",
        "Churn":            "Yes",
    }])


# ================================================================
# TEST 1
# customerID SHOULD BE REMOVED
# ================================================================

def test_clean_data_removes_customer_id(sample_raw_row):

    cleaned = clean_data(
        sample_raw_row.copy()
    )

    assert "customerID" not in cleaned.columns


# ================================================================
# TEST 2
# TotalCharges SHOULD BECOME NUMERIC
# ================================================================

def test_clean_data_converts_total_charges(sample_raw_row):

    cleaned = clean_data(
        sample_raw_row.copy()
    )

    assert pd.api.types.is_numeric_dtype(
        cleaned["TotalCharges"]
    )


# ================================================================
# TEST 3
# CHURN TARGET SHOULD BECOME 0 OR 1
# ================================================================

def test_clean_data_encodes_churn_target(sample_raw_row):

    cleaned = clean_data(
        sample_raw_row.copy()
    )

    assert cleaned["Churn"].isin(
        [0, 1]
    ).all()


# ================================================================
# TEST 4
# ENCODED DATA SHOULD BE NUMERIC
# ================================================================

def test_encode_data_produces_numeric_columns(sample_raw_row):

    cleaned = clean_data(
        sample_raw_row.copy()
    )

    encoded = encode_data(
        cleaned
    )

    non_numeric = [
        column
        for column in encoded.columns
        if not pd.api.types.is_numeric_dtype(
            encoded[column]
        )
    ]

    assert len(non_numeric) == 0, (
        f"Non-numeric columns found after encoding: "
        f"{non_numeric}"
    )


# ================================================================
# TEST 5
# FEATURE ENGINEERING SHOULD ADD 5 FEATURES
# ================================================================

def test_engineer_features_adds_five_features(sample_raw_row):

    cleaned = clean_data(
        sample_raw_row.copy()
    )

    encoded = encode_data(
        cleaned
    )

    n_before = encoded.shape[1]

    engineered = engineer_features(
        encoded
    )

    n_after = engineered.shape[1]

    assert n_after == n_before + 5, (
        f"Expected 5 new features, "
        f"got {n_after - n_before}"
    )


# ================================================================
# TEST 6
# HIGH-RISK CUSTOMER SHOULD BE FLAGGED
# ================================================================

def test_engineer_features_high_risk_flag(sample_raw_row):

    cleaned = clean_data(
        sample_raw_row.copy()
    )

    encoded = encode_data(
        cleaned
    )

    engineered = engineer_features(
        encoded
    )

    assert (
        engineered["high_risk_flag"].iloc[0]
        == 1
    )


# ================================================================
# TEST 7
# FEATURES AND TARGET SHOULD BE SEPARATED
# ================================================================

def test_get_features_and_target_separates_correctly(
    sample_raw_row
):

    cleaned = clean_data(
        sample_raw_row.copy()
    )

    encoded = encode_data(
        cleaned
    )

    engineered = engineer_features(
        encoded
    )

    X, y = get_features_and_target(
        engineered
    )

    assert "Churn" not in X.columns

    assert y.isin(
        [0, 1]
    ).all()