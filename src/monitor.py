# src/monitor.py
# ─────────────────────────────────────────────────────────────────
# Data drift monitoring using Evidently AI.
#
# Reference data:
#   Data representing what the model originally learned from.
#
# Current data:
#   New incoming production-style data.
#
# For demonstration, we simulate drift by deliberately changing
# several important customer features.
# ─────────────────────────────────────────────────────────────────

import os
import sys

import numpy as np
import pandas as pd

# ================================================================
# EVIDENTLY 0.7.x IMPORTS
# ================================================================

from evidently import Report
from evidently.presets import (
    DataDriftPreset,
    DataSummaryPreset,
)


# ================================================================
# IMPORT PROJECT PREPROCESSING
# ================================================================

sys.path.append(
    os.path.dirname(__file__)
)

from preprocess import (
    load_data,
    clean_data,
    encode_data,
    engineer_features,
    get_features_and_target,
)


# ================================================================
# CONFIGURATION
# ================================================================

DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

REPORTS_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "reports"
)

RANDOM_STATE = 42


# ================================================================
# LOAD AND PREPARE DATA
# ================================================================

def prepare_data():
    """
    Load and preprocess the complete Telco Churn dataset.

    Returns only model features.
    The churn target is not needed for data-drift analysis.
    """

    df = load_data(DATA_PATH)

    df = clean_data(df)

    df = encode_data(df)

    df = engineer_features(df)

    X, _ = get_features_and_target(df)

    return X


# ================================================================
# SIMULATE DATA DRIFT
# ================================================================

def simulate_drift(X: pd.DataFrame) -> pd.DataFrame:
    """
    Artificially create distribution drift.

    Simulated changes:

    MonthlyCharges:
        +15% increase plus small random noise

    charges_per_tenure:
        +20% increase

    tenure:
        -25% decrease

    In a real production system this function would NOT exist.
    Instead, 'current' would contain recent incoming customer data.
    """

    X_drifted = X.copy()

    np.random.seed(RANDOM_STATE)


    # ------------------------------------------------------------
    # MonthlyCharges drift
    # ------------------------------------------------------------

    if "MonthlyCharges" in X_drifted.columns:

        X_drifted["MonthlyCharges"] = (
            X_drifted["MonthlyCharges"] * 1.15
            + np.random.normal(
                0,
                3,
                len(X_drifted)
            )
        )


    # ------------------------------------------------------------
    # Charges per tenure drift
    # ------------------------------------------------------------

    if "charges_per_tenure" in X_drifted.columns:

        X_drifted["charges_per_tenure"] = (
            X_drifted["charges_per_tenure"] * 1.20
        )


    # ------------------------------------------------------------
    # Tenure drift
    # ------------------------------------------------------------

    if "tenure" in X_drifted.columns:

        X_drifted["tenure"] = (
            X_drifted["tenure"] * 0.75
        ).clip(lower=0)


    return X_drifted


# ================================================================
# RUN EVIDENTLY DRIFT REPORT
# ================================================================

def run_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame
) -> str:
    """
    Generate an Evidently HTML report comparing reference
    and current data.

    DataDriftPreset:
        Detects distribution changes across features and provides
        per-column drift visualizations.

    DataSummaryPreset:
        Provides descriptive statistics and data-quality information.
    """

    print("\n    Building Evidently report...")


    # ------------------------------------------------------------
    # Create Evidently report
    # ------------------------------------------------------------

    report = Report([
        DataDriftPreset(),
        DataSummaryPreset(),
    ])


    # ------------------------------------------------------------
    # Run Evidently report
    # ------------------------------------------------------------

    result = report.run(
        current_data=current,
        reference_data=reference
    )


    # ------------------------------------------------------------
    # Create reports folder if needed
    # ------------------------------------------------------------

    os.makedirs(
        REPORTS_DIR,
        exist_ok=True
    )


    # ------------------------------------------------------------
    # Set report path
    # ------------------------------------------------------------

    report_path = os.path.join(
        REPORTS_DIR,
        "drift_report.html"
    )


    # ------------------------------------------------------------
    # Save HTML report
    # ------------------------------------------------------------

    result.save_html(
        report_path
    )


    return os.path.abspath(report_path)


# ================================================================
# MAIN MONITORING PIPELINE
# ================================================================

def monitor():

    print("=" * 60)
    print("Telco Churn MLOps — Data Drift Monitor")
    print("=" * 60)


    # ============================================================
    # STEP 1 — LOAD DATA
    # ============================================================

    print(
        "\n[1/3] Loading and preparing data..."
    )

    X = prepare_data()

    print(
        f"    Total rows: {len(X)}"
    )

    print(
        f"    Total features: {X.shape[1]}"
    )


    # ============================================================
    # STEP 2 — REFERENCE VS CURRENT
    # ============================================================

    print(
        "\n[2/3] Splitting reference vs current data..."
    )


    # First 80% = reference distribution
    # Last 20% = simulated current production data

    split_idx = int(
        len(X) * 0.8
    )


    reference = (
        X.iloc[:split_idx]
        .copy()
        .reset_index(drop=True)
    )


    current_raw = (
        X.iloc[split_idx:]
        .copy()
        .reset_index(drop=True)
    )


    # Apply simulated drift
    current_drifted = simulate_drift(
        current_raw
    )


    print(
        f"    Reference rows : {len(reference)}"
    )

    print(
        f"    Current rows   : {len(current_drifted)}"
    )

    print()

    print(
        "    Drift simulation applied to:"
    )

    print(
        "      MonthlyCharges      : +15%"
    )

    print(
        "      charges_per_tenure  : +20%"
    )

    print(
        "      tenure              : -25%"
    )


    # ============================================================
    # STEP 3 — GENERATE REPORT
    # ============================================================

    print(
        "\n[3/3] Generating Evidently drift report..."
    )


    report_path = run_drift_report(
        reference=reference,
        current=current_drifted
    )


    print()

    print(
        "    Report saved to:"
    )

    print(
        f"    {report_path}"
    )


    # ============================================================
    # WHAT TO LOOK FOR
    # ============================================================

    print()

    print("=" * 60)
    print("What to look for in the report:")
    print("=" * 60)

    print(
        "  1. Dataset-level drift"
    )

    print(
        "     Check whether the overall feature distribution changed."
    )

    print()

    print(
        "  2. Drifted features"
    )

    print(
        "     Evidently compares every feature between reference"
    )

    print(
        "     and current data."
    )

    print()

    print(
        "  3. Pay particular attention to:"
    )

    print(
        "       - MonthlyCharges"
    )

    print(
        "       - charges_per_tenure"
    )

    print(
        "       - tenure"
    )

    print()

    print(
        "     These should show strong drift because we deliberately"
    )

    print(
        "     changed their distributions."
    )

    print()

    print(
        "  4. Data summary"
    )

    print(
        "     Review distributions, descriptive statistics,"
    )

    print(
        "     and missing-value information."
    )

    print("=" * 60)


    # ============================================================
    # DRIFT INTERPRETATION & RECOMMENDED ACTIONS
    # ============================================================

    print("\n" + "=" * 60)
    print("Drift Interpretation & Recommended Actions")
    print("=" * 60)

    print("""
  MonthlyCharges drifted (+15%):
    Customers are now paying more on average.
    Model may underestimate churn for high-charge segments.
    Action: Retrain with recent 3 months of data.

  tenure drifted (-25%):
    More short-tenure customers in the system.
    High-risk early-churn segment is growing.
    Action: Increase outreach budget for < 12-month customers.

  charges_per_tenure drifted (+20%):
    New customers paying more per month than historical average.
    Combined with short tenure — this is the highest-risk profile.
    Action: Trigger immediate retention review for this segment.

  Recommended retraining threshold: > 20% features drifted.
  Current status: RETRAIN RECOMMENDED.
    """)

    print("=" * 60)


# ================================================================
# RUN
# ================================================================

if __name__ == "__main__":
    monitor()