# src/train.py
# ─────────────────────────────────────────────────────────────────
# Production training script.
# Run this file to train the model.
# Every run is automatically logged to MLflow:
# parameters, metrics, and the trained XGBoost model.
# ─────────────────────────────────────────────────────────────────

import os
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)

from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    recall_score,
    precision_score
)

from xgboost import XGBClassifier


# Import preprocessing functions
from preprocess import (
    load_data,
    clean_data,
    encode_data,
    engineer_features,
    get_features_and_target
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

MLFLOW_EXPERIMENT = "telco-churn-prediction"

MODEL_NAME = "telco-churn-xgboost"

RANDOM_STATE = 42

TEST_SIZE = 0.2


# Best hyperparameters from GridSearchCV
PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
    "eval_metric": "logloss",
    "verbosity": 0,
}


# ================================================================
# CLASS IMBALANCE
# ================================================================

def compute_scale_pos_weight(y_train):
    """
    XGBoost uses scale_pos_weight to handle class imbalance.

    Formula:
        number of non-churn customers
        -----------------------------
        number of churn customers
    """

    n_stayed = (y_train == 0).sum()
    n_churned = (y_train == 1).sum()

    return round(
        n_stayed / n_churned,
        4
    )


# ================================================================
# TRAINING PIPELINE
# ================================================================

def train():

    print("=" * 60)
    print("Telco Churn MLOps — Training Pipeline")
    print("=" * 60)


    # ============================================================
    # 1. LOAD AND PREPARE DATA
    # ============================================================

    print("\n[1/5] Loading and preparing data...")


    df = load_data(DATA_PATH)

    df = clean_data(df)

    df = encode_data(df)

    df = engineer_features(df)

    X, y = get_features_and_target(df)


    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )


    print(
        f"    Train size: {X_train.shape[0]} "
        f"| Test size: {X_test.shape[0]}"
    )


    # ============================================================
    # 2. SET UP MLFLOW
    # ============================================================

    print("\n[2/5] Setting up MLflow experiment...")


    mlflow.set_experiment(
        MLFLOW_EXPERIMENT
    )


    # ============================================================
    # 3. TRAIN MODEL INSIDE MLFLOW RUN
    # ============================================================

    print(
        "\n[3/5] Training model and logging to MLflow..."
    )


    with mlflow.start_run(
        run_name="xgboost-tuned"
    ) as run:


        # --------------------------------------------------------
        # Compute scale_pos_weight
        # --------------------------------------------------------

        spw = compute_scale_pos_weight(
            y_train
        )


        # Create a copy so the global PARAMS dictionary
        # is not modified permanently
        model_params = PARAMS.copy()

        model_params[
            "scale_pos_weight"
        ] = spw


        # --------------------------------------------------------
        # LOG PARAMETERS
        # --------------------------------------------------------

        mlflow.log_params(
            model_params
        )


        mlflow.log_param(
            "test_size",
            TEST_SIZE
        )


        mlflow.log_param(
            "train_rows",
            X_train.shape[0]
        )


        mlflow.log_param(
            "test_rows",
            X_test.shape[0]
        )


        mlflow.log_param(
            "n_features",
            X_train.shape[1]
        )


        # --------------------------------------------------------
        # TRAIN MODEL
        # --------------------------------------------------------

        model = XGBClassifier(
            **model_params
        )


        model.fit(
            X_train,
            y_train
        )


        # ========================================================
        # 4. EVALUATE MODEL
        # ========================================================

        print(
            "\n[4/5] Evaluating model..."
        )


        # Probability of churn
        y_pred_proba = (
            model.predict_proba(X_test)[:, 1]
        )


        # Predicted class
        y_pred = model.predict(
            X_test
        )


        # --------------------------------------------------------
        # Test metrics
        # --------------------------------------------------------

        auc = round(
            roc_auc_score(
                y_test,
                y_pred_proba
            ),
            4
        )


        recall = round(
            recall_score(
                y_test,
                y_pred
            ),
            4
        )


        precision = round(
            precision_score(
                y_test,
                y_pred
            ),
            4
        )


        f1 = round(
            f1_score(
                y_test,
                y_pred
            ),
            4
        )


        # --------------------------------------------------------
        # 5-fold cross-validation
        # --------------------------------------------------------

        cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=RANDOM_STATE
        )


        cv_scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1
        )


        cv_auc_mean = round(
            cv_scores.mean(),
            4
        )


        cv_auc_std = round(
            cv_scores.std(),
            4
        )


        # --------------------------------------------------------
        # LOG METRICS TO MLFLOW
        # --------------------------------------------------------

        mlflow.log_metric(
            "test_auc",
            auc
        )


        mlflow.log_metric(
            "churn_recall",
            recall
        )


        mlflow.log_metric(
            "churn_precision",
            precision
        )


        mlflow.log_metric(
            "churn_f1",
            f1
        )


        mlflow.log_metric(
            "cv_auc_mean",
            cv_auc_mean
        )


        mlflow.log_metric(
            "cv_auc_std",
            cv_auc_std
        )


        # --------------------------------------------------------
        # PRINT METRICS
        # --------------------------------------------------------

        print(
            f"    AUC:         {auc}"
        )

        print(
            f"    Recall:      {recall}"
        )

        print(
            f"    Precision:   {precision}"
        )

        print(
            f"    F1:          {f1}"
        )

        print(
            f"    CV AUC:      "
            f"{cv_auc_mean} "
            f"(+/- {cv_auc_std})"
        )


        # --------------------------------------------------------
        # LOG FEATURE NAMES
        # --------------------------------------------------------

        feature_names = list(
            X_train.columns
        )


        mlflow.log_param(
            "feature_names",
            str(feature_names)
        )


        # ========================================================
        # LOG XGBOOST MODEL
        # ========================================================

        print(
            "\n    Logging XGBoost model to MLflow..."
        )


        mlflow.xgboost.log_model(
            xgb_model=model,
            name="model",
            registered_model_name=MODEL_NAME,
            input_example=X_train.iloc[:3]
        )

        
        # --------------------------------------------------------
        # SAVE MODEL AND FEATURE NAMES LOCALLY
        # This is what the API loads — it cannot rely on the
        # MLflow server being available inside Docker/Lambda.
        # --------------------------------------------------------

        import json

        model_dir = os.path.join(
            os.path.dirname(__file__), "..", "model"
        )
        os.makedirs(model_dir, exist_ok=True)

        # Save XGBoost model as a portable JSON file
        model.save_model(
            os.path.join(model_dir, "xgboost_model.json")
        )

        # Save the exact feature names and order used in training
        # The API uses this to align incoming data to model input
        with open(os.path.join(model_dir, "feature_names.json"), "w") as f:
            json.dump(feature_names, f, indent=2)

        print(
            "    Model saved to: model/xgboost_model.json"
        )
        print(
            "    Feature names saved to: model/feature_names.json"
        )

        # --------------------------------------------------------
        # RUN INFORMATION
        # --------------------------------------------------------

        run_id = run.info.run_id


        print(
            "\n[5/5] Run complete."
        )


        print(
            f"    MLflow Run ID: {run_id}"
        )


        print(
            f"    Experiment: {MLFLOW_EXPERIMENT}"
        )


        print(
            f"    Registered model: {MODEL_NAME}"
        )


    # ============================================================
    # FINISHED
    # ============================================================

    print()

    print(
        "To view the experiment in MLflow, run:"
    )

    print()

    print(
        "    mlflow ui"
    )

    print()

    print(
        "Then open:"
    )

    print(
        "    http://127.0.0.1:5000"
    )

    print()

    print("=" * 60)


# ================================================================
# RUN SCRIPT
# ================================================================

if __name__ == "__main__":
    train()