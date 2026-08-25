# Telco Churn MLOps Pipeline

![CI/CD](https://github.com/Orko-Dutta-13/telco-churn-mlops/actions/workflows/deploy.yml/badge.svg?branch=main)

> An end-to-end MLOps system that takes a churn prediction model from experiment to production — with a live REST API, automated CI/CD, data drift monitoring, and full experiment tracking.

**Live API endpoint:**
```text
https://zndftpq9ue.execute-api.us-east-1.amazonaws.com/prod/predict
```

---

## The Business Problem

The IBM Telco Customer Churn dataset contains roughly 26% churned customers. This project builds a production-oriented machine learning system that:

1. Predicts customer churn probability using XGBoost (test ROC AUC: 0.8288)
2. Serves predictions through a live REST API deployed on AWS
3. Monitors incoming data for drift so changes in feature distributions can be detected
4. Tests, builds, and redeploys the serving application automatically on every push to `main` via GitHub Actions

This is not just a notebook experiment. It is a deployable end-to-end MLOps project.

---

## System Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│                        Developer Push                           │
│                      (git push → main)                          │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    GitHub Actions CI/CD                         │
│   1. Run pytest unit tests  (blocks deploy if any fail)         │
│   2. Build Docker image     (Dockerfile.lambda)                 │
│   3. Push to Amazon ECR     (tagged with git SHA)               │
│   4. Update Lambda function                                    │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                   AWS API Gateway (HTTPS)                       │
│              /predict   /health   /                             │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│              AWS Lambda  (container image)                      │
│         FastAPI + Mangum + XGBoost model                        │
│         Memory: 2048 MB   Timeout: 30s                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Live API — Try It Now

**Health check:**
```bash
curl https://zndftpq9ue.execute-api.us-east-1.amazonaws.com/prod/health
```

**Predict churn for a single customer:**
```bash
curl -X POST \
  https://zndftpq9ue.execute-api.us-east-1.amazonaws.com/prod/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 4,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 91.20,
    "TotalCharges": 364.80
  }'
```

**Response:**
```json
{
  "churn_probability": 0.8354,
  "risk_tier": "High Risk",
  "recommended_action": "Priority call within 48 hours — offer annual plan upgrade",
  "model_version": "1.0.0"
}
```

Interactive API docs:

```text
https://zndftpq9ue.execute-api.us-east-1.amazonaws.com/prod/docs
```

---

## Model Performance

| Metric | Score |
|---|---:|
| ROC AUC | 0.8288 |
| Churn Recall | 0.7513 |
| Churn Precision | 0.5243 |
| Churn F1 | 0.6176 |
| CV ROC AUC (5-fold) | 0.8422 ± 0.0043 |

Model: XGBoost with `scale_pos_weight` for class imbalance. Training parameters, evaluation metrics, and model artifacts are tracked with MLflow.

---

## MLflow Experiment Tracking

Every training run is logged automatically with parameters, metrics, and the model artifact so experiments can be reproduced and compared.

```bash
mlflow ui
```

Then open:

```text
http://127.0.0.1:5000
```

Logged values include `n_estimators`, `learning_rate`, `max_depth`, `subsample`, `scale_pos_weight`, `test_auc`, `churn_recall`, `churn_precision`, `churn_f1`, `cv_auc_mean`, and `cv_auc_std`.

---

## Data Drift Monitoring

Evidently AI compares a reference dataset against current data and generates an HTML report showing distribution changes across model inputs.

```bash
python src/monitor.py
```

The report is saved to:

```text
reports/drift_report.html
```

The monitoring demo deliberately simulates drift in `MonthlyCharges`, `tenure`, and `charges_per_tenure`. Evidently also evaluates drift across the model input features. The project uses a demonstration retraining rule of more than 20% of features drifting.

---

## Feature Engineering

Five features are engineered on top of the cleaned Telco data:

| Feature | Description | Signal |
|---|---|---|
| `tenure_bucket` | Early / Mid / Loyal customer tenure grouping | Captures lifecycle stage |
| `service_count` | Number of subscribed add-on services | Measures customer service adoption |
| `charges_per_tenure` | Charges normalized by tenure | Highlights high spending relative to relationship length |
| `high_value_flag` | Flags high-value customers | Identifies customers with elevated spending |
| `high_risk_flag` | Combines month-to-month, fiber, and early-tenure risk signals | Captures a high-risk profile |

---

## Project Structure

```text
telco-churn-mlops/
│
├── src/
│   ├── preprocess.py          # Cleaning, encoding, and feature engineering
│   ├── train.py               # Training pipeline with MLflow tracking
│   └── monitor.py             # Evidently AI drift monitoring
│
├── api/
│   └── app.py                 # FastAPI serving endpoint + Mangum handler
│
├── model/
│   ├── xgboost_model.json     # Trained model in portable XGBoost JSON format
│   └── feature_names.json     # Exact model feature names and column order
│
├── tests/
│   ├── test_predict.py        # Unit tests used by CI
│   └── test_api_local.py      # Local API integration test
│
├── reports/
│   └── drift_report.html      # Generated Evidently report
│
├── .github/
│   └── workflows/
│       └── deploy.yml         # GitHub Actions CI/CD workflow
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── .dockerignore
├── .gitignore
├── Dockerfile                 # Local development image
├── Dockerfile.lambda          # AWS Lambda-compatible image
├── requirements.txt
├── requirements.lambda.txt
└── README.md
```

---

## How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Orko-Dutta-13/telco-churn-mlops.git
cd telco-churn-mlops
```

### 2. Create and activate the environment

```bash
conda create -n telco_mlops python=3.11
conda activate telco_mlops
pip install -r requirements.txt
```

### 3. Train the model

```bash
cd src
python train.py
```

### 4. Start the API

From the project root:

```bash
uvicorn api.app:app --reload --port 8000
```

Then open:

```text
http://127.0.0.1:8000/docs
```

### 5. Run tests

From the project root:

```bash
python -m pytest tests/test_predict.py -v
```

### 6. Run the drift monitor

From the project root:

```bash
python src/monitor.py
```

Then open:

```text
reports/drift_report.html
```

### 7. Run with Docker

```bash
docker build -t telco-churn-api .
docker run -p 8000:8000 telco-churn-api
```

---

## CI/CD Pipeline

Every push to `main` triggers the deployment workflow:

```text
push → tests → Docker build → ECR push → Lambda update → health check
```

The test job must pass before deployment begins. The deployment job builds the Lambda-compatible image, pushes commit-SHA and `latest` tags to Amazon ECR, updates the Lambda function, waits for the update to complete, and verifies the public health endpoint.

Pipeline status is available in the **Actions** tab on GitHub.

Required GitHub secrets:

```text
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_ACCOUNT_ID
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| ML Model | XGBoost |
| Experiment Tracking | MLflow |
| API Framework | FastAPI + Uvicorn |
| Serverless Adapter | Mangum |
| Model Monitoring | Evidently AI |
| Containerisation | Docker |
| Container Registry | Amazon ECR |
| Deployment | AWS Lambda (container image) |
| API Gateway | AWS API Gateway (HTTP API) |
| CI/CD | GitHub Actions |
| Testing | pytest |
| Data | IBM Telco Customer Churn — 7,032 cleaned customers |

---

## Dataset

The project uses the IBM Telco Customer Churn dataset. After cleaning, the training pipeline uses 7,032 customer records.

Source: [IBM Sample Data](https://www.ibm.com/communities/analytics/watson-analytics-blog/guide-to-sample-datasets/)

The dataset includes customer account information, services, contract type, internet service, monthly charges, tenure, payment method, demographic variables, and the churn target.

---

## Author

**Orko Dutta**

[GitHub](https://github.com/Orko-Dutta-13) · [LinkedIn](https://www.linkedin.com/in/orko-dutta/)
