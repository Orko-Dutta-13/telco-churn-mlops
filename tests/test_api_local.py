# tests/test_api_local.py
# Run this while the API server is running to test predictions.

import requests

# A high-risk customer: month-to-month, fiber optic, new tenure
high_risk_customer = {
    "gender":           "Male",
    "SeniorCitizen":    0,
    "Partner":          "No",
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
    "TotalCharges":     364.80
}

# A low-risk customer: two-year contract, long tenure
low_risk_customer = {
    "gender":           "Female",
    "SeniorCitizen":    0,
    "Partner":          "Yes",
    "Dependents":       "Yes",
    "tenure":           58,
    "PhoneService":     "Yes",
    "MultipleLines":    "Yes",
    "InternetService":  "DSL",
    "OnlineSecurity":   "Yes",
    "OnlineBackup":     "Yes",
    "DeviceProtection": "Yes",
    "TechSupport":      "Yes",
    "StreamingTV":      "Yes",
    "StreamingMovies":  "Yes",
    "Contract":         "Two year",
    "PaperlessBilling": "No",
    "PaymentMethod":    "Bank transfer (automatic)",
    "MonthlyCharges":   79.50,
    "TotalCharges":     4611.00
}

for label, customer in [("HIGH RISK", high_risk_customer), ("LOW RISK", low_risk_customer)]:
    response = requests.post(
        "http://127.0.0.1:8000/predict",
        json=customer
    )
    result = response.json()
    print(f"\n--- {label} CUSTOMER ---")
    print(f"Churn Probability : {result['churn_probability']:.1%}")
    print(f"Risk Tier         : {result['risk_tier']}")
    print(f"Action            : {result['recommended_action']}")