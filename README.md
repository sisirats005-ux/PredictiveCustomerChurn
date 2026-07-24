# ConnectTel Customer Churn Prediction Pipeline

This repository contains a modular machine learning and MLOps engineering pipeline designed to predict customer churn for ConnectTel, a telecommunications company. The system features modular data preprocessing, advanced feature engineering, hyperparameter tuning, model explainability (SHAP), a containerized REST API, interactive diagnostics (Streamlit), and an automated CI/CD validation workflow.

---

## 1. Project Objective & Scope

Customer attrition (churn) directly impacts recurring revenue and increases replacement marketing costs. The objective of this project is to build an empirical, end-to-end classification system that:
1. **Predicts Churn Risks**: Identifies at-risk accounts with high sensitivity (Recall).
2. **Identifies Key Drivers**: Explains prediction drivers using SHAP feature attributions.
3. **Operationalizes Inference**: Serves real-time risk predictions via a containerized REST API and interactive dashboard.
4. **Calculates ROI**: Models the financial benefit of targeting interventions at predicted churners.

---

## 2. Dataset Description

The dataset consists of **7,043 rows** and **21 columns** representing ConnectTel customer profiles.
- **Source**: Industry standard Telco Customer Churn dataset (`data/WA_Fn-UseC_-Telco-Customer-Churn.csv`).
- **Target Label**: `Churn` (Yes/No, imbalanced with a ~26.6% positive class rate).

### Key Features
- **Account Data**: `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`.
- **Demographics**: `gender`, `SeniorCitizen`, `Partner`, `Dependents`.
- **Services**: `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`.

---

## 3. Directory Structure

```
PredictiveCustomerChurn/
│
├── .github/
│   └── workflows/
│       └── ci.yml                     # CI/CD GitHub Actions workflow (linter + tests)
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Raw customer dataset
├── notebooks/
│   └── churn_analysis.ipynb           # Generated Jupyter Notebook report
├── models/
│   ├── registry/                      # Versioned model artifacts folder
│   │   └── v1/
│   │       ├── best_model.joblib
│   │       ├── preprocessing_pipeline.joblib
│   │       └── metadata.json
│   ├── best_model.joblib              # Serialized champion model (Tuned XGBoost)
│   ├── preprocessing_pipeline.joblib  # Serialized ColumnTransformer preprocessing object
│   └── feature_names.joblib           # Serialized feature list for deployment
├── src/
│   ├── __init__.py
│   ├── preprocessing.py               # Data ingestion, cleaning, splitting, and scaling
│   ├── feature_engineering.py         # Domain-specific advanced feature extraction
│   ├── hypothesis_testing.py          # Chi-Square / Welch's t-test statistical checks
│   ├── train.py                       # Baseline training and GridSearchCV tuning
│   ├── evaluate.py                    # Classification metrics and evaluation plotting
│   ├── explain.py                     # SHAP model explainability plots
│   ├── predict.py                     # Single-customer inference logic
│   ├── experiment_tracker.py          # JSON registry and MLflow experiment logging
│   ├── utils.py                       # Logging and directory helpers
│   └── generate_notebook.py           # Script to compile churn_analysis.ipynb
├── tests/
│   ├── conftest.py                    # Shared pytest mock customer fixtures
│   ├── test_preprocessing.py          # Preprocessing unit tests
│   ├── test_feature_engineering.py    # Feature engineering unit tests
│   ├── test_predict.py                # Inference path unit tests
│   └── test_api.py                    # FastAPI integration tests
├── outputs/
│   ├── plots/                         # Evaluation, SHAP, and EDA charts
│   └── metrics/
│       ├── model_comparison.csv       # Evaluation metrics table
│       ├── experiment_runs.json       # Historical experiment execution logs
│       └── hypothesis_tests.json      # Statistical significance p-values
├── reports/
│   ├── pipeline.log                   # Active pipeline run logs
│   └── business_insight_report.md     # Retention recommendations and ROI calculations
├── config.yaml                        # Central configuration file (hyperparameters, paths)
├── app.py                             # Streamlit web interface
├── api.py                             # FastAPI REST API deployment
├── Dockerfile                         # Containerization config for FastAPI app
├── pytest.ini                         # Pytest configuration
├── requirements.txt                   # Stable dependencies
└── main.py                            # End-to-end execution script
```

---

## 4. Advanced Feature Engineering

To improve classification performance, the pipeline constructs several advanced features from raw inputs:
1. **Interaction Features**:
   - `Fiber_x_MonthToMonth`: Identifies customers on month-to-month contracts subscribing to Fiber Optic services.
   - `Fiber_x_NoTechSupport`: Identifies Fiber users without technical support access.
   - `MonthToMonth_x_NoAutoPay`: Identifies month-to-month contracts paying via manual payment channels.
   - `HighRiskContractPay`: Intersects Month-to-Month contracts with Electronic Check payments.
2. **Customer Lifetime Value (CLV)**: Estimated value modeled as:
   \(\text{CLV} = \text{MonthlyCharges} \times (\text{tenure} + \text{ContractBonus})\)
   *(ContractBonus: Month-to-month = 6 months, 1-Year = 12 months, 2-Year = 24 months)*.
3. **Payment & Billing Risk**:
   - `ChargesRatio`: Density of monthly billing calculated as \(\text{MonthlyCharges} / (\text{TotalCharges} + \text{MonthlyCharges})\).
   - `BillingRisk`: Measures monthly cash-collection friction: \(\text{MonthlyCharges} \times (1 - \text{AutoPayment}) \times \text{IsMonthToMonth}\).
4. **Multi-Factor Risk Segmentation**:
   - `HighRiskProfile`: Flagged if an account has at least 2 of (Month-to-month contract, manual Electronic Check payment, Fiber Optic internet).
5. **Customer Personas**:
   - `Persona_New_HighSpend`: Tenure \(\le 12\) months and Monthly Charges > $70.
   - `Persona_Loyal_HighSpend`: Tenure > 12 months and Monthly Charges > $70.

---

## 5. Production & MLOps Infrastructure

This project contains several components designed for deployment in production environments:

### Central Configuration (`config.yaml`)
All operational variables, paths, train/test split fractions, GridSearchCV tuning parameters, and business model metrics are housed in a single configuration file.

### REST API Service (`api.py`)
Exposes the predictive model as a REST API built with **FastAPI**.
- **Endpoint `GET /health`**: Health status and model load checks.
- **Endpoint `POST /predict`**: Accepts customer JSON structures, validates inputs via **Pydantic**, and returns:
  - Churn probability.
  - Churn prediction.
  - Risk tier (Low, Medium, High).
  - Target retention recommendations.

### Containerization (`Dockerfile`)
Multi-stage builder configuration packaging the FastAPI application inside a lightweight `python:3.11-slim` runner.

### CI/CD Workflow (`.github/workflows/ci.yml`)
Runs Python syntax linter (`flake8`) and unit tests (`pytest`) automatically on every push or pull request to preserve build health.

### Experiment Tracking (`src/experiment_tracker.py`)
Tracks all training runs, parameters, and metrics in a persistent local registry (`outputs/metrics/experiment_runs.json`). It contains optional hook integrations to record runs inside **MLflow**.

---

## 6. Installation & Execution

### Setup Environment
Ensure you have Python 3.11+ installed:

```bash
# Navigate to project root
cd PredictiveCustomerChurn

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Pipeline
To clean data, run hypothesis testing, engineer features, tune models, serialize the champion model, and generate evaluation plots:

```bash
python main.py
```

### Serve the REST API
Start the FastAPI server locally:

```bash
python api.py
```
*API docs will be available at [http://localhost:8000/docs](http://localhost:8000/docs).*

### Launch the Streamlit Dashboard
Run the Streamlit interactive dashboard:

```bash
streamlit run app.py
```

### Run Tests
Execute the unit and integration test suite:

```bash
python -m pytest -v
```

---

## 7. Model Performance Results

Evaluation results obtained on a stratified 20% holdout test set:

| Model | Test Accuracy | Test Precision | Test Recall | Test F1-Score | Test ROC-AUC | Mean CV ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tuned XGBoost (Champion)** | **0.7495** | **0.5186** | **0.8128** | **0.6332** | **0.8493** | **0.8488** |
| Tuned Random Forest | 0.7686 | 0.5471 | 0.7620 | 0.6370 | 0.8459 | 0.8446 |
| Logistic Regression | 0.7410 | 0.5085 | 0.7995 | 0.6214 | 0.8451 | 0.8456 |
| Baseline XGBoost | 0.7630 | 0.5434 | 0.6845 | 0.6059 | 0.8267 | 0.8242 |
| Baseline Random Forest | 0.7892 | 0.6335 | 0.4973 | 0.5572 | 0.8281 | 0.8251 |
| Naive Majority-Class Baseline | 0.7346 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | -- |

### Selection Rationale
- We prioritized **Recall** (sensitivity) and **ROC-AUC** over accuracy because the cost of failing to catch a churner (customer loss + CAC re-acquisition) is higher than the cost of a promotional discount extended to a loyal customer.
- **Tuned XGBoost** achieves **81.3% Recall** and **0.849 Test ROC-AUC**, successfully identifying roughly 4 in 5 churners.
- **Model Selection Justification**: Although Logistic Regression achieved a slightly higher cross-validated ROC-AUC, Tuned XGBoost was selected because it achieved higher recall (81.3%), captured more potential churners, modeled complex non-linear interactions, and provided richer SHAP explanations. Since customer churn prediction prioritizes identifying at-risk customers, recall and business impact were considered alongside ROC-AUC.
