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
│   ├── best_model.joblib              # Evaluation champion (Tuned XGBoost) -- NOT served in production, see Section 7
│   ├── logistic_regression_model.joblib   # Deployed model -- what app.py / api.py actually score with
│   ├── logistic_regression_metadata.json  # Trained sklearn version + decision threshold, read by src/predict.py
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
├── .streamlit/
│   └── runtime.txt                    # Pins Python 3.11 for Streamlit Community Cloud (see Section 10)
├── config.yaml                        # Central configuration file (hyperparameters, paths)
├── app.py                             # Streamlit web interface
├── api.py                             # FastAPI REST API deployment
├── Dockerfile                         # Containerization config for FastAPI app
├── pytest.ini                         # Pytest configuration
├── requirements.txt                   # Runtime dependencies for app.py / api.py (see Section 9)
├── requirements-dev.txt               # Training/notebook/offline-explainability dependencies (see Section 9)
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
- **Tuned XGBoost** achieves **81.3% Recall** and **0.849 Test ROC-AUC**, successfully identifying roughly 4 in 5 churners, and is the table's evaluation champion.
- **Model Selection Justification**: Although Logistic Regression achieved a slightly higher cross-validated ROC-AUC, Tuned XGBoost was selected as the evaluation champion because it achieved higher recall (81.3%), captured more potential churners, modeled complex non-linear interactions, and provided richer SHAP explanations. Since customer churn prediction prioritizes identifying at-risk customers, recall and business impact were considered alongside ROC-AUC.

> **Evaluation champion vs. deployed model.** `models/best_model.joblib` (Tuned XGBoost) is the artifact produced by this evaluation, but the *deployed* model — the one `app.py`, `api.py`, and every prediction in this README actually run — is `models/logistic_regression_model.joblib`. Logistic Regression was chosen for production over the XGBoost champion because its linear coefficients drive the live, dependency-free "Explain AI" waterfall in `app.py` (`individual_shap_waterfall()`, computed directly from `model.coef_` — not a real-time SHAP call), giving every prediction an instant, exactly-reproducible explanation instead of requiring `shap`/`xgboost` at inference time. `src/predict.py` reads `models/logistic_regression_metadata.json` and hard-fails at startup if the installed scikit-learn version doesn't match what that model was trained with — see `_validate_runtime_compatibility()`.

---

## 8. Logging

Every runtime entrypoint (`app.py`, `api.py`, `src/predict.py`) shares one logging convention via `src/utils.py::setup_logger()`: each module gets its own named logger (`churn_project.app`, `churn_project.api`, `churn_project.predict`, ...) writing to both the console and `reports/pipeline.log`, at a level matched to what happened:
- `CRITICAL` — the service cannot serve predictions at all (model artifacts missing or unloadable at startup).
- `ERROR` — a scikit-learn version mismatch, or an unexpected failure while scoring a customer.
- `WARNING` — a recoverable fallback that silently changes behavior (e.g. the optimized decision threshold couldn't be read from metadata, so the app fell back to the naive 0.5 default).
- `INFO` — expected user-input validation failures, and successful artifact loads.

This means a production incident (for example: the Streamlit dashboard showing a generic "Inference failed" banner) always has a matching traceback in `reports/pipeline.log`, not just whatever text was shown in the browser at that moment.

## 9. Dependency Management: `requirements.txt` vs `requirements-dev.txt`

Runtime dependencies for the deployed services (`app.py` on Streamlit Community Cloud, `api.py` in the Docker container) live in `requirements.txt`. Both entrypoints resolve to `src/predict.py`, which loads only `models/logistic_regression_model.joblib` — so `requirements.txt` is intentionally scoped to what that inference path imports, and `scikit-learn` is pinned to the exact version the artifacts were trained with (see the callout in Section 7).

Training, offline SHAP/plot generation, and notebook tooling (`xgboost`, `shap`, `matplotlib`, `seaborn`, `mlflow`, `jupyter`) live in `requirements-dev.txt` instead, since none of it is imported by the deployed app or API. Install both for local development:

```bash
pip install -r requirements.txt -r requirements-dev.txt
```

## 10. Deployment Notes: Streamlit Community Cloud

`.streamlit/runtime.txt` pins `python-3.11` to match the Python version used by `Dockerfile` and `.github/workflows/ci.yml`, so all three environments run identical interpreter/dependency behavior. Streamlit Community Cloud has a known platform issue where `runtime.txt` is occasionally ignored in favor of a newer default Python — if a deploy log shows a Python version other than 3.11, explicitly select **Python 3.11** in the app's **Advanced settings** at deploy time (this can only be changed by deleting and redeploying the app, not edited in place afterward).

## 11. Known Limitations

- **Explain tab is linear-model-only.** The live per-customer explanation in `app.py` uses the deployed Logistic Regression's coefficients, not the offline SHAP analysis run against the XGBoost champion. The two are directionally consistent (see `reports/business_insight_report.md`) but are not numerically identical.
- **No batch/async endpoint.** `api.py`'s `POST /predict` scores one customer per request; the Executive Dashboard's portfolio-wide scoring is a Streamlit-only, in-process batch loop, not exposed over the REST API.
- **No drift monitoring in production.** See "Recommended Next Steps" below (Section 12) for planned follow-up work in this area.

---

## 12. Senior ML Engineering Review Notes

This repository has been reviewed as an end-to-end churn prediction system. The core design is strong for a portfolio/demo MLOps project: data cleaning, feature engineering, model comparison, explainability artifacts, API serving, dashboarding, containerization, CI, and tests are separated into clear modules. The latest hardening pass focused on production-facing reliability without changing the trained artifacts or existing model outputs.

### Architecture and Production Readiness Improvements
- **API contract hardening**: FastAPI now validates categorical fields with explicit enums instead of accepting arbitrary strings. This prevents silent one-hot `handle_unknown='ignore'` behavior from masking upstream data quality issues.
- **Inference explainability parity**: The REST response now exposes the decision threshold, top risk factors, and prioritized recommended action already produced by the inference layer, making the API output consistent with the Streamlit experience.
- **Feature engineering robustness**: Feature generation now validates required raw columns up front and guards `ChargesRatio` against `0 / 0` edge cases so scoring remains finite for zero-charge/new-customer records.
- **Container safety**: The Docker runtime now uses an unprivileged user and includes a `/health`-based health check, improving deployment observability and reducing container privilege risk.
- **Regression coverage**: Tests were added for strict API category validation, clear feature-engineering schema failures, and finite zero-charge ratio handling.

### Recommended Next Steps
1. Add a lightweight model-card file that records training data date, target definition, validation metrics, threshold policy, limitations, and owner sign-off.
2. Promote artifacts through an immutable registry path (for example `models/registry/v2/`) and load deployed artifacts by an environment variable such as `MODEL_VERSION`.
3. Add drift checks for categorical distribution shifts and score distribution shifts before batch outreach campaigns.
4. ~~Add request/response logging.~~ **Done** — `app.py`, `api.py`, and `src/predict.py` now share a consistent logger (see Section 8). Remaining follow-up: PII-safe redaction on logged request payloads, and latency/error-rate monitoring on top of the existing log stream.
5. Extend CI to include Docker image build validation when artifact size and CI runtime budgets allow it.

### Streamlit SaaS Dashboard Upgrade
The Streamlit application has been redesigned as **ConnectTel AI Churn Studio**, an enterprise-style decisioning workspace. It keeps the existing prediction backend and executive dashboard, while adding:
- A wide responsive layout with a branded hero section, modern KPI cards, improved spacing, and color-coded risk badges.
- A Plotly churn probability gauge and interactive portfolio charts for probability distribution, risk-tier mix, and risk concentration by contract type.
- Persistent post-submit results using `st.session_state`, which prevents results from disappearing during Streamlit reruns.
- A customer summary panel, AI explanation panel, recommended retention campaign card, single-customer ROI summary, loading spinner, validation messages, example/reset buttons, and downloadable CSV prediction report.
- Artifact/runtime validation in `src/predict.py` so scikit-learn compatibility problems fail fast with a clear remediation path instead of producing warnings or delayed prediction failures.