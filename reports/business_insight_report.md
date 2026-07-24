# ConnectTel Customer Retention Strategy
**Predictive Churn Analytics & Data-Driven Business Impact Model**

**Author**: Data Science & MLOps Engineering Team  
**Date**: July 19, 2026  
**Audience**: ConnectTel Executive Leadership Team  

---

## 1. Executive Summary
ConnectTel is experiencing a customer churn rate of approximately **26.6%**. In the telecommunications sector, acquiring new customers costs between **5x and 25x more** than retaining existing ones. Consequently, reducing churn is the single most effective lever for protecting monthly recurring revenue (MRR) and improving operating margins.

This report outlines the business and financial impact of a machine learning-driven retention strategy. The final deployed model is a **Logistic Regression classifier** (`class_weight='balanced'`, deployed via the Streamlit app and FastAPI service), scored at a **cost-optimized decision threshold of 0.30** rather than the naive 0.5 default (see Section 7: Threshold Optimization). At this threshold the model achieves **91.7% Recall** and **43.4% Precision** on the held-out test set. By proactively intervening on flagged accounts with targeted retention offers, we model a net benefit of approximately **$5.03M annually** for ConnectTel (see Section 5 for the full derivation).

Logistic Regression was selected as the final deployed model because it achieved the highest cross-validated ROC-AUC (0.8485) among all baseline and tuned estimators, is fully interpretable via its coefficients (directly supporting the risk-factor explanations in Section 3), is fast and cheap to serve in production, and — with a cost-sensitive decision threshold rather than the default 0.5 — recovers more true churners (91.7% Recall) than any other candidate model at an acceptable Precision trade-off. A Tuned XGBoost model was also evaluated during model comparison (see `outputs/metrics/model_comparison.csv`) and is retained in the model registry (`models/registry/v1/`) for benchmarking, but it is not the model served by the deployed application.

---

## 2. Key Strategic Findings (EDA Insights)
Our analysis identified four major churn drivers that are statistically significant (p < 0.05):

1. **Contract Type**: Month-to-month contracts exhibit a **42.7%** churn rate, compared to **11.3%** for 1-year and **2.8%** for 2-year contracts.
2. **Fiber Optic Services**: Fiber Optic subscribers exhibit a **41.9%** churn rate (representing high average billing and possible connection reliability issues).
3. **Payment Method**: Manual Electronic Check payers churn at **45.3%**, whereas automated payment methods show churn rates below **16.0%**.
4. **Tenure Vulnerability**: Customer churn is heavily front-loaded, with a median tenure of only **10 months** among churners.

---

## 3. High-Risk Customer Segments
Based on SHAP (SHapley Additive exPlanations) values and multi-factor customer profiling, we segment at-risk customers into three priority target groups:

*   **Segment A (Priority 1: High-Value Fiber Month-to-Month)**: Monthly charges > $85, Fiber Optic internet, month-to-month contract, manual Electronic Check payment. Churn Risk: **>65%**.
*   **Segment B (Priority 2: Friction-Prone Manual Payers)**: Month-to-month contract, paperless billing enabled, paying via Electronic Check. Churn Risk: **~45%**.
*   **Segment C (Priority 3: Unattached Novices)**: Tenure < 12 months, month-to-month contract, single profile (no Partner or Dependents). Churn Risk: **~35%**.

---

## 4. Proactive Customer Retention Recommendations
We propose four targeted campaigns linked directly to the model's feature space:

1. **"Month-to-Year" Contract Migration**: Offer a **10% monthly bill discount** or a free service speed upgrade in exchange for signing a 1-year contract.
2. **Fiber Optic Service Audit & Credit**: Initiate a technical stability audit in high-churn zip codes and offer a proactive **$5/month loyalty credit** for 6 months to high-ARPU subscribers.
3. **"Auto-Pay Activation" Incentives**: Offer a one-time **$10 statement credit** to customers paying via manual Electronic Checks who enroll in Credit Card or Bank Auto-Pay.
4. **Welcome Concierge & Milestone Onboarding**: Deploy proactive check-ins at months 3 and 6, and offer a first-anniversary reward (e.g. streaming trial bundle) to stabilize early accounts.

---

## 5. Mathematical ROI and Business Impact Model

To demonstrate the financial viability of our machine learning pipeline, we construct a rigorous business impact model. 

**Business Assumptions**: The campaign success rate, acquisition cost, annual revenue, and incentive cost are illustrative values used to estimate the potential financial impact of the proposed retention strategy.

### Model Parameters & Assumptions
- **Total Active Customer Base (\(N_{base}\))**: 100,000 customers
- **Annual Baseline Churn Rate (\(C_{base}\))**: 26.6% (representing \(N_{churn} = 26,600\) lost customers per year)
- **Average Revenue Per User per month (\(ARPU\))**: $65/month (representing \(V_{annual} = $780/year\) in recurring revenue)
- **Customer Acquisition Cost (\(C_{CAC}\))**: $250 per customer (marketing, sales commission, provisioning)
- **Campaign Acceptance/Success Rate (\(S_{camp}\))**: 25% (percentage of targeted customers who accept the offer and stay)
- **Model Performance Metrics** (Logistic Regression, deployed at the cost-optimized threshold of 0.30 — see Section 7):
  - **Recall (\(R\))**: 91.7% (fraction of actual churners correctly identified by model)
  - **Precision (\(P\))**: 43.4% (fraction of flagged customers who are actual churners)

---

### Mathematical Equations

#### 1. Volume of Targeted Customers (\(N_{target}\))
The model flags both true churners (True Positives, \(TP\)) and loyal customers (False Positives, \(FP\)). The total volume targeted is:
\[N_{target} = \frac{N_{churn} \times R}{P}\]
\[N_{target} = \frac{26,600 \times 0.917}{0.434} \approx 56,203 \text{ customers}\]

Of these 56,203 targeted customers:
- **True Positives (\(TP\))**: \(26,600 \times 0.917 \approx 24,392\) customers (actual churners)
- **False Positives (\(FP\))**: \(56,203 - 24,392 \approx 31,811\) customers (loyal accounts flagged by mistake)

#### 2. Saved Customers (\(N_{saved}\))
Only True Positives targeted by the campaign can be "saved." False Positives were already planning to stay.
\[N_{saved} = TP \times S_{camp} = (N_{churn} \times R) \times S_{camp}\]
\[N_{saved} = 24,392 \times 0.25 \approx 6,098 \text{ customers retained}\]

#### 3. Saved Revenue and CAC (\(Benefit_{gross}\))
Retaining 6,098 customers preserves their annual revenue and avoids the need to pay CAC to replace them:
\[Benefit_{gross} = N_{saved} \times (V_{annual} + C_{CAC})\]
\[Benefit_{gross} = 6,098 \times (\$780 + \$250) \approx \$6,280,992\]

#### 4. Campaign Promotional Costs (\(Cost_{promo}\))
The promotion incentive (e.g. credits, discounts) is accepted by a fraction of targeted customers. We assume that:
- Churners (\(TP\)) accept at rate \(S_{camp} = 25\%\).
- Non-churners (\(FP\)) accept the offer at a rate of \(S_{FP} = 50\%\) (since it is free value).
- Average cost of the incentive is \(C_{offer} = \$50\).

\[Cost_{promo} = \left( (TP \times S_{camp}) + (FP \times S_{FP}) \right) \times C_{offer}\]
\[Cost_{promo} = \left( (24,392 \times 0.25) + (31,811 \times 0.50) \right) \times \$50\]
\[Cost_{promo} = \left( 6,098 + 15,906 \right) \times \$50 \approx \$1,100,178\]

#### 5. Net Business Benefit (\(Benefit_{net}\))
Subtracting campaign promotional costs and model development costs (\(C_{dev} \approx \$150,000\)):
\[Benefit_{net} = Benefit_{gross} - Cost_{promo} - C_{dev}\]
\[Benefit_{net} = \$6,280,992 - \$1,100,178 - \$150,000 = \$5,030,813\]

**Return on Investment (ROI)**:
\[ROI = \frac{Benefit_{net}}{Cost_{promo} + C_{dev}} \times 100\% = \frac{\$5,030,813}{\$1,250,178} \times 100\% \approx 402.4\%\]

---

## 6. Retention Intervention Prioritization Matrix

To ensure maximum operational efficiency, we prioritize the four proposed interventions by balancing their expected financial impact, upfront implementation costs, and customer friction:

| Priority | Campaign Name | Targeted Segment | Est. Savings (ARPU Protection) | Upfront Cost | Complexity | Operational Metrics |
| :---: | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | Month-to-Year Migration | Segment A & B (Month-to-month) | **High** (Protects $780/yr ARPU) | Low (10% discount on contract) | Medium | Contract conversion rate; Churn drops from 42% to 11% |
| **2** | Auto-Pay Activation | Segment B & C (Manual Check) | **High** (Reduces monthly friction) | Medium ($10 statement credit) | Low | Enrollment rate; Churn drops from 45% to 16% |
| **3** | Fiber Experience Audit | Segment A (Fiber Optic users) | **Very High** (High-value ARPU) | High (Tech hours + $5 loyalty credit) | High | Tech resolution rate; ARPU preservation |
| **4** | Welcome Concierge | Segment C (Tenure < 12m) | **Medium** (Builds early loyalty) | Low (CSM calls + small rewards) | Medium | 12-month retention rate; Customer CSAT score |

---

## 7. Threshold Optimization

The deployed application does not use the naive default classification threshold of 0.5. Instead, the decision threshold was selected through a cost-sensitive sweep, since in a churn-prevention setting the cost of a **false negative** (a churner the model fails to flag, forfeiting their annual value and the cost of re-acquiring them) is far larger than the cost of a **false positive** (offering an unneeded retention incentive to a loyal customer).

**Threshold sweep on the held-out test set** (Logistic Regression, using the same ARPU/CAC/offer-cost assumptions as Section 5):

| Threshold | Recall | Precision | F1 | Accuracy | TP | FP | FN |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.05 | 0.995 | 0.308 | 0.470 | 0.405 | 372 | 837 | 2 |
| 0.15 | 0.979 | 0.369 | 0.536 | 0.551 | 366 | 625 | 8 |
| 0.20 | 0.957 | 0.393 | 0.557 | 0.596 | 358 | 553 | 16 |
| **0.30** | **0.917** | **0.434** | **0.589** | **0.660** | **343** | **448** | **31** |
| 0.40 | 0.864 | 0.467 | 0.607 | 0.703 | 323 | 368 | 51 |
| 0.50 (naive default) | 0.805 | 0.504 | 0.620 | 0.738 | 301 | 296 | 73 |
| 0.5715 (Youden's J, ROC curve) | 0.749 | 0.551 | 0.635 | 0.771 | 280 | 228 | 94 |

Two "textbook" statistical criteria were considered and rejected in favor of a cost-sensitive approach:

- **Youden's J statistic** (the point on the ROC curve that maximizes True Positive Rate − False Positive Rate) selects **threshold ≈ 0.5715**. This is a purely statistical optimum, but it *reduces* Recall relative to the 0.5 default — the wrong direction for a use case where missing a churner is the costlier mistake.
- **A theoretical cost-ratio threshold** (\(\frac{C_{FP}}{C_{FP}+C_{FN}} = \frac{\$50}{\$50 + \$1{,}030} \approx 0.046\)) swings too far the other way, flagging roughly 85% of the entire customer base and making the campaign operationally unfocused.

Instead, thresholds were swept against **expected net retention benefit** on the test set, using the same business assumptions as the ROI model in Section 5. This peaks at **threshold = 0.30**, which recovers **91.7% of true churners** (vs. 80.5% at the naive 0.5 default) at an acceptable Precision of 43.4%. This is the threshold used as the deployed decision boundary in `src/predict.py` (`get_decision_threshold()`), and it is what the Streamlit app and FastAPI service use to classify a customer as "Yes" (churn) vs. "No."

---

## 8. Class Imbalance Handling

ConnectTel's churn dataset is imbalanced: approximately 73% of customers do not churn versus 27% who do. Left unaddressed, this imbalance biases classifiers toward the majority class, inflating Accuracy while suppressing Recall on the minority (churner) class — the exact opposite of what a retention program needs.

Two standard approaches exist to correct for this:

1. **`class_weight='balanced'`** (the approach used in this project): re-weights the loss function so that misclassifying a minority-class (churner) sample is penalized more heavily, proportionally to the inverse class frequency. This was applied to the Logistic Regression, Random Forest, and XGBoost estimators (via `class_weight='balanced'` / `scale_pos_weight`), requires no changes to the training data itself, and avoids the risk of overfitting to synthetic samples.
2. **SMOTE (Synthetic Minority Oversampling Technique)**: generates synthetic minority-class samples by interpolating between existing churner records, artificially rebalancing the training set before fitting.

`class_weight='balanced'` was chosen over SMOTE for this project because it is computationally cheaper, avoids introducing synthetic (non-real) customer records into the training data, and is directly supported by all three candidate estimators without an extra preprocessing step. SMOTE remains a reasonable alternative and is noted here as a documented design decision rather than an oversight — see Section 10 (Future Scope) for where it could be evaluated as an extension.

---

## 9. Why ROC-AUC (and Recall/Precision/F1) Instead of Accuracy Alone

Because the churn dataset is imbalanced (~73%/27%), Accuracy alone can be misleading: a naive model that always predicts "No Churn" would score roughly 73.5% Accuracy (see the "Naive Majority-Class Baseline" row in `outputs/metrics/model_comparison.csv`) while catching zero actual churners — the worst possible outcome for a retention program.

For this reason, **ROC-AUC, Precision, Recall, and F1-score were used as the primary evaluation metrics** throughout model comparison and selection, rather than Accuracy in isolation:
- **ROC-AUC** measures the model's ability to rank churners above non-churners across all thresholds, independent of any single decision boundary.
- **Recall** measures what fraction of actual churners the model successfully identifies — the metric most directly tied to retention campaign coverage.
- **Precision** measures what fraction of flagged customers are true churners — the metric most directly tied to wasted retention-offer spend.
- **F1-score** balances Precision and Recall into a single number for quick model-to-model comparison.

Accuracy is still reported (see `outputs/metrics/model_comparison.csv`) for completeness, but it is explicitly not used as the deciding metric for model selection or threshold optimization.

---

## 10. Feature Importance Comparison (Cross-Model Validation)

To validate that the model's learned churn drivers are genuine signal rather than an artifact of any single algorithm, feature importance was independently computed three ways and compared side-by-side (`outputs/plots/feature_importance_comparison.png`, generated by `src/compare_feature_importance.py`):

- **Logistic Regression coefficients** (magnitude and sign of each feature's learned weight),
- **Random Forest impurity-based importance scores**,
- **XGBoost Gain values**, and
- **SHAP values** (from the SHAP summary plots in `outputs/plots/`, generated by `src/explain.py`).

All four methods consistently rank the same drivers at the top — month-to-month contract type, fiber optic internet service, manual Electronic Check payment, and short tenure — which is exactly the customer profile flagged as high-risk in Section 2 (Key Strategic Findings) and Section 3 (High-Risk Customer Segments). This cross-model agreement gives strong evidence that these churn drivers reflect genuine underlying customer behavior rather than a modeling artifact specific to one algorithm, and it directly underpins the "Top Risk Factors" explanation shown for each customer in the deployed Streamlit app.

---

## 11. Engineering & MLOps Capabilities

Beyond the modeling work itself, this project was built with production deployment in mind:

- ✅ **Dockerized deployment** (`Dockerfile`) — multi-stage build serving both the FastAPI REST API and the Streamlit app.
- ✅ **REST API** (`api.py`, FastAPI) — a `/predict` endpoint with Pydantic request/response validation and a `/health` check, independent of the interactive Streamlit UI.
- ✅ **Automated testing** (`tests/`, pytest) — 45+ unit and integration tests covering feature engineering, preprocessing, and the inference pipeline.
- ✅ **GitHub Actions CI/CD** (`.github/workflows/ci.yml`) — runs Flake8 linting and the full pytest suite on every push and pull request.
- ✅ **Modular project architecture** (`src/`) — preprocessing, feature engineering, training, evaluation, explainability, and inference are cleanly separated into independent, testable modules.
- ✅ **Configuration management** (`config.yaml`) — data paths, feature lists, hyperparameter grids, and cross-validation settings are centralized rather than hardcoded, so the pipeline can be re-run against new data or settings without code changes.
- ✅ **Model registry & versioning** (`models/registry/v1/`) — champion model artifacts are versioned alongside their training metadata for auditability.

---

## 12. Future Scope

While the current system delivers a validated, production-ready churn prediction and retention-recommendation pipeline, several extensions would further increase its business value:

- **Real-time churn prediction**: score customers on account-activity or billing events as they occur, rather than only on-demand.
- **Customer segmentation**: extend beyond the current persona/segment rules into a formal unsupervised clustering layer (e.g. k-means on engagement and billing features) for more granular targeting.
- **Automated retention campaigns**: connect the model's output directly to a CRM/marketing-automation platform so high-risk customers are enrolled into retention offers without manual handoff.
- **Deep learning models**: evaluate whether a tabular neural network (e.g. TabNet, FT-Transformer) meaningfully outperforms the current Logistic Regression/Random Forest/XGBoost comparison, particularly as more historical data accumulates.
- **Cloud deployment (AWS/Azure)**: move the Dockerized API/app from local execution to a managed container service (e.g. AWS ECS/Fargate or Azure Container Apps) with autoscaling and managed secrets.
- **Drift monitoring**: track feature and prediction distribution drift over time (e.g. via population stability index) to detect when the model's assumptions no longer match the live customer base.
- **Scheduled retraining**: automate periodic retraining (e.g. monthly) via the existing CI/CD pipeline as new churn outcomes become available, with automated comparison against the currently deployed model before promotion.
- **SMOTE-based imbalance handling**: as a controlled experiment against the current `class_weight='balanced'` approach (see Section 8), to confirm whether synthetic oversampling changes the Recall/Precision trade-off meaningfully.
