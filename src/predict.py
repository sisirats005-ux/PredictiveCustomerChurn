"""
ConnectTel Customer Churn Prediction Inference Module.
Loads the serialized final deployed model (Logistic Regression, as
documented in the project report and presentation) and its preprocessing
pipeline, and scores new/unseen customer records (as used by the
Streamlit deployment app and by the test suite).
"""

import json
import os
import warnings

import joblib
import pandas as pd
import numpy as np

from src.feature_engineering import create_features

MODEL_PATH = "models/logistic_regression_model.joblib"
PREPROCESSOR_PATH = "models/preprocessing_pipeline.joblib"
FEATURE_NAMES_PATH = "models/feature_names.joblib"
METADATA_PATH = "models/logistic_regression_metadata.json"


def _check_sklearn_version_match(metadata_path=METADATA_PATH):
    """
    Compare the installed scikit-learn version against the version the
    Logistic Regression model was trained/serialized with. Warns (does
    not raise) on a mismatch, since this is the leading cause of cryptic
    unpickling errors such as `'LogisticRegression' object has no
    attribute 'multi_class'`.
    """
    if not os.path.exists(metadata_path):
        return
    try:
        import sklearn

        with open(metadata_path, "r") as f:
            meta = json.load(f)

        trained_version = meta.get("sklearn_version")
        installed_version = sklearn.__version__

        if trained_version and trained_version != installed_version:
            warnings.warn(
                f"models/logistic_regression_model.joblib was saved with "
                f"scikit-learn {trained_version}, but scikit-learn "
                f"{installed_version} is installed. Mismatched scikit-learn "
                "versions can cause loading/prediction errors. Install the "
                "version pinned in requirements.txt (scikit-learn==1.4.2) "
                "or retrain the model with your current environment.",
                RuntimeWarning,
            )
    except Exception:
        # Best-effort check only; never block model loading over it.
        pass


def load_artifacts(model_path=MODEL_PATH, preprocessor_path=PREPROCESSOR_PATH,
                    feature_names_path=FEATURE_NAMES_PATH):
    """
    Load the serialized model, ColumnTransformer preprocessor, and the
    ordered list of feature names produced at training time.

    Parameters
    ----------
    model_path : str, optional
        Path to the serialized final deployed model (Logistic Regression).
    preprocessor_path : str, optional
        Path to the serialized ColumnTransformer.
    feature_names_path : str, optional
        Path to the serialized list of post-transform feature names.

    Returns
    -------
    model : estimator
        Fitted classifier.
    preprocessor : ColumnTransformer
        Fitted preprocessing pipeline.
    feature_names : list of str
        Post-transform feature names, in the order the model expects them.
    """
    missing = [p for p in (model_path, preprocessor_path, feature_names_path) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Missing required model artifact(s): " + ", ".join(missing) +
            ". Ensure the `models/` folder contains the trained Logistic "
            "Regression artifacts before launching the app."
        )

    _check_sklearn_version_match()

    try:
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        feature_names = joblib.load(feature_names_path)
    except AttributeError as exc:
        raise RuntimeError(
            "Failed to load model artifacts due to a scikit-learn version "
            f"mismatch ({exc}). Install the exact scikit-learn version listed "
            "in requirements.txt (scikit-learn==1.4.2), or retrain the model "
            "with your currently installed scikit-learn version."
        ) from exc

    return model, preprocessor, feature_names


DEFAULT_THRESHOLD = 0.5


def get_decision_threshold(metadata_path=METADATA_PATH):
    """
    Load the cost-optimized decision threshold from the Logistic Regression
    metadata file, if available. Falls back to the naive 0.5 default if the
    metadata or the threshold_optimization block is missing.

    The optimized threshold (see reports/business_insight_report.md, Section
    "Threshold Optimization") was chosen via a cost-sensitive sweep that
    maximizes expected net retention benefit, rather than defaulting to 0.5,
    since a missed churner (lost customer value) is far more expensive than
    an unnecessary retention offer to a loyal customer.
    """
    try:
        with open(metadata_path, "r") as f:
            meta = json.load(f)
        return float(meta["threshold_optimization"]["recommended_threshold"])
    except Exception:
        return DEFAULT_THRESHOLD


def prepare_customer_frame(customer):
    """
    Turn a single raw customer record into a one-row engineered DataFrame,
    ready to be passed through the fitted preprocessing pipeline.

    Mirrors the cleaning done in `preprocessing.clean_data` (TotalCharges
    blank/whitespace handling) and applies the same engineered features
    used at training time via `feature_engineering.create_features`.

    Parameters
    ----------
    customer : dict
        Raw feature values for a single customer, using the same field
        names/categories as the original Telco Churn dataset (excluding
        `customerID` and `Churn`).

    Returns
    -------
    pd.DataFrame
        Single-row engineered DataFrame.
    """
    df = pd.DataFrame([customer])

    # Mirror preprocessing.clean_data's TotalCharges handling
    df['TotalCharges'] = df['TotalCharges'].replace(r'^\s*$', np.nan, regex=True)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges']).fillna(0.0)

    df = create_features(df)
    return df


def top_risk_factors(df_row):
    """
    Identify the top contributing risk factors for a single engineered
    customer row, using the same directional relationships validated by the
    project's SHAP analysis (see outputs/plots/shap_summary_*.png and
    reports/business_insight_report.md, Section 2: Key Strategic Findings).

    This is a lightweight, rule-based explanation layer (not a live SHAP
    recomputation) intended for fast, human-readable "why" context in the
    Streamlit app — it mirrors the same drivers already validated offline.

    Parameters
    ----------
    df_row : pd.Series
        A single row from the engineered customer DataFrame.

    Returns
    -------
    list of str
        Up to four human-readable risk factor descriptions, ordered by the
        same priority used in the project's segment analysis.
    """
    factors = []

    if df_row.get("Contract") == "Month-to-month":
        factors.append("Month-to-month contract (highest churn segment: 42.7% vs 2.8% for 2-year)")
    if df_row.get("InternetService") == "Fiber optic":
        factors.append("Fiber optic internet service (41.9% churn rate)")
    if df_row.get("PaymentMethod") == "Electronic check":
        factors.append("Manual Electronic Check payment (45.3% churn rate)")
    if df_row.get("tenure", 99) <= 12:
        factors.append("Short tenure — new customer (churn is front-loaded, median 10 months)")
    if df_row.get("MonthlyCharges", 0) > 85:
        factors.append("High monthly charges (elevated billing sensitivity)")
    if df_row.get("TechSupport") == "No" and df_row.get("InternetService") != "No":
        factors.append("No Tech Support add-on (unresolved service friction)")

    return factors[:4] if factors else ["No dominant risk factors identified — profile appears stable."]


def recommend_action(risk_tier, df_row):
    """
    Map a customer's risk tier and profile to the specific retention
    campaign(s) defined in reports/business_insight_report.md, Section 4:
    Proactive Customer Retention Recommendations.

    Parameters
    ----------
    risk_tier : str
        'Low', 'Medium', or 'High', as returned by `predict_churn`.
    df_row : pd.Series
        A single row from the engineered customer DataFrame.

    Returns
    -------
    str
        A single recommended action, prioritized by expected business impact.
    """
    if risk_tier == "Low":
        return "No intervention needed — maintain standard engagement and account monitoring."

    if df_row.get("Contract") == "Month-to-month" and df_row.get("InternetService") == "Fiber optic" \
            and df_row.get("MonthlyCharges", 0) > 85:
        return ("Priority Segment A action: Offer a 10% annual-plan discount to migrate off "
                "month-to-month, plus a $5/month fiber loyalty credit for 6 months.")
    if df_row.get("Contract") == "Month-to-month":
        return ("Month-to-Year Contract Migration: Offer a 10% monthly discount or a free speed "
                "upgrade in exchange for a 1-year contract.")
    if df_row.get("PaymentMethod") == "Electronic check":
        return ("Auto-Pay Enrollment Campaign: Offer a one-time $10 statement credit to switch to "
                "Credit Card or Bank Auto-Pay billing.")
    if df_row.get("tenure", 99) <= 12:
        return ("Welcome Concierge Outreach: Priority check-in call at month 3/6 plus a "
                "first-anniversary loyalty offer.")

    return "Outbound Care Call: Assess satisfaction and offer a general loyalty discount."


def predict_churn(customer, model=None, preprocessor=None, feature_names=None, threshold=None):
    """
    Score a single customer record and return the churn prediction.

    Parameters
    ----------
    customer : dict
        Raw feature values for a single customer (see `prepare_customer_frame`).
    model : estimator, optional
        Fitted classifier. Loaded from disk if not provided.
    preprocessor : ColumnTransformer, optional
        Fitted preprocessing pipeline. Loaded from disk if not provided.
    feature_names : list of str, optional
        Post-transform feature names. Loaded from disk if not provided.
    threshold : float, optional
        Decision threshold for the Yes/No churn call. Defaults to the
        cost-optimized threshold from `get_decision_threshold()` (falls back
        to 0.5 if no metadata is available).

    Returns
    -------
    dict
        Dictionary with keys:
        - 'churn_probability' (float): predicted probability of churn (class 1)
        - 'churn_prediction' (str): 'Yes' or 'No' at the decision threshold used
        - 'decision_threshold' (float): the threshold actually applied
        - 'risk_tier' (str): 'Low', 'Medium', or 'High' based on probability
        - 'clv' (float): estimated customer lifetime value
        - 'billing_risk' (float): billing collection friction metric
        - 'high_risk_profile' (int): binary indicator for multi-factor risk
        - 'persona' (str): customer segment categorization description
        - 'top_risk_factors' (list of str): human-readable churn drivers for this customer
        - 'recommended_action' (str): suggested retention campaign for this customer
    """
    decision_threshold = threshold if threshold is not None else get_decision_threshold()

    if model is None or preprocessor is None or feature_names is None:
        loaded_model, loaded_preprocessor, loaded_feature_names = load_artifacts()
        model = model or loaded_model
        preprocessor = preprocessor or loaded_preprocessor
        feature_names = feature_names or loaded_feature_names

    df = prepare_customer_frame(customer)

    X_trans = preprocessor.transform(df)
    X_df = pd.DataFrame(X_trans, columns=feature_names, index=df.index)

    proba = float(model.predict_proba(X_df)[0, 1])
    prediction = 'Yes' if proba >= decision_threshold else 'No'

    if proba < 0.3:
        risk_tier = 'Low'
    elif proba < 0.6:
        risk_tier = 'Medium'
    else:
        risk_tier = 'High'

    # Extract new features for frontend display
    clv_val = float(df.loc[0, "CLV"])
    billing_risk_val = float(df.loc[0, "BillingRisk"])
    high_risk_profile_val = int(df.loc[0, "HighRiskProfile"])

    if df.loc[0, "Persona_New_HighSpend"] == 1:
        persona = "New Customer (High Spend)"
    elif df.loc[0, "Persona_Loyal_HighSpend"] == 1:
        persona = "Loyal Customer (High Spend)"
    elif df.loc[0, "tenure"] <= 12:
        persona = "New Customer (Low/Med Spend)"
    else:
        persona = "Loyal Customer (Low/Med Spend)"

    return {
        'churn_probability': proba,
        'churn_prediction': prediction,
        'decision_threshold': decision_threshold,
        'risk_tier': risk_tier,
        'clv': clv_val,
        'billing_risk': billing_risk_val,
        'high_risk_profile': high_risk_profile_val,
        'persona': persona,
        'top_risk_factors': top_risk_factors(df.loc[0]),
        'recommended_action': recommend_action(risk_tier, df.loc[0]),
    }
