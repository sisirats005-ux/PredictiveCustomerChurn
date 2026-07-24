"""
ConnectTel Feature Importance Comparison Module.
Compares native feature_importances_ from the tuned Random Forest and tuned
XGBoost models against mean |SHAP value| from the champion XGBoost model,
to check whether the models agree on what actually drives churn.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

from src.preprocessing import load_data, clean_data, split_data, preprocess_data
from src.feature_engineering import create_features
from src.train import tune_hyperparameters
from src.utils import setup_logger, ensure_directory, load_config

logger = setup_logger(name="compare_feature_importance", log_file="reports/pipeline.log")


def main():
    ensure_directory("outputs/plots")
    ensure_directory("outputs/metrics")

    # Load centralized config
    config = load_config()

    logger.info("Reloading data and re-fitting tuned RF/XGBoost for importance comparison...")
    df_raw = load_data(config["data"]["raw_path"])
    df_clean, _ = clean_data(df_raw)
    df_engineered = create_features(df_clean)
    
    # Split
    X_train, X_test, y_train, y_test = split_data(
        df_engineered,
        target_col=config["data"]["target_col"],
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"]
    )

    # Dynamic columns
    numeric_cols = config["features"]["numeric_cols"]
    categorical_cols = config["features"]["categorical_cols"]
    remainder_cols = config["features"]["remainder_cols"]

    X_train_df, X_test_df, preprocessor, feature_names = preprocess_data(
        X_train, X_test, numeric_cols, categorical_cols, remainder_cols
    )

    # Tune hyperparameters using config grids
    tuned_models, _ = tune_hyperparameters(
        X_train_df, y_train,
        rf_grid=config["hyperparameters"]["random_forest"],
        xgb_grid=config["hyperparameters"]["xgboost"],
        cv_folds=config["cv"]["folds"],
        scoring=config["cv"]["scoring"],
        random_state=config["data"]["random_state"]
    )
    
    rf_model = tuned_models["Tuned Random Forest"]
    xgb_model = tuned_models["Tuned XGBoost"]

    # Native feature importances (normalized to sum to 1 each, for fair comparison)
    rf_importance = pd.Series(rf_model.feature_importances_, index=feature_names)
    xgb_importance = pd.Series(xgb_model.feature_importances_, index=feature_names)

    # SHAP mean |value| importance for the champion XGBoost model
    X_sample = X_test_df.sample(200, random_state=42)
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer(X_sample)
    shap_importance = pd.Series(
        np.abs(shap_values.values).mean(axis=0), index=feature_names
    )
    shap_importance_norm = shap_importance / shap_importance.sum()

    comparison = pd.DataFrame({
        "RF_Importance": rf_importance,
        "XGB_Importance": xgb_importance,
        "SHAP_Importance": shap_importance_norm
    }).sort_values("XGB_Importance", ascending=False)

    comparison.to_csv("outputs/metrics/feature_importance_comparison.csv")
    logger.info("Saved feature_importance_comparison.csv")

    top_n = 12
    top_features = comparison.head(top_n)

    fig, ax = plt.subplots(figsize=(11, 7))
    x = np.arange(len(top_features))
    width = 0.27
    ax.barh(x + width, top_features["RF_Importance"][::-1], width, label="Random Forest", color="#4C72B0")
    ax.barh(x, top_features["XGB_Importance"][::-1], width, label="XGBoost", color="#DD8452")
    ax.barh(x - width, top_features["SHAP_Importance"][::-1], width, label="SHAP (mean |value|, normalized)", color="#55A868")
    ax.set_yticks(x)
    ax.set_yticklabels(top_features.index[::-1])
    ax.set_xlabel("Normalized Importance")
    ax.set_title(f"Figure 27: Feature Importance Comparison — Random Forest vs. XGBoost vs. SHAP (Top {top_n})")
    ax.legend()
    plt.tight_layout()
    plt.savefig("outputs/plots/feature_importance_comparison.png", dpi=150)
    plt.close()
    logger.info("Saved feature_importance_comparison.png")

    # Rank correlation between the three rankings (Spearman)
    rank_rf = comparison["RF_Importance"].rank(ascending=False)
    rank_xgb = comparison["XGB_Importance"].rank(ascending=False)
    rank_shap = comparison["SHAP_Importance"].rank(ascending=False)
    corr_rf_xgb = rank_rf.corr(rank_xgb, method="spearman")
    corr_xgb_shap = rank_xgb.corr(rank_shap, method="spearman")
    corr_rf_shap = rank_rf.corr(rank_shap, method="spearman")

    logger.info(f"Rank correlation RF vs XGB: {corr_rf_xgb:.3f}")
    logger.info(f"Rank correlation XGB vs SHAP: {corr_xgb_shap:.3f}")
    logger.info(f"Rank correlation RF vs SHAP: {corr_rf_shap:.3f}")

    print(comparison.head(top_n).round(4))
    print(f"\nSpearman rank correlation — RF vs XGBoost: {corr_rf_xgb:.3f}")
    print(f"Spearman rank correlation — XGBoost vs SHAP: {corr_xgb_shap:.3f}")
    print(f"Spearman rank correlation — RF vs SHAP: {corr_rf_shap:.3f}")

    return comparison, corr_rf_xgb, corr_xgb_shap, corr_rf_shap


if __name__ == "__main__":
    main()
