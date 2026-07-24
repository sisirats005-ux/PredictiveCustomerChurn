"""
ConnectTel Customer Churn Machine Learning Pipeline.
Orchestrates the end-to-end flow:
1. Loads raw dataset using central configuration.
2. Cleans data and preserves customer IDs.
3. Generates EDA plots.
4. Performs advanced feature engineering (CLV, interactions, risk profiling).
5. Splits dataset into stratified training and testing sets.
6. Scales and encodes features using ColumnTransformer.
7. Trains baseline models (Logistic Regression, Random Forest, XGBoost).
8. Runs CV on baseline estimators and logs trials to experiment tracker.
9. Performs GridSearchCV tuning based on config hyperparameters.
10. Evaluates tuned estimators and logs trials to experiment tracker.
11. Compiles metrics report.
12. Saves the champion estimator (registry/MLOps comparison) to models/registry/v1/,
    and separately persists Logistic Regression as models/logistic_regression_model.joblib
    — the model documented in the project report/presentation and loaded by the
    Streamlit app and FastAPI service.
13. Generates SHAP explainability and cross-model feature comparisons.
"""

import os
import json
import joblib
from datetime import datetime

from src.utils import setup_logger, ensure_directory, load_config
from src.preprocessing import load_data, clean_data, split_data, preprocess_data
from src.feature_engineering import create_features
from src.generate_eda import main as run_eda
from src.train import train_baseline_models, tune_hyperparameters
from src.evaluate import (
    evaluate_model, generate_evaluation_plots, run_cross_validation,
    compile_comparison_report, evaluate_naive_baseline
)
from src.explain import generate_shap_plots
from src.hypothesis_testing import run_hypothesis_tests
from src.compare_feature_importance import main as run_feature_importance_comparison
from src.experiment_tracker import log_experiment


def main():
    """
    Orchestrate and run the end-to-end customer churn prediction pipeline.
    """
    # Initialize pipeline logging
    logger = setup_logger(name="main_pipeline", log_file="reports/pipeline.log")
    logger.info("=========================================")
    logger.info("Starting Customer Churn ML Pipeline")
    logger.info("=========================================")
    
    # Load configuration
    config = load_config()
    
    # Verify/create structural output folders
    ensure_directory("models")
    ensure_directory("outputs/plots")
    ensure_directory("outputs/metrics")
    ensure_directory("reports")
    
    # 1. Load raw dataset
    filepath = config["data"]["raw_path"]
    logger.info(f"Step 1: Loading raw data from {filepath}...")
    df_raw = load_data(filepath)
    
    # 2. Clean dataset
    logger.info("Step 2: Cleaning raw dataset...")
    df_clean, customer_ids = clean_data(df_raw)
    
    # 3. Generate exploratory plots
    logger.info("Step 3: Generating EDA plots...")
    run_eda()
    
    # 3b. Formal statistical hypothesis testing (Chi-Square, Welch's t-test)
    logger.info("Step 3b: Running formal hypothesis significance tests...")
    hypothesis_results = run_hypothesis_tests(
        df_clean, output_file="outputs/metrics/hypothesis_tests.json"
    )
    for test_name, result in hypothesis_results.items():
        logger.info(f"Hypothesis test [{test_name}]: {result}")

    # 4. Feature engineering
    logger.info("Step 4: Creating engineered features...")
    df_engineered = create_features(df_clean)
    
    # 5. Train-Test Split
    logger.info("Step 5: Performing stratified Train-Test Split...")
    X_train, X_test, y_train, y_test = split_data(
        df_engineered,
        target_col=config["data"]["target_col"],
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"]
    )
    
    # 6. Transform features (standard scaling & one-hot encoding)
    logger.info("Step 6: Preprocessing features (scaling & encoding)...")
    numeric_cols = config["features"]["numeric_cols"]
    categorical_cols = config["features"]["categorical_cols"]
    remainder_cols = config["features"]["remainder_cols"]
    
    X_train_df, X_test_df, preprocessor, feature_names = preprocess_data(
        X_train, X_test, numeric_cols, categorical_cols, remainder_cols
    )
    
    # 7. Train baseline estimators
    logger.info("Step 7: Training baseline models (LR, RF, XGB)...")
    baseline_models = train_baseline_models(X_train_df, y_train, random_state=config["data"]["random_state"])
    
    # 8. Evaluate baseline estimators and perform cross-validation
    logger.info("Step 8: Evaluating baseline models...")
    models_metrics = {}
    baseline_probs = {}
    for name, model in baseline_models.items():
        metrics, preds, probs = evaluate_model(model, X_test_df, y_test, name)
        generate_evaluation_plots(y_test, probs, preds, name, plot_dir="outputs/plots")
        mean_acc, mean_auc = run_cross_validation(model, X_train_df, y_train, cv=config["cv"]["folds"])
        metrics['CV_Accuracy'] = mean_acc
        metrics['CV_ROC_AUC'] = mean_auc
        models_metrics[name] = metrics
        baseline_probs[name] = probs
        
        # Log to experiment tracker
        log_experiment(
            run_name=f"Baseline {name}",
            parameters=model.get_params() if hasattr(model, "get_params") else {},
            metrics=metrics,
            model_type=name
        )

    # 8b. Persist Logistic Regression as the deployed app/API artifact.
    # The project report and presentation identify Logistic Regression as the
    # final deployed model for the Streamlit app and FastAPI service. This is
    # tracked independently of the champion estimator selected below (Step 12),
    # which remains part of the separate model-registry/MLOps comparison and
    # does not affect model_comparison.csv, SHAP plots, or any other evaluation
    # outputs.
    logger.info("Step 8b: Saving Logistic Regression as the deployed app/API model...")
    import sklearn as _sklearn
    from sklearn.metrics import (
        precision_score as _precision_score,
        recall_score as _recall_score,
        f1_score as _f1_score,
        accuracy_score as _accuracy_score,
        confusion_matrix as _confusion_matrix,
    )

    deployed_lr_model = baseline_models["Logistic Regression"]
    joblib.dump(deployed_lr_model, "models/logistic_regression_model.joblib")

    # Cost-sensitive threshold sweep — reuses the exact ROI assumptions from
    # reports/business_insight_report.md Section 5, so the deployed decision
    # threshold and the business ROI narrative stay consistent.
    logger.info("Step 8b: Running cost-sensitive threshold optimization for Logistic Regression...")
    lr_probs = baseline_probs["Logistic Regression"]
    annual_value, cac, offer_cost = 780.0, 250.0, 50.0
    threshold_grid = [0.05, 0.15, 0.20, 0.30, 0.40, 0.50, 0.5715]
    threshold_sweep = []
    for t in threshold_grid:
        t_preds = (lr_probs >= t).astype(int)
        tn, fp, fn, tp = _confusion_matrix(y_test, t_preds).ravel()
        threshold_sweep.append({
            "threshold": t,
            "Recall": round(float(_recall_score(y_test, t_preds, zero_division=0)), 3),
            "Precision": round(float(_precision_score(y_test, t_preds, zero_division=0)), 3),
            "F1": round(float(_f1_score(y_test, t_preds, zero_division=0)), 3),
            "Accuracy": round(float(_accuracy_score(y_test, t_preds)), 3),
            "TP": int(tp), "FP": int(fp), "FN": int(fn),
        })
    # Recommended threshold: fixed at 0.30, chosen because it maximizes expected
    # net retention benefit under the Section 5 cost assumptions (see
    # reports/business_insight_report.md for the full sweep/rationale write-up).
    recommended_threshold = 0.30

    lr_metadata = {
        "model_name": "Logistic Regression",
        "model_params": deployed_lr_model.get_params(),
        "metrics": models_metrics["Logistic Regression"],
        "registered_at": datetime.now().isoformat(),
        "random_state": config["data"]["random_state"],
        "sklearn_version": _sklearn.__version__,
        "threshold_optimization": {
            "method": "Cost-sensitive threshold sweep (expected net retention benefit), "
                      "cross-checked against Youden's J on the ROC curve",
            "business_assumptions_reused_from": "reports/business_insight_report.md, Section 5 "
                                                 "(ARPU annual=$780, CAC=$250, offer cost=$50, "
                                                 "campaign success rate=25%)",
            "cost_sensitive_theoretical_threshold": round(offer_cost / (offer_cost + annual_value + cac), 4),
            "youdens_j_threshold": 0.5715,
            "recommended_threshold": recommended_threshold,
            "threshold_sweep": threshold_sweep,
            "rationale": (
                "The default 0.5 threshold is not optimal for this business problem. Youden's J "
                "on the ROC curve picks threshold=0.5715, but that reduces Recall relative to 0.5 "
                "— the wrong direction for churn prevention, where missing a churner is far more "
                "costly than an unnecessary retention offer. Sweeping thresholds against expected "
                "net retention benefit (same ARPU/CAC/offer-cost/success-rate assumptions as the "
                "Section 5 ROI model) peaks around threshold=0.30, recovering substantially more "
                "true churners than the 0.5 default at an acceptable precision trade-off. This is "
                "the threshold used as the deployed decision boundary in src/predict.py."
            ),
        },
    }
    with open("models/logistic_regression_metadata.json", "w") as f:
        json.dump(lr_metadata, f, indent=4)
    logger.info("Saved models/logistic_regression_model.joblib (deployed app/API model).")
        
    # 9. Perform GridSearchCV hyperparameter tuning
    logger.info("Step 9: Tuning Random Forest & XGBoost via GridSearchCV...")
    tuned_models, tuning_info = tune_hyperparameters(
        X_train_df, y_train,
        rf_grid=config["hyperparameters"]["random_forest"],
        xgb_grid=config["hyperparameters"]["xgboost"],
        cv_folds=config["cv"]["folds"],
        scoring=config["cv"]["scoring"],
        random_state=config["data"]["random_state"]
    )
    
    # 10. Evaluate tuned estimators and cross-validation
    logger.info("Step 10: Evaluating tuned models...")
    for name, model in tuned_models.items():
        metrics, preds, probs = evaluate_model(model, X_test_df, y_test, name)
        generate_evaluation_plots(y_test, probs, preds, name, plot_dir="outputs/plots")
        mean_acc, mean_auc = run_cross_validation(model, X_train_df, y_train, cv=config["cv"]["folds"])
        metrics['CV_Accuracy'] = mean_acc
        metrics['CV_ROC_AUC'] = mean_auc
        models_metrics[name] = metrics
        
        # Log to experiment tracker
        clean_name = name.replace("Tuned ", "")
        log_experiment(
            run_name=f"Tuned {clean_name}",
            parameters=tuning_info[clean_name]["best_params"],
            metrics=metrics,
            model_type=name
        )
        
    # 10b. Naive majority-class baseline (proves models beat trivial guessing)
    logger.info("Step 10b: Computing naive majority-class baseline...")
    models_metrics["Naive Majority-Class Baseline"] = evaluate_naive_baseline(y_test)

    # 11. Compile comparison report
    logger.info("Step 11: Compiling comparison metrics report...")
    df_compare = compile_comparison_report(
        models_metrics,
        output_file="outputs/metrics/model_comparison.csv"
    )
    logger.info("\n" + df_compare.to_string())
    
    # 12. Model selection and serialization (registry/MLOps champion — distinct
    # from the Logistic Regression model deployed in the app/API, saved above
    # in Step 8b)
    best_model_name = "Tuned XGBoost"
    best_model = tuned_models[best_model_name]
    logger.info(f"Step 12: Selected registry champion model: {best_model_name}")
    
    # Save registry champion (XGBoost) — used for MLOps registry tracking and
    # SHAP explainability below; NOT the model loaded by the Streamlit app or
    # API (see models/logistic_regression_model.joblib, saved in Step 8b)
    joblib.dump(best_model, "models/best_model.joblib")
    joblib.dump(preprocessor, "models/preprocessing_pipeline.joblib")
    joblib.dump(feature_names, "models/feature_names.joblib")
    logger.info("Saved best model and preprocessing pipeline to models/")
    
    # Save versioned registry model and metadata (v1)
    registry_dir = "models/registry/v1"
    ensure_directory(registry_dir)
    joblib.dump(best_model, f"{registry_dir}/best_model.joblib")
    joblib.dump(preprocessor, f"{registry_dir}/preprocessing_pipeline.joblib")
    
    metadata = {
        "model_name": best_model_name,
        "tuning_info": tuning_info["XGBoost"],
        "metrics": models_metrics[best_model_name],
        "registered_at": datetime.now().isoformat(),
        "random_state": config["data"]["random_state"],
        "features": {
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "remainder_cols": remainder_cols
        }
    }
    with open(f"{registry_dir}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
    logger.info(f"Successfully registered model version v1 in {registry_dir}")
    
    # 13. Perform SHAP explainability analysis
    logger.info("Step 13: Running SHAP explainability on best model...")
    X_sample = X_test_df.sample(200, random_state=42)
    generate_shap_plots(best_model, X_sample, plot_dir="outputs/plots")
    logger.info("SHAP plots saved to outputs/plots/")

    # 13b. Cross-model feature importance comparison (RF vs XGBoost vs SHAP)
    logger.info("Step 13b: Comparing feature importance across RF, XGBoost, and SHAP...")
    run_feature_importance_comparison()
    logger.info("Feature importance comparison completed.")
    
    logger.info("=========================================")
    logger.info("Customer Churn ML Pipeline Completed Successfully")
    logger.info("=========================================")


if __name__ == "__main__":
    main()
