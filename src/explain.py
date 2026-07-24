"""
ConnectTel Customer Churn Prediction Interpretability Module.
Computes SHAP (SHapley Additive exPlanations) values for tree-based models
and generates beeswarm summary, feature importance bar, and waterfall plots.
"""

import os
import matplotlib.pyplot as plt
import shap
import numpy as np


def generate_shap_plots(model, X, plot_dir="outputs/plots"):
    """
    Generate and save SHAP interpretability plots for a model.

    Generates three types of plots:
    1. beeswarm: A summary plot showing how feature values push model output
       risk higher (red) or lower (blue).
    2. bar: Global feature importance based on average absolute SHAP values.
    3. waterfall: Local prediction analysis for a single sample (index 0).

    Parameters
    ----------
    model : estimator
        Fitted tree-based classifier (e.g. XGBoost, Random Forest).
    X : pd.DataFrame
        Input features DataFrame aligned with model expectations.
    plot_dir : str, optional
        Target directory to save the plots, default is "outputs/plots".

    Returns
    -------
    summary_path : str
        File path of the saved SHAP beeswarm summary plot.
    importance_path : str
        File path of the saved SHAP feature importance plot.
    waterfall_path : str
        File path of the saved SHAP waterfall plot.
    """
    os.makedirs(plot_dir, exist_ok=True)
    
    # Create TreeExplainer
    explainer = shap.TreeExplainer(model)
    
    # Compute SHAP values as an Explanation object (preferred in modern SHAP)
    shap_explanation = explainer(X)
    
    # 1. SHAP Summary Plot
    plt.figure(figsize=(11, 7))
    try:
        # Modern Beeswarm API
        shap.plots.beeswarm(shap_explanation, show=False, max_display=12)
    except Exception:
        # Fallback to older summary_plot API
        shap_values_array = explainer.shap_values(X)
        shap.summary_plot(shap_values_array, X, show=False, max_display=12)
        
    plt.title("SHAP Summary Plot (Factors Influencing Churn)", fontsize=13, pad=15)
    plt.tight_layout()
    summary_path = os.path.join(plot_dir, "shap_summary_plot.png")
    plt.savefig(summary_path, dpi=150)
    plt.close()
    
    # 2. SHAP Feature Importance Plot (Bar)
    plt.figure(figsize=(11, 7))
    try:
        # Modern Bar API
        shap.plots.bar(shap_explanation, show=False, max_display=12)
    except Exception:
        # Fallback
        shap_values_array = explainer.shap_values(X)
        shap.summary_plot(shap_values_array, X, plot_type="bar", show=False, max_display=12)
        
    plt.title("SHAP Feature Importance (Bar Plot)", fontsize=13, pad=15)
    plt.tight_layout()
    importance_path = os.path.join(plot_dir, "shap_feature_importance_plot.png")
    plt.savefig(importance_path, dpi=150)
    plt.close()
    
    # 3. SHAP Waterfall Plot for a single customer (index 0)
    plt.figure(figsize=(11, 7))
    try:
        # Modern Waterfall API
        shap.plots.waterfall(shap_explanation[0], show=False, max_display=12)
    except Exception:
        # Legacy fallback
        shap_values_array = explainer.shap_values(X)
        expected_value = explainer.expected_value
        if isinstance(expected_value, np.ndarray):
            expected_value = expected_value[0] if len(expected_value) > 0 else expected_value
        if isinstance(shap_values_array, list):
            # Resolve binary class lists in some TreeExplainer versions
            shap_values_array = shap_values_array[1]
        shap.plots.waterfall(shap.Explanation(
            values=shap_values_array[0],
            base_values=expected_value,
            data=X.iloc[0],
            feature_names=X.columns.tolist()
        ), show=False, max_display=12)
        
    plt.title("SHAP Waterfall Plot (Single Customer Analysis)", fontsize=13, pad=15)
    plt.tight_layout()
    waterfall_path = os.path.join(plot_dir, "shap_waterfall_plot.png")
    plt.savefig(waterfall_path, dpi=150)
    plt.close()
    
    return summary_path, importance_path, waterfall_path
