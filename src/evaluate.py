"""
ConnectTel Customer Churn Prediction Evaluation Module.
Computes evaluation metrics (Accuracy, Precision, Recall, F1, ROC-AUC),
plots Confusion Matrices and ROC Curves, executes cross-validation,
and compiles comparison tables.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, classification_report
)
from sklearn.model_selection import cross_validate


def evaluate_model(model, X, y, model_name, threshold=0.5):
    """
    Compute standard classification evaluation metrics on a test set.

    Parameters
    ----------
    model : estimator
        Fitted classifier model.
    X : pd.DataFrame or np.ndarray
        Test features.
    y : pd.Series or np.ndarray
        True target labels.
    model_name : str
        Name of the model being evaluated.
    threshold : float, optional
        Classification threshold for mapping probabilities to binary labels,
        default is 0.5.

    Returns
    -------
    metrics : dict
        Dictionary of metrics containing Accuracy, Precision, Recall,
        F1_Score, and ROC_AUC.
    preds : np.ndarray
        Binary predictions for target labels.
    probs : np.ndarray
        Class probability outputs.
    """
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:, 1]
    else:
        probs = model.decision_function(X)
        # Normalization fallback for non-probability decision models
        probs = (probs - probs.min()) / (probs.max() - probs.min() + 1e-5)
        
    preds = (probs >= threshold).astype(int)
    
    metrics = {
        'Accuracy': float(accuracy_score(y, preds)),
        'Precision': float(precision_score(y, preds, zero_division=0)),
        'Recall': float(recall_score(y, preds, zero_division=0)),
        'F1_Score': float(f1_score(y, preds, zero_division=0)),
        'ROC_AUC': float(roc_auc_score(y, probs))
    }
    return metrics, preds, probs


def generate_evaluation_plots(y_true, probs, preds, model_name, plot_dir="outputs/plots"):
    """
    Generate and save Confusion Matrix and ROC Curve plots for a model.

    Parameters
    ----------
    y_true : pd.Series or np.ndarray
        True target labels.
    probs : np.ndarray
        Model output probabilities.
    preds : np.ndarray
        Model binary predictions.
    model_name : str
        Name of the model.
    plot_dir : str, optional
        Directory path to save the plots, default is "outputs/plots".

    Returns
    -------
    cm_path : str
        File path of the saved Confusion Matrix.
    roc_path : str
        File path of the saved ROC Curve.
    """
    os.makedirs(plot_dir, exist_ok=True)
    
    # 1. Confusion Matrix Plot
    cm = confusion_matrix(y_true, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', cbar=False,
        xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn']
    )
    plt.title(f"Confusion Matrix - {model_name}", fontsize=13, pad=15)
    plt.ylabel('Actual Class', fontsize=11)
    plt.xlabel('Predicted Class', fontsize=11)
    plt.tight_layout()
    cm_path = os.path.join(plot_dir, f"evaluation_{model_name.lower().replace(' ', '_')}_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    
    # 2. ROC Curve Plot
    fpr, tpr, _ = roc_curve(y_true, probs)
    auc_val = roc_auc_score(y_true, probs)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='#d95f02', lw=2.5, label=f'ROC Curve (AUC = {auc_val:.3f})')
    plt.plot([0, 1], [0, 1], color='#2b5c8f', lw=2, linestyle='--', label='Random Guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=11)
    plt.ylabel('True Positive Rate (TPR)', fontsize=11)
    plt.title(f'ROC Curve - {model_name}', fontsize=13, pad=15)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    roc_path = os.path.join(plot_dir, f"evaluation_{model_name.lower().replace(' ', '_')}_roc_curve.png")
    plt.savefig(roc_path, dpi=150)
    plt.close()
    
    return cm_path, roc_path


def evaluate_naive_baseline(y_true):
    """
    Compute metrics for a naive majority-class classifier that always predicts
    the majority class ("No Churn"). Used to prove that trained models provide
    real lift over a trivial baseline, since raw accuracy alone is misleading
    on an imbalanced churn dataset.

    Parameters
    ----------
    y_true : pd.Series or np.ndarray
        True target labels for the test set (1 = Churn, 0 = No Churn).

    Returns
    -------
    metrics : dict
        Dictionary of metrics containing Accuracy, Precision, Recall,
        F1_Score, and ROC_AUC for the naive majority-class baseline.
    """
    preds = np.zeros(len(y_true), dtype=int)
    probs = np.zeros(len(y_true), dtype=float)

    metrics = {
        'Accuracy': float(accuracy_score(y_true, preds)),
        'Precision': float(precision_score(y_true, preds, zero_division=0)),
        'Recall': float(recall_score(y_true, preds, zero_division=0)),
        'F1_Score': float(f1_score(y_true, preds, zero_division=0)),
        'ROC_AUC': float(roc_auc_score(y_true, probs))
    }
    return metrics


def run_cross_validation(model, X, y, cv=5):
    """
    Perform 5-Fold Cross Validation and calculate mean Accuracy and mean ROC-AUC.

    Parameters
    ----------
    model : estimator
        Classifier to evaluate.
    X : pd.DataFrame
        Training features.
    y : pd.Series
        Training target.
    cv : int, optional
        Number of cross validation folds, default is 5.

    Returns
    -------
    mean_accuracy : float
        Mean accuracy across folds.
    mean_roc_auc : float
        Mean ROC-AUC across folds.
    """
    cv_results = cross_validate(
        model, X, y, cv=cv,
        scoring=['accuracy', 'roc_auc'],
        return_train_score=False
    )
    
    mean_accuracy = float(np.mean(cv_results['test_accuracy']))
    mean_roc_auc = float(np.mean(cv_results['test_roc_auc']))
    
    return mean_accuracy, mean_roc_auc


def compile_comparison_report(models_metrics, output_file="reports/model_comparison.csv"):
    """
    Compile metrics from multiple models into a single summary comparison table.

    Saves the output as a CSV and orders the models based on ROC-AUC.

    Parameters
    ----------
    models_metrics : dict
        A nested dictionary of the format {model_name: {metric_name: value}}.
    output_file : str, optional
        Path to save the comparison CSV file, default is "reports/model_comparison.csv".

    Returns
    -------
    df_metrics : pd.DataFrame
        Compiled comparison DataFrame.
    """
    df_metrics = pd.DataFrame(models_metrics).T
    df_metrics = df_metrics.round(4)
    
    # Sort models by ROC_AUC and F1 Score descending to find best estimator
    df_metrics = df_metrics.sort_values(by=['ROC_AUC', 'F1_Score'], ascending=False)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_metrics.to_csv(output_file, index=True)
    
    return df_metrics
