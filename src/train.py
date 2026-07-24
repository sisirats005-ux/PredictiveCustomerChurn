"""
ConnectTel Customer Churn Prediction Training Module.
Handles baseline model initialization, training, and GridSearchCV hyperparameter tuning
for Random Forest and XGBoost classifiers.
Incorporates parameters from central configuration and includes robust logging.
"""

import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from src.utils import setup_logger

logger = setup_logger(name="train", log_file="reports/pipeline.log")


def train_baseline_models(X_train, y_train, random_state=42):
    """
    Train baseline estimators (Logistic Regression, Random Forest, and XGBoost).

    Parameters
    ----------
    X_train : pd.DataFrame
        Preprocessed training features.
    y_train : pd.Series
        Training target variable.
    random_state : int, optional
        Seed for reproducibility, default is 42.

    Returns
    -------
    models : dict of str: estimator
        Dictionary mapping model names to fitted scikit-learn/XGBoost estimators.
    """
    logger.info("Initializing baseline models training process...")
    try:
        # Calculate scale_pos_weight for XGBoost to handle target imbalance
        neg_count = sum(y_train == 0)
        pos_count = sum(y_train == 1)
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
        logger.info(f"Class counts in train - Negatives: {neg_count}, Positives: {pos_count}")
        logger.info(f"Computed scale_pos_weight for XGBoost: {scale_pos_weight:.3f}")
        
        # 1. Logistic Regression
        lr = LogisticRegression(
            class_weight='balanced',
            random_state=random_state,
            max_iter=1000
        )
        
        # 2. Random Forest
        rf = RandomForestClassifier(
            class_weight='balanced',
            random_state=random_state,
            n_estimators=100
        )
        
        # 3. XGBoost
        xgb = XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            eval_metric='logloss'
        )
        
        logger.info("Fitting Logistic Regression baseline model...")
        lr.fit(X_train, y_train)
        
        logger.info("Fitting Random Forest baseline model...")
        rf.fit(X_train, y_train)
        
        logger.info("Fitting XGBoost baseline model...")
        xgb.fit(X_train, y_train)
        
        models = {
            "Logistic Regression": lr,
            "Random Forest": rf,
            "XGBoost": xgb
        }
        
        logger.info("Baseline model training completed successfully.")
        return models
    except Exception as e:
        logger.error(f"Error during baseline training: {str(e)}")
        raise e


def tune_hyperparameters(X_train, y_train, rf_grid=None, xgb_grid=None, cv_folds=5, scoring='roc_auc', random_state=42):
    """
    Perform GridSearchCV hyperparameter tuning on Random Forest and XGBoost.

    Parameters
    ----------
    X_train : pd.DataFrame
        Preprocessed training features.
    y_train : pd.Series
        Training target variable.
    rf_grid : dict, optional
        Parameter grid for Random Forest. If None, uses default grids.
    xgb_grid : dict, optional
        Parameter grid for XGBoost. If None, uses default grids.
    cv_folds : int, optional
        Number of cross validation folds, default is 5.
    scoring : str, optional
        Scoring metric for GridSearch, default is 'roc_auc'.
    random_state : int, optional
        Seed for reproducibility, default is 42.

    Returns
    -------
    tuned_models : dict of str: estimator
        Dictionary mapping model names to their optimized, refitted estimators.
    tuning_info : dict
        Metadata containing GridSearchCV best parameters and cross-validation scores.
    """
    logger.info("Starting hyperparameter tuning process...")
    try:
        # Compute scale_pos_weight for XGBoost
        neg_count = sum(y_train == 0)
        pos_count = sum(y_train == 1)
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

        # Define default grids if none provided
        if rf_grid is None:
            rf_grid = {
                'n_estimators': [100, 200],
                'max_depth': [5, 8, None],
                'min_samples_split': [2, 5]
            }
        if xgb_grid is None:
            xgb_grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 5],
                'learning_rate': [0.05, 0.1]
            }

        # 1. Random Forest GridSearch
        rf_base = RandomForestClassifier(class_weight='balanced', random_state=random_state)
        logger.info(f"Tuning Random Forest via {cv_folds}-Fold GridSearchCV using grid: {rf_grid}")
        rf_cv = GridSearchCV(
            estimator=rf_base,
            param_grid=rf_grid,
            cv=cv_folds,
            scoring=scoring,
            n_jobs=-1
        )
        rf_cv.fit(X_train, y_train)
        logger.info(f"Random Forest Tuning Complete. Best Params: {rf_cv.best_params_}")
        logger.info(f"Random Forest Best CV ROC-AUC: {rf_cv.best_score_:.4f}")
        
        # 2. XGBoost GridSearch
        xgb_base = XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            eval_metric='logloss'
        )
        logger.info(f"Tuning XGBoost via {cv_folds}-Fold GridSearchCV using grid: {xgb_grid}")
        xgb_cv = GridSearchCV(
            estimator=xgb_base,
            param_grid=xgb_grid,
            cv=cv_folds,
            scoring=scoring,
            n_jobs=-1
        )
        xgb_cv.fit(X_train, y_train)
        logger.info(f"XGBoost Tuning Complete. Best Params: {xgb_cv.best_params_}")
        logger.info(f"XGBoost Best CV ROC-AUC: {xgb_cv.best_score_:.4f}")
        
        tuned_models = {
            "Tuned Random Forest": rf_cv.best_estimator_,
            "Tuned XGBoost": xgb_cv.best_estimator_
        }
        
        tuning_info = {
            "Random Forest": {
                "best_params": rf_cv.best_params_,
                "best_cv_auc": float(rf_cv.best_score_)
            },
            "XGBoost": {
                "best_params": xgb_cv.best_params_,
                "best_cv_auc": float(xgb_cv.best_score_)
            }
        }
        
        logger.info("Hyperparameter tuning process completed successfully.")
        return tuned_models, tuning_info
    except Exception as e:
        logger.error(f"Error during hyperparameter tuning: {str(e)}")
        raise e
