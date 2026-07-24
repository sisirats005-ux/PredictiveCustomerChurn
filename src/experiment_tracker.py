"""
ConnectTel Churn Prediction Experiment Tracking Module.
Logs experiment metrics, parameters, and metadata to a persistent local JSON registry
and optionally integrates with MLflow.
"""

import os
import json
import time
from datetime import datetime
import numpy as np
from src.utils import setup_logger, ensure_directory

logger = setup_logger(name="experiment_tracker", log_file="reports/pipeline.log")

# Optional MLflow Import
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


def log_experiment(run_name, parameters, metrics, model_type, log_dir="outputs/metrics"):
    """
    Log an experiment run containing hyperparameters, performance metrics,
    and metadata to a local JSON registry and MLflow (if active).

    Parameters
    ----------
    run_name : str
        Descriptive name of the experiment run.
    parameters : dict
        Hyperparameters used during training.
    metrics : dict
        Evaluation metrics (accuracy, precision, recall, AUC, etc.).
    model_type : str
        Type of algorithm (e.g. XGBoost, Random Forest).
    log_dir : str, optional
        Target directory to save the JSON log registry, default is 'outputs/metrics'.

    Returns
    -------
    dict
        The compiled run metadata dict that was logged.
    """
    ensure_directory(log_dir)
    json_path = os.path.join(log_dir, "experiment_runs.json")
    
    timestamp = datetime.now().isoformat()
    run_id = f"run_{int(time.time())}"
    
    run_data = {
        "run_id": run_id,
        "run_name": run_name,
        "timestamp": timestamp,
        "model_type": model_type,
        "parameters": parameters,
        "metrics": metrics
    }
    
    # 1. Local JSON File Logging
    logger.info(f"Logging experiment run '{run_name}' to local JSON registry...")
    try:
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                history = json.load(f)
        else:
            history = []
            
        history.append(run_data)
        
        with open(json_path, "w") as f:
            json.dump(history, f, indent=4)
        logger.info(f"Local logging complete. Registry has {len(history)} total entries.")
    except Exception as e:
        logger.error(f"Failed to write to local experiment registry: {str(e)}")
    
    # 2. MLflow Integration
    if MLFLOW_AVAILABLE:
        try:
            logger.info("Initializing MLflow experiment logging...")
            # Try to set experiment
            mlflow.set_experiment("ConnectTel_Customer_Churn")
            
            # Start run (autoclosed by context manager or nested handle)
            with mlflow.start_run(run_name=run_name, nested=True) as run:
                # Log tags
                mlflow.set_tag("model_type", model_type)
                mlflow.set_tag("logged_at", timestamp)
                
                # Log params
                # Flatten dictionary if nested
                flat_params = {}
                for k, v in parameters.items():
                    if isinstance(v, dict):
                        for sub_k, sub_v in v.items():
                            flat_params[f"{k}_{sub_k}"] = str(sub_v)
                    else:
                        flat_params[k] = str(v)
                mlflow.log_params(flat_params)
                
                # Log metrics
                # Filter metrics to only numeric values
                numeric_metrics = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float, np.integer, np.floating))}
                mlflow.log_metrics(numeric_metrics)
                
                logger.info(f"MLflow run logged successfully. Run ID: {run.info.run_id}")
        except Exception as e:
            logger.warning(f"MLflow experiment logging skipped/failed: {str(e)}. (Local logs preserved.)")
    else:
        logger.info("MLflow package not installed or import failed. Skipping MLflow tracking.")
        
    return run_data
