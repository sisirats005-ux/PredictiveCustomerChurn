"""
ConnectTel Customer Churn Prediction Utilities Module.
Contains helper utilities for setting up centralized logging systems,
managing directory architectures, and loading Yaml configuration files.
"""

import os
import logging
import yaml


def setup_logger(name="churn_project", log_file="reports/pipeline.log"):
    """
    Configure and return a dual-handler logging system.

    Logs are outputted simultaneously to the standard console stdout
    and a persistent text log file.

    Parameters
    ----------
    name : str, optional
        Unique identifier for the logger instance, default is "churn_project".
    log_file : str, optional
        File system path to write log logs, default is "reports/pipeline.log".

    Returns
    -------
    logger : logging.Logger
        Fitted logging service instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Check if handlers are already attached to prevent duplicate logging
    if not logger.handlers:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # 1. Console Stream Handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        # 2. File Write Handler
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
    return logger


def ensure_directory(path):
    """
    Verify the existence of a folder path, creating it recursively if absent.

    Parameters
    ----------
    path : str
        Directory folder path to verify or create.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def load_config(config_path="config.yaml"):
    """
    Load configuration parameters from a YAML file.

    Parameters
    ----------
    config_path : str, optional
        Path to the YAML configuration file, default is "config.yaml".

    Returns
    -------
    config : dict
        Parsed configuration dictionary.
    """
    if not os.path.exists(config_path):
        # Fallback search in parent directory if executed from src/
        alt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", config_path))
        if os.path.exists(alt_path):
            config_path = alt_path
        else:
            raise FileNotFoundError(f"Configuration file '{config_path}' not found.")
            
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config
