"""
ConnectTel Customer Churn Prediction Preprocessing Module.
Provides data loading, raw data inspection, cleaning, train-test splitting,
and feature transformation (scaling and categorical encoding).
Includes robust logging and exception handling for production readiness.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from src.utils import setup_logger

logger = setup_logger(name="preprocessing", log_file="reports/pipeline.log")


def load_data(filepath):
    """
    Load dataset from a CSV file.

    Parameters
    ----------
    filepath : str
        The file path to the CSV dataset.

    Returns
    -------
    pd.DataFrame
        Loaded raw DataFrame.
    """
    logger.info(f"Loading raw data from {filepath}...")
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Successfully loaded data. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Failed to load data from {filepath}: {str(e)}")
        raise e


def inspect_data(df):
    """
    Perform a complete inspection of the raw dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to inspect.

    Returns
    -------
    dict
        A dictionary containing metadata: shape, columns, types, nulls,
        duplicates, and target class distribution.
    """
    logger.info("Inspecting raw DataFrame statistics...")
    try:
        inspection = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'null_values': df.isnull().sum().to_dict(),
            'duplicates': int(df.duplicated().sum()),
            'class_distribution': df['Churn'].value_counts(dropna=False).to_dict() 
                                  if 'Churn' in df.columns else {}
        }
        logger.info("DataFrame inspection completed successfully.")
        return inspection
    except Exception as e:
        logger.error(f"Error inspecting data: {str(e)}")
        raise e


def clean_data(df):
    """
    Clean the raw Telco dataset:
    - Converts TotalCharges to numeric, handling empty/whitespace values by
      imputing them with 0.0 (new customers with 0 tenure).
    - Drops customerID from modelling but preserves it separately.

    Parameters
    ----------
    df : pd.DataFrame
        The raw input DataFrame.

    Returns
    -------
    df_clean : pd.DataFrame
        Cleaned DataFrame (without customerID).
    customer_ids : pd.Series
        Series containing the preserved customer IDs.
    """
    logger.info("Cleaning raw dataset...")
    try:
        df_clean = df.copy()
        
        # Verify required columns exist
        if 'customerID' not in df_clean.columns:
            logger.warning("'customerID' column missing from input data. Proceeding without dropping it.")
            customer_ids = pd.Series(df_clean.index, name='customerID')
        else:
            # Store customer ID separately for reference
            customer_ids = df_clean['customerID']
            df_clean = df_clean.drop(columns=['customerID'])
        
        if 'TotalCharges' not in df_clean.columns:
            raise KeyError("'TotalCharges' column is missing from raw dataset.")
            
        # TotalCharges contains blank spaces ' ' which make it an object type.
        # Replace empty spaces with NaN first
        df_clean['TotalCharges'] = df_clean['TotalCharges'].replace(r'^\s*$', np.nan, regex=True)
        
        # Impute missing TotalCharges with 0.0 (since these are new customers with tenure = 0)
        df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges']).fillna(0.0)
        
        logger.info("Data cleaning completed successfully.")
        return df_clean, customer_ids
    except Exception as e:
        logger.error(f"Error cleaning data: {str(e)}")
        raise e


def split_data(df, target_col='Churn', test_size=0.2, random_state=42):
    """
    Split the dataset into training and testing sets, stratifying on the target class.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame.
    target_col : str, optional
        Target column name, default is 'Churn'.
    test_size : float, optional
        Proportion of test set, default is 0.2.
    random_state : int, optional
        Seed for reproducibility, default is 42.

    Returns
    -------
    X_train : pd.DataFrame
        Training features.
    X_test : pd.DataFrame
        Testing features.
    y_train : pd.Series
        Training target.
    y_test : pd.Series
        Testing target.
    """
    logger.info(f"Splitting data with test_size={test_size}, stratifying on '{target_col}'...")
    try:
        if target_col not in df.columns:
            raise KeyError(f"Target column '{target_col}' is missing from DataFrame.")
            
        X = df.drop(columns=[target_col])
        # Map target 'Yes'/'No' to binary 1/0
        y = df[target_col].map({'Yes': 1, 'No': 0})
        
        if y.isna().any():
            logger.warning("Target column contains values other than 'Yes' or 'No'. Standardizing values...")
            # Fallback for alternative target mapping
            y = np.where(df[target_col].astype(str).str.lower().isin(['yes', '1', 'true']), 1, 0)
            y = pd.Series(y, index=df.index)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        logger.info(f"Split complete. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logger.error(f"Error splitting data: {str(e)}")
        raise e


def preprocess_data(X_train, X_test, numeric_cols, categorical_cols, remainder_cols):
    """
    Preprocess train and test features using a ColumnTransformer:
    - Scales numerical features using StandardScaler.
    - One-hot encodes categorical features dropping first category.
    - Passes through binary columns.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    X_test : pd.DataFrame
        Testing features.
    numeric_cols : list of str
        Numerical features to scale.
    categorical_cols : list of str
        Categorical features to encode.
    remainder_cols : list of str
        Binary/passthrough columns.

    Returns
    -------
    X_train_df : pd.DataFrame
        Processed training features DataFrame with correct column names.
    X_test_df : pd.DataFrame
        Processed testing features DataFrame with correct column names.
    preprocessor : ColumnTransformer
        Fitted preprocessing ColumnTransformer.
    clean_feature_names : list of str
        Cleaned column names corresponding to processed features.
    """
    logger.info("Initializing ColumnTransformer preprocessing pipeline...")
    try:
        # Validate that columns exist in input
        missing_train = [c for c in (numeric_cols + categorical_cols + remainder_cols) if c not in X_train.columns]
        if missing_train:
            raise KeyError(f"The following expected columns are missing from training data: {missing_train}")
            
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_cols),
                ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols),
                ('passthrough', 'passthrough', remainder_cols)
            ]
        )
        
        # Fit and transform
        logger.info("Fitting and transforming training features...")
        X_train_trans = preprocessor.fit_transform(X_train)
        
        logger.info("Transforming testing features...")
        X_test_trans = preprocessor.transform(X_test)
        
        # Retrieve feature names from transformer
        feature_names = preprocessor.get_feature_names_out()
        
        # Clean up prefixes from feature names
        clean_feature_names = []
        for name in feature_names:
            if name.startswith('num__'):
                clean_feature_names.append(name[5:])
            elif name.startswith('cat__'):
                clean_feature_names.append(name[5:])
            elif name.startswith('passthrough__'):
                clean_feature_names.append(name[13:])
            else:
                clean_feature_names.append(name)
                
        # Convert arrays back to DataFrames to preserve feature names for SHAP
        X_train_df = pd.DataFrame(X_train_trans, columns=clean_feature_names, index=X_train.index)
        X_test_df = pd.DataFrame(X_test_trans, columns=clean_feature_names, index=X_test.index)
        
        logger.info(f"Preprocessing completed. Encoded features dimensionality: {X_train_df.shape[1]}")
        return X_train_df, X_test_df, preprocessor, clean_feature_names
    except Exception as e:
        logger.error(f"Error during preprocessing steps: {str(e)}")
        raise e
