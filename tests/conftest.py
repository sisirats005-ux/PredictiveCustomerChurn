"""
Pytest configuration shared across the test suite.

Ensures the project root (which contains the `src` package) is importable
regardless of the working directory pytest is invoked from, and provides
small reusable fixtures for synthetic Telco-style data.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

# Make `src` importable when tests are run from any directory, e.g.
# `pytest` from the project root or `pytest tests/` from elsewhere.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


@pytest.fixture
def raw_customer_row():
    """A single valid raw customer record, matching the Telco schema."""
    return {
        "customerID": "0000-TEST",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 80.0,
        "TotalCharges": "960.0",
        "Churn": "No",
    }


@pytest.fixture
def raw_df(raw_customer_row):
    """A small synthetic DataFrame (20 rows) mimicking the raw Telco dataset,
    including a blank TotalCharges value (new-customer edge case) and a
    balanced mix of churned/retained labels so stratified splitting is
    exercised without tripping sklearn's minimum-class-size requirement."""
    rows = []
    for i in range(20):
        row = dict(raw_customer_row)
        row["customerID"] = f"{i:04d}-TEST"
        row["tenure"] = i
        row["Churn"] = "Yes" if i % 2 == 0 else "No"
        # Vary gender so one-hot encoding has more than one category to
        # encode (a constant column correctly produces zero columns under
        # drop='first', which would make that behavior untestable here).
        row["gender"] = "Male" if i % 3 == 0 else "Female"
        rows.append(row)

    # New customer: zero tenure and blank TotalCharges (whitespace), the
    # documented edge case that clean_data() must handle.
    rows[0]["tenure"] = 0
    rows[0]["TotalCharges"] = "  "

    return pd.DataFrame(rows)
