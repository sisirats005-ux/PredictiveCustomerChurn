"""
Unit tests for src/preprocessing.py.

Covers: raw data loading, dataset inspection, TotalCharges/customerID
cleaning (including the blank-TotalCharges edge case), stratified
train/test splitting, and the ColumnTransformer feature preprocessing.
"""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    load_data,
    inspect_data,
    clean_data,
    split_data,
    preprocess_data,
)

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_COLS = ["gender", "Contract", "PaymentMethod"]
REMAINDER_COLS = ["SeniorCitizen"]


class TestLoadData:
    def test_load_data_returns_dataframe(self, tmp_path, raw_df):
        csv_path = tmp_path / "sample.csv"
        raw_df.to_csv(csv_path, index=False)

        loaded = load_data(str(csv_path))

        assert isinstance(loaded, pd.DataFrame)
        assert list(loaded.columns) == list(raw_df.columns)
        assert len(loaded) == len(raw_df)


class TestInspectData:
    def test_inspection_reports_shape_and_class_distribution(self, raw_df):
        inspection = inspect_data(raw_df)

        assert inspection["shape"] == raw_df.shape
        assert set(inspection["columns"]) == set(raw_df.columns)
        assert inspection["class_distribution"] == raw_df["Churn"].value_counts().to_dict()

    def test_inspection_handles_missing_target_column(self, raw_df):
        df_no_target = raw_df.drop(columns=["Churn"])

        inspection = inspect_data(df_no_target)

        assert inspection["class_distribution"] == {}


class TestCleanData:
    def test_customer_id_removed_from_frame_but_preserved_separately(self, raw_df):
        df_clean, customer_ids = clean_data(raw_df)

        assert "customerID" not in df_clean.columns
        assert list(customer_ids) == list(raw_df["customerID"])

    def test_blank_total_charges_imputed_with_zero(self, raw_df):
        # The fixture's first row has a whitespace-only TotalCharges value,
        # matching the real dataset's known "new customer" edge case.
        df_clean, _ = clean_data(raw_df)

        assert df_clean.loc[0, "TotalCharges"] == 0.0
        assert pd.api.types.is_numeric_dtype(df_clean["TotalCharges"])

    def test_valid_total_charges_parsed_as_float(self, raw_df):
        df_clean, _ = clean_data(raw_df)

        # Row 1 in the fixture keeps its original numeric-string TotalCharges.
        assert df_clean.loc[1, "TotalCharges"] == pytest.approx(960.0)

    def test_clean_data_does_not_mutate_input(self, raw_df):
        original = raw_df.copy(deep=True)

        clean_data(raw_df)

        pd.testing.assert_frame_equal(raw_df, original)


class TestSplitData:
    def test_split_shapes_and_target_encoding(self, raw_df):
        df_clean, _ = clean_data(raw_df)

        X_train, X_test, y_train, y_test = split_data(df_clean, test_size=0.25, random_state=42)

        assert len(X_train) + len(X_test) == len(df_clean)
        assert "Churn" not in X_train.columns
        # Target should be mapped from Yes/No strings to binary 1/0
        assert set(y_train.unique()) <= {0, 1}
        assert set(y_test.unique()) <= {0, 1}

    def test_split_is_stratified(self, raw_df):
        df_clean, _ = clean_data(raw_df)
        original_rate = (df_clean["Churn"] == "Yes").mean()

        X_train, X_test, y_train, y_test = split_data(df_clean, test_size=0.5, random_state=42)

        # With a 50/50 balanced fixture, both splits should preserve the rate.
        assert y_train.mean() == pytest.approx(original_rate, abs=0.15)
        assert y_test.mean() == pytest.approx(original_rate, abs=0.15)

    def test_split_is_reproducible_with_fixed_seed(self, raw_df):
        df_clean, _ = clean_data(raw_df)

        X_train_1, X_test_1, _, _ = split_data(df_clean, random_state=42)
        X_train_2, X_test_2, _, _ = split_data(df_clean, random_state=42)

        pd.testing.assert_frame_equal(X_train_1, X_train_2)
        pd.testing.assert_frame_equal(X_test_1, X_test_2)


class TestPreprocessData:
    def test_output_shapes_and_no_missing_values(self, raw_df):
        df_clean, _ = clean_data(raw_df)
        X_train, X_test, y_train, y_test = split_data(df_clean, test_size=0.25, random_state=42)

        X_train_df, X_test_df, preprocessor, feature_names = preprocess_data(
            X_train, X_test, NUMERIC_COLS, CATEGORICAL_COLS, REMAINDER_COLS
        )

        assert X_train_df.shape[0] == len(X_train)
        assert X_test_df.shape[0] == len(X_test)
        assert X_train_df.shape[1] == len(feature_names)
        assert not X_train_df.isnull().any().any()
        assert not X_test_df.isnull().any().any()

    def test_numeric_columns_are_standardized(self, raw_df):
        df_clean, _ = clean_data(raw_df)
        X_train, X_test, y_train, y_test = split_data(df_clean, test_size=0.25, random_state=42)

        X_train_df, _, _, _ = preprocess_data(
            X_train, X_test, NUMERIC_COLS, CATEGORICAL_COLS, REMAINDER_COLS
        )

        # StandardScaler should center training data at ~0 mean.
        for col in NUMERIC_COLS:
            assert X_train_df[col].mean() == pytest.approx(0.0, abs=1e-8)

    def test_categorical_columns_are_one_hot_encoded_with_drop_first(self, raw_df):
        df_clean, _ = clean_data(raw_df)
        X_train, X_test, y_train, y_test = split_data(df_clean, test_size=0.25, random_state=42)

        _, _, _, feature_names = preprocess_data(
            X_train, X_test, NUMERIC_COLS, CATEGORICAL_COLS, REMAINDER_COLS
        )

        # drop='first' means gender (2 categories) contributes exactly 1 column.
        gender_cols = [f for f in feature_names if f.startswith("gender_")]
        assert len(gender_cols) == 1

    def test_remainder_columns_pass_through_unchanged(self, raw_df):
        df_clean, _ = clean_data(raw_df)
        X_train, X_test, y_train, y_test = split_data(df_clean, test_size=0.25, random_state=42)

        X_train_df, _, _, _ = preprocess_data(
            X_train, X_test, NUMERIC_COLS, CATEGORICAL_COLS, REMAINDER_COLS
        )

        # SeniorCitizen is passthrough (0/1 int), so it must not be scaled.
        assert set(X_train_df["SeniorCitizen"].unique()) <= {0, 1}

    def test_unseen_category_at_transform_time_is_handled_gracefully(self, raw_df):
        # handle_unknown='ignore' should prevent a crash when the test set
        # (or, by extension, a future production batch) contains a category
        # never seen during fit.
        df_clean, _ = clean_data(raw_df)
        X_train, X_test, y_train, y_test = split_data(df_clean, test_size=0.25, random_state=42)
        X_test = X_test.copy()
        X_test.iloc[0, X_test.columns.get_loc("PaymentMethod")] = "Cryptocurrency"

        # Should not raise.
        _, X_test_df, _, _ = preprocess_data(
            X_train, X_test, NUMERIC_COLS, CATEGORICAL_COLS, REMAINDER_COLS
        )

        assert not X_test_df.isnull().any().any()
