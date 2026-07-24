"""
ConnectTel Customer Churn Prediction Hypothesis Testing Module.
Performs formal statistical significance tests (Chi-Square Test of
Independence, Welch's t-test) for the hypotheses posed in the project
brief, rather than relying on descriptive EDA plots alone.
"""

import os
import json
import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind


def test_categorical_association(df, group_col, target_col='Churn',
                                   group_values=None, alpha=0.05):
    """
    Run a Chi-Square Test of Independence between a categorical feature
    and the churn target.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned input DataFrame.
    group_col : str
        Categorical column to test against churn (e.g. 'InternetService').
    target_col : str, optional
        Target column name, default is 'Churn'.
    group_values : list of str or None, optional
        If provided, restrict the test to these category values only
        (e.g. ['Fiber optic', 'DSL']) rather than all categories.
    alpha : float, optional
        Significance threshold, default is 0.05.

    Returns
    -------
    dict
        Test statistic, degrees of freedom, p-value, per-group churn
        rates, and a plain-language significance verdict.
    """
    data = df.copy()
    if group_values is not None:
        data = data[data[group_col].isin(group_values)]

    contingency = pd.crosstab(data[group_col], data[target_col])
    chi2_stat, p_val, dof, _ = chi2_contingency(contingency)

    churn_rates = (
        data.groupby(group_col)[target_col]
        .apply(lambda s: (s == 'Yes').mean())
        .to_dict()
    )

    return {
        'group_col': group_col,
        'chi2_statistic': float(chi2_stat),
        'dof': int(dof),
        'p_value': float(p_val),
        'churn_rates': churn_rates,
        'significant': bool(p_val < alpha),
    }


def test_numeric_difference(df, numeric_col, target_col='Churn', alpha=0.05):
    """
    Run Welch's t-test comparing a numeric column's mean between
    churned and retained customers.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned input DataFrame.
    numeric_col : str
        Numeric column to test (e.g. 'MonthlyCharges').
    target_col : str, optional
        Target column name, default is 'Churn'.
    alpha : float, optional
        Significance threshold, default is 0.05.

    Returns
    -------
    dict
        t-statistic, p-value, group means, and a significance verdict.
    """
    churn_vals = df.loc[df[target_col] == 'Yes', numeric_col]
    no_churn_vals = df.loc[df[target_col] == 'No', numeric_col]

    t_stat, p_val = ttest_ind(churn_vals, no_churn_vals, equal_var=False)

    return {
        'numeric_col': numeric_col,
        't_statistic': float(t_stat),
        'p_value': float(p_val),
        'mean_churn': float(churn_vals.mean()),
        'mean_no_churn': float(no_churn_vals.mean()),
        'significant': bool(p_val < alpha),
    }


def run_hypothesis_tests(df, output_file="outputs/metrics/hypothesis_tests.json"):
    """
    Run all formal hypothesis tests posed in the project brief and
    persist the results to a JSON file for reuse in reporting.

    Tests run:
    1. Internet Service (Fiber optic vs. DSL) vs. Churn -- Chi-Square.
    2. Partner status vs. Churn -- Chi-Square.
    3. MonthlyCharges (Churn vs. No Churn) -- Welch's t-test.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned input DataFrame (post `clean_data`, pre train/test split).
    output_file : str, optional
        Path to save the JSON results summary.

    Returns
    -------
    dict
        Nested dictionary of all test results, keyed by test name.
    """
    results = {
        'fiber_vs_dsl_churn': test_categorical_association(
            df, group_col='InternetService', group_values=['Fiber optic', 'DSL']
        ),
        'partner_vs_churn': test_categorical_association(
            df, group_col='Partner'
        ),
        'monthly_charges_vs_churn': test_numeric_difference(
            df, numeric_col='MonthlyCharges'
        ),
    }

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    return results

