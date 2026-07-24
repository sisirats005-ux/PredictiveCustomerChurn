"""
ConnectTel Customer Churn Prediction Feature Engineering Module.
Creates domain-specific customer indicators: ServiceCount, TotalChargesPerTenure,
HasFamily, LongTermCustomer, AutoPayment, and advanced interaction/segmentation metrics.
"""

import pandas as pd
import numpy as np


def create_features(df):
    """
    Create engineered features from the cleaned Telco Churn DataFrame.

    Engineered features:
    1. ServiceCount (int): Total number of active services the customer subscribes to.
    2. TotalChargesPerTenure (float): TotalCharges divided by tenure. Safe division.
    3. HasFamily (int): Binary indicator if the customer has a partner OR dependents.
    4. LongTermCustomer (int): Binary indicator if tenure is > 24 months.
    5. AutoPayment (int): Binary indicator if payment method is automatic.
    
    Advanced features:
    6. Fiber_x_MonthToMonth (int): Interaction between Fiber Internet and Month-to-Month contract.
    7. Fiber_x_NoTechSupport (int): Interaction between Fiber Internet and having no Tech Support.
    8. MonthToMonth_x_NoAutoPay (int): Interaction between Month-to-Month contract and manual payment.
    9. HighRiskContractPay (int): Month-to-month contract paying by Electronic check.
    10. CLV (float): Estimated customer lifetime value = MonthlyCharges * (tenure + ContractBonus).
        Where ContractBonus is: Month-to-month: 6, One year: 12, Two year: 24.
    11. ChargesRatio (float): MonthlyCharges / (TotalCharges + MonthlyCharges) - density of monthly billing.
    12. BillingRisk (float): MonthlyCharges * (1 - AutoPayment) * (Contract == 'Month-to-month').
    13. HighRiskProfile (int): 1 if customer has at least 2 of (Month-to-month, Electronic check, Fiber optic), else 0.
    14. Persona_New_HighSpend (int): Tenure <= 12 and MonthlyCharges > 70.
    15. Persona_Loyal_HighSpend (int): Tenure > 12 and MonthlyCharges > 70.

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned input DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame containing original features alongside newly engineered features.
    """
    df_engineered = df.copy()
    
    # 1. ServiceCount: Count active phone/internet/security/streaming lines
    services = [
        (df_engineered['PhoneService'] == 'Yes'),
        (df_engineered['MultipleLines'] == 'Yes'),
        df_engineered['InternetService'].isin(['DSL', 'Fiber optic']),
        (df_engineered['OnlineSecurity'] == 'Yes'),
        (df_engineered['OnlineBackup'] == 'Yes'),
        (df_engineered['DeviceProtection'] == 'Yes'),
        (df_engineered['TechSupport'] == 'Yes'),
        (df_engineered['StreamingTV'] == 'Yes'),
        (df_engineered['StreamingMovies'] == 'Yes')
    ]
    df_engineered['ServiceCount'] = sum(s.astype(int) for s in services)
    
    # 2. TotalChargesPerTenure: TotalCharges / tenure (safe division-by-zero)
    df_engineered['TotalChargesPerTenure'] = np.where(
        df_engineered['tenure'] > 0,
        df_engineered['TotalCharges'] / df_engineered['tenure'],
        df_engineered['MonthlyCharges']
    )
    
    # 3. HasFamily: 1 if Partner is 'Yes' OR Dependents is 'Yes', else 0
    df_engineered['HasFamily'] = np.where(
        (df_engineered['Partner'] == 'Yes') | (df_engineered['Dependents'] == 'Yes'),
        1,
        0
    )
    
    # 4. LongTermCustomer: 1 if tenure > 24, else 0
    df_engineered['LongTermCustomer'] = np.where(
        df_engineered['tenure'] > 24,
        1,
        0
    )
    
    # 5. AutoPayment: 1 if PaymentMethod is automatic, else 0
    df_engineered['AutoPayment'] = np.where(
        df_engineered['PaymentMethod'].str.contains('automatic', case=False, na=False),
        1,
        0
    )

    # 6. Fiber_x_MonthToMonth: Fiber Optic and Month-to-Month
    df_engineered['Fiber_x_MonthToMonth'] = np.where(
        (df_engineered['InternetService'] == 'Fiber optic') & (df_engineered['Contract'] == 'Month-to-month'),
        1,
        0
    )

    # 7. Fiber_x_NoTechSupport: Fiber Optic and No Tech Support
    df_engineered['Fiber_x_NoTechSupport'] = np.where(
        (df_engineered['InternetService'] == 'Fiber optic') & (df_engineered['TechSupport'] == 'No'),
        1,
        0
    )

    # 8. MonthToMonth_x_NoAutoPay: Month-to-Month and Manual Pay
    df_engineered['MonthToMonth_x_NoAutoPay'] = np.where(
        (df_engineered['Contract'] == 'Month-to-month') & (df_engineered['AutoPayment'] == 0),
        1,
        0
    )

    # 9. HighRiskContractPay: Month-to-month contract paying by Electronic check
    df_engineered['HighRiskContractPay'] = np.where(
        (df_engineered['Contract'] == 'Month-to-month') & (df_engineered['PaymentMethod'] == 'Electronic check'),
        1,
        0
    )

    # 10. CLV: Estimated lifetime value based on monthly charges and contract bonus + tenure
    contract_bonus = np.select(
        condlist=[
            df_engineered['Contract'] == 'Two year',
            df_engineered['Contract'] == 'One year',
            df_engineered['Contract'] == 'Month-to-month'
        ],
        choicelist=[24.0, 12.0, 6.0],
        default=6.0
    )
    df_engineered['CLV'] = df_engineered['MonthlyCharges'] * (df_engineered['tenure'] + contract_bonus)

    # 11. ChargesRatio: billing intensity (MonthlyCharges divided by Total + Monthly)
    df_engineered['ChargesRatio'] = df_engineered['MonthlyCharges'] / (df_engineered['TotalCharges'] + df_engineered['MonthlyCharges'])

    # 12. BillingRisk: billing friction metric
    df_engineered['BillingRisk'] = df_engineered['MonthlyCharges'] * (1.0 - df_engineered['AutoPayment']) * (df_engineered['Contract'] == 'Month-to-month').astype(float)

    # 13. HighRiskProfile: multi-factor risk segmentation
    risk_factors = (
        (df_engineered['Contract'] == 'Month-to-month').astype(int) +
        (df_engineered['PaymentMethod'] == 'Electronic check').astype(int) +
        (df_engineered['InternetService'] == 'Fiber optic').astype(int)
    )
    df_engineered['HighRiskProfile'] = np.where(risk_factors >= 2, 1, 0)

    # 14 & 15. Customer Personas (Clustering proxy)
    df_engineered['Persona_New_HighSpend'] = np.where(
        (df_engineered['tenure'] <= 12) & (df_engineered['MonthlyCharges'] > 70.0),
        1,
        0
    )
    df_engineered['Persona_Loyal_HighSpend'] = np.where(
        (df_engineered['tenure'] > 12) & (df_engineered['MonthlyCharges'] > 70.0),
        1,
        0
    )

    return df_engineered
