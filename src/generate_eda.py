"""
ConnectTel Customer Churn Prediction EDA Module.
Automates the generation of 13 professional, styled plots and saves
them to outputs/plots/ for document inclusion and visual analysis.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root directory to path for relative imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import load_data, clean_data
from src.utils import setup_logger, ensure_directory

logger = setup_logger(name="generate_eda", log_file="reports/pipeline.log")


def main():
    """
    Execute raw data loading, cleaning, and generate 13 publication-quality plots.
    Saves all visual plots under outputs/plots/.
    """
    filepath = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    logger.info(f"Loading data for EDA from {filepath}...")
    df_raw = load_data(filepath)
    df, _ = clean_data(df_raw)
    
    plot_dir = "outputs/plots"
    ensure_directory(plot_dir)
    logger.info(f"EDA plots will be saved to {plot_dir}/")
    
    # Establish Seaborn styling theme
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 13,
        'axes.titlesize': 15,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'figure.titlesize': 16,
        'figure.figsize': (10, 6)
    })
    
    # Color palette matching deep corporate blue/teal (No Churn) and soft red/coral (Churn)
    palette = {0: "#2b5c8f", 1: "#d95f02", "No": "#2b5c8f", "Yes": "#d95f02"}
    
    df_plot = df.copy()
    
    # 1. Target Distribution
    plt.figure()
    ax = sns.countplot(data=df_plot, x='Churn', hue='Churn', palette=palette, legend=False)
    total = len(df_plot)
    for p in ax.patches:
        height = p.get_height()
        percentage = f'{100 * height / total:.1f}%'
        ax.annotate(
            f'{height}\n({percentage})', (p.get_x() + p.get_width() / 2., height / 2),
            ha='center', va='center', xytext=(0, 0), textcoords='offset points',
            color='white', fontweight='bold'
        )
    plt.title("ConnectTel Customer Churn Distribution")
    plt.xlabel("Churn Status")
    plt.ylabel("Number of Customers")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/01_target_distribution.png", dpi=150)
    plt.close()
    
    # Grouped count plot helper
    def plot_grouped_count(df_data, x_col, title, filename):
        plt.figure(figsize=(10, 6))
        ax_plot = sns.countplot(data=df_data, x=x_col, hue='Churn', palette=palette)
        plt.title(title)
        plt.xlabel(x_col)
        plt.ylabel("Count")
        
        # Annotate labels
        for p in ax_plot.patches:
            height = p.get_height()
            if height > 0:
                ax_plot.annotate(
                    f'{int(height)}', (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom', xytext=(0, 3), textcoords='offset points', fontsize=9
                )
        
        plt.legend(title="Churn", labels=["No Churn", "Churn"])
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/{filename}", dpi=150)
        plt.close()
    
    # 2. Gender vs Churn
    plot_grouped_count(df_plot, 'gender', "Churn Distribution by Gender", "02_gender_vs_churn.png")
    
    # 3. Senior Citizen vs Churn
    df_plot['SeniorCitizen_Label'] = df_plot['SeniorCitizen'].map({0: "No", 1: "Yes"})
    plot_grouped_count(df_plot, 'SeniorCitizen_Label', "Churn Distribution by Senior Citizen Status", "03_senior_vs_churn.png")
    
    # 4. Partner vs Churn
    plot_grouped_count(df_plot, 'Partner', "Churn Distribution by Partner Status", "04_partner_vs_churn.png")
    
    # 5. Dependents vs Churn
    plot_grouped_count(df_plot, 'Dependents', "Churn Distribution by Dependents Status", "05_dependents_vs_churn.png")
    
    # 6. Internet Service vs Churn
    plot_grouped_count(df_plot, 'InternetService', "Churn Distribution by Internet Service Type", "06_internet_service_vs_churn.png")
    
    # 7. Contract Type vs Churn
    plot_grouped_count(df_plot, 'Contract', "Churn Distribution by Contract Type", "07_contract_vs_churn.png")
    
    # 8. Payment Method vs Churn
    plt.figure(figsize=(12, 6))
    ax = sns.countplot(data=df_plot, x='PaymentMethod', hue='Churn', palette=palette)
    plt.title("Churn Distribution by Payment Method")
    plt.xlabel("Payment Method")
    plt.ylabel("Count")
    plt.xticks(rotation=15, ha='right')
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(
                f'{int(height)}', (p.get_x() + p.get_width() / 2., height),
                ha='center', va='bottom', xytext=(0, 3), textcoords='offset points', fontsize=9
            )
    plt.legend(title="Churn", labels=["No Churn", "Churn"])
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/08_payment_method_vs_churn.png", dpi=150)
    plt.close()
    
    # 9. Monthly Charges Distribution
    plt.figure()
    sns.histplot(data=df_plot, x='MonthlyCharges', kde=True, color="#2b5c8f")
    plt.title("Distribution of Customer Monthly Charges")
    plt.xlabel("Monthly Charges ($)")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/09_monthly_charges_distribution.png", dpi=150)
    plt.close()
    
    # 10. Tenure Distribution
    plt.figure()
    sns.histplot(data=df_plot, x='tenure', kde=True, color="#2b5c8f")
    plt.title("Distribution of Customer Tenure (Months)")
    plt.xlabel("Tenure (Months)")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/10_tenure_distribution.png", dpi=150)
    plt.close()
    
    # 11. Monthly Charges vs Churn
    plt.figure()
    sns.boxplot(data=df_plot, x='Churn', y='MonthlyCharges', hue='Churn', palette=palette, legend=False)
    plt.title("Monthly Charges vs Churn")
    plt.xlabel("Churn Status")
    plt.ylabel("Monthly Charges ($)")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/11_monthly_charges_vs_churn.png", dpi=150)
    plt.close()
    
    # 12. Tenure vs Churn
    plt.figure()
    sns.boxplot(data=df_plot, x='Churn', y='tenure', hue='Churn', palette=palette, legend=False)
    plt.title("Tenure vs Churn")
    plt.xlabel("Churn Status")
    plt.ylabel("Tenure (Months)")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/12_tenure_vs_churn.png", dpi=150)
    plt.close()
    
    # 13. Correlation Heatmap
    plt.figure(figsize=(8, 6))
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']
    df_corr_df = df_plot[numeric_cols].copy()
    df_corr_df['Churn_Numeric'] = df_plot['Churn'].map({'Yes': 1, 'No': 0})
    
    corr_matrix = df_corr_df.corr()
    
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".3f", linewidths=0.5, vmin=-1, vmax=1)
    plt.title("Correlation Matrix Heatmap")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/13_correlation_heatmap.png", dpi=150)
    plt.close()
    
    logger.info("All 13 EDA plots successfully generated and saved.")


if __name__ == "__main__":
    main()
