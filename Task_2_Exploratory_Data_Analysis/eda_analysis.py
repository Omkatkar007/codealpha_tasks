"""
CodeAlpha Data Analytics Internship
Task 2: Exploratory Data Analysis (EDA)
Author: Om Katkar
Description:
    Comprehensive EDA on the Titanic Passenger Dataset.
    Includes question formulation, data structure exploration, missing value analysis,
    IQR outlier detection, hypothesis testing (Chi-square & Two-sample T-test),
    and generation of high-resolution analytical visualizations.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Ensure UTF-8 output on Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Configure plot styling
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "figure.titlesize": 15,
    "figure.titleweight": "bold",
})


def print_header(title: str):
    print("\n" + "=" * 65)
    print(f" {title.center(63)} ")
    print("=" * 65)


def load_dataset(filepath: str) -> pd.DataFrame:
    """Load the Titanic dataset from disk."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    df = pd.read_csv(filepath)
    print(f"[+] Loaded dataset successfully: {df.shape[0]} rows, {df.shape[1]} columns.")
    return df


def explore_data_structure(df: pd.DataFrame):
    """Inspect variables, datatypes, missing values, and descriptive statistics."""
    print_header("1. DATA STRUCTURE & SUMMARY INSPECTION")
    
    print("\n[A] Column Types and Non-Null Counts:")
    buffer = []
    for col in df.columns:
        buffer.append({
            "Column": col,
            "Non-Null Count": df[col].notnull().sum(),
            "Null Count": df[col].isnull().sum(),
            "Null %": f"{(df[col].isnull().mean() * 100):.2f}%",
            "Dtype": str(df[col].dtype)
        })
    info_df = pd.DataFrame(buffer)
    print(info_df.to_string(index=False))

    print("\n[B] Numerical Summary Statistics:")
    print(df.describe().round(2).to_string())

    print("\n[C] Categorical Value Counts:")
    for cat_col in ["sex", "pclass", "embark_town", "alone"]:
        if cat_col in df.columns:
            print(f"\nValue counts for '{cat_col}':")
            print(df[cat_col].value_counts().to_string())


def detect_outliers(df: pd.DataFrame, col: str = "fare") -> pd.DataFrame:
    """Detect anomalies/outliers using the Interquartile Range (IQR) method."""
    print_header(f"2. ANOMALY & OUTLIER DETECTION ({col.upper()})")
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = max(0, q1 - 1.5 * iqr)
    upper_bound = q3 + 1.5 * iqr

    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    print(f"Q1 (25th percentile) : {q1:.2f}")
    print(f"Q3 (75th percentile) : {q3:.2f}")
    print(f"IQR                  : {iqr:.2f}")
    print(f"Upper Outlier Cutoff : {upper_bound:.2f}")
    print(f"Identified Outliers  : {len(outliers)} records ({(len(outliers) / len(df) * 100):.2f}% of data)")
    print(f"Max Extreme Fare     : {df[col].max():.2f}")
    return outliers


def test_hypotheses(df: pd.DataFrame):
    """
    Perform statistical hypothesis testing:
    - Hypothesis 1: Chi-Square Test of Independence for Sex vs. Survival
    - Hypothesis 2: Chi-Square Test of Independence for Class (pclass) vs. Survival
    - Hypothesis 3: Independent Two-Sample T-test for Ticket Fare (Survivors vs. Non-Survivors)
    """
    print_header("3. STATISTICAL HYPOTHESIS TESTING")

    # Hypothesis 1: Sex vs Survival
    print("[H1] Association Between Gender and Survival:")
    contingency_sex = pd.crosstab(df["sex"], df["survived"])
    chi2_sex, p_sex, dof_sex, _ = stats.chi2_contingency(contingency_sex)
    print(contingency_sex)
    print(f"     Chi2 Statistic = {chi2_sex:.4f}, p-value = {p_sex:.4e}")
    if p_sex < 0.05:
        print("     -> Reject Null Hypothesis: Strong statistically significant association between gender and survival.")
    else:
        print("     -> Fail to reject Null Hypothesis.")

    # Hypothesis 2: Class vs Survival
    print("\n[H2] Association Between Passenger Class and Survival:")
    contingency_pclass = pd.crosstab(df["pclass"], df["survived"])
    chi2_pclass, p_pclass, dof_pclass, _ = stats.chi2_contingency(contingency_pclass)
    print(contingency_pclass)
    print(f"     Chi2 Statistic = {chi2_pclass:.4f}, p-value = {p_pclass:.4e}")
    if p_pclass < 0.05:
        print("     -> Reject Null Hypothesis: Strong statistically significant association between ticket class and survival.")
    else:
        print("     -> Fail to reject Null Hypothesis.")

    # Hypothesis 3: Fare difference between survivors and non-survivors
    print("\n[H3] Difference in Ticket Fare (Survivors vs. Non-Survivors):")
    survivor_fares = df[df["survived"] == 1]["fare"].dropna()
    victim_fares = df[df["survived"] == 0]["fare"].dropna()
    t_stat, p_val = stats.ttest_ind(survivor_fares, victim_fares, equal_var=False)
    print(f"     Mean Fare - Survivors     : ${survivor_fares.mean():.2f} (std: ${survivor_fares.std():.2f})")
    print(f"     Mean Fare - Non-Survivors : ${victim_fares.mean():.2f} (std: ${victim_fares.std():.2f})")
    print(f"     Welch's T-Statistic       : {t_stat:.4f}, p-value = {p_val:.4e}")
    if p_val < 0.05:
        print("     -> Reject Null Hypothesis: Survivors paid a significantly higher ticket fare than non-survivors.")
    else:
        print("     -> Fail to reject Null Hypothesis.")


def generate_visualizations(df: pd.DataFrame, output_dir: str):
    """Generate high-resolution analytical plots and save to outputs directory."""
    print_header("4. GENERATING EDA VISUALIZATIONS")
    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: Survival Rate by Gender and Class
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1A: Gender Survival Rate
    gender_survival = df.groupby("sex")["survived"].mean().reset_index()
    sns.barplot(data=gender_survival, x="sex", y="survived", hue="sex", legend=False, ax=axes[0], palette=["#2b5c8f", "#d95f02"])
    axes[0].set_title("Survival Rate by Gender")
    axes[0].set_ylabel("Survival Probability")
    axes[0].set_ylim(0, 1)
    for p in axes[0].patches:
        axes[0].annotate(f"{p.get_height()*100:.1f}%", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                         ha='center', va='center', color='white', fontweight='bold', fontsize=12)

    # 1B: Class Survival Rate
    pclass_survival = df.groupby(["pclass", "sex"])["survived"].mean().reset_index()
    sns.barplot(data=pclass_survival, x="pclass", y="survived", hue="sex", ax=axes[1], palette=["#2b5c8f", "#d95f02"])
    axes[1].set_title("Survival Rate by Ticket Class & Gender")
    axes[1].set_ylabel("Survival Probability")
    axes[1].set_ylim(0, 1)
    axes[1].legend(title="Gender")

    plt.tight_layout()
    p1_path = os.path.join(output_dir, "survival_by_class_gender.png")
    plt.savefig(p1_path, dpi=300)
    plt.close()
    print(f"[+] Saved: {p1_path}")

    # Plot 2: Age Distribution and Survival
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.kdeplot(data=df, x="age", hue="survived", common_norm=False, fill=True, alpha=0.4,
                palette=["#c44e52", "#4c72b0"], ax=ax)
    ax.set_title("Age Distribution: Survivors (1) vs Non-Survivors (0)")
    ax.set_xlabel("Age (Years)")
    ax.set_ylabel("Density")
    plt.tight_layout()
    p2_path = os.path.join(output_dir, "age_distribution.png")
    plt.savefig(p2_path, dpi=300)
    plt.close()
    print(f"[+] Saved: {p2_path}")

    # Plot 3: Fare Boxplot & Outlier Distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.boxplot(data=df, x="pclass", y="fare", hue="survived", palette=["#c44e52", "#55a868"], ax=axes[0])
    axes[0].set_title("Ticket Fare Distribution by Class & Survival")
    axes[0].set_ylabel("Fare ($)")
    axes[0].set_yscale("log")  # Log scale to clearly show outliers

    # Traveling Alone vs Family
    sns.barplot(data=df, x="alone", y="survived", hue="alone", legend=False, ax=axes[1], palette=["#4c72b0", "#dd8452"])
    axes[1].set_title("Survival Rate: Traveling Alone vs. With Family")
    axes[1].set_ylabel("Survival Probability")
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(["With Family", "Alone"])
    for p in axes[1].patches:
        axes[1].annotate(f"{p.get_height()*100:.1f}%", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                         ha='center', va='center', color='white', fontweight='bold', fontsize=12)

    plt.tight_layout()
    p3_path = os.path.join(output_dir, "fare_boxplot.png")
    plt.savefig(p3_path, dpi=300)
    plt.close()
    print(f"[+] Saved: {p3_path}")

    # Plot 4: Correlation Matrix
    numeric_df = df.select_dtypes(include=[np.number])
    fig, ax = plt.subplots(figsize=(8, 6))
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax, vmin=-1, vmax=1)
    ax.set_title("Correlation Heatmap of Numerical Features")
    plt.tight_layout()
    p4_path = os.path.join(output_dir, "correlation_heatmap.png")
    plt.savefig(p4_path, dpi=300)
    plt.close()
    print(f"[+] Saved: {p4_path}")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, "dataset", "titanic.csv")
    output_dir = os.path.join(base_dir, "outputs")

    df = load_dataset(dataset_path)
    explore_data_structure(df)
    detect_outliers(df, "fare")
    test_hypotheses(df)
    generate_visualizations(df, output_dir)
    print("\n[+] Exploratory Data Analysis completed successfully!")


if __name__ == "__main__":
    main()
