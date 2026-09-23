# Task 2: Exploratory Data Analysis (EDA) — Titanic Dataset

## Overview
This project presents an in-depth Exploratory Data Analysis (EDA) on the Titanic passenger survival dataset (`891` rows, `15` attributes). 

The goal is to systematically understand the factors influencing passenger survival, identify data structure nuances, handle anomalies/outliers, and formally test analytical hypotheses using inferential statistics.

---

## Pre-Analysis Analytical Questions
1. **Gender Impact**: Did female passengers have a significantly higher survival rate than male passengers?
2. **Socioeconomic Advantage**: Did higher ticket classes (1st vs 2nd vs 3rd) translate directly to higher survival probability?
3. **Age Vulnerability**: Were children prioritized over adults and elderly passengers?
4. **Ticket Fare Economics**: How skewed was the ticket fare distribution, and did paying higher fares correlate with survival?
5. **Companionship & Solitude**: Did passengers traveling alone have different survival outcomes compared to those traveling with family?

---

## Data Structure & Data Quality Findings
- **Sample Size**: 891 records.
- **Missing Values**:
  - `age`: 177 missing values (19.87%) — requires imputation (e.g. median by class/gender) for modeling.
  - `deck`: 688 missing values (77.22%) — highly sparse; best handled by missing-category indicator.
  - `embarked` / `embark_town`: 2 missing values (0.22%) — easily imputed using mode (`Southampton`).
- **Anomalies & Outliers**:
  - `fare` exhibited severe right-skewness.
  - Using the **Interquartile Range (IQR)** method ($Q1 = \$7.91$, $Q3 = \$31.00$, $IQR = \$23.09$), the upper cutoff was calculated as **$65.63**.
  - **116 records (13.02%)** were identified as upper fare outliers, reaching an extreme high of **$512.33**.

---

## Statistical Hypothesis Testing

### Hypothesis 1: Gender vs. Survival
- **Test**: Pearson's Chi-Square Test of Independence ($\alpha = 0.05$).
- **Contingency Table**:
  - Female: 233 Survived / 81 Perished (**74.2%** survival rate)
  - Male: 109 Survived / 468 Perished (**18.9%** survival rate)
- **Result**: $\chi^2 = 260.72$, $p\text{-value} = 1.19 \times 10^{-58}$.
- **Conclusion**: Strongly reject the null hypothesis. Gender played a decisive role in survival outcomes.

### Hypothesis 2: Passenger Class vs. Survival
- **Test**: Pearson's Chi-Square Test of Independence.
- **Survival by Class**:
  - 1st Class: **62.96%**
  - 2nd Class: **47.28%**
  - 3rd Class: **24.24%**
- **Result**: $\chi^2 = 102.89$, $p\text{-value} = 4.55 \times 10^{-23}$.
- **Conclusion**: Strongly reject the null hypothesis. Ticket class (socioeconomic status and cabin proximity to boat decks) significantly governed survival chances.

### Hypothesis 3: Ticket Fare Comparison (Survivors vs. Non-Survivors)
- **Test**: Welch's Two-Sample Independent T-Test (unequal variances).
- **Means**:
  - Survivors Mean Fare: **$48.40** ($\pm \$66.60$)
  - Non-Survivors Mean Fare: **$22.12** ($\pm \$31.39$)
- **Result**: $t = 6.8391$, $p\text{-value} = 2.69 \times 10^{-11}$.
- **Conclusion**: Reject null hypothesis. Survivors paid more than double the average fare of non-survivors.

---

## Visual Outputs

The generated high-resolution visualizations are stored in [`outputs/`](outputs/):
- `survival_by_class_gender.png`: Survival comparison across passenger class and gender cohorts.
- `age_distribution.png`: Kernel Density Estimation (KDE) comparing age profiles of survivors vs. non-survivors.
- `fare_boxplot.png`: Boxplot illustrating log-scale fare distribution across classes and companionship status.
- `correlation_heatmap.png`: Pearson correlation matrix among all numerical variables.

---

## How to Run

### Python Script:
```bash
python eda_analysis.py
```

### Jupyter Notebook:
Open `eda_notebook.ipynb` in VS Code or JupyterLab and execute all cells.
