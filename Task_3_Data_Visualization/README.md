# Task 3: Data Visualization — Superstore Sales & Profitability

## Overview
This task transforms transactional sales records into a suite of publication-quality visual formats and dashboards using `matplotlib` and `seaborn`. 

Beyond rendering charts, the core deliverable is **data storytelling**: transforming raw financial figures into clear strategic insights regarding product line profitability, customer segments, regional drivers, and discounting risks.

---

## Executive Data Story & Strategic Insights

### 1. Revenue & Profit Trends
- **Total Gross Revenue**: **$2,274,379.82**
- **Total Net Profit**: **$289,191.20**
- **Overall Profit Margin**: **12.72%**
- Trajectory shows robust sequential quarter-over-quarter expansion with notable peaks in Q4 driven by holiday retail volume.

### 2. Category Performance
- **Technology** represents the most lucrative segment, generating high absolute dollar revenue with healthy profit margins (~**20-25%**).
- **Office Supplies** demonstrates stable volume with low variance in margins.
- **Furniture** exhibits higher revenue variance and compressed margins due to bulky logistics and deeper promotional discounts.

### 3. The Discounting Tipping Point
- Regression analysis of discount rates against profit margin reveals that discounts exceeding **25%** frequently cause net operating margins to plummet into negative territory.
- **Recommendation**: Establish automated pricing guardrails capping promotional discounts at **20%** unless explicitly approved for inventory clearance.

### 4. Sub-Category Deep-Dive
- **Top Profit Generators**: *Copiers*, *Phones*, *Laptops*, *Paper*.
- **Margin Drag / Losses**: *Tables*, *Bookcases* (under high promotional discounting).

---

## Visual Deliverables

All high-DPI (300 DPI) visual assets are saved in [`outputs/`](outputs/):
1. **`sales_overview_dashboard.png`**:
   - Panel A: Monthly Sales & Profit Growth Trendline.
   - Panel B: Revenue and Net Profit grouped bars by Category with margin labels.
   - Panel C: Regional contribution (West, East, Central, South).
   - Panel D: Discount Rate vs. Profit Margin regression with threshold line.
2. **`subcategory_profitability.png`**:
   - Diverging horizontal bar chart ranking sub-categories from largest loss to highest gain.
3. **`customer_segment_analysis.png`**:
   - Donut chart depicting total sales share across Consumer, Corporate, and Home Office segments, alongside Average Order Value (AOV) comparisons.

---

## How to Run

### Python Script:
```bash
python data_visualization.py
```

### Jupyter Notebook:
Open `visualization_notebook.ipynb` in VS Code or JupyterLab and execute all cells.
