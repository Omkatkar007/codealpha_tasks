"""
CodeAlpha Data Analytics Internship
Task 3: Data Visualization
Author: Om Katkar
Description:
    Generates publication-quality business intelligence charts and an Executive Sales Dashboard
    from the Superstore Sales dataset using Matplotlib and Seaborn.
    Crafts a compelling data story detailing revenue trends, category profitability,
    regional performance, and discount sensitivity.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# Ensure UTF-8 output on Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Global styling
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.sans-serif": "Arial",
    "font.family": "sans-serif",
    "figure.titlesize": 16,
    "figure.titleweight": "bold",
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})


def load_and_preprocess(filepath: str) -> pd.DataFrame:
    """Load sales data and compute derived analytical attributes."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Sales dataset not found at {filepath}")
    df = pd.read_csv(filepath)
    df["order_date"] = pd.to_datetime(df["order_date"])
    df["year_month"] = df["order_date"].dt.to_period("M").dt.to_timestamp()
    df["profit_margin_pct"] = (df["profit"] / df["sales"]) * 100
    print(f"[+] Loaded and preprocessed {len(df)} sales records.")
    return df


def plot_executive_dashboard(df: pd.DataFrame, output_dir: str):
    """Create a 4-panel comprehensive executive dashboard."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Executive Sales & Profit Performance Dashboard (2023 - 2024)", fontsize=18, y=0.98)

    # ---------------- Panel 1: Monthly Trends ----------------
    monthly = df.groupby("year_month")[["sales", "profit"]].sum().reset_index()
    ax1 = axes[0, 0]
    ax1.plot(monthly["year_month"], monthly["sales"], marker="o", color="#1f77b4", linewidth=2.5, label="Monthly Sales ($)")
    ax1.plot(monthly["year_month"], monthly["profit"], marker="s", color="#2ca02c", linewidth=2.5, label="Monthly Profit ($)")
    ax1.fill_between(monthly["year_month"], monthly["profit"], color="#2ca02c", alpha=0.15)
    ax1.set_title("A. Monthly Revenue & Profit Growth")
    ax1.set_xlabel("Order Month")
    ax1.set_ylabel("Total USD ($)")
    ax1.yaxis.set_major_formatter(ticker.StrMethodFormatter("${x:,.0f}"))
    ax1.legend(loc="upper left")
    ax1.tick_params(axis="x", rotation=35)

    # ---------------- Panel 2: Category Breakdown ----------------
    cat_summary = df.groupby("category")[["sales", "profit"]].sum().reset_index()
    ax2 = axes[0, 1]
    x_pos = np.arange(len(cat_summary["category"]))
    bar_width = 0.35
    ax2.bar(x_pos - bar_width/2, cat_summary["sales"], width=bar_width, label="Sales ($)", color="#4575b4")
    ax2.bar(x_pos + bar_width/2, cat_summary["profit"], width=bar_width, label="Profit ($)", color="#74add1")
    ax2.set_title("B. Revenue & Net Profit by Product Category")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(cat_summary["category"])
    ax2.set_ylabel("USD ($)")
    ax2.yaxis.set_major_formatter(ticker.StrMethodFormatter("${x:,.0f}"))
    ax2.legend()
    # Annotate margin percentage
    for i, row in cat_summary.iterrows():
        margin = (row["profit"] / row["sales"]) * 100
        ax2.annotate(f"Margin: {margin:.1f}%", (i, max(row["sales"], row["profit"]) * 0.5),
                     ha="center", va="center", color="white", fontweight="bold", fontsize=10)

    # ---------------- Panel 3: Regional Performance ----------------
    reg_summary = df.groupby("region")[["sales", "profit"]].sum().sort_values(by="sales", ascending=True).reset_index()
    ax3 = axes[1, 0]
    bars = ax3.barh(reg_summary["region"], reg_summary["sales"], color="#313695", alpha=0.85, label="Sales")
    ax3.barh(reg_summary["region"], reg_summary["profit"], color="#fdae61", alpha=0.9, label="Profit")
    ax3.set_title("C. Regional Contribution: Sales vs. Net Profit")
    ax3.set_xlabel("USD ($)")
    ax3.xaxis.set_major_formatter(ticker.StrMethodFormatter("${x:,.0f}"))
    ax3.legend()

    # ---------------- Panel 4: Discount vs Profit Margin ----------------
    ax4 = axes[1, 1]
    sns.regplot(data=df, x="discount", y="profit_margin_pct", ax=ax4,
                scatter_kws={"alpha": 0.3, "color": "#4575b4", "s": 30},
                line_kws={"color": "#d73027", "linewidth": 2.5})
    ax4.axhline(0, color="gray", linestyle="--", linewidth=1.2)
    ax4.axvline(0.25, color="red", linestyle=":", linewidth=1.5, label="Risk Threshold (25% Discount)")
    ax4.set_title("D. Impact of Discounting on Profit Margin (%)")
    ax4.set_xlabel("Discount Rate (Decimal)")
    ax4.set_ylabel("Profit Margin (%)")
    ax4.xaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    ax4.legend(loc="lower left")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    dash_path = os.path.join(output_dir, "sales_overview_dashboard.png")
    plt.savefig(dash_path, dpi=300)
    plt.close()
    print(f"[+] Saved Executive Dashboard: {dash_path}")


def plot_subcategory_profitability(df: pd.DataFrame, output_dir: str):
    """Generate diverging horizontal bar chart of sub-categories sorted by profit."""
    sub_df = df.groupby(["category", "sub_category"])[["sales", "profit"]].sum().reset_index()
    sub_df = sub_df.sort_values(by="profit", ascending=True)

    fig, ax = plt.subplots(figsize=(12, 7))
    colors = ["#d73027" if p < 0 else "#1a9850" for p in sub_df["profit"]]
    bars = ax.barh(sub_df["sub_category"], sub_df["profit"], color=colors, edgecolor="black", linewidth=0.5)

    ax.axvline(0, color="black", linewidth=1.2)
    ax.set_title("Sub-Category Net Profit Ranking (Green = Profitable, Red = Deficit)", pad=15)
    ax.set_xlabel("Net Profit ($)")
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("${x:,.0f}"))

    # Annotate value labels
    for bar in bars:
        width = bar.get_width()
        ha = "left" if width >= 0 else "right"
        offset = 500 if width >= 0 else -500
        ax.annotate(f"${width:,.0f}", (width + offset, bar.get_y() + bar.get_height() / 2),
                    ha=ha, va="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    chart_path = os.path.join(output_dir, "subcategory_profitability.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"[+] Saved Sub-Category Profitability Chart: {chart_path}")


def plot_customer_segments(df: pd.DataFrame, output_dir: str):
    """Plot customer segment sales share donut chart and average order value."""
    seg_summary = df.groupby("segment").agg(
        total_sales=("sales", "sum"),
        avg_order_value=("sales", "mean"),
        order_count=("order_id", "count")
    ).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Segment Sales Share Donut
    colors = ["#2b83ba", "#abdda4", "#fdae61"]
    wedges, texts, autotexts = axes[0].pie(
        seg_summary["total_sales"], labels=seg_summary["segment"], autopct="%1.1f%%",
        startangle=140, colors=colors, pctdistance=0.75,
        textprops={"fontsize": 11, "fontweight": "bold"}
    )
    # Donut center circle
    centre_circle = plt.Circle((0, 0), 0.55, fc="white")
    axes[0].add_artist(centre_circle)
    axes[0].set_title("A. Total Sales Share by Customer Segment")

    # Average Order Value (AOV)
    sns.barplot(data=seg_summary, x="segment", y="avg_order_value", hue="segment", legend=False,
                palette=colors, ax=axes[1])
    axes[1].set_title("B. Average Order Value (AOV) by Segment")
    axes[1].set_ylabel("Average Order Value ($)")
    axes[1].yaxis.set_major_formatter(ticker.StrMethodFormatter("${x:,.0f}"))
    for p in axes[1].patches:
        axes[1].annotate(f"${p.get_height():,.2f}", (p.get_x() + p.get_width() / 2., p.get_height() * 0.9),
                         ha="center", va="center", color="white", fontweight="bold", fontsize=11)

    plt.tight_layout()
    seg_path = os.path.join(output_dir, "customer_segment_analysis.png")
    plt.savefig(seg_path, dpi=300)
    plt.close()
    print(f"[+] Saved Customer Segment Analysis: {seg_path}")


def print_executive_story(df: pd.DataFrame):
    """Print high-level executive data story."""
    total_rev = df["sales"].sum()
    total_prof = df["profit"].sum()
    overall_margin = (total_prof / total_rev) * 100

    print("\n" + "=" * 65)
    print("           EXECUTIVE DATA STORY & BUSINESS INSIGHTS          ")
    print("=" * 65)
    print(f"Total Gross Revenue    : ${total_rev:,.2f}")
    print(f"Total Net Profit       : ${total_prof:,.2f}")
    print(f"Overall Profit Margin  : {overall_margin:.2f}%")
    print("\nKey Takeaways:")
    print("1. Technology products drive the highest revenue and gross profit margins.")
    print("2. Aggressive discounting (>25%) consistently erodes margins, turning sales unprofitable.")
    print("3. Consumer segment accounts for half of all sales, but Corporate orders yield competitive AOV.")
    print("4. Furniture items (specifically Tables/Bookcases) carry the highest risk of negative margin under heavy discounts.")
    print("=" * 65 + "\n")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, "dataset", "superstore_sales.csv")
    output_dir = os.path.join(base_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    df = load_and_preprocess(dataset_path)
    plot_executive_dashboard(df, output_dir)
    plot_subcategory_profitability(df, output_dir)
    plot_customer_segments(df, output_dir)
    print_executive_story(df)


if __name__ == "__main__":
    main()
