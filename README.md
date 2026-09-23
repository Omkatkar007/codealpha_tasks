# CodeAlpha Data Analytics Internship — Project Portfolio

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-darkblue?logo=pandas)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange?logo=scikit-learn)
![Seaborn](https://img.shields.io/badge/Seaborn-0.13%2B-teal)
![Status](https://img.shields.io/badge/Internship-Completed-success)

This repository contains the complete, production-ready deliverables for the **CodeAlpha Data Analytics Internship**. 

Although the minimum completion requirement was **2 or 3 tasks**, this repository contains **all 4 tasks** (including runnable Python scripts, interactive Jupyter notebooks, datasets, high-resolution visual outputs, and comprehensive analysis reports).

---

## 📑 Tasks Summary

| Task # | Domain | Core Tools & Techniques | Deliverables | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Task 1** | **Web Scraping** | `requests`, `BeautifulSoup4`, `pandas`, DOM Parsing | `web_scraper.py`, `web_scraping_notebook.ipynb`, `books_data.csv`, `books_data.json` | ✅ Completed |
| **Task 2** | **Exploratory Data Analysis (EDA)** | `pandas`, `seaborn`, `scipy.stats`, Hypothesis Testing, IQR Outlier Detection | `eda_analysis.py`, `eda_notebook.ipynb`, `titanic.csv`, 4 High-Res Plots | ✅ Completed |
| **Task 3** | **Data Visualization** | `matplotlib`, `seaborn`, Executive Storytelling, Multi-Panel Dashboard | `data_visualization.py`, `visualization_notebook.ipynb`, `superstore_sales.csv`, Dashboard & Charts | ✅ Completed |
| **Task 4** | **Sentiment Analysis** *(Bonus)* | NLP, `scikit-learn`, `TF-IDF`, Logistic Regression, Model Interpretation | `sentiment_analysis.py`, `sentiment_notebook.ipynb`, `product_reviews.csv`, Confusion Matrix & Keyword Plots | ✅ Completed |

---

## 📂 Repository Structure

```text
codealpha_tasks/
│
├── requirements.txt                   # Project-wide Python dependencies
├── .gitignore                         # Python, environment & cache ignore rules
├── README.md                          # Repository portfolio overview
│
├── Task_1_Web_Scraping/
│   ├── web_scraper.py                 # Automated multi-page web scraper
│   ├── web_scraping_notebook.ipynb    # Interactive scraping walkthrough
│   ├── books_data.csv                 # Scraped book catalog in CSV
│   ├── books_data.json                # Scraped book catalog in JSON
│   └── README.md                      # Detailed Task 1 documentation
│
├── Task_2_Exploratory_Data_Analysis/
│   ├── eda_analysis.py                # EDA pipeline & statistical hypothesis tests
│   ├── eda_notebook.ipynb             # Interactive exploratory notebook
│   ├── dataset/
│   │   └── titanic.csv                # Titanic passenger survival dataset
│   ├── outputs/                       # Saved high-resolution figures (300 DPI)
│   │   ├── survival_by_class_gender.png
│   │   ├── age_distribution.png
│   │   ├── fare_boxplot.png
│   │   └── correlation_heatmap.png
│   └── README.md                      # Statistical findings & data insights
│
├── Task_3_Data_Visualization/
│   ├── data_visualization.py          # Dashboard & visual analytics generator
│   ├── visualization_notebook.ipynb   # Data storytelling notebook
│   ├── dataset/
│   │   └── superstore_sales.csv       # Retail sales & profit transactions
│   ├── outputs/                       # 300 DPI visual assets
│   │   ├── sales_overview_dashboard.png
│   │   ├── subcategory_profitability.png
│   │   └── customer_segment_analysis.png
│   └── README.md                      # Executive takeaways & strategic advice
│
└── Task_4_Sentiment_Analysis/
    ├── sentiment_analysis.py          # End-to-end NLP & ML classification pipeline
    ├── sentiment_notebook.ipynb       # Interactive model development notebook
    ├── dataset/
    │   └── product_reviews.csv        # Curated e-commerce customer reviews
    ├── outputs/                       # Model evaluation charts
    │   ├── sentiment_distribution.png
    │   ├── confusion_matrix.png
    │   └── top_sentiment_words.png
    └── README.md                      # Model performance & product recommendations
```

---

## 🚀 Quick Start & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Omkatkar007/codealpha_tasks.git
cd codealpha_tasks
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Any Task

#### Task 1: Web Scraping
```bash
python Task_1_Web_Scraping/web_scraper.py
```

#### Task 2: Exploratory Data Analysis
```bash
python Task_2_Exploratory_Data_Analysis/eda_analysis.py
```

#### Task 3: Data Visualization
```bash
python Task_3_Data_Visualization/data_visualization.py
```

#### Task 4: Sentiment Analysis
```bash
python Task_4_Sentiment_Analysis/sentiment_analysis.py
```

---

## 📊 Key Highlights & Takeaways

### Task 1: Web Scraping
- Scraped **100 books** across 5 catalogue pages from `books.toscrape.com`.
- Extracted book titles, clean floating-point prices in GBP, numeric star ratings (1 to 5), stock status, and resolved absolute URLs.
- Dual export to clean CSV and formatted JSON.

### Task 2: Exploratory Data Analysis
- Analyzed 891 records from the Titanic passenger dataset.
- Detected 116 upper fare outliers ($13.02\%$) with maximum fare reaching **$512.33** using the IQR method.
- Statistically validated:
  - **Gender Effect**: Female survival ($74.2\%$) vs Male survival ($18.9\%$) ($\chi^2 = 260.72, p < 0.001$).
  - **Class Effect**: 1st Class ($63.0\%$) vs 3rd Class ($24.2\%$) ($\chi^2 = 102.89, p < 0.001$).
  - **Fare Advantage**: Survivors paid an average of **$48.40** vs non-survivors **$22.12** ($t = 6.84, p < 0.001$).

### Task 3: Data Visualization
- Generated an **Executive Sales & Profit Dashboard** representing **$2.27M** gross sales and **$289K** net profit.
- Revealed that promotional discounts greater than **25%** erode profit margins into severe deficits.
- Recommended automated discounting ceilings and focused marketing on high-margin Technology categories.

### Task 4: Sentiment Analysis
- Developed a TF-IDF + Logistic Regression NLP classification pipeline on 1,200 customer product reviews.
- Achieved **100% test set accuracy** across Positive, Neutral, and Negative sentiments.
- Extracted top predictive keywords (`battery drains`, `defective scratches` vs `phenomenal build`, `sleek design`) to guide product engineering and customer support.

---

## 👤 Author & Contact
- **Intern**: Om Katkar
- **GitHub**: [@Omkatkar007](https://github.com/Omkatkar007)
- **Internship**: CodeAlpha Data Analytics Internship
