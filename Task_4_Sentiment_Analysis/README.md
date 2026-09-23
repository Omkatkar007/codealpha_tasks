# Task 4: Sentiment Analysis (NLP & Machine Learning)

## Overview
This task implements an end-to-end Natural Language Processing (NLP) and supervised classification pipeline to categorize e-commerce customer product reviews into **Positive**, **Neutral**, and **Negative** sentiments.

The system tokenizes customer reviews, extracts n-gram TF-IDF textual features, trains a multiclass classifier, and evaluates performance through precision, recall, F1-score, and confusion matrices. It also interprets model weights to extract key driver vocabulary for product and marketing teams.

---

## Technical Architecture & Pipeline
1. **Data Ingestion**: 1,200 curated e-commerce reviews across audio, wearables, office, and computer peripherals.
2. **Text Preprocessing**: Lowercasing, stripping punctuation/symbols, collapsing whitespace, and stop-word filtering.
3. **Feature Engineering**: `TfidfVectorizer` utilizing unigrams and bigrams (`ngram_range=(1, 2)`) with a maximum vocabulary size of 2,000 features.
4. **Classification Algorithm**: Multinomial Logistic Regression trained on a stratified 80/20 train/test split.
5. **Interpretability**: Extraction of highest positive and negative coefficient values to determine sentiment-driving words.

---

## Performance & Evaluation Metrics

| Metric | Score |
| :--- | :--- |
| **Overall Test Accuracy** | **100.00%** |
| **Negative Precision / Recall / F1** | **1.00 / 1.00 / 1.00** |
| **Neutral Precision / Recall / F1** | **1.00 / 1.00 / 1.00** |
| **Positive Precision / Recall / F1** | **1.00 / 1.00 / 1.00** |

---

## Actionable Business Recommendations
1. **Product Development & Quality Assurance**:
   - High-frequency negative indicators include `battery drains`, `defective scratches`, and `software crashes`.
   - Action: Firmware team should prioritize power-saving updates; packaging team should add extra protective padding for transit.
2. **Customer Service & Warranty Resolution**:
   - Mentions of `warranty` and `customer service` in negative contexts highlight claim hurdles.
   - Action: Deploy automated ticketing for return merchandise authorizations (RMAs) to salvage at-risk customers.
3. **Marketing & Conversion Optimization**:
   - `phenomenal build`, `sleek design`, and `crystal clear` are the strongest positive conversion drivers.
   - Action: Use these exact keywords in Amazon PPC ad copy and storefront hero banners to increase click-through rates.

---

## Visual Outputs

Saved in [`outputs/`](outputs/):
- **`sentiment_distribution.png`**: Breakdown of reviews by sentiment class and star rating (1-5).
- **`confusion_matrix.png`**: Confusion matrix heatmap on test set predictions.
- **`top_sentiment_words.png`**: Top positive vs. negative vocabulary coefficients.

---

## How to Run

### Standalone Script:
```bash
python sentiment_analysis.py
```

### Interactive Notebook:
Open `sentiment_notebook.ipynb` in VS Code or JupyterLab and execute all cells.
