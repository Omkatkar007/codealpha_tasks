"""
CodeAlpha Data Analytics Internship
Task 4: Sentiment Analysis (Bonus / Extra Learning)
Author: Om Katkar
Description:
    End-to-End Natural Language Processing (NLP) Sentiment Analysis pipeline
    on e-commerce product reviews. Classifies customer feedback into Positive,
    Neutral, and Negative sentiments using TF-IDF feature extraction and
    Logistic Regression. Evaluates model performance and provides actionable business insights.
"""

import os
import sys
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Ensure UTF-8 output on Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Global plot styling
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.sans-serif": "Arial",
    "figure.titlesize": 15,
    "figure.titleweight": "bold",
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
})


def print_header(title: str):
    print("\n" + "=" * 65)
    print(f" {title.center(63)} ")
    print("=" * 65)


def clean_text(text: str) -> str:
    """Preprocess review text: lowercase, remove non-alphanumeric chars, normalize whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_and_prepare_data(filepath: str) -> pd.DataFrame:
    """Load reviews dataset and preprocess text."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Reviews dataset not found at {filepath}")
    df = pd.read_csv(filepath)
    df["cleaned_text"] = df["review_text"].apply(clean_text)
    print(f"[+] Loaded {len(df)} customer reviews from {filepath}")
    return df


def train_sentiment_model(df: pd.DataFrame):
    """Train TF-IDF + Logistic Regression NLP classification pipeline."""
    print_header("1. TRAINING NLP SENTIMENT CLASSIFICATION MODEL")

    X = df["cleaned_text"]
    y = df["sentiment"]

    # Stratified 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # TF-IDF Feature Extraction with unigrams and bigrams
    vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2), stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train Logistic Regression
    classifier = LogisticRegression(max_iter=1000, random_state=42)
    classifier.fit(X_train_vec, y_train)

    # Predictions & Metrics
    y_pred = classifier.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    classes = classifier.classes_

    print(f"Model Architecture       : TF-IDF Vectorizer + Multinomial Logistic Regression")
    print(f"Training Samples         : {X_train_vec.shape[0]}")
    print(f"Testing Samples          : {X_test_vec.shape[0]}")
    print(f"Vocabulary Size          : {len(vectorizer.get_feature_names_out())} features")
    print(f"Overall Test Accuracy    : {accuracy * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    cm = confusion_matrix(y_test, y_pred, labels=classes)
    return classifier, vectorizer, X_test, y_test, y_pred, cm, classes


def extract_top_keywords(classifier, vectorizer, n_words: int = 10):
    """Extract strongest positive and negative indicator keywords from model coefficients."""
    feature_names = np.array(vectorizer.get_feature_names_out())
    classes = list(classifier.classes_)

    results = {}
    if "Positive" in classes and "Negative" in classes:
        pos_idx = classes.index("Positive")
        neg_idx = classes.index("Negative")

        pos_coef = classifier.coef_[pos_idx]
        neg_coef = classifier.coef_[neg_idx]

        top_pos_idx = np.argsort(pos_coef)[-n_words:][::-1]
        top_neg_idx = np.argsort(neg_coef)[-n_words:][::-1]

        results["Positive"] = [(feature_names[i], pos_coef[i]) for i in top_pos_idx]
        results["Negative"] = [(feature_names[i], neg_coef[i]) for i in top_neg_idx]

    return results


def generate_visualizations(df: pd.DataFrame, cm, classes, top_words, output_dir: str):
    """Generate and save visual outputs for sentiment analysis."""
    print_header("2. GENERATING SENTIMENT VISUALIZATIONS")
    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: Sentiment & Rating Distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    palette = {"Positive": "#2ca02c", "Neutral": "#7f7f7f", "Negative": "#d62728"}

    sent_counts = df["sentiment"].value_counts()
    axes[0].bar(sent_counts.index, sent_counts.values, color=[palette[c] for c in sent_counts.index])
    axes[0].set_title("A. Overall Sentiment Class Distribution")
    axes[0].set_ylabel("Number of Reviews")
    for p in axes[0].patches:
        axes[0].annotate(f"{int(p.get_height())} ({(p.get_height()/len(df)*100):.1f}%)",
                         (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                         ha="center", va="center", color="white", fontweight="bold")

    # Rating distribution by sentiment
    sns.countplot(data=df, x="rating", hue="sentiment", palette=palette, ax=axes[1])
    axes[1].set_title("B. Customer Rating (1-5 Stars) by Sentiment")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    p1_path = os.path.join(output_dir, "sentiment_distribution.png")
    plt.savefig(p1_path, dpi=300)
    plt.close()
    print(f"[+] Saved Sentiment Distribution: {p1_path}")

    # Plot 2: Confusion Matrix Heatmap
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_title("Test Set Confusion Matrix")
    ax.set_xlabel("Predicted Sentiment")
    ax.set_ylabel("Actual Ground Truth")
    plt.tight_layout()
    p2_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(p2_path, dpi=300)
    plt.close()
    print(f"[+] Saved Confusion Matrix: {p2_path}")

    # Plot 3: Top Discriminative Keywords
    if top_words:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Positive Words
        pos_df = pd.DataFrame(top_words["Positive"], columns=["Keyword", "Weight"]).sort_values(by="Weight", ascending=True)
        axes[0].barh(pos_df["Keyword"], pos_df["Weight"], color="#2ca02c")
        axes[0].set_title("Top Positive Sentiment Keywords")
        axes[0].set_xlabel("Logistic Regression Coefficient")

        # Negative Words
        neg_df = pd.DataFrame(top_words["Negative"], columns=["Keyword", "Weight"]).sort_values(by="Weight", ascending=True)
        axes[1].barh(neg_df["Keyword"], neg_df["Weight"], color="#d62728")
        axes[1].set_title("Top Negative Sentiment Keywords")
        axes[1].set_xlabel("Logistic Regression Coefficient")

        plt.tight_layout()
        p3_path = os.path.join(output_dir, "top_sentiment_words.png")
        plt.savefig(p3_path, dpi=300)
        plt.close()
        print(f"[+] Saved Top Sentiment Words: {p3_path}")


def print_business_recommendations():
    """Print strategic insights derived from sentiment analysis."""
    print_header("3. STRATEGIC BUSINESS RECOMMENDATIONS")
    print("1. Product Development: Customers frequently express negative sentiment around")
    print("   'battery drains', 'crashes', and 'defective scratches'. QA teams must prioritize")
    print("   battery longevity and rugged packaging.")
    print("2. Customer Support: Reviews mentioning 'warranty' and 'customer service' in negative")
    print("   contexts indicate warranty claim friction. Automating RMA ticket routing will salvage churn.")
    print("3. Marketing & Conversion: Highlight 'phenomenal build', 'sleek design', and 'fast shipping'")
    print("   in ad campaigns, as these terms strongly drive 5-star customer conversions.")
    print("=" * 65 + "\n")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, "dataset", "product_reviews.csv")
    output_dir = os.path.join(base_dir, "outputs")

    df = load_and_prepare_data(dataset_path)
    classifier, vectorizer, X_test, y_test, y_pred, cm, classes = train_sentiment_model(df)
    top_words = extract_top_keywords(classifier, vectorizer, n_words=8)
    generate_visualizations(df, cm, classes, top_words, output_dir)
    print_business_recommendations()


if __name__ == "__main__":
    main()
