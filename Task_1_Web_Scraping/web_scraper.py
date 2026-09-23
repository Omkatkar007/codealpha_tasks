"""
CodeAlpha Data Analytics Internship
Task 1: Web Scraping
Author: Om Katkar
Description:
    An automated, robust web scraper using BeautifulSoup and Requests to extract
    structured book catalog data from 'http://books.toscrape.com/'.
    Handles pagination, HTML parsing, data cleaning, type conversion,
    and exports datasets to CSV and JSON formats.
"""

import os
import sys
import re
import json
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# Ensure UTF-8 output on Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_URL = "http://books.toscrape.com/catalogue/"
START_URL = "http://books.toscrape.com/catalogue/page-1.html"

RATING_MAP = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


def fetch_page(url: str, retries: int = 3, delay: float = 1.0) -> BeautifulSoup | None:
    """Fetch an HTML page with retry mechanism and return a BeautifulSoup object."""
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, headers=HEADERS, timeout=10)
            if response.status_code == 200:
                return BeautifulSoup(response.content, "html.parser")
            elif response.status_code == 404:
                # Reached end of pagination
                return None
            else:
                print(f"[!] Warning: HTTP {response.status_code} for URL: {url} (Attempt {attempt}/{retries})")
        except requests.RequestException as e:
            print(f"[!] Network error on {url}: {e} (Attempt {attempt}/{retries})")
        time.sleep(delay)
    return None


def parse_rating(article_tag) -> int:
    """Extract and map star rating class to integer (1-5)."""
    p_tag = article_tag.find("p", class_="star-rating")
    if p_tag:
        classes = p_tag.get("class", [])
        for c in classes:
            c_lower = c.lower()
            if c_lower in RATING_MAP:
                return RATING_MAP[c_lower]
    return 0


def parse_price(price_text: str) -> float:
    """Extract numeric float from price string (e.g., '£51.77' -> 51.77)."""
    match = re.search(r"[\d.]+", price_text)
    return float(match.group()) if match else 0.0


def scrape_books(max_pages: int = 5) -> list[dict]:
    """Scrape book catalog across multiple pages."""
    all_books = []
    print(f"[*] Starting web scraping for up to {max_pages} pages...")

    for page_num in range(1, max_pages + 1):
        page_url = f"{BASE_URL}page-{page_num}.html"
        print(f"[*] Scraping Page {page_num}: {page_url}")
        
        soup = fetch_page(page_url)
        if not soup:
            print(f"[*] No more pages found or request failed at page {page_num}. Stopping.")
            break

        product_articles = soup.find_all("article", class_="product_pod")
        if not product_articles:
            print(f"[*] No product articles found on page {page_num}.")
            break

        for article in product_articles:
            # Title & Link
            h3_tag = article.find("h3")
            a_tag = h3_tag.find("a") if h3_tag else None
            title = a_tag.get("title") if a_tag and a_tag.get("title") else (a_tag.text.strip() if a_tag else "Unknown")
            rel_link = a_tag.get("href") if a_tag else ""
            product_url = urljoin(page_url, rel_link)

            # Image URL
            img_tag = article.find("img")
            img_rel = img_tag.get("src") if img_tag else ""
            image_url = urljoin(page_url, img_rel)

            # Price
            price_tag = article.find("p", class_="price_color")
            price_gbp = parse_price(price_tag.text) if price_tag else 0.0

            # Rating
            star_rating = parse_rating(article)

            # Availability
            avail_tag = article.find("p", class_="instock availability")
            availability = avail_tag.text.strip() if avail_tag else "Unknown"

            book_record = {
                "title": title,
                "price_gbp": price_gbp,
                "star_rating": star_rating,
                "availability": availability,
                "product_url": product_url,
                "image_url": image_url,
                "page_scraped": page_num
            }
            all_books.append(book_record)

        print(f"    -> Extracted {len(product_articles)} books from Page {page_num}.")
        time.sleep(0.5)  # Polite crawling delay

    print(f"[+] Scraping complete! Total books extracted: {len(all_books)}")
    return all_books


def save_datasets(books: list[dict], output_dir: str):
    """Save the scraped records as both CSV and JSON."""
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "books_data.csv")
    json_path = os.path.join(output_dir, "books_data.json")

    # Save CSV
    df = pd.DataFrame(books)
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"[+] Saved CSV dataset to: {csv_path}")

    # Save JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(books, f, indent=4, ensure_ascii=False)
    print(f"[+] Saved JSON dataset to: {json_path}")

    return df


def print_summary_analytics(df: pd.DataFrame):
    """Print high-level descriptive summary of the scraped dataset."""
    print("\n" + "="*50)
    print("           SCRAPED DATASET SUMMARY           ")
    print("="*50)
    print(f"Total Books Collected : {len(df)}")
    print(f"Average Book Price    : GBP {df['price_gbp'].mean():.2f}")
    print(f"Median Book Price     : GBP {df['price_gbp'].median():.2f}")
    print(f"Min Book Price        : GBP {df['price_gbp'].min():.2f}")
    print(f"Max Book Price        : GBP {df['price_gbp'].max():.2f}")
    print(f"In-Stock Percentage   : {(df['availability'].str.contains('In stock', case=False).mean() * 100):.1f}%")
    print("\nRating Distribution:")
    for rating, count in df["star_rating"].value_counts().sort_index().items():
        print(f"  {rating} Star: {count} books")

    most_expensive = df.loc[df["price_gbp"].idxmax()]
    least_expensive = df.loc[df["price_gbp"].idxmin()]
    print(f"\nMost Expensive Book   : '{most_expensive['title']}' at GBP {most_expensive['price_gbp']:.2f}")
    print(f"Least Expensive Book  : '{least_expensive['title']}' at GBP {least_expensive['price_gbp']:.2f}")
    print("="*50 + "\n")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    books = scrape_books(max_pages=5)  # 5 pages = 100 books
    if books:
        df = save_datasets(books, script_dir)
        print_summary_analytics(df)
    else:
        print("[!] No data was scraped.")


if __name__ == "__main__":
    main()
