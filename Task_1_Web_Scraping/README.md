# Task 1: Web Scraping — Book Catalog Extraction

## Overview
This task demonstrates automated data collection from the public e-commerce sandbox [Books to Scrape](http://books.toscrape.com/) using Python, `BeautifulSoup4`, and `requests`.

The pipeline extracts multi-page book listings, handles HTML structure navigation and pagination, cleans messy textual data into typed values (e.g. numeric prices and star ratings), and exports the structured dataset to both CSV and JSON formats.

---

## Key Features & Methodology
1. **Multi-Page Web Traversal**: Programmatically navigates through catalogue pages (`page-1.html` to `page-5.html`) using robust status code checking and pagination termination.
2. **Resilient Request Handling**: Custom User-Agent headers, polite crawl delays (`0.5s` - `1.0s`), and retry logic for network timeouts.
3. **Structured HTML DOM Parsing**:
   - Title extraction from `<h3>` tags and full title attributes.
   - Price cleaning via regular expressions (`£51.77` -> `51.77` float).
   - Star rating text mapping (`One` -> `1`, `Two` -> `2`, ..., `Five` -> `5`).
   - Stock status identification (`In stock`).
   - URL resolution using `urllib.parse.urljoin` to generate absolute URLs for products and thumbnails.
4. **Data Formats**:
   - `books_data.csv`: Tabular dataset for spreadsheet and data analysis pipelines.
   - `books_data.json`: Nested document format suitable for NoSQL databases and web APIs.

---

## Dataset Schema

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `title` | `string` | Full title of the book |
| `price_gbp` | `float` | Price in British Pounds (£ / GBP) |
| `star_rating` | `int` | Rating score ranging from 1 to 5 |
| `availability` | `string` | In-stock / Out-of-stock inventory status |
| `product_url` | `string` | Absolute URL to the individual book product page |
| `image_url` | `string` | Absolute URL to the book cover thumbnail image |
| `page_scraped`| `int` | Catalogue page number where the item was listed |

---

## Scraping Summary & Statistics
- **Total Books Scraped**: 100 books across 5 catalogue pages.
- **Average Price**: £34.56 (Min: £10.16, Max: £58.11, Median: £34.78).
- **Availability**: 100% In-stock.
- **Star Rating Distribution**:
  - 1 Star: 22 books
  - 2 Star: 19 books
  - 3 Star: 22 books
  - 4 Star: 18 books
  - 5 Star: 19 books

---

## How to Run

### Run the standalone script:
```bash
python web_scraper.py
```

### Run the interactive notebook:
Open `web_scraping_notebook.ipynb` in VS Code or Jupyter Notebook / JupyterLab and execute all cells.
