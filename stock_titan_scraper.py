import time
import requests
from bs4 import BeautifulSoup
import re
import csv
import json
from finbert_utils import estimate_sentiment
# from llama_utils import estimate_sentiment

def fetch_live_blog_updates(url):
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    
    json_scripts = soup.find_all("script", type="application/ld+json")
    updates = []
    for script in json_scripts:
        try:
            data = json.loads(script.string)
            if isinstance(data, list):
                for item in data:
                    if item.get("@type") == "LiveBlogPosting" and "liveBlogUpdate" in item:
                        updates.extend(item["liveBlogUpdate"])
            elif isinstance(data, dict) and data.get("@type") == "LiveBlogPosting":
                updates.extend(data.get("liveBlogUpdate", []))
        except Exception:
            continue
    return updates

def scrape_and_extract(url):
    """
    1) Try to pull the English summary div.
    2) If none, fall back to the <title> tag.
    Returns: (text_to_analyze, list_of_tickers, source_type)
      or (None, None, None) if fetch failed.
    """
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"    ✖ Failed to fetch {url!r}: {e}")
        return None, None, None

    soup = BeautifulSoup(resp.text, "html.parser")

    # Try summary
    summary_div = soup.find("div", {"id": "summary", "lang": "en"})
    if summary_div:
        text = summary_div.get_text(separator=" ", strip=True)
        source = "summary"
    else:
        # fallback → use page <title>
        title = soup.title.string if soup.title else ""
        text = title.strip()
        source = "title"

    # extract tickers from the chosen text
    tickers = re.findall(r'\b(?:TSX|NYSE|NASDAQ):\s?[A-Z]{1,5}\b', text)
    tickers = [t.replace(" ", "") for t in tickers]
    return text, tickers, source

def store_results(url, text, tickers, sentiment, source):
    with open('scraped_data.csv', 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow([url, source, text, ",".join(tickers), sentiment['label'], sentiment['score']])

def run_pipeline():
    live_update_url = "https://www.stocktitan.net/news/live.html"
    updates = fetch_live_blog_updates(live_update_url)
    
    for update in updates:
        url = update.get("url")
        if not url:
            continue
        
        print(f"Processing {url} …")
        text, tickers, source = scrape_and_extract(url)
        if text is None:
            # fetch failed; already logged inside scrape_and_extract
            continue

        print(f"   → using {source!r}, found tickers: {tickers}")
        if not text:
            print("   → no text found, skipping sentiment.")
            continue

        # sentiment
        try:
            prob, label = estimate_sentiment(text)
        except Exception as e:
            print(f"   ✖ Sentiment analysis failed: {e}")
            continue

        print(f"   Sentiment ({source}): {label} ({prob:.4f})")
        store_results(url, text, tickers, {"label": label, "score": prob}, source)

        # polite
        time.sleep(2)

if __name__ == "__main__":
    run_pipeline()