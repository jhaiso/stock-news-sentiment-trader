import sqlite3
import requests

from googlenewsdecoder import gnewsdecoder
from tradingview_gainers_scraper import run_scraper_pipeline
from google_search import fetch_google_news_feed_sorted
from concurrent.futures import ThreadPoolExecutor, as_completed
from article_sentiment import extract_main_content
from finbert_utils import estimate_sentiment as finbert_sentiment
from llama_utils import estimate_sentiment as llama_sentiment
from gpt_utils import estimate_sentiment as gpt_sentiment

DB_FILE = "gainers.db"
TRADE_DB_FILE = "potential_trades.db"
SENTIMENT_THRESHOLD = 0.7
minutes_back = 10
max_news = 1
MAX_WORKERS = 100
TITLE_PENALTY_FACTOR = 0.85

def get_latest_gainers():
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("SELECT ticker, company_name, pct_change, rel_volume FROM gainers ORDER BY ts DESC")
    rows = [
        {"ticker": row[0], "company_name": row[1], "pct_change": row[2], "rel_volume": row[3]}
        for row in cur.fetchall()
    ]
    conn.close()
    return rows

DB_FILE = "gainers.db"
MAX_WORKERS = 100
SENTIMENT_THRESHOLD = 0.7
minutes_back = 60
max_news = 1

def resolve_actual_url(google_news_url):
    try:
        # Decode the Google News URL using googlenewsdecoder
        status = gnewsdecoder(google_news_url)
        print(f"[↪️] Decoded URL found: {status['decoded_url']}")
        return status['decoded_url']
    except Exception as e:
        print(f"[ERROR] Failed to resolve article URL using googlenewsdecoder: {e}")
        return google_news_url


def analyze_article(url, fallback_text=None):
    print(f"[INFO] Extracting and summarizing article: {url}")
    content = extract_main_content(url)
    used_fallback = False

    if content is None:
        if fallback_text:
            print("[WARN] Content extraction failed. Using fallback title.")
            summary = fallback_text.strip()
            used_fallback = True
        else:
            print("[WARN] Content extraction failed and no fallback title provided.")
            return None
    else:
        summary = content.get("summary", "").strip()
        if not summary:
            if fallback_text:
                print("[WARN] No summary extracted. Using fallback title.")
                summary = fallback_text.strip()
                used_fallback = True
            else:
                print("[WARN] No summary extracted and no fallback title provided.")
                return None

    if not summary:
        print("[WARN] Summary (or fallback) is empty.")
        return None

    results = []
    for fn in [finbert_sentiment, llama_sentiment, gpt_sentiment]:
        try:
            prob, sentiment = fn(summary)
            results.append((prob, sentiment))
        except Exception as e:
            print(f"[ERROR] Sentiment estimation failed: {e}")

    if not results:
        print("[WARN] No sentiment functions succeeded.")
        return None

    avg_prob = sum(p for p, _ in results) / len(results)
    pos_count = sum(1 for _, s in results if s.lower() == "positive")
    majority_sentiment = "positive" if pos_count >= 2 else "negative"

    if used_fallback:
        penalty_factor = TITLE_PENALTY_FACTOR
        print(f"[⚠️] Fallback headline used. Applying confidence penalty ({penalty_factor*100:.0f}%).")
        avg_prob *= penalty_factor

    print(f"[INFO] Sentiment: {majority_sentiment} (avg prob: {avg_prob:.2f})")
    return (avg_prob, majority_sentiment, used_fallback)



def save_trade_candidate(ticker, probability):
    conn = sqlite3.connect(TRADE_DB_FILE)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT,
            probability REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cur.execute("INSERT INTO trades (ticker, probability) VALUES (?, ?)", (ticker, probability))
    conn.commit()
    conn.close()
    print(f"[💾] Saved {ticker} with probability {probability:.2f} to potential_trades.db")

def fetch_news_for_company(row):
    company = row.get("company_name")
    ticker = row.get("ticker")
    if not company:
        return (ticker, company, [])

    news = fetch_google_news_feed_sorted(company, max_results=max_news, minutes_back=minutes_back)
    return (ticker, company, news)

if __name__ == "__main__":
    print("🔄 Running TradingView scraper...")
    run_scraper_pipeline()

    print("\n🗂️  Fetching latest gainers from database...")
    rows = get_latest_gainers()

    print(f"\n🚀 Searching Google News for {len(rows)} companies using {MAX_WORKERS} threads...\n")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_company = {executor.submit(fetch_news_for_company, row): row for row in rows}
        for future in as_completed(future_to_company):
            ticker, company, news = future.result()
            if not company:
                continue
            print(f"\n🔍 {company} ({ticker})")

            if not news:
                print("No recent news found.")
                continue

            article_probs = []

            for article in news:
                print(f"• {article['published']} | {article['title']}\n  {article['link']}")
                raw_url = article['link']
                resolved_url = resolve_actual_url(raw_url)
                print(f"[🔗] Resolved article URL:\n  {resolved_url}")
                result = analyze_article(resolved_url)
                if result:
                    article_probs.append(result[0])

            if article_probs:
                avg_prob = sum(article_probs) / len(article_probs)
                if avg_prob >= SENTIMENT_THRESHOLD:
                    save_trade_candidate(ticker, avg_prob)
                else:
                    print(f"[INFO] {ticker} did not meet threshold ({avg_prob:.2f} < {SENTIMENT_THRESHOLD})")
            else:
                print(f"[INFO] No sentiment data collected for {ticker}")
