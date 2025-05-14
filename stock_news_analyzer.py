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

DB_FILE           = "gainers.db"
TRADE_DB_FILE     = "potential_trades.db"
SENTIMENT_THRESHOLD = 0.7
minutes_back      = 20
max_news          = 1
MAX_WORKERS       = 100
TITLE_PENALTY_FACTOR = 0.85

def init_url_cache(db_file=TRADE_DB_FILE):
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    c.execute("""
      CREATE TABLE IF NOT EXISTS analyzed_urls (
        url         TEXT    PRIMARY KEY,
        probability REAL,
        sentiment   TEXT,
        timestamp   DATETIME DEFAULT CURRENT_TIMESTAMP
      );
    """)
    conn.commit()
    conn.close()

def has_url_been_analyzed(db_file, url):
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    c.execute("SELECT 1 FROM analyzed_urls WHERE url = ?", (url,))
    found = c.fetchone() is not None
    conn.close()
    return found

def get_cached_sentiment(db_file, url):
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    c.execute("SELECT probability, sentiment FROM analyzed_urls WHERE url = ?", (url,))
    row = c.fetchone()
    conn.close()
    return (row[0], row[1]) if row else None

def mark_url_as_analyzed(db_file, url, probability, sentiment):
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    c.execute("""
      INSERT OR REPLACE INTO analyzed_urls (url, probability, sentiment)
      VALUES (?, ?, ?)
    """, (url, probability, sentiment))
    conn.commit()
    conn.close()

def get_latest_gainers():
    conn = sqlite3.connect(DB_FILE)
    cur  = conn.cursor()
    cur.execute("SELECT ticker, company_name, pct_change, rel_volume FROM gainers ORDER BY ts DESC")
    rows = [
      {"ticker": row[0], "company_name": row[1], "pct_change": row[2], "rel_volume": row[3]}
      for row in cur.fetchall()
    ]
    conn.close()
    return rows

def resolve_actual_url(google_news_url):
    try:
        status = gnewsdecoder(google_news_url)
        print(f"[↪️] Decoded URL found: {status['decoded_url']}")
        return status['decoded_url']
    except Exception as e:
        print(f"[ERROR] Failed to resolve article URL: {e}")
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
            print("[WARN] Content extraction failed and no fallback provided.")
            return None
    else:
        summary = content.get("summary", "").strip()
        if not summary:
            if fallback_text:
                print("[WARN] No summary extracted. Using fallback title.")
                summary = fallback_text.strip()
                used_fallback = True
            else:
                print("[WARN] No summary extracted and no fallback provided.")
                return None

    if not summary:
        print("[WARN] Summary (or fallback) is empty.")
        return None

    results = []
    for fn in (finbert_sentiment, llama_sentiment, gpt_sentiment):
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
    majority_sent = "positive" if pos_count >= 2 else "negative"

    if used_fallback:
        print(f"[⚠️] Fallback used, applying penalty {TITLE_PENALTY_FACTOR:.2f}")
        avg_prob *= TITLE_PENALTY_FACTOR

    print(f"[INFO] Sentiment: {majority_sent} (avg prob: {avg_prob:.2f})")
    return (avg_prob, majority_sent, used_fallback)

def clean_ticker(ticker: str) -> str:
    return ticker.split(":", 1)[-1].strip()

def save_trade_candidate(ticker: str, probability: float):
    clean = clean_ticker(ticker)
    conn  = sqlite3.connect(TRADE_DB_FILE)
    cur   = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker      TEXT    UNIQUE,
            probability REAL,
            timestamp   DATETIME DEFAULT (DATETIME('now','localtime'))
        );
    """)
    cur.execute("""
        INSERT INTO trades (ticker, probability)
        VALUES (?, ?)
        ON CONFLICT(ticker) DO UPDATE
          SET probability = excluded.probability,
              timestamp   = CURRENT_TIMESTAMP;
    """, (clean, probability))
    conn.commit()
    conn.close()
    print(f"[💾] Saved {clean} @ {probability:.2f}")

def fetch_news_for_company(row):
    company = row.get("company_name")
    ticker  = row.get("ticker")
    if not company:
        return (ticker, company, [])
    news = fetch_google_news_feed_sorted(
        company,
        max_results = max_news,
        minutes_back= minutes_back
    )
    return (ticker, company, news)

def process_articles_for_ticker(ticker: str, articles: list):
    def _analyze_one(art):
        raw_url = art.get("link")
        title   = art.get("title", "")
        url     = resolve_actual_url(raw_url)

        if has_url_been_analyzed(TRADE_DB_FILE, url):
            prob, sent = get_cached_sentiment(TRADE_DB_FILE, url)
            print(f"   ↳ [cache] {url}: {prob:.2f} {sent}")
            return prob, sent

        res = analyze_article(url, fallback_text=title)
        if not res:
            return None

        prob, sent, _ = res
        mark_url_as_analyzed(TRADE_DB_FILE, url, prob, sent)
        return prob, sent

    probs = []
    sentiments = []
    with ThreadPoolExecutor(max_workers=8) as exe:
        futures = {exe.submit(_analyze_one, art): art for art in articles}
        for fut in as_completed(futures):
            result = fut.result()
            if result:
                prob, sent = result
                probs.append(prob)
                sentiments.append(sent)

    if not probs:
        print(f"[INFO] {ticker} no usable sentiment data.")
        return

    avg_prob = sum(probs) / len(probs)
    pos_count = sum(1 for s in sentiments if s.lower() == "positive")
    majority_sent = "positive" if pos_count >= 2 else "negative"

    print(f"[INFO] {ticker}: averaged prob = {avg_prob:.2f}, sentiment = {majority_sent}")

    if avg_prob >= SENTIMENT_THRESHOLD and majority_sent == "positive":
        save_trade_candidate(ticker, avg_prob)
    else:
        print(f"[INFO] {ticker} did not meet sentiment requirements "
              f"({avg_prob:.2f}, sentiment: {majority_sent})")

if __name__ == "__main__":
    init_url_cache(TRADE_DB_FILE)

    print("🔄 Running TradingView scraper...")
    run_scraper_pipeline()

    print("\n🗂️ Fetching latest gainers from database...")
    rows = get_latest_gainers()

    print(f"\n🚀 Searching Google News for {len(rows)} companies using {MAX_WORKERS} threads…\n")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(fetch_news_for_company, row): row
            for row in rows
        }
        for future in as_completed(futures):
            ticker, company, news = future.result()
            if not company or not news:
                continue

            print(f"\n🔍 {company} ({ticker})")
            process_articles_for_ticker(ticker, news)
