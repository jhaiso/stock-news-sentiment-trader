import sqlite3
from tradingview_gainers_scraper import run_scraper_pipeline
from google_search import fetch_google_news_feed_sorted
from concurrent.futures import ThreadPoolExecutor, as_completed

DB_FILE = "gainers.db"
minutes_back = 60
max_news = 1
MAX_WORKERS = 100

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
            if news:
                for article in news:
                    print(f"• {article['published']} | {article['title']}\n  {article['link']}")
            else:
                print("No recent news found.")
