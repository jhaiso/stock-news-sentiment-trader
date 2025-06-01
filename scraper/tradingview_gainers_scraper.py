"""_summary_
Script that scrapes TradingView's Daily Screeners
Returns:
    None - Saves information from each stock in the following format
           to gainers.db
    "ticker":       ticker,
    "company_name": company_name,
    "pct_change":   pct_change,
    "rel_volume":   rel_volume
    
"""

import sqlite3
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from bs4 import BeautifulSoup
from datetime import datetime, time as dtime

try:
    from zoneinfo import ZoneInfo  # Python 3.9+
    NY = ZoneInfo("America/New_York")
except ImportError:
    import pytz
    NY = pytz.timezone("America/New_York")

PRE_MARKET_URL     = "https://www.tradingview.com/markets/stocks-usa/market-movers-pre-market-gainers/"
REGULAR_MARKET_URL = "https://www.tradingview.com/markets/stocks-usa/market-movers-gainers/"
AFTER_HOURS_URL    = "https://www.tradingview.com/markets/stocks-usa/market-movers-after-hours-gainers/"

DB_FILE = "gainers.db"

def pick_gainers_url(now=None):
    if now is None:
        now = datetime.now(NY).time()
    if dtime(4, 0) <= now < dtime(9, 30):
        return PRE_MARKET_URL
    if dtime(9, 30) <= now < dtime(16, 0):
        return REGULAR_MARKET_URL
    if dtime(16, 0) <= now < dtime(20, 0):
        return AFTER_HOURS_URL
    return REGULAR_MARKET_URL

def init_db(db_file):
    conn = sqlite3.connect(db_file, timeout=30, check_same_thread=False)
    c = conn.cursor()
    c.execute("PRAGMA journal_mode=WAL;")
    c.execute("""
    CREATE TABLE IF NOT EXISTS gainers (
        ts            TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        ticker        TEXT      PRIMARY KEY,
        company_name  TEXT,
        pct_change    TEXT,
        rel_volume    TEXT
    );
    """)
    conn.commit()
    conn.close()

def save_to_db(db_file, rows):
    conn = sqlite3.connect(db_file, timeout=30, check_same_thread=False)
    c = conn.cursor()
    c.execute("PRAGMA journal_mode=WAL;")
    sql = """
      INSERT OR REPLACE INTO gainers 
        (ticker, company_name, pct_change, rel_volume)
      VALUES (?, ?, ?, ?);
    """
    data = [
        (
            item["ticker"],
            item["company_name"],
            item["pct_change"],
            item["rel_volume"]
        )
        for item in rows
    ]
    c.executemany(sql, data)
    conn.commit()
    conn.close()
    
def clear_db(db_file):
    conn = sqlite3.connect(db_file, timeout=30, check_same_thread=False)
    c    = conn.cursor()
    # ensure WAL mode
    c.execute("PRAGMA journal_mode=WAL;")

    # check if the table "gainers" exists
    c.execute("""
      SELECT name FROM sqlite_master
       WHERE type='table' AND name='gainers';
    """)
    if c.fetchone():
        c.execute("DELETE FROM gainers;")
        print("[INFO] Cleared gainers table.")
    else:
        print("[INFO] Table 'gainers' does not exist; nothing to clear.")

    conn.commit()
    conn.close()

def scrape_gainers(page_url, timeout=15):
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")

    driver = webdriver.Chrome(options=chrome_options)
    driver.get(page_url)

    WebDriverWait(driver, timeout).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "tr.listRow"))
    )

    soup = BeautifulSoup(driver.page_source, "html.parser")
    driver.quit()

    results = []
    for tr in soup.select("tr.listRow"):
        ticker = tr.get("data-rowkey")
        name_tag = tr.find("sup", class_="tickerDescription-GrtoTeat")
        company_name = name_tag.get_text(strip=True) if name_tag else None
        pct_tag = tr.find_all("td")[1].find("span")
        pct_change = pct_tag.get_text(strip=True) if pct_tag else None
        rel_vol_td = tr.find_all("td")[4]
        rel_volume = rel_vol_td.get_text(strip=True) if rel_vol_td else None

        results.append({
            "ticker":       ticker,
            "company_name": company_name,
            "pct_change":   pct_change,
            "rel_volume":   rel_volume
        })

    return results

def run_scraper_pipeline(db_file=DB_FILE):
    """Runs the full scraping + DB save pipeline and returns the data."""
    clear_db(db_file)  # Clear the existing rows before inserting new data
    init_db(db_file)
    url = pick_gainers_url()
    print("Scraping:", url)
    rows = scrape_gainers(url)
    save_to_db(db_file, rows)
    print(f"Saved {len(rows)} rows to {db_file}")
    return rows

if __name__ == "__main__":
    results = run_scraper_pipeline()
    # Optionally print all stored rows
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("SELECT * FROM gainers ORDER BY ts DESC;")
    for r in cur.fetchall():
        print(r)
    conn.close()
