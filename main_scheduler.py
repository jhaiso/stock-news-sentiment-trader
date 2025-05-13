#!/usr/bin/env python3
import time
import sqlite3
from datetime import datetime, time as dtime, timedelta
import zoneinfo
from concurrent.futures import ThreadPoolExecutor, as_completed

# tradingview scraper
from tradingview_gainers_scraper import run_scraper_pipeline

# news + sentiment analyzer
from stock_news_analyzer import (
    init_url_cache,
    fetch_news_for_company,
    process_articles_for_ticker,
    TRADE_DB_FILE,
    MAX_WORKERS
)

# alpaca trader bits
from trader import (
    get_minute_bars,
    compute_indicators,
    entry_signal,
    size_position,
    submit_split_exit
)



# ── CONFIG ────────────────────────────────────────────────────────────────
INTERVAL_MINUTES = 10  # run every X minutes
TRADER_START = dtime(8, 0)   # 08:00 Eastern
TRADER_END   = dtime(19, 0)  # 19:00 Eastern
TZ_NY = zoneinfo.ZoneInfo("America/New_York")

# ── HELPER: run trading logic ──────────────────────────────────────────────
def run_trader():
    conn = sqlite3.connect(TRADE_DB_FILE)
    c = conn.cursor()
    c.execute("SELECT ticker FROM trades")
    symbols = [r[0] for r in c.fetchall()]
    conn.close()

    for symbol in symbols:
        end = datetime.now(TZ_NY)
        start = end - timedelta(minutes=60)
        df = get_minute_bars(symbol, start.isoformat(), end.isoformat())
        if df.empty:
            continue
        df = compute_indicators(df)
        if entry_signal(symbol, df):
            qty = size_position(symbol)
            submit_split_exit(symbol, qty)
        else:
            print(f"[{symbol}] no entry signal.")

# ── MAIN LOOP ─────────────────────────────────────────────────────────────
def main():
    print("Initializing URL cache…")
    init_url_cache()

    while True:
        now = datetime.now(TZ_NY)
        print(f"\n[{now.isoformat()}] Starting cycle…")

        # 1) scrape top gainers
        gainers = run_scraper_pipeline()  # returns list of dicts with ticker, company_name, news

        # 2) fetch + analyze news for each gainer
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as exe:
            futures = {
                exe.submit(fetch_news_for_company, row): row
                for row in gainers
            }
            for future in as_completed(futures):
                ticker, company, news = future.result()
                if not company or not news:
                    continue
                process_articles_for_ticker(ticker, news)

        # 3) run the trader if within trading hours
        now_time = now.time()
        if TRADER_START <= now_time <= TRADER_END:
            print("→ Within trading hours. Running trader…")
            run_trader()
        else:
            print("→ Outside trading hours. Skipping trader.")

        # 4) sleep until next cycle
        print(f"Sleeping {INTERVAL_MINUTES} minutes…")
        time.sleep(INTERVAL_MINUTES * 60)


if __name__ == "__main__":
    main()