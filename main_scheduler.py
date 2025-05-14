#!/usr/bin/env python3
import os, time, sqlite3
from datetime import datetime, time as dtime, timedelta
import zoneinfo
from concurrent.futures import ThreadPoolExecutor, as_completed

from tradingview_gainers_scraper import run_scraper_pipeline
from stock_news_analyzer import (
    init_url_cache,
    fetch_news_for_company,
    process_articles_for_ticker,
    TRADE_DB_FILE,
    MAX_WORKERS
)
from trader import (
    api,
    get_minute_bars,
    compute_indicators,
    entry_signal,
    size_position,
    submit_split_exit
)

# ── CONFIG ────────────────────────────────────────────────────────────────
INTERVAL_MINUTES    = 10
TRADER_START        = dtime(8, 0)
TRADER_END          = dtime(19, 0)
TZ_NY               = zoneinfo.ZoneInfo("America/New_York")

MAX_OPEN_TRADES     = 3
MAX_CHECKED_SYMBOLS = 5

# NEW: your risk parameters
RISK_PCT_PER_TRADE  = 0.02   # lose no more than  2% of cash
STOP_PCT_PER_TRADE  = 0.02   # hard stop at 2% below entry

def run_trader():
    # 1) how many are already open?
    open_positions = api.list_positions()
    open_syms      = {p.symbol for p in open_positions}
    slots_left     = MAX_OPEN_TRADES - len(open_syms)
    if slots_left <= 0:
        print(f"🔒 max open trades ({MAX_OPEN_TRADES}) reached; skipping entries.")
        return

    # 2) pick top symbols by probability
    conn = sqlite3.connect(TRADE_DB_FILE)
    cur  = conn.cursor()
    cur.execute(
        "SELECT ticker FROM trades "
        "ORDER BY probability DESC LIMIT ?",
        (MAX_CHECKED_SYMBOLS,)
    )
    rows = [r[0] for r in cur.fetchall()]
    conn.close()

    # 3) exclude already‐open & cap by slots_left
    candidates = [s for s in rows if s not in open_syms][:slots_left]

    # 4) test entry & size position
    for symbol in candidates:
        end   = datetime.now(TZ_NY)
        start = end - timedelta(minutes=60)
        df = get_minute_bars(symbol, start.isoformat(), end.isoformat())
        if df.empty:
            print(f"[{symbol}] no minute‐data; skipping.")
            continue

        df = compute_indicators(df)
        if entry_signal(symbol, df):
            qty = size_position(symbol,
                                risk_pct=RISK_PCT_PER_TRADE,
                                stop_pct=STOP_PCT_PER_TRADE)
            if qty > 0:
                submit_split_exit(symbol,
                                  qty,
                                  stop_pct=STOP_PCT_PER_TRADE)
            else:
                print(f"[{symbol}] not enough cash to size a {RISK_PCT_PER_TRADE*100:.1f}% risk trade.")
        else:
            print(f"[{symbol}] no entry signal.")

def main():
    print("Initializing URL cache…")
    init_url_cache()

    while True:
        now = datetime.now(TZ_NY)
        print(f"\n[{now.isoformat()}] Starting cycle…")

        gainers = run_scraper_pipeline()
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as exe:
            futures = {exe.submit(fetch_news_for_company, row): row
                       for row in gainers}
            for future in as_completed(futures):
                ticker, company, news = future.result()
                if not company or not news:
                    continue
                process_articles_for_ticker(ticker, news)

        now_time = now.time()
        if TRADER_START <= now_time <= TRADER_END:
            print("→ Within trading hours. Running trader…")
            run_trader()
        else:
            print("→ Outside trading hours. Skipping trader.")

        print(f"Sleeping {INTERVAL_MINUTES} minutes…")
        time.sleep(INTERVAL_MINUTES * 60)

if __name__ == "__main__":
    main()