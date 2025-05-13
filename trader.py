#!/usr/bin/env python3
import os
import threading
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
import pandas as pd
import numpy as np
setattr(np, "NaN", np.nan)
import pandas_ta as ta
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import APIError

# Lumibot data‐source for Alpaca
from lumibot.data.data_sources.alpaca_data_source import AlpacaDataSource

# ── API setup ───────────────────────────────────────────────────────────────
load_dotenv()
API_KEY    = os.environ['APCA_API_KEY_ID']
API_SECRET = os.environ['APCA_API_SECRET_KEY']
API_BASE   = os.environ.get(
    'APCA_API_BASE_URL',
    'https://paper-api.alpaca.markets'
)
api = tradeapi.REST(API_KEY, API_SECRET, API_BASE, api_version='v2')

# ── Market‐data helper ─────────────────────────────────────────────────────────
def get_minute_bars(symbol: str,
                    start:  str,
                    end:    str,
                    limit:  int = 500
                   ) -> pd.DataFrame:
    """
    Try Lumibot’s historical fetch (unlimited lookback on IEX feed).
    If that fails, fall back to Alpaca.get_bars().
    Returns a tz‐aware NY‐time DataFrame: [open,high,low,close,volume].
    """
    # 1) Try Lumibot first
    try:
        df = lumibot_ds.get_price_history(
            symbol     = symbol,
            timeframe  = "1Min",
            start_date = pd.Timestamp(start),
            end_date   = pd.Timestamp(end),
            limit      = limit
        )
        # Lumibot returns a DataFrame with a DatetimeIndex already
        return df.tz_convert("America/New_York")

    except Exception as e:
        print(f"[WARN] Lumibot failed ({e}), falling back to Alpaca…")

    # 2) Fallback: Alpaca.get_bars (still limited to last 15 min on free tier)
    syms = [symbol]
    try:
        raw = api.get_bars(
            syms,
            tradeapi.TimeFrame.Minute,
            start=start,
            end=end,
            limit=limit,
            feed="iex"
        ).df
    except APIError as e:
        print(f"[WARN] Alpaca.get_bars failed for {symbol}: {e}")
        return pd.DataFrame(columns=["open","high","low","close","volume"])

    if raw.empty:
        return raw

    # strip off MultiIndex if present
    if isinstance(raw.index, pd.MultiIndex):
        df = raw.xs(symbol, level=0)
    else:
        df = raw

    return df.tz_convert("America/New_York")

# ── Indicators & entry signal ─────────────────────────────────────────────────
def compute_indicators(df):
    df['vwap']   = ta.vwap(df['high'], df['low'], df['close'], df['volume'])
    df['ema5']   = ta.ema(df['close'], length=5)
    df['rsi14']  = ta.rsi(df['close'], length=14)
    return df

def entry_signal(symbol, df):
    """
    Return True if all conditions met on the last bar:
      1) Breaks above yesterday's high
      2) Volume > 1.5× 20-period avg
      3) Pullback to VWAP or EMA5 in last 5 bars
      4) RSI14 < 70
    """
    last = df.iloc[-1]

    # ── fetch the last two daily bars with v2 get_bars ──────────────────────────
    end_d   = pd.Timestamp.now(tz='America/New_York')
    start_d = (end_d - pd.Timedelta(days=2)).isoformat()
    try:
        raw_day = api.get_bars(
            [symbol],
            tradeapi.TimeFrame.Day,
            start=start_d,
            end=end_d.isoformat(),
            limit=2
        ).df
    except APIError as e:
        print(f"[WARN] Couldn't fetch daily bars for {symbol}: {e}")
        return False

    # slice out symbol if MultiIndex
    if isinstance(raw_day.index, pd.MultiIndex):
        day_df = raw_day.xs(symbol, level=0)
    else:
        day_df = raw_day

    # pick yesterday's high
    today_floor = end_d.floor('D')
    if last.name.floor('D') == today_floor:
        y_high = day_df['high'].iloc[-2]  # bar from the prior day
    else:
        y_high = day_df['high'].iloc[-1]

    if last.close <= y_high:
        return False

    # volume filter
    vol20 = df['volume'].rolling(20).mean().iloc[-1]
    if last.volume < 1.5 * vol20:
        return False

    # pullback test
    window = df.tail(5)
    pb_ok = ((window['close'] - window['vwap']).abs().min() < 0.003 * last.close) \
         or ((window['close'] - window['ema5']).abs().min() < 0.003 * last.close)
    if not pb_ok:
        return False

    # RSI
    if last.rsi14 >= 70:
        return False

    return True

# ── Position sizing (example) ───────────────────────────────────────────────
def size_position(symbol, dollar_risk=100):
    return 10

# ── Split‐exit order logic ───────────────────────────────────────────────────
def submit_split_exit(symbol: str,
                      qty: int,
                      stop_pct: float = 0.02,
                      rr: float     = 2.0,
                      ema_len: int  = 5):
    order = api.submit_order(
        symbol=symbol,
        qty=qty,
        side='buy',
        type='market',
        time_in_force='day'
    )
    entry_price = None
    while entry_price is None:
        time.sleep(0.2)
        o = api.get_order(order.id)
        if o.filled_avg_price:
            entry_price = float(o.filled_avg_price)

    stop_price   = round(entry_price * (1 - stop_pct), 2)
    target_price = round(entry_price * (1 + stop_pct * rr), 2)
    half_qty     = qty // 2
    trail_qty    = qty - half_qty

    api.submit_order(
        symbol=symbol,
        qty=half_qty,
        side='sell',
        type='limit',
        time_in_force='day',
        limit_price=target_price
    )
    print(f"→ {symbol}: Bought {qty} @ {entry_price:.2f}, "
          f"placed limit‐sell {half_qty} @ {target_price:.2f}, "
          f"hard stop @ {stop_price:.2f}")

    def _trail_exit():
        current_stop = stop_price
        print(f"🔄 Starting EMA({ema_len}) trail for {trail_qty} shares of {symbol}")
        while True:
            now   = datetime.now(tz=pd.Timestamp.now().tz)
            start = now - timedelta(minutes=ema_len * 10)
            df    = get_minute_bars(symbol, start.isoformat(), now.isoformat())
            if df.shape[0] < ema_len:
                time.sleep(5)
                continue
            ema_val = ta.ema(df['close'], length=ema_len).iloc[-1]
            last_px = df['close'].iloc[-1]
            if last_px < ema_val or last_px < current_stop:
                reason = "EMA" if last_px < ema_val else "STOP"
                print(f"⚠️  Trail exit: {symbol} at {last_px:.2f} (broke {reason})")
                api.submit_order(
                    symbol=symbol,
                    qty=trail_qty,
                    side='sell',
                    type='market',
                    time_in_force='day'
                )
                break
            time.sleep(10)

    t = threading.Thread(target=_trail_exit, daemon=True)
    t.start()

# ── CLI smoke‐test for get_minute_bars ─────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser("Smoke‐test get_minute_bars()")
    p.add_argument("--symbol",
                   default="AAPL",
                   help="Ticker to fetch")
    p.add_argument("--minutes",
                   type=int,
                   default=60,
                   help="How many minutes back to fetch")
    p.add_argument("--end",
                   default=None,
                   help="End timestamp (ISO) — default=now NY")
    args = p.parse_args()

    end = (pd.Timestamp.now(tz="America/New_York")
           if args.end is None
           else pd.Timestamp(args.end))
    start = end - pd.Timedelta(minutes=args.minutes)

    print(f"\n→ Fetching {args.symbol} from {start.isoformat()} to {end.isoformat()} …")
    df = get_minute_bars(
        args.symbol,
        start.isoformat(),
        end.isoformat(),
        limit=args.minutes
    )

    if df.empty:
        print("→ got NO data back (empty DataFrame).")
    else:
        print(f"→ received {len(df)} rows.  HEAD:")
        print(df.head())
        print("\nTAIL:")
        print(df.tail())