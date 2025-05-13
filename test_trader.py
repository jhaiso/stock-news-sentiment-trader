# ====================================================================
# Hack: inject a fake pandas_ta into sys.modules so that
#       `import pandas_ta as ta` inside trader.py never fails
# ====================================================================
import sys
import types
import pandas as pd

fake_ta = types.ModuleType("pandas_ta")

# a simple EMA implementation
def fake_ema(close_series, length):
    # just use pandas' EWM for test‐purposes
    return close_series.ewm(span=length, adjust=False).mean()

# a dummy VWAP (just returns close for simplicity)
def fake_vwap(high, low, close, volume):
    return pd.Series(close.values, index=close.index)

# a dummy RSI (just returns 50 everywhere)
def fake_rsi(close_series, length):
    return pd.Series(50.0, index=close_series.index)

fake_ta.ema  = fake_ema
fake_ta.vwap = fake_vwap
fake_ta.rsi  = fake_rsi

sys.modules['pandas_ta'] = fake_ta
# ====================================================================

# now we can safely import the rest
import pytest
import pandas as pd
import trader   # your trader.py


# -----------------------------------------------------------------------------
# A fake Alpaca client to inject into trader.api
# -----------------------------------------------------------------------------
class FakeBarFrame:
    def __init__(self, df):
        self.df = df


class FakeOrder:
    def __init__(self, oid, filled_price=None):
        self.id = oid
        # string or number
        self.filled_avg_price = filled_price


class FakeAPI:
    def __init__(self):
        self.bars_calls = []
        self.barset_calls = []
        self.order_book = {}
        self.submit_calls = []
        self.get_order_calls = 0

    # Simulate get_bars(...)
    def get_bars(self, symbol, timeframe, start, end, limit):
        self.bars_calls.append((symbol, timeframe, start, end, limit))
        # We'll store a DataFrame in self._next_bars if caller wants it
        df = getattr(self, "_next_bars", pd.DataFrame())
        return FakeBarFrame(df.copy())

    # Simulate get_barset for daily bars
    def get_barset(self, syms, timeframe, limit):
        self.barset_calls.append((tuple(syms), timeframe, limit))
        # caller expects .df[symbol]
        # Use stored self._yday_bars[symbol]
        data = {}
        for s in syms:
            data[s] = self._yday_bars[s].copy()
        # build MultiIndex for DataFrame
        df = pd.concat(data, axis=1)
        return FakeBarFrame(df)

    def submit_order(self, **kwargs):
        # record the kwargs
        self.submit_calls.append(kwargs)
        oid = f"OID{len(self.submit_calls)}"
        # Simulate immediate fill if market order
        if kwargs.get("type") == "market" and kwargs.get("side") == "buy":
            # We'll let get_order return a price next
            self.order_book[oid] = True
        return FakeOrder(oid)

    def get_order(self, oid):
        # Called repeatedly until filled_avg_price is set
        self.get_order_calls += 1
        # once called, always return a filled price
        return FakeOrder(oid, filled_price="100.00")


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def patch_api(monkeypatch):
    """
    Every test runs with trader.api replaced by our FakeAPI.
    """
    fake = FakeAPI()
    monkeypatch.setattr(trader, "api", fake)
    return fake


@pytest.fixture(autouse=True)
def patch_thread(monkeypatch):
    """
    Prevent real threads from starting: run the target synchronously once.
    """
    class DummyThread:
        def __init__(self, target, daemon):
            self.target = target
        def start(self):
            # call once synchronously
            self.target()

    monkeypatch.setattr(trader.threading, "Thread", DummyThread)
    return DummyThread


# -----------------------------------------------------------------------------
# Tests for get_minute_bars
# -----------------------------------------------------------------------------
def test_get_minute_bars_empty(patch_api):
    patch_api._next_bars = pd.DataFrame()  # no data
    df = trader.get_minute_bars("FOO", "s", "e", limit=123)
    assert df.empty
    # check it called the API with the right args
    assert patch_api.bars_calls[0][0] == "FOO"
    assert patch_api.bars_calls[0][4] == 123


def test_get_minute_bars_nonempty(patch_api):
    # build a fake multi-index DataFrame
    idx = pd.date_range("2023-01-01 10:00", periods=3, freq="T", tz="UTC")
    df0 = pd.DataFrame({
        ("FOO", "open"): [1,2,3],
        ("FOO", "high"): [1,2,3],
        ("FOO", "low"):  [1,2,3],
        ("FOO", "close"):[1,2,3],
        ("FOO", "volume"):[10,20,30],
    }, index=idx)
    df0.columns = pd.MultiIndex.from_tuples(df0.columns)
    patch_api._next_bars = df0
    out = trader.get_minute_bars("FOO", "s", "e")
    # should have index in New York tz
    assert out.index.tz.zone == "America/New_York"
    # should have the symbol-level removed
    assert set(out.columns) == {"open","high","low","close","volume"}


# -----------------------------------------------------------------------------
# Tests for compute_indicators
# -----------------------------------------------------------------------------
def test_compute_indicators_basic():
    d = pd.DataFrame({
        "high":  [10,11,12,13,14,15],
        "low":   [9,10,11,12,13,14],
        "close": [9.5,10.5,11.5,12.5,13.5,14.5],
        "volume":[100,110,120,130,140,150]
    })
    out = trader.compute_indicators(d.copy())
    # should have the three new columns
    assert "vwap" in out
    assert "ema5" in out
    assert "rsi14" in out
    # values should be numeric
    assert pd.api.types.is_numeric_dtype(out["vwap"])
    assert pd.api.types.is_numeric_dtype(out["ema5"])
    assert pd.api.types.is_numeric_dtype(out["rsi14"])


# -----------------------------------------------------------------------------
# Tests for entry_signal
# -----------------------------------------------------------------------------
def make_bar_df(close_values, volume_values, vwap_values, ema5_values, rsi_values, ts=None):
    """
    Build a DataFrame with needed columns and a monochronological index.
    """
    n = len(close_values)
    if ts is None:
        ts = pd.date_range("2023-01-10 14:30", periods=n, freq="T", tz="America/New_York")
    df = pd.DataFrame({
        "close": close_values,
        "volume": volume_values,
        "vwap": vwap_values,
        "ema5": ema5_values,
        "rsi14": rsi_values
    }, index=ts)
    return df

def test_entry_signal_fails_on_price_not_breaking_high(patch_api):
    # Yesterday's high = 100
    yesterday = pd.DataFrame({
        "h": [98, 100]
    }, index=pd.date_range("2023-01-09", periods=2, freq="D"))
    patch_api._yday_bars = {"ABC": yesterday}

    # last close is 99 <= 100 → fail early
    df = make_bar_df([95,96,97,98,99], [1]*5, [0]*5, [0]*5, [50]*5)
    assert not trader.entry_signal("ABC", df)

def test_entry_signal_fails_on_low_volume(patch_api):
    # Yesterday's high = 90
    yesterday = pd.DataFrame({"h":[80,90]},
                             index=pd.date_range("2023-01-09", periods=2, freq="D"))
    patch_api._yday_bars = {"XYZ": yesterday}

    # close breaks above, but volume is too small
    last_close = 95
    df = make_bar_df(
        close_values=[90,91,92,93,95],
        volume_values=[1,1,1,1,1],    # avg20 ~1 → need 1.5
        vwap_values=[0]*5,
        ema5_values=[0]*5,
        rsi_values=[50]*5
    )
    assert not trader.entry_signal("XYZ", df)

def test_entry_signal_fails_on_no_pullback(patch_api):
    yesterday = pd.DataFrame({"h":[40,50]},
                             index=pd.date_range("2023-01-09", periods=2, freq="D"))
    patch_api._yday_bars = {"DEF": yesterday}

    # volume is big enough, close>high, rsi ok, but no bar within 0.3% of vwap or ema5
    close = [50,51,52,53,55]
    vol   = [100]*5
    df = make_bar_df(
        close_values=close,
        volume_values=vol,
        vwap_values=[10,10,10,10,10],
        ema5_values=[10,10,10,10,10],
        rsi_values=[50]*5
    )
    assert not trader.entry_signal("DEF", df)

def test_entry_signal_fails_on_rsi_too_high(patch_api):
    yesterday = pd.DataFrame({"h":[100,110]},
                             index=pd.date_range("2023-01-09", periods=2, freq="D"))
    patch_api._yday_bars = {"GGG": yesterday}

    # break high, vol ok, pb ok, but last rsi >=70
    close = [110,111,112,113,115]
    vol   = [200]*5
    vwap  = close.copy()
    ema5  = close.copy()
    rsi   = [30,30,30,30,70]
    df = make_bar_df(close, vol, vwap, ema5, rsi)
    assert not trader.entry_signal("GGG", df)

def test_entry_signal_success(patch_api):
    yesterday = pd.DataFrame({"h":[50,60]},
                             index=pd.date_range("2023-01-09", periods=2, freq="D"))
    patch_api._yday_bars = {"HHH": yesterday}

    # break high=60, vol>1.5*avg(20)=150, rsi<70, and last 5 have a point within 0.3% of vwap
    close = [61,62,63,64,65]
    vol   = [200]*5
    vwap  = [60.9,61.5,62.3,63.2,64.8]  # last bar 65, 64.8 is within 0.3%
    ema5  = vwap
    rsi   = [50]*5
    df = make_bar_df(close, vol, vwap, ema5, rsi)
    assert trader.entry_signal("HHH", df)


# -----------------------------------------------------------------------------
# Test size_position
# -----------------------------------------------------------------------------
def test_size_position_default():
    assert trader.size_position("ANY") == 10


# -----------------------------------------------------------------------------
# Test submit_split_exit
# -----------------------------------------------------------------------------
def test_submit_split_exit_full_flow(patch_api, patch_thread, monkeypatch):
    """
    We simulate:
     - market buy fills at 100.00
     - limit sell half at target = 100*(1+0.02*2)=104.00
     - trailing thread runs once, sees last px<EMA and submits a market sell
    """
    # Prepare a minute-bar feed that will cause an immediate trail exit
    def fake_minute_bars(symbol, start, end, limit=500):
        # build DataFrame with close below the EMA so we exit by EMA
        idx = pd.date_range("2023-01-10 14:00", periods=5, freq="T", tz="UTC")
        closes = [100, 101, 102,  90,  90]  # drop to 90 < EMA
        df = pd.DataFrame({"close": closes}, index=idx)
        return df

    monkeypatch.setattr(trader, "get_minute_bars", fake_minute_bars)

    # now call the function
    trader.submit_split_exit("BBB", qty=10, stop_pct=0.02, rr=2.0, ema_len=5)

    # 1) first call: market buy
    buy_call = patch_api.submit_calls[0]
    assert buy_call["symbol"] == "BBB"
    assert buy_call["qty"] == 10
    assert buy_call["type"] == "market"

    # 2) limit sell half at target
    limit_call = patch_api.submit_calls[1]
    assert limit_call["symbol"] == "BBB"
    assert limit_call["qty"] == 5  # half
    assert limit_call["type"] == "limit"
    # target_price = round(100*(1+0.02*2),2) = 104.00
    assert float(limit_call["limit_price"]) == pytest.approx(104.00)

    # 3) trailing exit: should detect close<ema and do a market sell of the other half
    trail_call = patch_api.submit_calls[2]
    assert trail_call["symbol"] == "BBB"
    assert trail_call["qty"] == 5
    assert trail_call["type"] == "market"


# If you want to run coverage, just do:
#    pytest --maxfail=1 --disable-warnings -q