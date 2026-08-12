"""Pine Script v5 → Python port of the SAIYAN OCC (XXX) indicator.

Ports the SIGNAL/CONDITION/TP-SL state machine only. Visual decoration (supply/demand
boxes, BOS, fractals, SR zones, KC bands, RSI panels, EMA144, channel-balance fills,
bar coloring, label rendering) is NOT ported — only what drives entries, exits, TPs.

Faithfulness highlights:
* ALMA weights index Pine's `i=0` (current bar) onto the LAST element of a Python
  rolling window (oldest..newest). Inverse mapping is encoded in `alma()`.
* HTF MA is computed by pandas resampling LTF candles into the higher timeframe, then
  forward-filling onto the LTF index. This is `lookahead_off` semantics — for live
  alerts it matches TradingView's realtime. For historical bars it differs slightly
  from `lookahead_on`, which Pine uses here for the explainer (we accept a small dev).
* Signal evaluation happens on closed bars. A signal on bar `i` is recorded with
  bar `i`'s close timestamp, matching Pine's `freq_once_per_bar_close` +
  `process_orders_on_close`.
* The `switch` statement's branching order (TP3 → TP2 → TP1 → SL → entry for the
  opposing side) is preserved. If `high` jumps over both TP1 and TP2 in a single bar,
  only TP1 fires — this is what Pine does too.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Event / config model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Plan:
    """The trade plan attached to events of a single position.

    `entry_price` is the close of the bar that triggered the entry; tp1/tp2/tp3 and
    sl are the four engineered exit prices recoded at entry time using
    cfg.tp_levels_pct and cfg.sl_level_pct. Persisted forward through the state
    machine so TP/SL cross events can show the ORIGINAL plan alongside the level
    that just got hit. Frozen: this is descriptive state, not mutable machinery.
    """
    side: str                # "Long" or "Short"
    entry_price: float
    tp1: float
    tp2: float
    tp3: float
    sl: float

@dataclass
class Event:
    time: pd.Timestamp       # bar close time the event fires on
    symbol: str
    kind: str                # 'Long Entry' | 'Short Entry' | 'Long TP1/2/3' | 'Short TP1/2/3' | 'Long SL' | 'Short SL'
    price: float             # bar close for entries; bar high/low for crosses
    level: Optional[float]   # crossed TP/SL level for cross events; None for entries
    plan: Optional[Plan] = None  # the trade plan from this position's entry; None for legacy/test-fired events without plan info
    # TP targets already filled BY THE TIME this event is emitted (NOT including the
    # current event's contribution). For SL events this captures whether TP1 / TP2
    # had already been hit BEFORE the SL fired — populated from the prior state in
    # run_indicator so format_event doesn't reverse-engineer fill state from kind
    # alone. Defaults to empty frozenset for legacy events and Entry events (nothing
    # filled BEFORE Entry fires).
    filled: frozenset = field(default_factory=frozenset)
    # Institutional filter grade + tags, set on Entry events when the filter
    # stack is active. None for legacy/management events.
    confidence: Optional[int] = None
    context: Optional[tuple] = None



@dataclass
class IndicatorConfig:
    ma_type: str = "ALMA"
    basis_len: int = 2
    offset_sigma: float = 5.0
    offset_alma: float = 0.85
    use_res: bool = True
    intres_multiplier: int = 8
    chart_timeframe_min: int = 15
    delay_offset: int = 0
    heikin_ashi: bool = False
    tp_levels_pct: Tuple[float, float, float] = (1.0, 1.5, 2.0)
    sl_level_pct: float = 0.5
    trade_type: str = "BOTH"  # LONG / SHORT / BOTH / NONE


# ---------------------------------------------------------------------------
# MA variant functions — the 12 variants from Pine's `variant()`
# ---------------------------------------------------------------------------

def _to_float(a) -> np.ndarray:
    return np.asarray(a, dtype=float)


def sma(a, length: int) -> np.ndarray:
    return pd.Series(_to_float(a)).rolling(length, min_periods=length).mean().to_numpy()


def ema(a, length: int) -> np.ndarray:
    """Standard EMA α=2/(L+1) seeded with X_0.

    NOTE on Pine parity: TradingView's `ta.ema(src, length)` uses a 2-phase recursive
    init that subtly differs from a vanilla SMA-seeded EMA on the very first `length`
    bars. With the SAIYAN OCC defaults (`basis_len=2`, ALMA-driven) the difference is
    effectively zero; users who flip to DEMA/TEMA/HullMA with longer `basis_len` should
    eyeball-compare a replay against TradingView before going live."""
    return pd.Series(_to_float(a)).ewm(span=length, adjust=False, min_periods=length).mean().to_numpy()


def dema(a, length: int) -> np.ndarray:
    e = ema(a, length)
    return 2 * e - ema(e, length)


def tema(a, length: int) -> np.ndarray:
    e1 = ema(a, length)
    e2 = ema(e1, length)
    e3 = ema(e2, length)
    return 3 * (e1 - e2) + e3


def wma(a, length: int) -> np.ndarray:
    arr = _to_float(a)
    out = np.full_like(arr, np.nan)
    if arr.shape[0] < length:
        return out
    # Pine semantics: src[0] = current, weight = (len - i). Newest gets weight=length, oldest=1.
    # In Python's windowed array (oldest..newest), arr[-1] is newest → weight=length, arr[i-length+1] oldest → weight=1.
    weights = np.arange(1.0, length + 1)  # [1, 2, ..., length]
    for i in range(length - 1, arr.shape[0]):
        out[i] = np.dot(weights, arr[i - length + 1:i + 1]) / weights.sum()
    return out


def vwma(prices, volumes, length: int) -> np.ndarray:
    pv = _to_float(prices) * _to_float(volumes)
    s_pv = pd.Series(pv).rolling(length, min_periods=length).sum().to_numpy()
    s_v = pd.Series(_to_float(volumes)).rolling(length, min_periods=length).sum().to_numpy()
    return s_pv / s_v


def smma(a, length: int) -> np.ndarray:
    arr = _to_float(a)
    out = np.full_like(arr, np.nan)
    if arr.shape[0] < length:
        return out
    out[length - 1] = np.nanmean(arr[:length])
    for i in range(length, arr.shape[0]):
        out[i] = (out[i - 1] * (length - 1) + arr[i]) / length
    return out


def hullma(a, length: int) -> np.ndarray:
    half = max(1, int(length / 2))
    arr = _to_float(a)
    if arr.shape[0] < length:
        return np.full_like(arr, np.nan)
    h = wma(arr, half)
    f = wma(arr, length)
    raw = 2 * h - f
    return wma(_to_float(raw), int(round(math.sqrt(length))))


def linreg(a, length: int, offset: int = 0) -> np.ndarray:
    """Linear regression over `length` bars. Pine offset=0 emits the fitted line at
    the most recent bar; offset > 0 projects `offset` bars forward."""
    arr = _to_float(a)
    out = np.full_like(arr, np.nan)
    if arr.shape[0] < length:
        return out
    x = np.arange(1.0, length + 1)
    sum_x = x.sum()
    sum_x2 = (x * x).sum()
    for i in range(length - 1, arr.shape[0]):
        window = arr[i - length + 1:i + 1]
        sum_y = window.sum()
        sum_xy = (x * window).sum()
        slope = (length * sum_xy - sum_x * sum_y) / (length * sum_x2 - sum_x * sum_x)
        intercept = (sum_y - slope * sum_x) / length
        out[i] = intercept + slope * (length - 1 + offset)
    return out


def alma(a, length: int, offset: float = 0.85, sigma: float = 5.0) -> np.ndarray:
    """ALMA. Pine semantics: `src[i]` with i=0 == newest. We map arr[k] (oldest..newest)
    to Pine's i = (length - 1 - k). Weight peak at i = offset * (length - 1)."""
    arr = _to_float(a)
    out = np.full_like(arr, np.nan)
    if arr.shape[0] < length or sigma <= 0:
        return out
    weights = np.array([
        math.exp(-(((length - 1 - k) - offset * (length - 1)) ** 2) / (2.0 * sigma * sigma))
        for k in range(length)
    ])
    for i in range(length - 1, arr.shape[0]):
        out[i] = np.dot(weights, arr[i - length + 1:i + 1]) / weights.sum()
    return out


def ssma(a, length: int) -> np.ndarray:
    """Ehlers SuperSmoother via second-order IIR — the SSMA in Pine's `variant()`."""
    arr = _to_float(a)
    out = np.full_like(arr, np.nan)
    if arr.shape[0] < 3:
        return out
    a1 = math.exp(-1.414 * math.pi / length)
    b1 = 2 * a1 * math.cos(1.414 * math.pi / length)
    c1 = 1 - b1 + a1 * a1
    c2 = b1
    c3 = -a1 * a1
    # Pine seed via `nz(v12[1])` fallback on the first valid bar.
    out[1] = c1 * (arr[1] + arr[0]) / 2
    for i in range(2, arr.shape[0]):
        out[i] = c1 * (arr[i] + arr[i - 1]) / 2 + c2 * out[i - 1] + c3 * out[i - 2]
    return out


def tma(a, length: int) -> np.ndarray:
    return sma(sma(a, length), length)


def apply_variant(prices, volumes, ma_type: str, length: int, off_sig: float, off_alma: float):
    if ma_type == "VWMA":
        return vwma(prices, volumes, length)
    if ma_type == "SMA":   return sma(prices, length)
    if ma_type == "EMA":   return ema(prices, length)
    if ma_type == "DEMA":  return dema(prices, length)
    if ma_type == "TEMA":  return tema(prices, length)
    if ma_type == "WMA":   return wma(prices, length)
    if ma_type == "SMMA":  return smma(prices, length)
    if ma_type == "HullMA": return hullma(prices, length)
    if ma_type == "LSMA":  return linreg(prices, length, int(off_sig))
    if ma_type == "ALMA":  return alma(prices, length, offset=off_alma, sigma=off_sig)
    if ma_type == "TMA":   return tma(prices, length)
    if ma_type == "SSMA":  return ssma(prices, length)
    raise ValueError(f"Unknown MA type: {ma_type}")


# ---------------------------------------------------------------------------
# Heikin-Ashi (used only if `cfg.heikin_ashi`); needs warmup for convergence
# ---------------------------------------------------------------------------

def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ha_close = (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0
    ha_open = pd.Series(np.nan, index=df.index, dtype=float)
    ha_open.iloc[0] = (df["open"].iloc[0] + df["close"].iloc[0]) / 2.0
    for i in range(1, len(df)):
        ha_open.iloc[i] = (ha_open.iloc[i - 1] + ha_close.iloc[i - 1]) / 2.0
    ha_high = pd.concat([df["high"], ha_open, ha_close], axis=1).max(axis=1)
    ha_low = pd.concat([df["low"], ha_open, ha_close], axis=1).min(axis=1)
    out["open"], out["high"], out["low"], out["close"] = ha_open, ha_high, ha_low, ha_close
    return out


# ---------------------------------------------------------------------------
# HTF aggregation (resample + ffill back to LTF index)
# ---------------------------------------------------------------------------

def aggregate_htf(df: pd.DataFrame, ltf_minutes: int, intres: int) -> pd.DataFrame:
    """Aggregate LTF OHLCV into HTF bars (HTF minutes = ltf_minutes * intres).
    Drops the last (developing) HTF bar AND the first HTF bucket when `df` doesn't
    start on an HTF boundary — leading buckets would otherwise aggregate only a partial
    slice of LTF bars and produce a misleading early MA."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df.index must be a pandas DatetimeIndex.")
    htf_minutes = ltf_minutes * intres
    if htf_minutes < 1440:
        rule = f"{htf_minutes}min"
    else:
        days = htf_minutes // 1440
        rule = f"{days}D"
    agg = df.resample(rule, origin="start_day").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna()
    # Drop the leading (potentially partial) bucket if df begins inside the first HTF bucket.
    # Guard: only drop when at least 2 buckets remain, so we don't yield an empty agg that
    # silently retires the HTF MA. The downstream `if len(htf_df) < cfg.basis_len` check
    # inside `run_indicator` is the second safety net for the single-bucket edge case.
    if len(agg) >= 2:
        first_boundary = agg.index[0] - pd.Timedelta(minutes=htf_minutes)
        if df.index[0] > first_boundary:
            agg = agg.iloc[1:]
    return agg


# ---------------------------------------------------------------------------
# Crossover / crossunder
# ---------------------------------------------------------------------------

def crossover(a: pd.Series, b: pd.Series) -> pd.Series:
    return ((a > b) & (a.shift(1) <= b.shift(1))).fillna(False)


def crossunder(a: pd.Series, b: pd.Series) -> pd.Series:
    return ((a < b) & (a.shift(1) >= b.shift(1))).fillna(False)


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

def _shift_for_delay(arr: np.ndarray, delay: int) -> np.ndarray:
    out = np.full_like(arr, np.nan, dtype=float)
    if delay <= 0:
        return arr.astype(float).copy()
    if delay < arr.shape[0]:
        out[delay:] = arr[:-delay]
    return out


def run_indicator(df: pd.DataFrame, cfg: IndicatorConfig, symbol: str = "?",
                  filter_cfg=None, block_log=None) -> list:
    """Run the indicator over `df` (OHLCV, DatetimeIndex, UTC). Returns events.
    Signal evaluation uses bar i's high/low/close against bar i-1's high/low against
    state from bar i-1 — i.e., closed-bar semantics.

    Entry-bar cross checks: on the bar where `leTrigger`/`seTrigger` fires, `tp*_val`
    and `sl_val` are still NaN/0 from before any position existed, so any TP1/TP2/TP3
    cross on the SAME bar is suppressed by our `np.isfinite(...)` guard. This matches
    Pine's `switch` priority (entry branch is checked before the TP branches on bar `i`),
    so the *emitted* signal is identical. We document this rather than faithfully reproduce
    Pine's intermediate "TP hit but entry wins anyway" intermediate state.
    """
    df = df.copy()
    if cfg.heikin_ashi:
        df = heikin_ashi(df)

    # 1. Build the variant series on LTF
    n = len(df)
    if n < max(60, cfg.basis_len * 4):  # rough warmup guard
        return []

    close_src = _shift_for_delay(df["close"].to_numpy(dtype=float), cfg.delay_offset)
    open_src  = _shift_for_delay(df["open"].to_numpy(dtype=float),  cfg.delay_offset)
    volume    = df["volume"].to_numpy(dtype=float)

    close_series_ltf = apply_variant(close_src, volume, cfg.ma_type, cfg.basis_len,
                                     cfg.offset_sigma, cfg.offset_alma)
    open_series_ltf  = apply_variant(open_src,  volume, cfg.ma_type, cfg.basis_len,
                                     cfg.offset_sigma, cfg.offset_alma)

    # 2. Build the variant series on HTF (if enabled) and forward-fill to LTF index
    if cfg.use_res:
        htf_df = aggregate_htf(df, ltf_minutes=cfg.chart_timeframe_min, intres=cfg.intres_multiplier)
        if len(htf_df) < cfg.basis_len:
            close_aligned = close_series_ltf
            open_aligned  = open_series_ltf
        else:
            htf_close = _shift_for_delay(htf_df["close"].to_numpy(dtype=float), cfg.delay_offset)
            htf_open  = _shift_for_delay(htf_df["open"].to_numpy(dtype=float),  cfg.delay_offset)
            htf_vol   = htf_df["volume"].to_numpy(dtype=float)
            close_htf = apply_variant(htf_close, htf_vol, cfg.ma_type, cfg.basis_len,
                                      cfg.offset_sigma, cfg.offset_alma)
            open_htf  = apply_variant(htf_open,  htf_vol, cfg.ma_type, cfg.basis_len,
                                      cfg.offset_sigma, cfg.offset_alma)
            series_close_htf = pd.Series(close_htf, index=htf_df.index).reindex(df.index, method="ffill")
            series_open_htf  = pd.Series(open_htf,  index=htf_df.index).reindex(df.index, method="ffill")
            # Backfill any leading NaN (first HTF bar fills first LTF bars).
            close_aligned = pd.Series(close_series_ltf, index=df.index).where(series_close_htf.isna(),
                                                                              series_close_htf).to_numpy()
            open_aligned  = pd.Series(open_series_ltf,  index=df.index).where(series_open_htf.isna(),
                                                                              series_open_htf).to_numpy()
    else:
        close_aligned = close_series_ltf
        open_aligned  = open_series_ltf

    # If leading NaN remains (insufficient warmup), keep it — crossovers against NaN are False.
    c = pd.Series(close_aligned, index=df.index)
    o = pd.Series(open_aligned,  index=df.index)
    le_bo = crossover(c, o).to_numpy()
    se_bo = crossunder(c, o).to_numpy()

    # 3. Walk the bars
    events = []
    state = 0.0
    entry_val = sl_val = tp1_val = tp2_val = tp3_val = np.nan
    # `current_plan` is the active trade's TP/SL lineup, set on entry transitions
    # and carried forward to every emitted Event so the alert formatter can render
    # the full plan alongside the single crossed level. Naturally overwritten on
    # the next entry; left intact across TP walks so a TP1 cross still shows the
    # TP2/TP3/SL targets that haven't fired yet.
    current_plan: Optional[Plan] = None

    opens  = df["open"].to_numpy(dtype=float)
    highs  = df["high"].to_numpy(dtype=float)
    lows   = df["low"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)
    # Pre-shift high/low into "previous bar" arrays for fast cross checks
    prev_highs = np.concatenate([[np.nan], highs[:-1]])
    prev_lows  = np.concatenate([[np.nan], lows[:-1]])

    is_long  = cfg.trade_type in ("LONG", "BOTH")
    is_short = cfg.trade_type in ("SHORT", "BOTH")
    is_any   = cfg.trade_type != "NONE"

    # Institutional filter stack (causal confluence + hard gates). When
    # active, candidate entries are gated per bar; a blocked entry leaves
    # the state machine flat, so no phantom TP/SL events follow downstream.
    entry_filter = None
    _conf_fn = None
    if filter_cfg is not None and getattr(filter_cfg, "enabled", True):
        from filters import EntryFilter, confidence as _conf_fn_import
        entry_filter = EntryFilter(filter_cfg, df, symbol=symbol, block_log=block_log)
        _conf_fn = _conf_fn_import

    for i in range(1, n):  # start at 1 — need prior bar for cross checks
        le_i = bool(le_bo[i])
        se_i = bool(se_bo[i])

        # Gate the entry (strictly causal): a blocked entry opens nothing.
        le_ok = le_i and state <= 0
        se_ok = se_i and state >= 0
        passed_tags = None
        if entry_filter is not None:
            if le_ok:
                le_ok, passed_tags = entry_filter(i, "Long")
            if se_ok:
                se_ok, passed_tags = entry_filter(i, "Short")

        # Update entry / SL / TP lines from current bar
        if le_ok:
            entry_i = closes[i]
            sl_i    = entry_i * (1 - cfg.sl_level_pct / 100.0)
            tp1_i   = entry_i * (1 + cfg.tp_levels_pct[0] / 100.0)
            tp2_i   = entry_i * (1 + cfg.tp_levels_pct[1] / 100.0)
            tp3_i   = entry_i * (1 + cfg.tp_levels_pct[2] / 100.0)
            current_plan = Plan(side="Long", entry_price=entry_i,
                                tp1=tp1_i, tp2=tp2_i, tp3=tp3_i, sl=sl_i)
        elif se_ok:
            entry_i = closes[i]
            sl_i    = entry_i * (1 + cfg.sl_level_pct / 100.0)
            tp1_i   = entry_i * (1 - cfg.tp_levels_pct[0] / 100.0)
            tp2_i   = entry_i * (1 - cfg.tp_levels_pct[1] / 100.0)
            tp3_i   = entry_i * (1 - cfg.tp_levels_pct[2] / 100.0)
            current_plan = Plan(side="Short", entry_price=entry_i,
                                tp1=tp1_i, tp2=tp2_i, tp3=tp3_i, sl=sl_i)
        else:
            entry_i = entry_val
            sl_i    = sl_val
            tp1_i   = tp1_val
            tp2_i   = tp2_val
            tp3_i   = tp3_val

        # Cross checks
        def crossed(prev_, cur_, level_prev, level_cur):
            if not (np.isfinite(level_prev) and np.isfinite(level_cur)):
                return False
            return bool((cur_ > level_cur) and (prev_ <= level_prev)), bool((cur_ < level_cur) and (prev_ >= level_prev))

        # Long-side TP1/2/3 crosses (high > tp AND prev high <= tp)
        tp1_long  = (highs[i] > tp1_i)   and (prev_highs[i] <= tp1_val)   if np.isfinite(tp1_i)   and np.isfinite(tp1_val)   else False
        tp1_short = (lows[i]  < tp1_i)   and (prev_lows[i]  >= tp1_val)   if np.isfinite(tp1_i)   and np.isfinite(tp1_val)   else False
        tp2_long  = (highs[i] > tp2_i)   and (prev_highs[i] <= tp2_val)   if np.isfinite(tp2_i)   and np.isfinite(tp2_val)   else False
        tp2_short = (lows[i]  < tp2_i)   and (prev_lows[i]  >= tp2_val)   if np.isfinite(tp2_i)   and np.isfinite(tp2_val)   else False
        tp3_long  = (highs[i] > tp3_i)   and (prev_highs[i] <= tp3_val)   if np.isfinite(tp3_i)   and np.isfinite(tp3_val)   else False
        tp3_short = (lows[i]  < tp3_i)   and (prev_lows[i]  >= tp3_val)   if np.isfinite(tp3_i)   and np.isfinite(tp3_val)   else False
        sl_long   = (lows[i]  < sl_i)    and (prev_lows[i]  >= sl_val)    if np.isfinite(sl_i)    and np.isfinite(sl_val)    else False
        sl_short  = (highs[i] > sl_i)    and (prev_highs[i] <= sl_val)    if np.isfinite(sl_i)    and np.isfinite(sl_val)    else False

        # Switch (priority: entry > TP3 > TP2 > TP1 > SL; per Pine ordering)
        new_state = state
        if le_ok:
            new_state = 1.0
        elif se_ok:
            new_state = -1.0
        elif tp3_long  and state ==  1.2: new_state =  1.3
        elif tp3_short and state == -1.2: new_state = -1.3
        elif tp2_long  and state ==  1.1: new_state =  1.2
        elif tp2_short and state == -1.1: new_state = -1.2
        elif tp1_long  and state ==  1.0: new_state =  1.1
        elif tp1_short and state == -1.0: new_state = -1.1
        elif sl_long   and state >=  1.0: new_state =  0.0
        elif sl_short  and state <= -1.0: new_state =  0.0

        # Feed the filter's sequential state: cooldowns count from the
        # actual close bars (a stop-out or a fully-banked TP3), never from
        # entry bars.
        if entry_filter is not None:
            if (state ==  1.2 and new_state ==  1.3) or (state == -1.2 and new_state == -1.3):
                entry_filter.mark_tp(i)
            if new_state == 0.0 and ((sl_long and state >= 1.0) or (sl_short and state <= -1.0)):
                entry_filter.mark_sl(i)

        evt_time = df.index[i]
        # Emit events in switch-priority order (a single bar yields at most one event).
        if   new_state ==  1.3 and state ==  1.2 and is_long:
            events.append(Event(evt_time, symbol, "Long TP3",  highs[i], tp3_i, plan=current_plan, filled=frozenset({"TP1", "TP2", "TP3"})))
        elif new_state == -1.3 and state == -1.2 and is_short:
            events.append(Event(evt_time, symbol, "Short TP3", lows[i],  tp3_i, plan=current_plan, filled=frozenset({"TP1", "TP2", "TP3"})))
        elif new_state ==  1.2 and state ==  1.1 and is_long:
            events.append(Event(evt_time, symbol, "Long TP2",  highs[i], tp2_i, plan=current_plan, filled=frozenset({"TP1", "TP2"})))
        elif new_state == -1.2 and state == -1.1 and is_short:
            events.append(Event(evt_time, symbol, "Short TP2", lows[i],  tp2_i, plan=current_plan, filled=frozenset({"TP1", "TP2"})))
        elif new_state ==  1.1 and state ==  1.0 and is_long:
            events.append(Event(evt_time, symbol, "Long TP1",  highs[i], tp1_i, plan=current_plan, filled=frozenset({"TP1"})))
        elif new_state == -1.1 and state == -1.0 and is_short:
            events.append(Event(evt_time, symbol, "Short TP1", lows[i],  tp1_i, plan=current_plan, filled=frozenset({"TP1"})))
        elif new_state == 0.0  and state >=  1.0  and sl_long and is_long:
            # SL can fire from state 1.0/1.1/1.2 — capture prior TP fills so the
            # alert shows what was already realised before the stop-out. The state
            # machine encodes progress in the fractional part of |state|: 1.0=enter,
            # 1.1=after TP1, 1.2=after TP2 (1.3=after TP3 — but SL doesn't fire there).
            # Use `int(round((abs(state)-1.0)*10))` to map back to a 0/1/2 count.
            sl_filled = frozenset(("TP1", "TP2", "TP3")[:int(round((abs(state) - 1.0) * 10))])
            events.append(Event(evt_time, symbol, "Long SL",   lows[i],  sl_i, plan=current_plan, filled=sl_filled))
            current_plan = None  # position closed at SL; future bars carry no plan until next entry
        elif new_state == 0.0  and state <= -1.0 and sl_short and is_short:
            sl_filled = frozenset(("TP1", "TP2", "TP3")[:int(round((abs(state) - 1.0) * 10))])
            events.append(Event(evt_time, symbol, "Short SL",  highs[i], sl_i, plan=current_plan, filled=sl_filled))
            current_plan = None  # position closed at SL; future bars carry no plan until next entry
        elif new_state ==  1.0 and state <= 0 and is_any:
            ev = Event(evt_time, symbol, "Long Entry",  closes[i], closes[i], plan=current_plan)
            if entry_filter is not None and passed_tags is not None:
                ev.confidence = _conf_fn(passed_tags)
                ev.context = tuple(passed_tags)
            events.append(ev)
        elif new_state == -1.0 and state >= 0 and is_any:
            ev = Event(evt_time, symbol, "Short Entry", closes[i], closes[i], plan=current_plan)
            if entry_filter is not None and passed_tags is not None:
                ev.confidence = _conf_fn(passed_tags)
                ev.context = tuple(passed_tags)
            events.append(ev)

        state = new_state
        entry_val, sl_val, tp1_val, tp2_val, tp3_val = entry_i, sl_i, tp1_i, tp2_i, tp3_i

    return events
