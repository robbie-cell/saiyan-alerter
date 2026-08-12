"""Institutional filter stack for SAIYAN signals.

A raw signal from `run_indicator` is an MA-crossover entry. This module is the
confluence layer that decides whether that entry is *tradeable* — and grades it
for display. Every filter is strictly causal (uses only data available at the
bar being evaluated), so results are identical whether evaluated live in the
cron, in the local bot, or in the historical research harness.

Tiers:
  * Hard filters — block the entry outright: extreme-volatility regime
    (dead / chaotic markets), re-entry cooldowns after a stop or a fully-banked
    trade, and an optional UTC session window.
  * Confluence — contribute to the displayed confidence grade but do not block:
    higher-timeframe trend alignment (the big one), a lighter LTF trend, and
    entry-bar momentum.

`FilterConfig` mirrors the `filters:` block in config.yaml; `build_context`
precomputes the vectorised series once per DataFrame and `EntryFilter` closes
over them plus sequential state (cooldown counters), exposing the per-bar
callable used by `run_indicator` plus `mark_sl` / `mark_tp` hooks that the
state machine calls when a stop or a full TP3 completes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from indicator import aggregate_htf


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class FilterConfig:
    enabled: bool = True

    # HTF trend confluence — resample LTF into htf_minutes bars, EMA(htf_ema_len)
    # on their closes, forward-fill onto the LTF index (same lookahead-off
    # semantics as the indicator's own HTF MA). Long allowed only when the HTF
    # trend is up (close > EMA), short only when down. 0 disables.
    htf_minutes: int = 240          # 4h — one rung above the indicator's 2h HTF
    htf_ema_len: int = 50           # ~8.3 days of 4h bars
    # True = misaligned trend BLOCKS the entry; False = alignment only raises
    # the displayed confidence grade.
    hard_htf: bool = False

    # LTF trend confluence — EMA(ltf_ema_len) on 15m closes. 0 disables.
    ltf_ema_len: int = 0

    # Volatility regime — ATR(14) vs the 10th/90th percentile of the trailing
    # `vol_window` bars (percentiles shifted back 1 bar: strictly causal).
    # LOW  = atr < q10  (dead market, no follow-through)
    # HIGH = atr > q90  (chaos, slippage, whipsaw)
    # Both block by default; set a flag false to tolerate that extreme.
    vol_window: int = 960           # 10 days at 15m
    vol_low_pct: float = 10.0
    vol_high_pct: float = 90.0
    block_vol_low: bool = True
    block_vol_high: bool = True

    # Entry-bar momentum:  off | impulse | candle
    #   impulse       — close must move in the trade direction vs the prior close
    #   candle        — close must be on the trade side of the bar's open (bullish
    #                   candle for Long, bearish for Short)
    momentum: str = "off"
    # True = a failed momentum check BLOCKS the entry; False = soft (grade only).
    hard_momentum: bool = False

    # Re-entry discipline: bars to wait after an SL (a loss) or a full TP3
    # (all thirds banked) before the same pair may be re-entered.
    cooldown_sl_bars: int = 0
    cooldown_tp_bars: int = 0

    # Session window in UTC hours [start, end) — only entries whose bar closes
    # inside the window are allowed. None = 24/7.
    session_utc: Optional[Tuple[int, int]] = None


# ---------------------------------------------------------------------------
# Context computation (vectorised, causal)
# ---------------------------------------------------------------------------

def _ema_ffill(close: pd.Series, length: int) -> np.ndarray:
    if length <= 0:
        return np.full(len(close), np.nan)
    return close.ewm(span=length, adjust=False, min_periods=length).mean().to_numpy()


def build_context(df: pd.DataFrame, cfg: FilterConfig) -> dict:
    """Precompute per-bar filter series over the full DataFrame.

    All series are aligned to df.index and strictly causal: rolling windows are
    shifted back one bar where a same-bar value would leak the current bar.
    """
    ctx: dict = {}

    # --- HTF trend (4h EMA, ffilled from the last completed HTF bucket) ---
    if cfg.htf_ema_len > 0 and cfg.htf_minutes > 0:
        htf = aggregate_htf(df, ltf_minutes=15, intres=max(1, cfg.htf_minutes // 15))
        if len(htf) >= cfg.htf_ema_len + 2:
            htf_ema = _ema_ffill(htf["close"], cfg.htf_ema_len)
            trend = pd.Series(
                np.where(np.isfinite(htf_ema), htf["close"].to_numpy() > htf_ema, False),
                index=htf.index,
            )
            ctx["htf_up"] = trend.reindex(df.index, method="ffill", fill_value=False).to_numpy(dtype=bool)
        else:
            ctx["htf_up"] = np.full(len(df), False, dtype=bool)

    # --- LTF trend (15m EMA) ---
    if cfg.ltf_ema_len > 0:
        ema = _ema_ffill(df["close"], cfg.ltf_ema_len)
        ctx["ltf_up"] = np.where(np.isfinite(ema), df["close"].to_numpy() > ema, False)

    # --- Volatility regime (ATR(14) vs shifted rolling percentiles) ---
    if cfg.block_vol_low or cfg.block_vol_high:
        h = df["high"].to_numpy(float)
        l = df["low"].to_numpy(float)
        c = df["close"].to_numpy(float)
        pc = np.concatenate([[c[0]], c[:-1]])
        tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])
        atr = pd.Series(tr).rolling(14, min_periods=14).mean()
        warm = max(50, cfg.vol_window // 4)
        q_low = atr.rolling(cfg.vol_window, min_periods=warm).quantile(cfg.vol_low_pct / 100.0).shift(1)
        q_high = atr.rolling(cfg.vol_window, min_periods=warm).quantile(cfg.vol_high_pct / 100.0).shift(1)
        regime = np.full(len(df), "MID", dtype=object)
        regime[np.where(np.isfinite(q_low) & (atr.to_numpy() < q_low.to_numpy()))[0]] = "LOW"
        regime[np.where(np.isfinite(q_high) & (atr.to_numpy() > q_high.to_numpy()))[0]] = "HIGH"
        ctx["vol_regime"] = regime
        ctx["atr_pct"] = atr.to_numpy()

    return ctx


# ---------------------------------------------------------------------------
# Per-bar entry filter
# ---------------------------------------------------------------------------

class EntryFilter:
    """Causal per-bar gate over indicator entries.

    Construct once per DataFrame (per pair per run), then feed it to
    `run_indicator`. `run_indicator` calls `f(i, side)` for every candidate
    entry bar and `mark_sl(i)` / `mark_tp(i)` when a stop-out or a fully-banked
    TP3 completes, so re-entry cooldowns count from the actual close bar.
    """

    def __init__(self, cfg: FilterConfig, df: pd.DataFrame, symbol: str = "?",
                 block_log: Optional[list] = None):
        self.cfg = cfg
        self.symbol = symbol
        self.block_log = block_log
        self._ctx = build_context(df, cfg) if cfg.enabled else {}
        self._closes = df["close"].to_numpy(float)
        self._opens = df["open"].to_numpy(float)
        self._index = df.index
        self._last_sl = -10**9
        self._last_tp = -10**9

        if cfg.session_utc:
            hours = df.index.hour.to_numpy()
            s, e = cfg.session_utc
            self._in_session = ((hours >= s) & (hours < e)) if s <= e else ((hours >= s) | (hours < e))
        else:
            self._in_session = None

    # -- state hooks called by run_indicator --------------------------------
    def mark_sl(self, i: int) -> None:
        self._last_sl = i

    def mark_tp(self, i: int) -> None:
        self._last_tp = i

    # -- the gate -----------------------------------------------------------
    def __call__(self, i: int, side: str) -> Tuple[bool, list]:
        passed: list = []

        # --- hard filters (block) ---
        if self._in_session is not None and not bool(self._in_session[i]):
            self._log(i, side, ["session"])
            return False, []
        if i - self._last_sl <= self.cfg.cooldown_sl_bars:
            self._log(i, side, [f"cooldown_sl:{i - self._last_sl}bar"])
            return False, []
        if i - self._last_tp <= self.cfg.cooldown_tp_bars:
            self._log(i, side, [f"cooldown_tp:{i - self._last_tp}bar"])
            return False, []

        regime = self._ctx.get("vol_regime")
        if regime is not None:
            r = regime[i] if i < len(regime) else "MID"
            if r == "LOW" and self.cfg.block_vol_low:
                self._log(i, side, ["vol:LOW"])
                return False, []
            if r == "HIGH" and self.cfg.block_vol_high:
                self._log(i, side, ["vol:HIGH"])
                return False, []
            if r in ("LOW", "HIGH"):
                pass  # tolerated extreme — no confluence credit
            else:
                passed.append("vol:OK")

        # --- HTF trend: hard gate or confluence (soft) ---
        if "htf_up" in self._ctx:
            up = bool(self._ctx["htf_up"][i])
            aligned = (up if side == "Long" else not up)
            if self.cfg.hard_htf and not aligned:
                self._log(i, side, ["htf_misaligned"])
                return False, []
            if aligned:
                passed.append("htf")
        if "ltf_up" in self._ctx:
            up = bool(self._ctx["ltf_up"][i])
            if up if side == "Long" else not up:
                passed.append("ltf")
        if self.cfg.momentum == "impulse":
            if i > 0:
                ok = (self._closes[i] > self._closes[i - 1]) if side == "Long" else (self._closes[i] < self._closes[i - 1])
            else:
                ok = False
            if self.cfg.hard_momentum and not ok:
                self._log(i, side, ["momentum"])
                return False, []
            if ok:
                passed.append("mom")
        elif self.cfg.momentum == "candle":
            ok = (self._closes[i] > self._opens[i]) if side == "Long" else (self._closes[i] < self._opens[i])
            if self.cfg.hard_momentum and not ok:
                self._log(i, side, ["momentum"])
                return False, []
            if ok:
                passed.append("mom")

        return True, passed

    # -- telemetry ----------------------------------------------------------
    def _log(self, i: int, side: str, reasons: list) -> None:
        if self.block_log is not None:
            self.block_log.append({
                "symbol": self.symbol,
                "time": self._index[i].isoformat(),
                "side": side,
                "reasons": reasons,
            })


def confidence(passed: list, total_filters: int = 4) -> int:
    """Grade an allowed entry 2–5.

    Base 2 for the raw indicator signal, +1 per confluence check that passed
    (htf / ltf / vol:OK / mom), capped at 5.
    """
    credit = {"htf", "ltf", "vol:OK", "mom"}
    score = 2 + sum(1 for p in passed if p in credit)
    return min(5, score)
