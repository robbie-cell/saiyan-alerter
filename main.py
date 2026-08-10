"""CLI entry point.

Subcommands:
    replay — fetch historical candles, run the indicator, dump events to CSV.
    live   — poll closed bars and forward events to Telegram (with dedupe + quiet hours).
    check  — run-once scan for cron (GitHub Actions): report bars closed since the
             last run, persisting per-pair progress in a state file.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import time as time_mod
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import ccxt
import pandas as pd

from config import PairConfig, RuntimeConfig, load_config
from indicator import Event, IndicatorConfig, Plan, run_indicator
from telegram import TelegramClient, format_event, log as tg_log

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def fetch_candles(exchange_id: str, symbol: str, timeframe: str,
                  since_ms: Optional[int] = None, limit: int = 1000) -> pd.DataFrame:
    """Fetch OHLCV candles, paginating past the exchange's per-call cap.

    Returns a DataFrame with DatetimeIndex (UTC). Most exchanges — Binance
    included — return at most ~1000 bars per `fetch_ohlcv` call regardless of
    how much history `since_ms` would permit. This function paginates by
    advancing `since_ms` one bar past the previous page's last timestamp and
    looping until we have `limit` bars or the exchange returns a partial page
    (= we've reached the head of available history).

    Args:
        since_ms: lower-bound bar timestamp in ms. `None` returns the most recent.
        limit:    upper bound on total bars to return; per-call cap is 1000.
    """
    if not hasattr(ccxt, exchange_id):
        raise ValueError(f"Exchange `{exchange_id}` not supported by ccxt ({ccxt.exchanges}...)")
    ex_class = getattr(ccxt, exchange_id)
    ex = ex_class({"enableRateLimit": True})
    # Most exchanges cap one response at 1000 bars regardless of `limit`.
    per_call_cap = min(limit, 1000)
    tf_minutes = timeframe_to_minutes(timeframe)

    if since_ms is None:
        # Most-recent-bars query: one page (up to 1000 bars ≈ 10+ days at 15m) is
        # far more than the indicator's warmup needs (~50 bars). Deliberately do
        # NOT paginate forward from here — the next cursor would lie in the
        # future, which Binance tolerates (empty page) but Gate.io rejects with
        # INVALID_PARAM_VALUE "invalid time range".
        raw = ex.fetch_ohlcv(symbol, timeframe, since=None, limit=per_call_cap)
        if not raw:
            empty = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
            return empty.set_index("timestamp")
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return df.drop_duplicates(subset="timestamp").set_index("timestamp").sort_index()

    seen: set = set()
    batches: list = []
    cursor = since_ms
    # Pages needed = ceil(limit / per_call_cap), plus a small buffer for partial-page
    # detection at the head of history. A buggy exchange that repeats the same bar
    # also exits this loop — `df.empty : break` after dedupe handles that case.
    safety_cap = -(-limit // per_call_cap) + 5

    for _ in range(safety_cap):
        raw = ex.fetch_ohlcv(symbol, timeframe, since=cursor, limit=per_call_cap)
        if not raw:
            break
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        # Dedupe across pages — exchanges occasionally re-emit the boundary bar on overlap.
        df = df[~df["timestamp"].isin(seen)]
        if df.empty:
            break
        seen.update(df["timestamp"].tolist())
        batches.append(df)
        # Two ways to stop: we've hit the head of history OR we've collected `limit` bars.
        if len(raw) < per_call_cap:
            break  # partial page = exchange has no more bars beyond this one
        if len(seen) >= limit:
            break  # we have what was asked for
        cursor = int(df["timestamp"].iloc[-1]) + tf_minutes * 60 * 1000

    if not batches:
        empty = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        return empty.set_index("timestamp")
    full = pd.concat(batches, ignore_index=True)
    full["timestamp"] = pd.to_datetime(full["timestamp"], unit="ms", utc=True)
    full = full.drop_duplicates(subset="timestamp").set_index("timestamp").sort_index()
    return full


def timeframe_to_minutes(tf: str) -> int:
    if tf.endswith("m"): return int(tf[:-1])
    if tf.endswith("h"): return int(tf[:-1]) * 60
    if tf.endswith("d"): return int(tf[:-1]) * 60 * 24
    if tf.endswith("w"): return int(tf[:-1]) * 60 * 24 * 7
    raise ValueError(f"Unsupported timeframe: {tf}")


def minutes_to_timeframe(minutes: int) -> str:
    if minutes < 60: return f"{minutes}m"
    if minutes < 1440:
        h = minutes // 60
        m = minutes % 60
        return f"{h}h" if m == 0 else f"{h}h{m}m"
    d = minutes // 1440
    return f"{d}d"


# --------------------------------------------------------------------------
# TradingView deep-link generation
# --------------------------------------------------------------------------
#
# Generates a URL of the form
#   https://www.tradingview.com/chart/?symbol=BINANCE%3ABTCUSDT&interval=15
# which loads a fresh chart with the right pair and timeframe so the user can
# one-click jump from a CSV row to a TradingView candle for eyeball comparison.
# TV's URL `time` anchor is unstable and poorly documented — leaving it out
# keeps the link robust against chart-engine changes.

_TV_INTERVAL = {
    "1m": "1", "2m": "2", "3m": "3", "5m": "5", "10m": "10", "15m": "15",
    "30m": "30", "45m": "45", "1h": "60", "2h": "120", "3h": "180", "4h": "240",
    "1d": "D", "1w": "W", "1M": "M",
}

# ccxt exchange IDs that TradingView maps onto its `BINANCE` provider. We strip
# the suffix so `binanceusdm` → `BINANCE:` (TV defaults to the perp suffix when
# the symbol is suffixed `.P`; this helper leaves that decision to the user).
_TV_EXCHANGE_ALIASES = {
    "binance": "BINANCE",
    "binanceusdm": "BINANCE",
    "binancecoinm": "BINANCE",
    "binanceus": "BINANCE",
    "bybit": "BYBIT",
    "okx": "OKX",
    "okex": "OKX",
    "kraken": "KRAKEN",
    "bitstamp": "BITSTAMP",
    "coinbase": "COINBASE",
    "coinbasepro": "COINBASE",
    "bitfinex": "BITFINEX",
    "huobi": "HTX",          # TradingView rebranded Huobi → HTX in late 2024
    "htx": "HTX",
    "kucoin": "KUCOIN",
    "gate": "GATEIO",
    "gateio": "GATEIO",
}


def tv_url_for(symbol: str, exchange: str, timeframe: str) -> str:
    """Build a TradingView deep link to the pair+TF. `BTC/USDT` → `BINANCE:BTCUSDT`.

    Unknown exchanges fall back to the ccxt id upper-cased. Unknown timeframes
    fall back to a numeric minutes-based interval. Both fallback paths emit a
    one-line log so the operator notices instead of silently opening the wrong
    chart."""
    tv_exchange = _TV_EXCHANGE_ALIASES.get(exchange.lower(), exchange.upper())
    pair_no_slash = symbol.replace("/", "")
    tv_symbol = f"{tv_exchange}:{pair_no_slash}"
    interval = _TV_INTERVAL.get(timeframe)
    if interval is None:
        # Fall back to minutes-based numeric interval for anything unmapped.
        try:
            interval = str(timeframe_to_minutes(timeframe))
        except ValueError:
            # INFO, not WARNING — the link still works; we just picked a default.
            log.info(
                "tv_url_for: timeframe '%s' unrecognised — falling back to 15m interval.",
                timeframe,
            )
            interval = "15"
    # Encode ':' as %3A so URLs render cleanly in spreadsheet apps.
    return f"https://www.tradingview.com/chart/?symbol={quote(tv_symbol, safe='')}&interval={interval}"


# --------------------------------------------------------------------------
# Dedupe + quiet-hours + pair health
# --------------------------------------------------------------------------

class Dedupe:
    """Per-(symbol, kind) dedupe: ignore repeat signals within `minutes`.
    Only advances the timestamp after a successful Telegram send — failed sends do not
    consume the dedupe window, so a transient failure can still retry within window."""
    def __init__(self, minutes: int):
        self.minutes = minutes
        self.last = {}

    def ok(self, e: Event) -> bool:
        key = (e.symbol, e.kind)
        prev = self.last.get(key)
        if prev is None or (e.time - prev).total_seconds() > self.minutes * 60:
            return True
        return False

    def mark(self, e: Event) -> None:
        self.last[(e.symbol, e.kind)] = e.time


class QuietHours:
    def __init__(self, hours: Optional[tuple], tz_name: str = "UTC"):
        self.hours = hours
        try:
            import zoneinfo
            self.tz = zoneinfo.ZoneInfo(tz_name)
        except Exception as exc:
            log.warning(
                "Timezone '%s' unavailable (%s); quiet-hours will run in UTC. "
                "On Windows: `pip install tzdata` resolves this.",
                tz_name, exc.__class__.__name__)
            self.tz = timezone.utc

    def is_quiet(self) -> bool:
        if self.hours is None:
            return False
        start, end = self.hours
        local_hour = pd.Timestamp.now(tz=self.tz).hour
        if start <= end:
            return start <= local_hour < end
        # wraps midnight
        return local_hour >= start or local_hour < end


class PairHealth:
    """Per-pair error suppression.

    After `suppress_after` consecutive failures, errors are silenced for
    `suppress_minutes` minutes with a single warning emitted at the threshold.
    A successful poll clears the counter, the suppression window, and the
    streak-start timestamp.
    """
    def __init__(self, suppress_after: int = 3, suppress_minutes: int = 15):
        self.suppress_after = suppress_after
        self.suppress_minutes = suppress_minutes
        self.errors: dict[str, int] = {}
        self.suppressed_until: dict[str, pd.Timestamp] = {}
        self.first_error_at: dict[str, pd.Timestamp] = {}

    def should_log_error(self, symbol: str, now: pd.Timestamp) -> bool:
        until = self.suppressed_until.get(symbol)
        if until and now < until:
            return False
        err = self.errors.get(symbol, 0) + 1
        self.errors[symbol] = err
        self.first_error_at.setdefault(symbol, now)  # capture streak start on error #1
        if err > self.suppress_after:
            self.suppressed_until[symbol] = now + pd.Timedelta(minutes=self.suppress_minutes)
            log.warning(
                "Pair %s: %d consecutive errors since %s UTC — silencing further error logs for %d minutes.",
                symbol, err, self.first_error_at[symbol].strftime("%H:%M:%S"), self.suppress_minutes,
            )
            return True  # log the threshold-crossing warning once
        return True

    def record_success(self, symbol: str) -> None:
        # Clear all per-symbol state so the dict doesn't grow over long uptimes.
        self.errors.pop(symbol, None)
        self.suppressed_until.pop(symbol, None)
        self.first_error_at.pop(symbol, None)


# --------------------------------------------------------------------------
# Replay mode
# --------------------------------------------------------------------------

def run_replay(cfg: RuntimeConfig, args):
    exchange_id = args.exchange or cfg.exchange
    log.info("Replay %s [%s on %s, last %d hours]", args.symbol, args.timeframe, exchange_id.upper(), args.hours)
    tf_minutes = timeframe_to_minutes(args.timeframe)
    # Deep-copy to avoid mutating the shared cfg (so subsequent `live` runs are unaffected).
    cfg_local = copy.deepcopy(cfg)
    cfg_local.cfg.chart_timeframe_min = tf_minutes

    # Compute how many bars `--hours` actually requires. The CLI default
    # `--limit 2000` is silently too small for `--hours 720` at 15m (which needs
    # 2880 bars); expand to cover the requested history with a small buffer. The
    # expanded limit feeds the paginated fetcher — Binance's 1000-bar-per-call
    # cap is handled internally by fetch_candles.
    needed_bars = int(args.hours * 60 / tf_minutes) + 100
    effective_limit = max(args.limit, needed_bars)
    since_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000) - args.hours * 3600 * 1000
    df = fetch_candles(exchange_id, args.symbol, args.timeframe, since_ms, limit=effective_limit)
    log.info("Loaded %d candles from %s to %s", len(df), df.index[0], df.index[-1])
    if len(df) < needed_bars - 100:
        # The exchange ran out of history before reaching the requested window —
        # surface this instead of silently truncating.
        log.warning(
            "Replay returned %d candles but --hours %d implies ≥%d bars — "
            "the exchange's available history was shorter than requested.",
            len(df), args.hours, needed_bars - 100,
        )

    events = run_indicator(df, cfg_local.cfg, symbol=args.symbol)
    rows = [(e.time.isoformat(), e.symbol, e.kind, e.price, e.level) for e in events]
    cols = ["time", "symbol", "kind", "price", "level"]
    if args.tv_urls:
        url = tv_url_for(args.symbol, cfg.exchange, args.timeframe)
        rows = [r + (url,) for r in rows]
        cols.append("tv_url")
        log.info("TradingView deep-link enabled: %s", url)
    out = pd.DataFrame(rows, columns=cols)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    # Sidecar with the indicator config used (timestamp, pair, TF, MA inputs) for
    # eyeball-comparison against the TradingView chart.
    meta_path = Path(args.output).with_suffix(".meta.json")
    meta = {
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "candle_count": int(len(df)),
        "candle_from": df.index[0].isoformat() if not df.empty else None,
        "candle_to":   df.index[-1].isoformat() if not df.empty else None,
        "indicator": {
            "ma_type": cfg_local.cfg.ma_type,
            "basis_len": cfg_local.cfg.basis_len,
            "offset_sigma": cfg_local.cfg.offset_sigma,
            "offset_alma": cfg_local.cfg.offset_alma,
            "use_res": cfg_local.cfg.use_res,
            "intres_multiplier": cfg_local.cfg.intres_multiplier,
            "chart_timeframe_min": cfg_local.cfg.chart_timeframe_min,
            "tp_levels_pct": list(cfg_local.cfg.tp_levels_pct),
            "sl_level_pct": cfg_local.cfg.sl_level_pct,
            "trade_type": cfg_local.cfg.trade_type,
            "heikin_ashi": cfg_local.cfg.heikin_ashi,
        },
        "event_count": int(len(events)),
        "tv_url": tv_url_for(args.symbol, cfg.exchange, args.timeframe) if args.tv_urls else None,
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    out.to_csv(args.output, index=False)
    log.info("Wrote %d events to %s (meta: %s)", len(events), args.output, meta_path)
    print(out.to_string(index=False))


# --------------------------------------------------------------------------
# Summarize mode (pure read-only — useful after every replay)
# --------------------------------------------------------------------------

def summarize_csv(path: Path) -> dict:
    """Read a replay CSV and produce a kind histogram, L/S balance, and timeframe stats.

    Pure read-only helper, no network. Returns a dict; pretty-printing lives in
    run_summarize() so callers asking for --json get machine-readable output.
    """
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        return {"file": str(path), "event_count": 0}

    df["time"] = pd.to_datetime(df["time"], utc=True)
    event_count = len(df)
    first = df["time"].min()
    last = df["time"].max()
    span_hours = max(0.0, (last - first).total_seconds() / 3600.0)

    kind_counts: dict = df["kind"].value_counts().to_dict()
    long_count = int(df["kind"].str.startswith("Long").sum())
    short_count = int(df["kind"].str.startswith("Short").sum())
    # Match entries by suffix rather than a fixed list — future-proof against adding
    # new entry kinds (e.g. "Long DCA") without silently dropping them from the count.
    entry_count = int(df["kind"].str.endswith(" Entry").sum())
    tp_count = int(df["kind"].str.contains("TP").sum())
    sl_count = int(df["kind"].str.contains("SL").sum())

    events_per_day = (event_count * 24.0 / span_hours) if span_hours > 0 else 0.0
    closed_by_tp_or_sl = tp_count + sl_count
    # `closes_per_entry` is a raw event-count ratio, NOT a 0–1 probability.
    # The dominant reason to exceed 1.0: one Entry event can produce up to 3
    # closes in sequence (TP1 → TP2 → TP3) — a fully-realised trade gives a
    # ratio near 3.0. Pre-window carry-through (positions entered before
    # `first_event` that close in-window) is a second-order cause. Rename keeps
    # the semantics honest; see README §"Summarize a replay CSV".
    closes_per_entry = (closed_by_tp_or_sl / entry_count) if entry_count > 0 else 0.0

    return {
        "file": str(path),
        "event_count": int(event_count),
        "first_event": first.isoformat(),
        "last_event": last.isoformat(),
        "span_hours": round(span_hours, 2),
        "events_per_day": round(events_per_day, 2),
        "long_count": int(long_count),
        "short_count": int(short_count),
        "entries": int(entry_count),
        "tps": int(tp_count),
        "sls": int(sl_count),
        "closes_per_entry": round(closes_per_entry, 4),
        "kinds": {k: int(v) for k, v in kind_counts.items()},
    }


def run_summarize(args) -> None:
    """Pretty-print or JSON-dump a replay CSV's summary."""
    summary = summarize_csv(Path(args.input))
    if args.json:
        print(json.dumps(summary, indent=2, default=str))
        return
    if summary["event_count"] == 0:
        print(f"{summary['file']}: no events.")
        return
    print(f"=== Replay summary: {summary['file']} ===")
    print(f"Events:        {summary['event_count']}")
    print(f"Timespan:      {summary['first_event']}  →  {summary['last_event']}")
    print(f"Window hours:  {summary['span_hours']}")
    print(f"Events / day:  {summary['events_per_day']}")
    print(
        f"Long: {summary['long_count']}   Short: {summary['short_count']}   "
        f"Entries: {summary['entries']}   TPs: {summary['tps']}   SLs: {summary['sls']}"
    )
    cpe = summary["closes_per_entry"]
    print(f"Closes/entry:  {cpe:.3f}  ((TP+SL) events per Entry; can exceed 1.0 because one Entry can fire TP1+TP2+TP3)")
    print("\nBy kind:")
    for kind, count in sorted(summary["kinds"].items(), key=lambda kv: (-kv[1], kv[0])):
        pct = count / summary["event_count"] * 100.0
        bar = "█" * int(pct / 2)
        print(f"  {kind:<14} {count:>4}   {pct:>5.1f}%   {bar}")


# --------------------------------------------------------------------------
# Live mode
# --------------------------------------------------------------------------

def _process_symbol(cfg: RuntimeConfig, pair: PairConfig, tg: TelegramClient, dedupe: Dedupe,
                    quiet: QuietHours) -> list:
    """Fetch trailing candles for `pair.symbol` from `pair.exchange` (or the
    default exchange when `pair.exchange is None`), run the indicator, return
    NEW events (those whose time is on the most recently CLOSED bar). Closed =
    bar timestamp is at least one `chart_timeframe_min` + 30s grace in the past."""
    exchange_id = pair.exchange or cfg.exchange
    symbol = pair.symbol
    tf_minutes = cfg.cfg.chart_timeframe_min
    timeframe_str = minutes_to_timeframe(tf_minutes)
    limit = 1500
    df = fetch_candles(exchange_id, symbol, timeframe_str, limit=limit)
    if df.empty:
        return []
    now_utc = pd.Timestamp.now(tz="UTC")
    # Drop the in-progress bar — a 15m bar at ts=14:00 closes at 14:14:59.99; we wait
    # until 14:15:30 (15m + 30s grace) before evaluating it. Mirrors Pine's
    # `process_orders_on_close=true`, which only fires on confirmed bars.
    cutoff = now_utc - pd.Timedelta(minutes=tf_minutes) - pd.Timedelta(seconds=30)
    df_closed = df[df.index <= cutoff]
    if df_closed.empty:
        return []
    events = run_indicator(df_closed, cfg.cfg, symbol=symbol)
    last_bar_time = df_closed.index[-1]
    new_events = [e for e in events if e.time == last_bar_time]
    if quiet.is_quiet():
        return []
    sent = []
    for e in new_events:
        if not dedupe.ok(e):
            continue
        msg = format_event(e)
        log.info("alert: %s", msg, extra={"symbol": symbol, "exchange": exchange_id})
        if tg.send_message(msg):
            dedupe.mark(e)
            sent.append(e)
        else:
            tg_log.error("Failed to deliver event: %s", msg)
    return sent


def _send_startup_message(tg: TelegramClient, cfg: RuntimeConfig, focus: Optional[PairConfig]) -> None:
    """Send a 'bot started' Telegram message so the user can confirm reachability
    and see what the bot is currently configured to monitor. Per-pair exchange
    overrides are surfaced with `@(exchange)` so the user knows when a non-default
    exchange is being polled."""
    pairs_monitored = [focus] if focus else list(cfg.pairs)
    default_ex = (cfg.exchange or "").lower()
    parts: list = []
    for p in pairs_monitored:
        if p.exchange and p.exchange.lower() != default_ex:
            parts.append(f"{p.symbol}@({p.exchange.lower()})")
        else:
            parts.append(p.symbol)
    pairs_str = ", ".join(parts)
    tf_str = minutes_to_timeframe(cfg.cfg.chart_timeframe_min)
    htf_min = cfg.cfg.chart_timeframe_min * cfg.cfg.intres_multiplier
    htf_str = minutes_to_timeframe(htf_min)
    msg = (
        f"🟢 SAIYAN alerter started\n"
        f"Exchange:  {cfg.exchange}\n"
        f"Pairs ({len(pairs_monitored)}):  {pairs_str}\n"
        f"Timeframe: {tf_str} (HTF: {htf_str})\n"
        f"MA:        {cfg.cfg.ma_type} (len={cfg.cfg.basis_len})\n"
        f"TradeType: {cfg.cfg.trade_type}"
    )
    if tg.send_message(msg):
        log.info("Sent startup Telegram message.")
    else:
        log.warning("Could not deliver startup Telegram message.")


def run_live(cfg: RuntimeConfig, once: bool = False, focus_pair: Optional[str] = None):
    if not cfg.telegram_token or not cfg.telegram_chat_id or cfg.telegram_token == "":
        raise SystemExit("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set in .env for live mode.")
    tg = TelegramClient(cfg.telegram_token, cfg.telegram_chat_id)
    dedupe = Dedupe(cfg.dedupe_minutes)
    quiet = QuietHours(cfg.quiet_hours, cfg.timezone)
    health = PairHealth()

    # Hugging Face Spaces: keep-alive + platform health checks need an HTTP
    # endpoint on $PORT (7860). Pings reset the 48h-inactivity sleep timer, so
    # the Space stays awake 24/7. No-op unless HF_SPACE=1 (set in the Space's
    # Dockerfile); runs in a daemon thread, never blocks the polling loop.
    if os.getenv("HF_SPACE") == "1":
        from healthserver import start_health_server
        start_health_server()

    # Resolve --pair CLI focus to a PairConfig. Force user to register the symbol
    # in config.yaml (with a per-pair `exchange` override if needed) so we don't
    # silently poll the default exchange for a symbol it doesn't carry.
    if focus_pair:
        match: Optional[PairConfig] = next(
            (p for p in cfg.pairs if p.symbol == focus_pair), None
        )
        if match is None:
            raise SystemExit(
                f"--pair '{focus_pair}' is not registered in config.yaml. "
                f"Add it under `pairs:` as either '{focus_pair}' (uses default exchange) "
                f"or a dict like '{{symbol: \"{focus_pair}\", exchange: \"<name>\"}}' "
                f"to override the exchange for that pair."
            )
        pairs: list = [match]
    else:
        pairs = list(cfg.pairs)

    # When focused down to a single pair, the startup message should reflect
    # that focus and its per-pair exchange override, not the full registry.
    startup_focus = pairs[0] if len(pairs) == 1 else None
    _send_startup_message(tg, cfg, focus=startup_focus)

    if once:
        total = 0
        for pair in pairs:
            try:
                total += len(_process_symbol(cfg, pair, tg, dedupe, quiet))
                health.record_success(pair.symbol)
            except Exception as e:
                if health.should_log_error(pair.symbol, pd.Timestamp.now(tz="UTC")):
                    log.exception("Error processing %s: %s", pair.symbol, e)
        log.info("Live (once): sent %d alerts.", total)
        return

    tf_minutes = cfg.cfg.chart_timeframe_min
    tick_seconds = 5 if tf_minutes < 60 else 30
    log.info("Live loop starting: %d pairs, %s candles, %ds tick.",
             len(pairs), minutes_to_timeframe(tf_minutes), tick_seconds)
    while True:
        start = time_mod.time()
        try:
            for pair in pairs:
                try:
                    _process_symbol(cfg, pair, tg, dedupe, quiet)
                    health.record_success(pair.symbol)
                except Exception as e:
                    now = pd.Timestamp.now(tz="UTC")
                    if health.should_log_error(pair.symbol, now):
                        log.exception("Error processing %s: %s", pair.symbol, e)
        except Exception as e:
            log.exception("Outer loop error: %s", e)
        elapsed = time_mod.time() - start
        sleep_for = max(1.0, tick_seconds - elapsed)
        time_mod.sleep(sleep_for)


# --------------------------------------------------------------------------
# Check mode — run-once signal scan for cron scheduling (GitHub Actions)
# --------------------------------------------------------------------------

def _load_state(path: Path) -> dict:
    """Read the per-pair last-processed-bar map. Returns {} when absent/corrupt."""
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        last_bar = data.get("last_bar") if isinstance(data, dict) else None
        if isinstance(last_bar, dict):
            return last_bar
    except Exception as exc:
        log.warning("Ignoring unreadable state file %s: %s", path, exc)
    return {}


def _save_state(path: Path, last_bar: dict) -> None:
    """Write the state file atomically (tmp + rename) so a crash never corrupts it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1, "updated": datetime.now(timezone.utc).isoformat(), "last_bar": last_bar}
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def run_check(cfg: RuntimeConfig, args) -> int:
    """One-shot scan: recompute the full indicator over trailing candles and send
    only events on bars closed after the last run (per pair). Exit 1 on any pair
    failure so the GitHub Actions job goes red and the next scheduled run retries
    the un-advanced pairs — state only advances for pairs that fully succeeded,
    so a crash/network blip never silently drops a signal.

    Why no TP/SL position state needs persisting: `run_indicator` derives the
    whole state machine (position, plan, TP/SL levels) from the candle history on
    every call and is deterministic on overlapping windows, so a later run
    recomputes identical events for the same bars. The only thing we must
    remember is how far through history we've already reported.
    """
    if not cfg.telegram_token or not cfg.telegram_chat_id:
        raise SystemExit("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set for check mode.")
    state_path = Path(args.state)
    first_run = not state_path.exists()
    last_bar = _load_state(state_path)
    tg = TelegramClient(cfg.telegram_token, cfg.telegram_chat_id)
    quiet = QuietHours(cfg.quiet_hours, cfg.timezone)
    tf_minutes = cfg.cfg.chart_timeframe_min
    timeframe_str = minutes_to_timeframe(tf_minutes)
    dry_run = bool(args.dry_run)
    total_sent = 0
    failed_pairs: list = []

    if first_run and not dry_run:
        # One-time 'online' confirmation on the very first run (like live mode's
        # startup message) — never repeated, so cron doesn't spam Telegram.
        _send_startup_message(tg, cfg, focus=None)

    for pair in cfg.pairs:
        symbol = pair.symbol
        exchange_id = pair.exchange or cfg.exchange
        try:
            df = fetch_candles(exchange_id, symbol, timeframe_str, limit=1500)
            if df.empty:
                log.warning("Check %s: no candles returned — will retry next run.", symbol)
                failed_pairs.append(symbol)
                continue
            now_utc = pd.Timestamp.now(tz="UTC")
            cutoff = now_utc - pd.Timedelta(minutes=tf_minutes) - pd.Timedelta(seconds=30)
            df_closed = df[df.index <= cutoff]
            if df_closed.empty:
                log.warning("Check %s: no fully-closed bars yet — will retry next run.", symbol)
                failed_pairs.append(symbol)
                continue
            events = run_indicator(df_closed, cfg.cfg, symbol=symbol)
            last_closed = df_closed.index[-1]
            prev_raw = last_bar.get(symbol)

            if prev_raw is None or pd.Timestamp(prev_raw) < df_closed.index[0]:
                # Fresh baseline (first run, or state older than the fetched
                # window): only report the most recent closed bar so a fresh
                # start never replays old signals.
                new_events = [e for e in events if e.time == last_closed]
                log.info("Check %s: fresh baseline at %s (%d event(s) on that bar).",
                         symbol, last_closed, len(new_events))
            else:
                prev = pd.Timestamp(prev_raw)
                new_events = [e for e in events if e.time > prev]
                new_bars = int((last_closed - prev) / pd.Timedelta(minutes=tf_minutes))
                log.info("Check %s: %d new closed bar(s) since %s → %d event(s).",
                         symbol, new_bars, prev, len(new_events))

            if quiet.is_quiet():
                # Match live mode: quiet hours suppress delivery entirely (and
                # state still advances, so old bars are never replayed later).
                log.info("Check %s: quiet hours — skipping %d event(s).", symbol, len(new_events))
            elif new_events:
                ok = True
                for e in new_events:
                    msg = format_event(e)
                    if dry_run:
                        log.info("Check %s (dry-run) would send: %s", symbol, msg)
                        continue
                    log.info("alert: %s", msg, extra={"symbol": symbol, "exchange": exchange_id})
                    if not tg.send_message(msg):
                        ok = False
                        tg_log.error("Failed to deliver event: %s", msg)
                        break
                if not ok:
                    # Leave state untouched → the next run re-processes this pair.
                    failed_pairs.append(symbol)
                    continue
                if not dry_run:
                    total_sent += len(new_events)

            # Advance only on full success (dry-run and quiet are deterministic).
            last_bar[symbol] = last_closed.isoformat()
        except Exception:
            log.exception("Check %s: error processing pair — will retry next run.", symbol)
            failed_pairs.append(symbol)

    _save_state(state_path, last_bar)
    if failed_pairs:
        log.warning("Check complete: sent %d alert(s); %d pair(s) deferred for retry: %s.",
                    total_sent, len(failed_pairs), ", ".join(failed_pairs))
        return 1
    log.info("Check complete: sent %d alert(s). State advanced for all %d pair(s).",
             total_sent, len(cfg.pairs))
    return 0


# --------------------------------------------------------------------------
# Entry
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# Test-fire mode (synthetic signal → Telegram end-to-end check)
# --------------------------------------------------------------------------

# Indicator-emitted kinds, used as argparse choices for --kind so the caller's
# `--kind` value matches what the real pipeline emits (avoid badge fallbacks).
_VALID_KINDS = (
    "Long Entry", "Short Entry",
    "Long TP1", "Short TP1", "Long TP2", "Short TP2", "Long TP3", "Short TP3",
    "Long SL", "Short SL",
)

# Synthetic-fill map for run_test_fire: ties the ✓-marker logic in `format_event`
# (which reads `event.filled`) to realistic fill-state. SL test-events default to
# the "SL after TP1" scenario — the most-common real-world path — so the
# generated alert shows `✓ TP1` (realised gains) and the `🛑 LONG SL` badge.
_FILLED_BY_KIND = {
    "Long Entry":  frozenset(),
    "Short Entry": frozenset(),
    "Long TP1":    frozenset({"TP1"}),
    "Short TP1":   frozenset({"TP1"}),
    "Long TP2":    frozenset({"TP1", "TP2"}),
    "Short TP2":   frozenset({"TP1", "TP2"}),
    "Long TP3":    frozenset({"TP1", "TP2", "TP3"}),
    "Short TP3":   frozenset({"TP1", "TP2", "TP3"}),
    "Long SL":     frozenset({"TP1"}),
    "Short SL":    frozenset({"TP1"}),
}

def run_test_fire(cfg: RuntimeConfig, args) -> None:
    """Construct a synthetic Event and push it through the live Telegram pipeline.

    Useful right after wiring Telegram credentials: a single `test-fire` invocation
    proves the bot can talk to the chat, the formatter renders badges/prices/levels
    the way you expect, and any `format_event()` regressions surface immediately —
    all without waiting 5-30 minutes for a real 15m market cross on Binance.

    Side effect: emits ONE Telegram message to the configured chat. Re-running
    is fine — Telegram allows identical messages back-to-back.
    """
    if not cfg.telegram_token or not cfg.telegram_chat_id:
        raise SystemExit("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set in .env.")
    tg = TelegramClient(cfg.telegram_token, cfg.telegram_chat_id)
    sym = (args.symbol or "BTC/USDT").upper()
    kind = args.kind
    # Per-pair round-number price floors so a bare `test-fire` produces a
    # recognizable-but-fake baseline message; --price overrides per run.
    floor = {"BTC/USDT": 60000.0, "ETH/USDT": 3000.0, "SOL/USDT": 150.0, "PAXG/USDT": 4000.0}
    base_price = args.price if args.price is not None else floor.get(sym, 100.0)

    is_entry = kind.endswith(" Entry")
    is_long = kind.startswith("Long")
    side_sign = 1.0 if is_long else -1.0
    side = "Long" if is_long else "Short"

    # Build the synthetic Plan BEFORE the cross branch below — that branch reads
    # plan_tp1/tp2/tp3/sl to pick its pierced level, so construction order matters.
    # Mirrors what `run_indicator` would attach to a real event of this kind:
    # entry_price = base_price, TPs/SL derived from cfg percentages, all signed
    # consistently with `side_sign`. Plan is immutable (frozen=True).
    plan_tp1 = base_price * (1 + side_sign * cfg.cfg.tp_levels_pct[0] / 100.0)
    plan_tp2 = base_price * (1 + side_sign * cfg.cfg.tp_levels_pct[1] / 100.0)
    plan_tp3 = base_price * (1 + side_sign * cfg.cfg.tp_levels_pct[2] / 100.0)
    plan_sl  = base_price * (1 - side_sign * cfg.cfg.sl_level_pct / 100.0)  # Long SL below entry; Short SL above (sign flips).
    plan = Plan(side=side, entry_price=base_price,
                tp1=plan_tp1, tp2=plan_tp2, tp3=plan_tp3, sl=plan_sl)

    if is_entry:
        # Entries have price == level (the bar close).
        price, level = base_price, base_price
    else:
        # TP/SL crosses: pick the level we just pierced from the matching plan line
        # so test-fire's (price, level) pair is precisely self-consistent with `plan`.
        if "TP1" in kind:   pierce = plan_tp1
        elif "TP2" in kind: pierce = plan_tp2
        elif "TP3" in kind: pierce = plan_tp3
        elif "SL" in kind:  pierce = plan_sl
        else:               pierce = plan_tp1  # unreachable; kinds are validated by argparse choices
        level = args.level if args.level is not None else pierce
        # TP crosses put `price` past the level on the same side as the cross:
        # Long TP → bar high above level (sign +), Short TP → bar low below level (sign -).
        # SL crosses flip the side: Long SL → bar low below level (-sign), Short SL → bar
        # high above level (+sign). Mirrors `run_indicator`'s price-vs-level relationship
        # for all four cross kinds so test-fire output matches real alert rows.
        price = level * (1 + side_sign * 0.001 if "TP" in kind else -side_sign * 0.001)

    e = Event(
        time=pd.Timestamp.now(tz="UTC"),
        symbol=sym,
        kind=kind,
        price=float(price),
        level=float(level),
        plan=plan,
        filled=_FILLED_BY_KIND[kind],
    )
    msg = format_event(e)
    print("--- Synthesized event ---")
    print(msg)
    print("--- Sending to Telegram ---")
    ok = tg.send_message(msg)
    print(f"Telegram delivery: {'SUCCESS' if ok else 'FAILED (see logs)'}")
    if not ok:
        raise SystemExit(2)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
    p = argparse.ArgumentParser(description="SAIYAN OCC (XXX) — Pine→Python port → Telegram alerter")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("replay", help="Replay historical candles and dump events to CSV.")
    pr.add_argument("--symbol", required=True)
    pr.add_argument("--timeframe", required=True)
    pr.add_argument("--hours", type=int, default=24 * 30)
    pr.add_argument("--limit", type=int, default=2000)
    pr.add_argument("--output", default="replay.csv")
    pr.add_argument("--exchange", default=None,
                    help="Override the exchange for this one replay. Defaults to `config.yaml`'s "
                         "exchange. Useful for fetching XAU/USD on bitfinex while the live bot "
                         "normally polls binance crypto pairs.")
    pr.add_argument("--tv-urls", action="store_true",
                    help="Add a `tv_url` column with a TradingView deep link to the same pair+TF.")

    pl = sub.add_parser("live", help="Poll closed bars and forward events to Telegram.")
    pl.add_argument("--once", action="store_true",
                    help="Run a single polling cycle and exit (useful for testing).")
    pl.add_argument("--pair", default=None,
                    help="Restrict live polling to a single symbol (e.g. 'BTC/USDT'). "
                         "Accepts any symbol on the configured exchange, not just those "
                         "listed in config.yaml — useful for focus-mode testing of related "
                         "charts without editing config.")

    pc = sub.add_parser("check",
        help="Run-once signal scan for cron scheduling (GitHub Actions): report events "
             "on bars closed since the last run; persists per-pair progress in a state file.")
    pc.add_argument("--state", default="state.json",
                    help="Path to the state file holding per-pair last-processed bar. "
                         "Default: state.json")
    pc.add_argument("--dry-run", action="store_true",
                    help="Print would-be alerts instead of sending to Telegram; state still "
                         "advances (use for testing).")

    ps = sub.add_parser("summarize", help="Pretty-print kind histogram + stats for one replay CSV.")
    ps.add_argument("--input", required=True, help="Path to a replay CSV (e.g. replays/btc_replay.csv).")
    ps.add_argument("--json", action="store_true",
                    help="Emit machine-readable JSON instead of a friendly table.")

    ptf = sub.add_parser("test-fire",
        help="Send a synthetic event through the live Telegram pipeline. "
             "Useful for verifying the alert format end-to-end without "
             "waiting for a real 15m market cross.")
    ptf.add_argument("--symbol", default="BTC/USDT",
                     help="Pair label, e.g. BTC/USDT, PAXG/USDT. Default: BTC/USDT.")
    ptf.add_argument("--kind", default="Long Entry", choices=_VALID_KINDS,
                     help="Event kind — restricted to indicator-emitted values so the format "
                          "matches real alerts. Default: 'Long Entry'.")
    ptf.add_argument("--price", type=float, default=None,
                     help="Reference price. Defaults to a per-pair round-number floor. "
                          "Entries: price == level. TP/SL: level is offset 0.5%% (SL) or "
                          "cfg.tp_levels_pct[0]%% (TP) from the reference price; price "
                          "is set just past the level so the message shows two distinct lines.")
    ptf.add_argument("--level", type=float, default=None,
                     help="Override the engineered level. Default is auto-computed from "
                          "the per-pair reference price plus cfg's TP/SL percentage.")

    args = p.parse_args()
    cfg = load_config()
    if args.cmd == "replay":
        run_replay(cfg, args)
    elif args.cmd == "live":
        run_live(cfg, once=args.once, focus_pair=args.pair)
    elif args.cmd == "check":
        raise SystemExit(run_check(cfg, args))
    elif args.cmd == "summarize":
        run_summarize(args)
    elif args.cmd == "test-fire":
        run_test_fire(cfg, args)


if __name__ == "__main__":
    main()
