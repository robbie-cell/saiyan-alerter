"""Load config.yaml + .env into a typed RuntimeConfig."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import yaml
from dotenv import load_dotenv

from indicator import IndicatorConfig

ROOT = Path(__file__).parent
DEFAULT_CFG_PATH = ROOT / "config.yaml"

load_dotenv(ROOT / ".env")  # safe if missing


@dataclass
class PairConfig:
    """One monitored symbol. `exchange=None` falls back to RuntimeConfig.exchange.

    Accept either a YAML string `"BASE/QUOTE"` or a dict with the same content
    plus optional overrides like `exchange: bitfinex`. Per-pair timeframe or
    indicator overrides are reserved for future work; not yet wired up.
    """
    symbol: str
    exchange: Optional[str] = None


@dataclass
class ExecutionConfig:
    """Automatic execution settings (config.yaml `execution:` block).

    mode: off | paper | testnet | live. `off` = alerts only. `paper` = internal
    simulation (no keys, no money). `testnet` = real orders on Gate's testnet
    (needs GATE_API_KEY/GATE_API_SECRET). `live` = REAL MONEY and requires
    `live_confirmation: true`.
    """
    mode: str = "off"
    size_usdt: float = 25.0
    max_positions: int = 3
    daily_loss_limit_usd: float = 50.0
    quiet_pause: bool = True   # don't trade during quiet_hours
    live_confirmation: bool = False
    tp_levels_pct: Tuple[float, float, float] = (1.0, 1.5, 2.0)  # set from indicator block
    sl_level_pct: float = 0.5


@dataclass
class RuntimeConfig:
    pairs: list  # list[PairConfig]
    exchange: str
    cfg: IndicatorConfig
    telegram_token: Optional[str]
    telegram_chat_id: Optional[str]
    dedupe_minutes: int = 5
    quiet_hours: Optional[Tuple[int, int]] = None  # (start_hour, end_hour) local
    timezone: str = "UTC"
    execution: Optional[ExecutionConfig] = None


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing config file at {path}. Did you copy .env.example to .env and create config.yaml?")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _parse_pair(p) -> PairConfig:
    """Accept either a plain-string pair or a dict with optional overrides.

    Forward-compatible: dict-level fields beyond `symbol`/`exchange` are passed
    through transparently; future per-pair TF and indicator overrides can ride
    here without a parsing change.
    """
    if isinstance(p, str):
        if not p or "/" not in p:
            raise ValueError(f"Pair string {p!r} must be in 'BASE/QUOTE' format like 'BTC/USDT'.")
        return PairConfig(symbol=p)
    if isinstance(p, dict):
        sym = p.get("symbol")
        if not sym or not isinstance(sym, str) or "/" not in sym:
            raise ValueError(f"Pair dict {p!r} must include 'symbol' in 'BASE/QUOTE' format.")
        ex = p.get("exchange")
        if ex is not None and not isinstance(ex, str):
            raise ValueError(f"Pair dict {p!r} 'exchange' must be a string if provided.")
        return PairConfig(symbol=sym, exchange=(ex or None))
    raise ValueError(f"Pair entry must be a string or dict, got {type(p).__name__}: {p!r}")


def load_config(path: Path = DEFAULT_CFG_PATH) -> RuntimeConfig:
    raw = _load_yaml(path)

    raw_pairs = raw.get("pairs") or []
    if not isinstance(raw_pairs, list):
        raise ValueError("`pairs` in config.yaml must be a list of strings or pair dicts.")

    pairs = [_parse_pair(p) for p in raw_pairs]
    if not pairs:
        raise ValueError("`pairs` is empty — list at least one symbol like 'BTC/USDT'.")

    exchange = str(raw.get("exchange", "binance"))

    ind = raw.get("indicator") or {}
    tp_levels = ind.get("tp_levels_pct", [1.0, 1.5, 2.0])
    if not (isinstance(tp_levels, list) and len(tp_levels) == 3):
        raise ValueError("`tp_levels_pct` must be a list of exactly three numbers, e.g. [1.0, 1.5, 2.0].")

    cfg = IndicatorConfig(
        ma_type=str(ind.get("ma_type", "ALMA")),
        basis_len=int(ind.get("basis_len", 2)),
        offset_sigma=float(ind.get("offset_sigma", 5)),
        offset_alma=float(ind.get("offset_alma", 0.85)),
        use_res=bool(ind.get("use_res", True)),
        intres_multiplier=int(ind.get("intres_multiplier", 8)),
        chart_timeframe_min=int(ind.get("chart_timeframe_min", 15)),
        delay_offset=int(ind.get("delay_offset", 0)),
        heikin_ashi=bool(ind.get("heikin_ashi", False)),
        tp_levels_pct=tuple(float(x) for x in tp_levels),
        sl_level_pct=float(ind.get("sl_level_pct", 0.5)),
        trade_type=str(ind.get("trade_type", "BOTH")),
    )

    quiet_hours = raw.get("quiet_hours")
    if quiet_hours is not None:
        if not (isinstance(quiet_hours, list) and len(quiet_hours) == 2):
            raise ValueError("`quiet_hours` must be null or [start_hour, end_hour] (24h local time).")
        quiet_hours = (int(quiet_hours[0]), int(quiet_hours[1]))

    ex = raw.get("execution") or {}
    mode = str(ex.get("mode", "off")).lower()
    if mode not in ("off", "paper", "testnet", "live"):
        raise ValueError("`execution.mode` must be one of off|paper|testnet|live.")
    execution = ExecutionConfig(
        mode=mode,
        size_usdt=float(ex.get("size_usdt", 25.0)),
        max_positions=int(ex.get("max_positions", 3)),
        daily_loss_limit_usd=float(ex.get("daily_loss_limit_usd", 50.0)),
        quiet_pause=bool(ex.get("quiet_pause", True)),
        live_confirmation=bool(ex.get("live_confirmation", False)),
    )
    execution.tp_levels_pct = cfg.tp_levels_pct
    execution.sl_level_pct = cfg.sl_level_pct

    return RuntimeConfig(
        pairs=pairs,
        exchange=exchange,
        cfg=cfg,
        telegram_token=os.getenv("TELEGRAM_BOT_TOKEN"),
        telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID"),
        dedupe_minutes=int(raw.get("dedupe_minutes", 5)),
        quiet_hours=quiet_hours,
        timezone=str(raw.get("timezone", "UTC")),
        execution=execution,
    )
