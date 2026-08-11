"""Automatic execution of SAIYAN signals with safety rails.

Modes (config ``execution.mode``):

* ``off``      — alerts only; nothing trades (default).
* ``paper``    — internal simulation. No API keys, no money. Fills mirror the
                 backtest reconstruction exactly (1/3 of the position per TP,
                 SL closes the remainder, a flip closes + reopens at bar close)
                 so paper PnL is directly comparable to ``backtest.py``.
* ``testnet``  — real orders on Gate's testnet (api-testnet.gateapi.io) via
                 ccxt sandbox mode. Requires ``GATE_API_KEY`` / ``GATE_API_SECRET``
                 in the environment. TP/SL are *monitored levels*: the bot
                 closes fractions when the live tick price crosses a level.
* ``live``     — same as testnet but against Gate mainnet — REAL MONEY.
                 Requires ``execution.live_confirmation: true`` in config.yaml.

Safety rails (all configurable):

* ``size_usdt``            — notional (USDT) per entry.
* ``max_positions``        — cap on concurrently open positions (flips allowed
                             at the cap; brand-new symbols are skipped).
* ``daily_loss_limit_usd`` — realized losses below -limit halt NEW entries
                             until UTC midnight (or ``/resume``).
* Telegram admin commands  — ``/status`` ``/stop`` ``/resume`` ``/flat``.

Paper fill model (matches backtest.py): each TP banks a fixed percentage of
the position (tp_pct[idx] / 3), SL banks -sl_pct * remaining, and a flip
closes the remainder at the bar close price. PnL is reported in USDT by
multiplying those fractions by the position notional.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

THIRD = 1.0 / 3.0


# ---------------------------------------------------------------------------
# Position bookkeeping (shared by every backend)
# ---------------------------------------------------------------------------

@dataclass
class Position:
    symbol: str
    side: str                  # "Long" | "Short"
    entry: float
    size: float                # notional in USDT
    tp1: float
    tp2: float
    tp3: float
    sl: float
    opened_at: str
    remaining: float = 1.0     # fraction still open
    tp_taken: int = 0
    partial_ret: float = 0.0   # return as fraction of notional (backtest model)
    realized: float = 0.0      # realized USDT (paper: size * partial_ret)
    closed: bool = False
    reason: Optional[str] = None
    closed_at: Optional[str] = None

    def bank_fraction(self, frac: float, price: float, tp_idx: Optional[int] = None,
                      tp_pcts=None, sl_pct: Optional[float] = None) -> float:
        """Bank PnL for a fraction of the position.

        Mirrors backtest.py: TP events bank tp_pcts[idx]/3 of notional (fixed
        percentage, independent of fill price); SL events bank -sl_pct *
        remaining. Returns the USDT PnL banked.
        """
        if tp_idx is not None and tp_pcts is not None:
            ret = tp_pcts[tp_idx] / 100.0 * THIRD
        elif sl_pct is not None:
            ret = -(sl_pct / 100.0) * self.remaining
        else:  # flip / flat: price-based on the remaining fraction
            sign = 1.0 if self.side == "Long" else -1.0
            ret = sign * (price - self.entry) / self.entry * self.remaining
        self.partial_ret += ret
        self.realized += ret * self.size
        self.remaining -= frac
        if self.remaining <= 1e-9:
            self.remaining = 0.0
        return ret * self.size

    def close(self, reason: str) -> None:
        self.closed = True
        self.reason = reason
        self.closed_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        d = {
            "symbol": self.symbol, "side": self.side, "entry": self.entry,
            "size": self.size, "tp1": self.tp1, "tp2": self.tp2, "tp3": self.tp3,
            "sl": self.sl, "opened_at": self.opened_at, "remaining": self.remaining,
            "tp_taken": self.tp_taken, "partial_ret": self.partial_ret,
            "realized": self.realized, "closed": self.closed,
            "reason": self.reason, "closed_at": self.closed_at,
        }
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Position":
        return cls(**{k: d[k] for k in (
            "symbol", "side", "entry", "size", "tp1", "tp2", "tp3", "sl",
            "opened_at", "remaining", "tp_taken", "partial_ret", "realized",
            "closed", "reason", "closed_at")})


# ---------------------------------------------------------------------------
# Base executor: shared rails + persistence + admin commands
# ---------------------------------------------------------------------------

class Executor:
    mode = "off"

    def __init__(self, ex_cfg, state_path: Path):
        self.cfg = ex_cfg
        self.state_path = Path(state_path)
        self.positions: list[Position] = []
        self.halted = False
        self.halt_reason: Optional[str] = None
        self.day = ""
        self.realized_today = 0.0
        self.last_prices: dict[str, float] = {}
        self._update_offset: Optional[int] = None
        self._pnl_snapshot = 0.0   # sum of position realized at last ledger update
        self.load()
        self._pnl_snapshot = sum(p.realized for p in self.positions)

    # -- persistence --------------------------------------------------------

    def load(self) -> None:
        if not self.state_path.exists():
            self.day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            return
        try:
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
            self.positions = [Position.from_dict(p) for p in raw.get("positions", [])]
            self.halted = bool(raw.get("halted", False))
            self.halt_reason = raw.get("halt_reason")
            self.day = raw.get("day", "")
            self.realized_today = float(raw.get("realized_today", 0.0))
        except Exception:
            log.exception("Could not load execution state %s — starting fresh.", self.state_path)
            self.day = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def save(self) -> None:
        payload = {
            "version": 1,
            "updated": datetime.now(timezone.utc).isoformat(),
            "mode": self.mode,
            "day": self.day,
            "realized_today": self.realized_today,
            "halted": self.halted,
            "halt_reason": self.halt_reason,
            "positions": [p.to_dict() for p in self.positions],
        }
        tmp = self.state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self.state_path)

    # -- rails --------------------------------------------------------------

    def _roll_day(self) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self.day:
            self.day = today
            self.realized_today = 0.0
            self._pnl_snapshot = sum(p.realized for p in self.positions)
            if self.halt_reason == "daily_loss":
                self.halted = False
                self.halt_reason = None
                log.info("New UTC day — daily loss halt lifted.")

    def _update_day_pnl(self) -> None:
        """Roll the realized-PnL ledger and enforce the daily loss cap.

        `realized_today` accumulates the delta of all position realized PnL
        since the last ledger update (snapshot), so it survives restarts and
        only counts PnL banked today.
        """
        self._roll_day()
        total = sum(p.realized for p in self.positions)
        self.realized_today += total - self._pnl_snapshot
        self._pnl_snapshot = total
        if self.realized_today <= -self.cfg.daily_loss_limit_usd and not self.halted:
            self.halted = True
            self.halt_reason = "daily_loss"
            log.warning("Daily loss cap hit: today $%.2f <= -$%.2f — halting new entries.",
                        self.realized_today, self.cfg.daily_loss_limit_usd)

    def _can_open(self, symbol: str) -> tuple[bool, str]:
        if self.halted:
            return False, f"halted ({self.halt_reason or 'kill switch'})"
        if any(p.symbol == symbol and not p.closed for p in self.positions):
            return True, ""  # flip/replacement allowed — handled by caller
        if sum(1 for p in self.positions if not p.closed) >= self.cfg.max_positions:
            return False, f"max positions ({self.cfg.max_positions}) reached"
        return True, ""

    # -- admin commands -----------------------------------------------------

    def check_commands(self, tg) -> None:
        """Poll Telegram for admin commands from the configured chat."""
        if tg is None:
            return
        try:
            updates = tg.fetch_updates(offset=self._update_offset)
        except Exception as exc:
            log.debug("Command poll failed: %s", exc)
            return
        for u in updates:
            self._update_offset = u.get("update_id", 0) + 1
            msg = u.get("message") or {}
            text = (msg.get("text") or "").strip()
            if not text.startswith("/"):
                continue
            self._run_command(text.lower(), tg)

    def _run_command(self, text: str, tg) -> None:
        self._update_day_pnl()
        if text == "/status":
            tg.send_message(self.status())
        elif text == "/stop":
            self.halted = True
            self.halt_reason = "kill switch"
            self.save()
            tg.send_message("🛑 SAIYAN execution HALTED — no new entries. Open positions keep their exits.")
        elif text == "/resume":
            self.halted = False
            self.halt_reason = None
            self.save()
            tg.send_message("▶️ SAIYAN execution resumed.")
        elif text == "/flat":
            n = self.close_all()
            self.save()
            tg.send_message(f"🏁 /flat: closed {n} position(s)." if n else "🏁 /flat: no open positions.")
        else:
            tg.send_message("Commands: /status /stop /resume /flat")

    # -- status -------------------------------------------------------------

    def status(self) -> str:
        self._roll_day()
        open_pos = [p for p in self.positions if not p.closed]
        lines = [
            f"🤖 Execution: {self.mode.upper()}"
            f" {'⛔ HALTED' if self.halted else '🟢'}",
            f"Positions: {len(open_pos)}/{self.cfg.max_positions} · "
            f"Size: ${self.cfg.size_usdt:g}/trade · "
            f"Day PnL: ${self.realized_today:+.2f} "
            f"(cap ${-self.cfg.daily_loss_limit_usd:g})",
        ]
        for p in open_pos:
            lines.append(
                f"  {p.symbol} {p.side} @ {p.entry:g} "
                f"rem {p.remaining:.0%} TP{p.tp_taken}/3 "
                f"PnL ${p.realized:+.2f}")
        if self.halt_reason:
            lines.append(f"Halt reason: {self.halt_reason}")
        return "\n".join(lines)

    # -- backend hooks ------------------------------------------------------

    def process_events(self, events, tg) -> None:
        """React to new events on a closed bar. Base impl = paper model."""
        raise NotImplementedError

    def tick(self, prices: dict[str, float], tg) -> None:
        """Per-loop maintenance: update prices, poll backend, check commands."""
        self.last_prices.update(prices)
        self.check_commands(tg)
        self.save()

    def close_all(self) -> int:
        """Close every open position (paper: at last known price)."""
        n = 0
        for p in self.positions:
            if p.closed:
                continue
            price = self.last_prices.get(p.symbol, p.entry)
            p.bank_fraction(p.remaining, price)
            p.close("flat")
            n += 1
        self._update_day_pnl()
        self.save()
        return n


# ---------------------------------------------------------------------------
# Paper executor — internal simulation, matches backtest.py
# ---------------------------------------------------------------------------

class PaperExecutor(Executor):
    mode = "paper"

    def __init__(self, ex_cfg, state_path: Path):
        super().__init__(ex_cfg, state_path)
        self.tp_pcts = tuple(float(x) for x in ex_cfg.tp_levels_pct)
        self.sl_pct = float(ex_cfg.sl_level_pct)

    def process_events(self, events, tg) -> None:
        self._roll_day()
        for e in events:
            kind = e.kind
            symbol = e.symbol
            price = float(e.price)

            if kind.endswith(" Entry"):
                side = "Long" if kind.startswith("Long") else "Short"
                open_p = next((p for p in self.positions
                               if p.symbol == symbol and not p.closed), None)
                if open_p is not None:
                    # Flip: close remainder at bar close, then reopen.
                    pnl = open_p.bank_fraction(open_p.remaining, price)
                    open_p.close("flip")
                    self._notify_close(tg, open_p, price, "flip", pnl)
                ok, why = self._can_open(symbol)
                if not ok:
                    self._notify_skip(tg, symbol, side, price, why)
                    continue
                plan = e.plan
                pos = Position(
                    symbol=symbol, side=side, entry=price, size=self.cfg.size_usdt,
                    tp1=float(plan.tp1) if plan else None,
                    tp2=float(plan.tp2) if plan else None,
                    tp3=float(plan.tp3) if plan else None,
                    sl=float(plan.sl) if plan else None,
                    opened_at=datetime.now(timezone.utc).isoformat(),
                )
                self.positions.append(pos)
                self._notify_open(tg, pos, "paper")
                continue

            open_p = next((p for p in self.positions
                           if p.symbol == symbol and not p.closed), None)
            if open_p is None:
                continue  # stray TP/SL without a position (window edge)
            if not kind.startswith(open_p.side):
                continue  # cross-side event — defensive

            if "TP" in kind and open_p.remaining > 1e-9:
                idx = {"TP1": 0, "TP2": 1, "TP3": 2}[kind.split(" ")[1]]
                pnl = open_p.bank_fraction(THIRD, price, tp_idx=idx, tp_pcts=self.tp_pcts)
                open_p.tp_taken += 1
                if open_p.remaining <= 1e-9:
                    open_p.close("TP3")
                    self._notify_close(tg, open_p, price, "TP3", pnl)
                else:
                    self._notify_tp(tg, open_p, idx + 1, pnl)
                continue

            if "SL" in kind and open_p.remaining > 1e-9:
                pnl = open_p.bank_fraction(open_p.remaining, price, sl_pct=self.sl_pct)
                open_p.close("SL")
                self._notify_close(tg, open_p, price, "SL", pnl)
        self._update_day_pnl()
        self.save()

    # -- notifications ------------------------------------------------------

    def _notify_open(self, tg, pos: Position, where: str) -> None:
        msg = (
            f"✅ [{where}] OPEN {pos.side} {pos.symbol} @ {pos.entry:g} "
            f"(${pos.size:g})\n"
            f"   TP1 {pos.tp1:g} · TP2 {pos.tp2:g} · TP3 {pos.tp3:g} · SL {pos.sl:g}"
        )
        log.info("exec: %s", msg)
        if tg:
            tg.send_message(msg)

    def _notify_tp(self, tg, pos: Position, idx: int, pnl: float) -> None:
        msg = f"🎯 [{self.mode}] {pos.symbol} TP{idx} hit — banked ${pnl:+.2f} (rem {pos.remaining:.0%})"
        log.info("exec: %s", msg)
        if tg:
            tg.send_message(msg)

    def _notify_close(self, tg, pos: Position, price: float, reason: str, pnl: float) -> None:
        msg = f"🏁 [{self.mode}] CLOSED {pos.symbol} ({reason}) @ {price:g} — ${pos.realized:+.2f}"
        log.info("exec: %s", msg)
        if tg:
            tg.send_message(msg)

    def _notify_skip(self, tg, symbol: str, side: str, price: float, why: str) -> None:
        msg = f"⏭️ [{self.mode}] SKIPPED {side} {symbol} @ {price:g} — {why}"
        log.info("exec: %s", msg)
        if tg:
            tg.send_message(msg)


# ---------------------------------------------------------------------------
# Gate executor — real orders via ccxt (testnet or live)
# ---------------------------------------------------------------------------

class GateExecutor(Executor):
    """Real orders on Gate (sandbox testnet or mainnet).

    TP/SL are monitored levels: on each tick the live price is checked against
    the plan and fractions are closed with market orders. Resting bracket
    orders are NOT used (Gate spot stop-order support is uneven); the 5s poll
    is the protection latency. Orders are reduce-only by construction.
    """

    mode = "gate"

    def __init__(self, ex_cfg, state_path: Path, sandbox: bool = True):
        super().__init__(ex_cfg, state_path)
        import ccxt
        self.tp_pcts = tuple(float(x) for x in ex_cfg.tp_levels_pct)
        self.sl_pct = float(ex_cfg.sl_level_pct)
        self.exchange = ccxt.gate({
            "apiKey": os.getenv("GATE_API_KEY", ""),
            "secret": os.getenv("GATE_API_SECRET", ""),
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        })
        self.exchange.set_sandbox_mode(sandbox)
        self.venue = "testnet" if sandbox else "live"
        self.mode = f"gate-{self.venue}"
        if not self.exchange.apiKey:
            raise SystemExit(
                f"GATE_API_KEY/GATE_API_SECRET must be set for {self.venue} mode.")

    def process_events(self, events, tg) -> None:
        self._roll_day()
        for e in events:
            kind = e.kind
            symbol = e.symbol
            price = float(e.price)

            if kind.endswith(" Entry"):
                side = "Long" if kind.startswith("Long") else "Short"
                open_p = next((p for p in self.positions
                               if p.symbol == symbol and not p.closed), None)
                if open_p is not None:
                    self._market_close(open_p, tg, "flip")
                ok, why = self._can_open(symbol)
                if not ok:
                    self._notify_skip(tg, symbol, side, price, why)
                    continue
                plan = e.plan
                pos = Position(
                    symbol=symbol, side=side, entry=price, size=self.cfg.size_usdt,
                    tp1=float(plan.tp1) if plan else None,
                    tp2=float(plan.tp2) if plan else None,
                    tp3=float(plan.tp3) if plan else None,
                    sl=float(plan.sl) if plan else None,
                    opened_at=datetime.now(timezone.utc).isoformat(),
                )
                self.positions.append(pos)
                self._place_entry(pos, tg)
                continue

            open_p = next((p for p in self.positions
                           if p.symbol == symbol and not p.closed), None)
            if open_p is None or not kind.startswith(open_p.side):
                continue
            # The exchange fills intrabar; our monitored levels also fire on
            # tick price. If the bar event says a level was hit but our monitor
            # missed it, reconcile now.
            if "TP" in kind and open_p.remaining > 1e-9:
                idx = {"TP1": 0, "TP2": 1, "TP3": 2}[kind.split(" ")[1]]
                self._close_fraction(open_p, idx + 1, tg)
            elif "SL" in kind and open_p.remaining > 1e-9:
                self._market_close(open_p, tg, "SL")
        self._update_day_pnl()
        self.save()

    def tick(self, prices: dict[str, float], tg) -> None:
        """Monitor TP/SL levels against live prices."""
        self.last_prices.update(prices)
        for p in self.positions:
            if p.closed:
                continue
            px = prices.get(p.symbol)
            if px is None:
                continue
            if p.side == "Long":
                if p.remaining > 1e-9 and px >= p.tp1 and p.tp_taken == 0:
                    self._close_fraction(p, 1, tg)
                elif p.remaining > 1e-9 and px >= p.tp2 and p.tp_taken == 1:
                    self._close_fraction(p, 2, tg)
                elif p.remaining > 1e-9 and px >= p.tp3 and p.tp_taken == 2:
                    self._close_fraction(p, 3, tg)
                elif p.remaining > 1e-9 and px <= p.sl:
                    self._market_close(p, tg, "SL")
            else:
                if p.remaining > 1e-9 and px <= p.tp1 and p.tp_taken == 0:
                    self._close_fraction(p, 1, tg)
                elif p.remaining > 1e-9 and px <= p.tp2 and p.tp_taken == 1:
                    self._close_fraction(p, 2, tg)
                elif p.remaining > 1e-9 and px <= p.tp3 and p.tp_taken == 2:
                    self._close_fraction(p, 3, tg)
                elif p.remaining > 1e-9 and px >= p.sl:
                    self._market_close(p, tg, "SL")
        self._update_day_pnl()
        self.check_commands(tg)
        self.save()

    # -- order plumbing -----------------------------------------------------

    def _units(self, pos: Position) -> float:
        return pos.size / pos.entry if pos.entry else 0.0

    def _place_entry(self, pos: Position, tg) -> None:
        try:
            units = self._units(pos)
            order = self.exchange.create_order(
                pos.symbol, "market", "buy" if pos.side == "Long" else "sell", units)
            log.info("gate %s: entry %s %s %s units=%s -> %s",
                     self.venue, pos.side, pos.symbol, pos.entry, units, order.get("id"))
        except Exception as exc:
            pos.close("entry_failed")
            self._notify_error(tg, pos.symbol, f"entry order failed: {exc}")
            return
        self._notify_open(tg, pos, self.venue)

    def _close_fraction(self, pos: Position, tp_idx: int, tg) -> None:
        frac_units = self._units(pos) * THIRD
        try:
            self.exchange.create_order(
                pos.symbol, "market",
                "sell" if pos.side == "Long" else "buy", frac_units)
        except Exception as exc:
            self._notify_error(tg, pos.symbol, f"TP{tp_idx} close failed: {exc}")
            return
        pnl = pos.bank_fraction(THIRD, self.last_prices.get(pos.symbol, pos.entry),
                                tp_idx=tp_idx - 1, tp_pcts=self.tp_pcts)
        pos.tp_taken += 1
        if pos.remaining <= 1e-9:
            pos.close("TP3")
            self._notify_close(tg, pos, self.last_prices.get(pos.symbol, pos.entry), "TP3", pnl)
        else:
            self._notify_tp(tg, pos, tp_idx, pnl)
        self.save()

    def _market_close(self, pos: Position, tg, reason: str) -> None:
        rem_units = self._units(pos) * pos.remaining
        try:
            if rem_units > 1e-12:
                self.exchange.create_order(
                    pos.symbol, "market",
                    "sell" if pos.side == "Long" else "buy", rem_units)
        except Exception as exc:
            self._notify_error(tg, pos.symbol, f"close ({reason}) failed: {exc}")
            return
        pnl = pos.bank_fraction(pos.remaining, self.last_prices.get(pos.symbol, pos.entry))
        pos.close(reason)
        self._notify_close(tg, pos, self.last_prices.get(pos.symbol, pos.entry), reason, pnl)
        self.save()

    # -- notifications ------------------------------------------------------

    def _notify_open(self, tg, pos: Position, where: str) -> None:
        msg = (
            f"✅ [{where}] OPEN {pos.side} {pos.symbol} @ {pos.entry:g} "
            f"(${pos.size:g}) — bracket TP1 {pos.tp1:g} · TP2 {pos.tp2:g} · "
            f"TP3 {pos.tp3:g} · SL {pos.sl:g}"
        )
        log.info("exec: %s", msg)
        if tg:
            tg.send_message(msg)

    def _notify_tp(self, tg, pos: Position, idx: int, pnl: float) -> None:
        msg = f"🎯 [{self.venue}] {pos.symbol} TP{idx} filled — ${pnl:+.2f} (rem {pos.remaining:.0%})"
        log.info("exec: %s", msg)
        if tg:
            tg.send_message(msg)

    def _notify_close(self, tg, pos: Position, price: float, reason: str, pnl: float) -> None:
        msg = f"🏁 [{self.venue}] CLOSED {pos.symbol} ({reason}) @ {price:g} — ${pos.realized:+.2f}"
        log.info("exec: %s", msg)
        if tg:
            tg.send_message(msg)

    def _notify_skip(self, tg, symbol: str, side: str, price: float, why: str) -> None:
        msg = f"⏭️ [{self.venue}] SKIPPED {side} {symbol} @ {price:g} — {why}"
        log.info("exec: %s", msg)
        if tg:
            tg.send_message(msg)

    def _notify_error(self, tg, symbol: str, detail: str) -> None:
        msg = f"⚠️ [{self.venue}] {symbol}: {detail}"
        log.warning("exec: %s", msg)
        if tg:
            tg.send_message(msg)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_executor(ex_cfg, state_path: Path) -> Optional[Executor]:
    mode = (ex_cfg.mode or "off").lower()
    if mode == "off":
        return None
    if mode == "paper":
        return PaperExecutor(ex_cfg, state_path)
    if mode == "testnet":
        return GateExecutor(ex_cfg, state_path, sandbox=True)
    if mode == "live":
        if not getattr(ex_cfg, "live_confirmation", False):
            raise SystemExit(
                "execution.mode: live requires execution.live_confirmation: true — "
                "real money. Re-read deploy/gh-actions/README.md before flipping this.")
        return GateExecutor(ex_cfg, state_path, sandbox=False)
    raise SystemExit(f"execution.mode must be one of off|paper|testnet|live, got {mode!r}")
