"""Automatic execution of SAIYAN signals with safety rails.

Modes (config ``execution.mode``):

* ``off``      — alerts only; nothing trades (default).
* ``paper``    — internal simulation. No API keys, no money. Fills mirror the
                 research reconstruction EXACTLY: each TP banks
                 ``tp_fractions[i]`` of the position at the PLAN level from the
                 entry event (ATR-scaled when exits=atr), SL closes the
                 remainder at the plan stop, a flip closes + reopens at bar
                 close. Paper PnL is directly comparable to ``research.py``.
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
* Telegram admin commands  — ``/status`` ``/stop`` ``/resume`` ``/flat``
                             ``/recap [N]`` ``/balance`` ``/help``.

Paper fill model (matches research.py): TP events bank the plan level's
percentage of ``tp_fractions[idx]`` of the position; SL banks the plan stop's
loss on the remainder; a flip closes the remainder at the bar close price.
PnL is reported in USDT by multiplying those fractions by the position
notional.

Gate executor fills are RECONCILED against the exchange: after every market
order the actual fill price/fee is fetched and booked, so PnL reflects real
execution, not assumptions. Partial fills are handled by sizing closes to the
exchange-reported remaining amount.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


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
    partial_ret: float = 0.0   # return as fraction of notional (research model)
    realized: float = 0.0      # realized USDT (size * partial_ret)
    closed: bool = False
    reason: Optional[str] = None
    closed_at: Optional[str] = None
    venue: str = "paper"       # paper | testnet | live
    fill_ts: str = ""          # last order fill timestamp (for reconciliation)

    def bank_tp(self, frac: float, tp_idx: int, level: float,
                advance: bool = True) -> float:
        """Bank PnL for a TP at its PLAN level (ATR-scaled when exits=atr).

        Mirrors research.reconstruct: gain = |level - entry| / entry * frac.
        `advance=False` (partial fill) keeps tp_taken unchanged so the level
        stays open and the remainder is retried. Returns the USDT PnL banked.
        """
        if not level or self.entry <= 0:
            return 0.0
        ret = abs(level - self.entry) / self.entry * frac
        self.partial_ret += ret
        self.realized += ret * self.size
        self.remaining -= frac
        if advance:
            self.tp_taken += 1
        if self.remaining <= 1e-9:
            self.remaining = 0.0
        return ret * self.size

    def bank_sl(self, price: float, level: Optional[float] = None,
                frac: Optional[float] = None) -> float:
        """Bank the SL loss on a fraction of the position (default: all of
        the remainder).

        Charges the ACTUAL crossed level when provided (research.py charges
        e.level — a breakeven-trailed stop moves the stop to entry without
        mutating plan.sl, and the loss must reflect where it really filled).
        Falls back to the plan stop, then to price-based.
        """
        frac = self.remaining if frac is None else frac
        lvl = level if level else self.sl
        if lvl and self.entry > 0:
            ret = -(abs(lvl - self.entry) / self.entry) * frac
        else:
            sign = 1.0 if self.side == "Long" else -1.0
            ret = sign * (price - self.entry) / self.entry * frac
        self.partial_ret += ret
        self.realized += ret * self.size
        self.remaining -= frac
        if self.remaining <= 1e-9:
            self.remaining = 0.0
        return ret * self.size

    def bank_price(self, frac: float, price: float) -> float:
        """Bank price-based PnL for a fraction (flip / flat close)."""
        sign = 1.0 if self.side == "Long" else -1.0
        ret = sign * (price - self.entry) / self.entry * frac
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
        return {
            "symbol": self.symbol, "side": self.side, "entry": self.entry,
            "size": self.size, "tp1": self.tp1, "tp2": self.tp2, "tp3": self.tp3,
            "sl": self.sl, "opened_at": self.opened_at, "remaining": self.remaining,
            "tp_taken": self.tp_taken, "partial_ret": self.partial_ret,
            "realized": self.realized, "closed": self.closed,
            "reason": self.reason, "closed_at": self.closed_at,
            "venue": self.venue, "fill_ts": self.fill_ts,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Position":
        keys = ("symbol", "side", "entry", "size", "tp1", "tp2", "tp3", "sl",
                "opened_at", "remaining", "tp_taken", "partial_ret", "realized",
                "closed", "reason", "closed_at", "venue", "fill_ts")
        return cls(**{k: d.get(k) for k in keys})


# ---------------------------------------------------------------------------
# Base executor: shared rails + persistence + admin commands
# ---------------------------------------------------------------------------

class Executor:
    mode = "off"

    def __init__(self, ex_cfg, ind_cfg, state_path: Path):
        self.cfg = ex_cfg
        self.ind_cfg = ind_cfg
        self.state_path = Path(state_path)
        self.positions: list[Position] = []
        self.halted = False
        self.halt_reason: Optional[str] = None
        self.day = ""
        self.realized_today = 0.0
        self.starting_balance = float(getattr(ex_cfg, "starting_balance_usdt", 0.0) or 0.0)
        self.last_prices: dict[str, float] = {}
        self._update_offset: Optional[int] = None
        self._pnl_snapshot = 0.0
        self.tp_fractions = tuple(getattr(ind_cfg, "tp_fractions", (1 / 3, 1 / 3, 1 / 3)))
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
            uid = raw.get("last_update_id")
            if isinstance(uid, int):
                self._update_offset = uid
        except Exception:
            log.exception("Could not load execution state %s — starting fresh.", self.state_path)
            self.day = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def save(self) -> None:
        payload = {
            "version": 2,
            "updated": datetime.now(timezone.utc).isoformat(),
            "mode": self.mode,
            "day": self.day,
            "realized_today": self.realized_today,
            "halted": self.halted,
            "halt_reason": self.halt_reason,
            "last_update_id": self._update_offset,
            "positions": [p.to_dict() for p in self.positions],
        }
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
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
        """Roll the realized-PnL ledger and enforce the daily loss cap."""
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
            if str((msg.get("chat") or {}).get("id")) != str(tg.chat_id):
                continue
            self._run_command(text, tg)

    def _run_command(self, text: str, tg) -> None:
        self._update_day_pnl()
        cmd, _, arg = text.partition(" ")
        cmd = cmd.lower().strip()
        if cmd == "/status":
            tg.send_message(self.status())
        elif cmd == "/stop":
            self.halted = True
            self.halt_reason = "kill switch"
            self.save()
            tg.send_message("🛑 SAIYAN execution HALTED — no new entries. Open positions keep their exits.")
        elif cmd == "/resume":
            self.halted = False
            self.halt_reason = None
            self.save()
            tg.send_message("▶️ SAIYAN execution resumed.")
        elif cmd == "/flat":
            n = self.close_all()
            tg.send_message(f"🏁 /flat: closed {n} position(s)." if n else "🏁 /flat: no open positions.")
        elif cmd == "/recap":
            n = 10
            if arg.strip():
                try:
                    n = max(1, min(int(arg.strip()), 50))
                except ValueError:
                    pass
            tg.send_message(self.recap(n))
        elif cmd == "/balance":
            tg.send_message(self.balance())
        else:
            tg.send_message(self.help_text())

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

    def recap(self, n: int = 10) -> str:
        closed = sorted((p for p in self.positions if p.closed),
                        key=lambda p: p.closed_at or "", reverse=True)[:n]
        lines = [f"📜 Recent trades (last {len(closed)} closed):"]
        if not closed:
            lines.append("   No closed trades yet — /status for open positions.")
        for p in closed:
            at = (p.closed_at or "")[11:16]  # HH:MM from ISO timestamp (UTC)
            lines.append(f"   {at}Z {p.symbol} {p.side:<5} {p.reason:<8} ${p.realized:+.2f}")
        if closed:
            wins = sum(1 for p in closed if p.realized > 0)
            net = sum(p.realized for p in closed)
            lines.append(f"   Win rate {wins / len(closed):.0%} · Net ${net:+.2f}")
            open_n = sum(1 for p in self.positions if not p.closed)
            if open_n:
                lines.append(f"   ({open_n} open — /status for details)")
        return "\n".join(lines)

    def balance(self) -> str:
        self._update_day_pnl()
        realized = sum(p.realized for p in self.positions)
        open_pos = [p for p in self.positions if not p.closed]
        unreal = 0.0
        notional = 0.0
        for p in open_pos:
            px = self.last_prices.get(p.symbol, p.entry)
            sign = 1.0 if p.side == "Long" else -1.0
            unreal += sign * (px - p.entry) / p.entry * p.remaining * p.size
            notional += p.size * p.remaining
        lines = [f"💰 Balance [{self.mode.upper()}]"]
        if self.starting_balance > 0:
            equity = self.starting_balance + realized + unreal
            lines.append(f"   Equity: ${equity:,.2f}  (start ${self.starting_balance:,.0f})")
        lines.append(f"   Realized (all time): ${realized:+,.2f}")
        lines.append(f"   Today: ${self.realized_today:+,.2f} (cap ${-self.cfg.daily_loss_limit_usd:g})")
        lines.append(f"   Open: {len(open_pos)} · unrealized ${unreal:+,.2f} · at risk ${notional:,.2f}")
        if self.halted:
            lines.append(f"   ⛔ HALTED ({self.halt_reason})")
        return "\n".join(lines)

    def help_text(self) -> str:
        return (
            "🤖 SAIYAN bot commands:\n"
            "/status — mode, open positions, day PnL\n"
            "/recap [N] — last N closed trades + win rate (default 10)\n"
            "/balance — equity, realized, open risk\n"
            "/stop — halt new entries (kill switch)\n"
            "/resume — re-enable entries\n"
            "/flat — close all open positions now"
        )

    # -- backend hooks ------------------------------------------------------

    def process_events(self, events, tg) -> None:
        raise NotImplementedError

    def tick(self, prices: dict[str, float], tg) -> None:
        self.last_prices.update(prices)
        self.check_commands(tg)
        self.save()

    def close_all(self) -> int:
        n = 0
        for p in self.positions:
            if p.closed:
                continue
            price = self.last_prices.get(p.symbol, p.entry)
            p.bank_price(p.remaining, price)
            p.close("flat")
            n += 1
        self._update_day_pnl()
        self.save()
        return n


# ---------------------------------------------------------------------------
# Paper executor — internal simulation, matches research.py to the cent
# ---------------------------------------------------------------------------

class PaperExecutor(Executor):
    mode = "paper"

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
                    pnl = open_p.bank_price(open_p.remaining, price)
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
                    venue="paper",
                )
                self.positions.append(pos)
                self._notify_open(tg, pos, "paper")
                continue

            open_p = next((p for p in self.positions
                           if p.symbol == symbol and not p.closed), None)
            if open_p is None:
                continue  # stray TP/SL without a position (window edge)
            if not kind.startswith(open_p.side):
                continue

            if "TP" in kind and open_p.remaining > 1e-9:
                idx = {"TP1": 0, "TP2": 1, "TP3": 2}[kind.split(" ")[1]]
                frac = self.tp_fractions[idx] if idx < len(self.tp_fractions) else self.tp_fractions[-1]
                lvl = (open_p.tp1, open_p.tp2, open_p.tp3)[idx]
                pnl = open_p.bank_tp(frac, idx, lvl)
                if open_p.remaining <= 1e-9:
                    open_p.close(f"TP{idx + 1}")
                    self._notify_close(tg, open_p, lvl or price, f"TP{idx + 1}", pnl)
                else:
                    self._notify_tp(tg, open_p, idx + 1, pnl)
                continue

            if "SL" in kind and open_p.remaining > 1e-9:
                pnl = open_p.bank_sl(price, level=e.level if e.level is not None else None)
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
# Gate executor — real orders via ccxt (testnet or live), reconciled fills
# ---------------------------------------------------------------------------

class GateExecutor(Executor):
    """Real orders on Gate (sandbox testnet or mainnet).

    TP/SL are monitored levels: on each tick the live price is checked against
    the plan and fractions are closed with market orders. Resting bracket
    orders are NOT used (Gate spot stop-order support is uneven); the 5s poll
    is the protection latency.

    FILL RECONCILIATION: every order is fetched back after submission and the
    ACTUAL average fill price + fee is booked into the position, so PnL
    reflects real execution. Partial fills are handled by closing only what
    the exchange reports as filled; the remaining fraction is retried on the
    next tick/bar event. On restart, open positions are re-synced against the
    exchange's live balances so a missed close is never double-booked.
    """

    mode = "gate"

    def __init__(self, ex_cfg, ind_cfg, state_path: Path, sandbox: bool = True):
        super().__init__(ex_cfg, ind_cfg, state_path)
        import ccxt
        # Venue is configurable (`execution.exchange`) so testnet can run on any
        # ccxt exchange with a sandbox — e.g. `gate` (api-testnet.gateapi.io) or
        # `bybit` (api-testnet.bybit.com, no KYC). Market data always comes from
        # the configured `exchange:` in config.yaml (Gate candles); only order
        # placement + balance checks hit the sandbox venue.
        exchange_id = (getattr(ex_cfg, "exchange", None) or "gate").lower()
        if not hasattr(ccxt, exchange_id):
            raise SystemExit(f"Unknown execution exchange: {exchange_id}")
        self.exchange = getattr(ccxt, exchange_id)({
            "apiKey": os.getenv("EXCHANGE_API_KEY", ""),
            "secret": os.getenv("EXCHANGE_API_SECRET", ""),
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        })
        self.exchange.set_sandbox_mode(sandbox)
        self.venue = "testnet" if sandbox else "live"
        self.mode = f"{exchange_id}-{self.venue}"
        if not self.exchange.apiKey:
            raise SystemExit(
                f"EXCHANGE_API_KEY/EXCHANGE_API_SECRET must be set for {self.venue} mode.")
        # Restart reconciliation: re-sync open positions against real balances
        # so a close that happened while we were down is never re-traded or
        # double-booked.
        self._sync_from_exchange()

    # -- reconciliation -----------------------------------------------------

    def _market_order(self, pos: Position, side: str, units: float) -> Optional[dict]:
        """Place a market order and return the AVERAGE FILL price + fee."""
        if units <= 1e-12:
            return None
        order = self.exchange.create_order(pos.symbol, "market", side, units)
        filled = None
        for attempt in range(4):
            try:
                filled = self.exchange.fetch_order(order["id"], pos.symbol)
                if filled.get("filled") and float(filled["filled"]) > 0:
                    break
            except Exception:
                pass
            time.sleep(0.25 * (attempt + 1))
        if filled is None:
            filled = order
        cost = 0.0
        amount = 0.0
        fee = 0.0
        for f in (filled.get("trades") or []):
            cost += float(f.get("cost") or 0.0)
            amount += float(f.get("amount") or 0.0)
            fee += float((f.get("fee") or {}).get("cost") or 0.0)
        if amount > 0 and cost > 0:
            avg_price = cost / amount
        else:
            avg_price = float(filled.get("average") or filled.get("price") or pos.entry)
            amount = float(filled.get("filled") or amount or units)
        fee_c = float((filled.get("fee") or {}).get("cost") or fee or 0.0)
        return {"price": avg_price, "amount": amount, "fee": fee_c,
                "id": order.get("id"), "ts": str(filled.get("timestamp") or "")}

    def _sync_from_exchange(self) -> None:
        """Re-sync open positions against real balances so a restart never
        double-books or misses an exchange-side close. Called on init."""
        if not self.exchange.apiKey:
            return
        try:
            balances = self.exchange.fetch_balance()
        except Exception as exc:
            log.warning("gate: balance sync failed (%s) — trusting local state.", exc)
            return
        for p in self.positions:
            if p.closed:
                continue
            base = p.symbol.split("/")[0]
            free = float(balances.get(base, {}).get("free", 0.0) or 0.0)
            if free <= 1e-12 and p.remaining > 1e-9:
                # Exchange says we hold nothing — a close happened while we
                # were down. Book it flat at the last known price.
                log.warning("gate: %s free balance is 0 but position %s open — "
                            "assuming exchange-side close.", base, p.symbol)
                p.bank_price(p.remaining, self.last_prices.get(p.symbol, p.entry))
                p.close("reconciled")
                p.venue = self.venue

    # -- event processing ---------------------------------------------------

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
                    if not open_p.closed:
                        # Flip close only partially filled — the old position
                        # still holds exposure. Opening the new side now would
                        # double up on the symbol; defer until the flip fully
                        # closes (retried by the monitor pass / next tick).
                        self._notify_error(tg, symbol,
                                           f"flip deferred — {open_p.remaining:.1%} of the "
                                           f"old position still open after partial fill")
                        continue
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
                    venue=self.venue,
                )
                self.positions.append(pos)
                self._place_entry(pos, tg)
                continue

            open_p = next((p for p in self.positions
                           if p.symbol == symbol and not p.closed), None)
            if open_p is None or not kind.startswith(open_p.side):
                continue
            if "TP" in kind and open_p.remaining > 1e-9:
                idx = {"TP1": 0, "TP2": 1, "TP3": 2}[kind.split(" ")[1]]
                self._close_fraction(open_p, idx + 1, tg)
            elif "SL" in kind and open_p.remaining > 1e-9:
                self._market_close(open_p, tg, "SL")
        # Self-healing monitor pass: check mode (GitHub cron) has no tick loop,
        # so partial fills / missed crosses are caught here against the last
        # known prices — same level checks tick() performs.
        for p in self.positions:
            if p.closed:
                continue
            px = self.last_prices.get(p.symbol)
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
        self.save()

    def tick(self, prices: dict[str, float], tg) -> None:
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
            fill = self._market_order(pos, "buy" if pos.side == "Long" else "sell", units)
        except Exception as exc:
            pos.close("entry_failed")
            self._notify_error(tg, pos.symbol, f"entry order failed: {exc}")
            return
        if fill is not None:
            pos.entry = fill["price"]  # book the REAL fill price
            pos.fill_ts = fill["ts"]
            fee_usd = fill["fee"] * pos.entry
            log.info("gate %s: entry %s %s units=%.6g avg=%.6g fee=%.6g USD",
                     self.venue, pos.side, pos.symbol, units, fill["price"], fee_usd)
        self._notify_open(tg, pos, self.venue)
        self.save()

    def _close_fraction(self, pos: Position, tp_idx: int, tg) -> None:
        frac = self.tp_fractions[tp_idx - 1] if tp_idx - 1 < len(self.tp_fractions) else self.tp_fractions[-1]
        frac_units = self._units(pos) * frac
        try:
            fill = self._market_order(pos, "sell" if pos.side == "Long" else "buy", frac_units)
        except Exception as exc:
            self._notify_error(tg, pos.symbol, f"TP{tp_idx} close failed: {exc}")
            return
        lvl = (pos.tp1, pos.tp2, pos.tp3)[tp_idx - 1]
        # Book ONLY what the exchange actually filled (partial fills leave the
        # remainder open — the next tick/bar retries it). Bank the plan-level
        # gain on the filled fraction, minus the real exchange fee.
        if fill is not None and fill["amount"] < frac_units - 1e-12:
            filled_frac = frac * (fill["amount"] / frac_units) if frac_units > 0 else 0.0
            self._notify_error(tg, pos.symbol,
                               f"TP{tp_idx} partial fill {fill['amount']:.6g}/{frac_units:.6g} — "
                               f"remainder stays open, retried next tick")
        else:
            filled_frac = frac
        pnl = pos.bank_tp(filled_frac, tp_idx - 1, lvl if lvl else (fill["price"] if fill else pos.entry),
                          advance=filled_frac >= frac - 1e-12)
        if fill is not None:
            pos.fill_ts = fill["ts"]
            fee_usd = fill["fee"] * (fill["price"] if fill["price"] else pos.entry)
            pos.realized -= fee_usd
            pos.partial_ret -= fee_usd / pos.size if pos.size else 0.0
        if pos.remaining <= 1e-9:
            pos.close(f"TP{tp_idx}")
            self._notify_close(tg, pos, fill["price"] if fill else pos.entry, f"TP{tp_idx}", pnl)
        else:
            self._notify_tp(tg, pos, tp_idx, pnl)
        self.save()

    def _market_close(self, pos: Position, tg, reason: str) -> None:
        rem_units = self._units(pos) * pos.remaining
        try:
            fill = self._market_order(pos, "sell" if pos.side == "Long" else "buy", rem_units) if rem_units > 1e-12 else None
        except Exception as exc:
            self._notify_error(tg, pos.symbol, f"close ({reason}) failed: {exc}")
            return
        px = fill["price"] if fill else self.last_prices.get(pos.symbol, pos.entry)
        # Bank only what the exchange filled; the rest stays open.
        if fill is not None and fill["amount"] < rem_units - 1e-12:
            closed_frac = pos.remaining * (fill["amount"] / rem_units) if rem_units > 0 else 0.0
            self._notify_error(tg, pos.symbol,
                               f"close ({reason}) partial fill {fill['amount']:.6g}/{rem_units:.6g} — "
                               f"remainder stays open")
        else:
            closed_frac = pos.remaining
        if reason == "SL":
            pnl = pos.bank_sl(px, level=pos.sl, frac=closed_frac)
        else:
            pnl = pos.bank_price(closed_frac, px)
        if fill is not None:
            pos.fill_ts = fill["ts"]
            fee_usd = fill["fee"] * (fill["price"] if fill["price"] else pos.entry)
            pos.realized -= fee_usd
            pos.partial_ret -= fee_usd / pos.size if pos.size else 0.0
        if pos.remaining <= 1e-9:
            pos.close(reason)
            self._notify_close(tg, pos, px, reason, pnl)
        else:
            self._notify_error(tg, pos.symbol, f"close ({reason}) still has {pos.remaining:.1%} open — retried next tick")
        self.save()

    def close_all(self) -> int:
        """/flat on real money: submit actual market closes for every open
        position (never book at last price). Partially-filled closes retry on
        the next command poll."""
        n = 0
        for p in self.positions:
            if p.closed:
                continue
            self._market_close(p, None, "flat")
            n += 1
        self._update_day_pnl()
        self.save()
        return n

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

def make_executor(ex_cfg, ind_cfg, state_path: Path) -> Optional[Executor]:
    mode = (ex_cfg.mode or "off").lower()
    if mode == "off":
        return None
    if mode == "paper":
        return PaperExecutor(ex_cfg, ind_cfg, state_path)
    if mode == "testnet":
        return GateExecutor(ex_cfg, ind_cfg, state_path, sandbox=True)
    if mode == "live":
        if not getattr(ex_cfg, "live_confirmation", False):
            raise SystemExit(
                "execution.mode: live requires execution.live_confirmation: true — "
                "real money. Re-read deploy/gh-actions/README.md before flipping this.")
        return GateExecutor(ex_cfg, ind_cfg, state_path, sandbox=False)
    raise SystemExit(f"Unknown execution.mode: {mode}")
