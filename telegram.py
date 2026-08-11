"""Telegram Bot API helper."""
from __future__ import annotations

import logging
import time
from typing import Optional
from zoneinfo import ZoneInfo

import requests

log = logging.getLogger("telegram")


class TelegramClient:
    BASE = "https://api.telegram.org"

    def __init__(self, bot_token: str, chat_id: str, timeout: float = 10.0, max_retries: int = 3):
        if not bot_token or "replace_me" in bot_token:
            raise ValueError("Telegram bot token is not configured (check your .env).")
        if not chat_id or "replace_me" in chat_id:
            raise ValueError("Telegram chat id is not configured (check your .env).")
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.timeout = timeout
        self.max_retries = max_retries
        # Best-effort cache of our own @username so error messages (notably the
        # 403 short-circuit) can name the precise bot to open in Telegram.
        # Non-fatal on init: failure here leaves bot_username as None and the
        # caller falls back to a generic "your bot" reference.
        self.bot_username: Optional[str] = None
        self._fetch_bot_username()

    def fetch_updates(self, offset: Optional[int] = None,
                      timeout: float = 2.0) -> list:
        """Long-poll getUpdates; returns raw update dicts (empty list on any error).

        Used by the live loop to receive admin commands (/status, /stop, …).
        `offset` is exclusive-acked: pass the previous update_id + 1 so already
        consumed updates are never re-delivered.
        """
        url = f"{self.BASE}/bot{self.bot_token}/getUpdates"
        params = {"timeout": int(max(1, min(timeout, 50))), "allowed_updates": ["message"]}
        if offset:
            params["offset"] = offset
        try:
            r = requests.get(url, params=params, timeout=timeout + 5)
            if not r.ok:
                log.debug("getUpdates HTTP %s: %s", r.status_code, r.text[:200])
                return []
            payload = r.json()
            return payload.get("result") or [] if payload.get("ok") else []
        except requests.RequestException as exc:
            log.debug("getUpdates failed: %s", exc)
            return []

    def _fetch_bot_username(self) -> None:
        """One-shot getMe at startup to learn our @username for clearer error messages."""
        url = f"{self.BASE}/bot{self.bot_token}/getMe"
        try:
            r = requests.get(url, timeout=self.timeout)
            if not r.ok:
                return
            payload = r.json()
            username = (payload.get("result") or {}).get("username")
            if isinstance(username, str) and username:
                self.bot_username = username
        except (requests.RequestException, ValueError) as exc:
            # Network blip / malformed JSON / not-yet-registered bot — ignore.
            log.debug("getMe lookup failed (username will be None): %s", exc.__class__.__name__)

    def send_message(self, text: str, parse_mode: Optional[str] = None,
                     disable_notification: bool = False) -> bool:
        url = f"{self.BASE}/bot{self.bot_token}/sendMessage"
        payload = {"chat_id": self.chat_id, "text": text}
        if parse_mode:
            payload["parse_mode"] = parse_mode
        if disable_notification:
            payload["disable_notification"] = True

        for attempt in range(1, self.max_retries + 1):
            try:
                r = requests.post(url, json=payload, timeout=self.timeout)
                if r.ok:
                    return True
                if r.status_code == 403:
                    # 403 is permanent — retrying won't help; bail fast with an
                    # actionable hint. The most common cause is "user has not yet
                    # messaged the bot" on private chats. We use the cached
                    # @username from __init__'s one-shot getMe (falls back to a
                    # generic "your bot" reference if the cache lookup didn't run).
                    bot_ref = f"@{self.bot_username}" if self.bot_username else "your bot"
                    log.error(
                        "Telegram 403 Forbidden: %s does not have access to chat %s. "
                        "If chat %s is a private chat, open Telegram, message %s, send /start, "
                        "then restart the alerter. If chat %s is a group, the bot must be a "
                        "member and an admin.",
                        bot_ref, self.chat_id, self.chat_id, bot_ref, self.chat_id,
                    )
                    return False
                if r.status_code == 429:
                    retry_after = int(r.headers.get("Retry-After", "1"))
                    log.warning("Telegram 429: sleeping %ds (attempt %d/%d)",
                                retry_after, attempt, self.max_retries)
                    time.sleep(retry_after)
                    continue
                log.error("Telegram send failed: %s — %s", r.status_code, r.text[:300])
            except requests.RequestException as e:
                log.exception("Telegram send exception (attempt %d/%d): %s",
                               attempt, self.max_retries, e)
                time.sleep(1.5 * attempt)
        return False


_BADGE = {
    "Long Entry":  "🟢⬆️ LONG ENTRY",
    "Short Entry": "🔴⬇️ SHORT ENTRY",
    "Long TP1":    "🎯 LONG TP1",
    "Short TP1":   "🎯 SHORT TP1",
    "Long TP2":    "🎯 LONG TP2",
    "Short TP2":   "🎯 SHORT TP2",
    "Long TP3":    "🎯 LONG TP3",
    "Short TP3":   "🎯 SHORT TP3",
    "Long SL":     "🛑 LONG SL",
    "Short SL":    "🛑 SHORT SL",
    "Long Exit":   "🚪 LONG EXIT",
    "Short Exit":  "🚪 SHORT EXIT",
}


def _fmt_price(p: float) -> str:
    """Adaptive precision based on magnitude — crypto pairs vary widely."""
    if not isinstance(p, float):
        return str(p)
    a = abs(p)
    if a >= 1000: return f"{p:.2f}"
    if a >= 10:   return f"{p:.4f}"
    if a >= 0.1:  return f"{p:.5f}"
    return f"{p:.8f}"


def format_event(event, tz_name: str = "UTC") -> str:
    """Format an Event into a plain-text Telegram message (no Markdown escaping).

    `tz_name` (default "UTC") controls the display timezone of the `Time:` line;
    a best-effort tz abbreviation (e.g. AEST/AEDT) is appended when available.
    Falls back to UTC if the timezone name is invalid.

    When `event.plan is not None`, appends a `Plan:` block listing entry price + the
    three TP targets + the SL price. Percentage is relative to the reference price:
    entry_price for Entry events (price == entry), `event.price` for cross events.
    Sign convention: `(+X.XX%)` means the target is in the trade's direction from
    reference; `(-X.XX%)` means it's against. Lets a reader glance at the alert
    and immediately see where the next stop is, without mentally re-running the
    indicator's TP/SL math.
    """
    badge = _BADGE.get(event.kind, f"⚪ {event.kind}")
    # Display time in the configured timezone (best effort; UTC on any error).
    try:
        local = event.time
        if local.tzinfo is None:
            local = local.tz_localize("UTC")
        local = local.tz_convert(ZoneInfo(tz_name))
        tz_label = local.tzname() or ""
    except Exception:
        local, tz_label = event.time, ""
    time_line = f"Time:   {local.strftime('%Y-%m-%d %H:%M:%S')}"
    if tz_label:
        time_line += f" {tz_label}"
    lines = [
        f"{badge}",
        f"Pair:   {event.symbol}",
        time_line,
        f"Price:  {_fmt_price(event.price)}",
    ]
    if event.level is not None and not (isinstance(event.level, float) and event.level != event.level):
        if abs(event.level - event.price) > 1e-12:
            lines.append(f"Level:  {_fmt_price(event.level)}")

    if event.plan is not None:
        is_entry = event.kind.endswith(" Entry")
        # Reference price for % calculation: entry events have price == entry_price,
        # so entries should never show "Entry: X (+Y%)" against itself; cross events
        # show distance from current bar price.
        reference_price = event.plan.entry_price if is_entry else event.price
        # Side sign flips the signed-pct calculation: Long targets above entry, Short
        # targets below. Apply * side_sign so the displayed % always reads "+X%" when
        # the target is in trade direction (good) and "-X%" against (bad).
        side_sign = 1.0 if event.plan.side == "Long" else -1.0

        lines.append("")
        lines.append("Plan:")
        if not is_entry:
            # Cross events: show where the entry sat relative to current price.
            entry_pct = (event.plan.entry_price - reference_price) / reference_price * 100 * side_sign
            lines.append(f"  Entry:  {_fmt_price(event.plan.entry_price):>10}   ({entry_pct:+.2f}%)")
        # `event.filled` is populated by run_indicator as a frozenset of TP targets
        # already hit BEFORE this event was emitted. Entry events have empty; TP
        # events include the just-filled target; SL events reflect prior fills so
        # the reader sees what got realised before the stop-out (e.g. a `Long SL`
        # after TP1 shows `✓ TP1  ...  ✓` even though the SL is the headline).
        for label, level in [("TP1", event.plan.tp1),
                             ("TP2", event.plan.tp2),
                             ("TP3", event.plan.tp3),
                             ("SL",  event.plan.sl)]:
            pct = (level - reference_price) / reference_price * 100 * side_sign
            mark = "✓" if label in event.filled else " "
            lines.append(f"  {mark} {label:<6} {_fmt_price(level):>10}   ({pct:+.2f}%)")

    return "\n".join(lines)
