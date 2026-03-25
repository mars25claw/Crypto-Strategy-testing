"""Alert delivery system with rate limiting and multiple channels.

Supports dashboard (in-memory), Telegram, Discord, and email delivery.
All channels are optional and configured via :class:`AlertingConfig`.
Delivery failures are logged but never raise exceptions.
"""

from __future__ import annotations

import asyncio
import logging
import smtplib
import time
from collections import deque
from dataclasses import dataclass, field
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import IntEnum
from typing import Any, Deque, Dict, List, Optional

import httpx

from shared.config_loader import AlertingConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Alert levels
# ---------------------------------------------------------------------------

class AlertLevel(IntEnum):
    """Severity levels in ascending order."""
    INFO = 0
    WARNING = 1
    CRITICAL = 2
    EMERGENCY = 3


_LEVEL_EMOJI = {
    AlertLevel.INFO: "ℹ️",
    AlertLevel.WARNING: "⚠️",
    AlertLevel.CRITICAL: "🔴",
    AlertLevel.EMERGENCY: "🚨",
}

_LEVEL_NAMES = {
    "info": AlertLevel.INFO,
    "warning": AlertLevel.WARNING,
    "critical": AlertLevel.CRITICAL,
    "emergency": AlertLevel.EMERGENCY,
}


def _parse_level(level: str | AlertLevel) -> AlertLevel:
    if isinstance(level, AlertLevel):
        return level
    return _LEVEL_NAMES.get(str(level).lower(), AlertLevel.INFO)


# ---------------------------------------------------------------------------
# Alert record
# ---------------------------------------------------------------------------

@dataclass
class AlertRecord:
    level: AlertLevel
    title: str
    message: str
    data: Optional[dict]
    strategy_id: str
    timestamp: float
    delivered_to: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Alert manager
# ---------------------------------------------------------------------------

class AlertManager:
    """Deliver alerts through multiple channels with rate limiting.

    Parameters:
        config: An :class:`AlertingConfig` from the shared config loader.
        strategy_id: Identifier for the strategy sending alerts.
    """

    MAX_HISTORY = 1000
    RATE_LIMIT_SECONDS = 60  # max 1 alert per title per this window

    def __init__(self, config: AlertingConfig, strategy_id: str) -> None:
        self._config = config
        self._strategy_id = strategy_id
        self._history: Deque[AlertRecord] = deque(maxlen=self.MAX_HISTORY)
        self._last_sent: Dict[str, float] = {}  # title -> last send timestamp
        self._client: Optional[httpx.AsyncClient] = None

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=10.0)
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------

    def _is_rate_limited(self, title: str) -> bool:
        last = self._last_sent.get(title, 0.0)
        return (time.time() - last) < self.RATE_LIMIT_SECONDS

    def _record_sent(self, title: str) -> None:
        self._last_sent[title] = time.time()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def send(
        self,
        level: str | AlertLevel,
        title: str,
        message: str,
        data: Optional[dict] = None,
    ) -> None:
        """Send an alert through all configured channels.

        Alerts with the same *title* are rate-limited to one per 60 seconds.
        """
        if not self._config.enabled:
            return

        parsed_level = _parse_level(level)

        if self._is_rate_limited(title):
            logger.debug("Rate-limited alert: %s", title)
            return

        record = AlertRecord(
            level=parsed_level,
            title=title,
            message=message,
            data=data,
            strategy_id=self._strategy_id,
            timestamp=time.time(),
        )

        # Always store in dashboard history
        self._history.append(record)
        record.delivered_to.append("dashboard")

        # Dispatch to external channels concurrently
        tasks = []
        if self._config.telegram_bot_token and self._config.telegram_chat_id:
            tasks.append(self._send_telegram(record))
        if self._config.discord_webhook_url:
            tasks.append(self._send_discord(record))
        if self._config.email_smtp_host and self._config.email_to:
            tasks.append(self._send_email(record))

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for r in results:
                if isinstance(r, Exception):
                    logger.warning("Alert delivery error: %s", r)

        self._record_sent(title)

    async def send_info(self, title: str, message: str, **kwargs: Any) -> None:
        await self.send(AlertLevel.INFO, title, message, data=kwargs.get("data"))

    async def send_warning(self, title: str, message: str, **kwargs: Any) -> None:
        await self.send(AlertLevel.WARNING, title, message, data=kwargs.get("data"))

    async def send_critical(self, title: str, message: str, **kwargs: Any) -> None:
        await self.send(AlertLevel.CRITICAL, title, message, data=kwargs.get("data"))

    async def send_emergency(self, title: str, message: str, **kwargs: Any) -> None:
        await self.send(AlertLevel.EMERGENCY, title, message, data=kwargs.get("data"))

    def get_recent_alerts(self, limit: int = 50) -> List[dict]:
        """Return the most recent alerts as plain dicts (newest first)."""
        alerts = list(self._history)
        alerts.reverse()
        return [
            {
                "level": a.level.name,
                "title": a.title,
                "message": a.message,
                "data": a.data,
                "strategy_id": a.strategy_id,
                "timestamp": a.timestamp,
                "delivered_to": a.delivered_to,
            }
            for a in alerts[:limit]
        ]

    # ------------------------------------------------------------------
    # Delivery channels
    # ------------------------------------------------------------------

    def _format_text(self, record: AlertRecord) -> str:
        emoji = _LEVEL_EMOJI.get(record.level, "")
        lines = [
            f"{emoji} [{record.level.name}] {record.title}",
            f"Strategy: {record.strategy_id}",
            "",
            record.message,
        ]
        if record.data:
            lines.append("")
            for k, v in record.data.items():
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)

    async def _send_telegram(self, record: AlertRecord) -> None:
        """Send alert via Telegram Bot API."""
        try:
            client = await self._ensure_client()
            url = f"https://api.telegram.org/bot{self._config.telegram_bot_token}/sendMessage"
            payload = {
                "chat_id": self._config.telegram_chat_id,
                "text": self._format_text(record),
                "parse_mode": "HTML",
            }
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            record.delivered_to.append("telegram")
            logger.debug("Telegram alert sent: %s", record.title)
        except Exception as exc:
            logger.warning("Telegram delivery failed for '%s': %s", record.title, exc)

    async def _send_discord(self, record: AlertRecord) -> None:
        """Send alert via Discord webhook."""
        try:
            client = await self._ensure_client()
            payload = {
                "content": self._format_text(record),
                "username": f"Trading Bot ({record.strategy_id})",
            }
            resp = await client.post(self._config.discord_webhook_url, json=payload)
            resp.raise_for_status()
            record.delivered_to.append("discord")
            logger.debug("Discord alert sent: %s", record.title)
        except Exception as exc:
            logger.warning("Discord delivery failed for '%s': %s", record.title, exc)

    async def _send_email(self, record: AlertRecord) -> None:
        """Send alert via SMTP email.

        Runs the blocking SMTP call in a thread executor to avoid blocking
        the event loop.
        """
        try:
            await asyncio.get_event_loop().run_in_executor(
                None, self._send_email_sync, record
            )
            record.delivered_to.append("email")
            logger.debug("Email alert sent: %s", record.title)
        except Exception as exc:
            logger.warning("Email delivery failed for '%s': %s", record.title, exc)

    def _send_email_sync(self, record: AlertRecord) -> None:
        """Blocking SMTP send (called from executor)."""
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"[{record.level.name}] {record.title} — {record.strategy_id}"
        msg["From"] = self._config.email_from
        msg["To"] = self._config.email_to

        body = self._format_text(record)
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP(self._config.email_smtp_host, self._config.email_smtp_port) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            if self._config.email_password:
                server.login(self._config.email_from, self._config.email_password)
            server.sendmail(self._config.email_from, [self._config.email_to], msg.as_string())
