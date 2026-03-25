"""
Heartbeat Monitoring — STRAT-001 Section 8.3.

Emits periodic heartbeat payloads, monitors health, and implements
auto-restart logic with rate limiting.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Callable, Coroutine, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class HeartbeatMonitor:
    """
    Heartbeat emission and health monitoring for trading strategies.

    Emits a heartbeat payload every `interval` seconds. If no heartbeat
    is emitted within `timeout` seconds, triggers the restart callback.
    Halts the strategy if restarts exceed `max_restarts_per_hour`.

    Args:
        strategy_id: Unique identifier for the strategy.
        interval: Seconds between heartbeat emissions (default 10).
        timeout: Seconds without a heartbeat before triggering restart (default 30).
        max_restarts_per_hour: Maximum restart attempts per rolling hour (default 3).
        on_restart: Async or sync callback invoked on restart. Receives the
            strategy_id as its sole argument. If it returns a coroutine, it
            will be awaited.
    """

    def __init__(
        self,
        strategy_id: str,
        interval: int = 10,
        timeout: int = 30,
        max_restarts_per_hour: int = 3,
        on_restart: Optional[Callable] = None,
    ) -> None:
        self.strategy_id = strategy_id
        self.interval = interval
        self.timeout = timeout
        self.max_restarts_per_hour = max_restarts_per_hour
        self._on_restart = on_restart

        # State
        self._start_time: float = 0.0
        self._last_heartbeat: Optional[dict] = None
        self._last_heartbeat_time: float = 0.0
        self._running: bool = False
        self._halted: bool = False

        # Restart tracking
        self._restart_timestamps: List[float] = []

        # Position / PnL state (updated by strategy)
        self._positions_count: int = 0
        self._unrealized_pnl: float = 0.0
        self._stream_timestamps: Dict[str, int] = {}

        # Tasks
        self._emit_task: Optional[asyncio.Task] = None
        self._watchdog_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the heartbeat emission loop and watchdog."""
        if self._running:
            logger.warning("[%s] Heartbeat already running", self.strategy_id)
            return

        if self._halted:
            logger.error(
                "[%s] Strategy is halted due to excessive restarts. "
                "Manual intervention required.",
                self.strategy_id,
            )
            return

        self._running = True
        self._start_time = time.time()
        self._last_heartbeat_time = time.time()

        logger.info(
            "[%s] Starting heartbeat monitor (interval=%ds, timeout=%ds)",
            self.strategy_id,
            self.interval,
            self.timeout,
        )

        self._emit_task = asyncio.create_task(self._emission_loop())
        self._watchdog_task = asyncio.create_task(self._watchdog_loop())

    def stop(self) -> None:
        """Stop the heartbeat monitor gracefully."""
        self._running = False
        if self._emit_task and not self._emit_task.done():
            self._emit_task.cancel()
        if self._watchdog_task and not self._watchdog_task.done():
            self._watchdog_task.cancel()
        logger.info("[%s] Heartbeat monitor stopped", self.strategy_id)

    def emit(self) -> dict:
        """
        Manually emit a heartbeat and return the payload.

        This resets the watchdog timer.
        """
        self._last_heartbeat_time = time.time()
        payload = self._build_payload()
        self._last_heartbeat = payload
        return payload

    def get_last_heartbeat(self) -> Optional[dict]:
        """Return the most recently emitted heartbeat payload."""
        return self._last_heartbeat

    def get_uptime(self) -> float:
        """Return uptime in seconds since start() was called."""
        if self._start_time == 0:
            return 0.0
        return time.time() - self._start_time

    def is_healthy(self) -> bool:
        """
        Check if the monitor is healthy.

        Healthy means the last heartbeat was emitted within `timeout` seconds
        and the strategy is not halted.
        """
        if self._halted:
            return False
        if self._last_heartbeat_time == 0:
            return False
        elapsed = time.time() - self._last_heartbeat_time
        return elapsed < self.timeout

    def set_positions_count(self, count: int) -> None:
        """Update the active positions count reported in heartbeats."""
        self._positions_count = count

    def set_unrealized_pnl(self, pnl: float) -> None:
        """Update the unrealized PnL reported in heartbeats."""
        self._unrealized_pnl = pnl

    def set_stream_timestamps(self, timestamps: Dict[str, int]) -> None:
        """
        Update the last-data timestamps per stream.

        Args:
            timestamps: Dict mapping stream name to last data timestamp (ms).
        """
        self._stream_timestamps = dict(timestamps)

    @property
    def is_halted(self) -> bool:
        """True if the strategy has been halted due to excessive restarts."""
        return self._halted

    @property
    def restart_count(self) -> int:
        """Number of restarts in the last hour."""
        cutoff = time.time() - 3600
        return sum(1 for ts in self._restart_timestamps if ts > cutoff)

    # ------------------------------------------------------------------
    # Private: loops
    # ------------------------------------------------------------------

    async def _emission_loop(self) -> None:
        """Periodically emit heartbeats at the configured interval."""
        try:
            while self._running:
                self.emit()
                logger.debug(
                    "[%s] Heartbeat emitted (uptime=%.1fs)",
                    self.strategy_id,
                    self.get_uptime(),
                )
                await asyncio.sleep(self.interval)
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("[%s] Heartbeat emission loop error", self.strategy_id)

    async def _watchdog_loop(self) -> None:
        """Monitor heartbeat freshness and trigger restarts when stale."""
        try:
            while self._running:
                await asyncio.sleep(self.timeout / 3)  # Check frequently

                if not self._running:
                    break

                elapsed = time.time() - self._last_heartbeat_time
                if elapsed >= self.timeout:
                    logger.warning(
                        "[%s] No heartbeat for %.1fs (timeout=%ds). Triggering restart.",
                        self.strategy_id,
                        elapsed,
                        self.timeout,
                    )
                    await self._handle_restart()
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("[%s] Watchdog loop error", self.strategy_id)

    # ------------------------------------------------------------------
    # Private: restart logic
    # ------------------------------------------------------------------

    async def _handle_restart(self) -> None:
        """
        Handle a missed-heartbeat restart.

        If restarts in the last hour exceed max_restarts_per_hour, halt
        the strategy and alert the operator.
        """
        now = time.time()
        self._restart_timestamps.append(now)

        # Prune timestamps older than 1 hour
        cutoff = now - 3600
        self._restart_timestamps = [
            ts for ts in self._restart_timestamps if ts > cutoff
        ]

        recent_count = len(self._restart_timestamps)

        if recent_count >= self.max_restarts_per_hour:
            logger.critical(
                "[%s] HALTING: %d restarts in the last hour (max=%d). "
                "Manual intervention required.",
                self.strategy_id,
                recent_count,
                self.max_restarts_per_hour,
            )
            self._halted = True
            self._running = False
            return

        logger.warning(
            "[%s] Restart %d/%d in the last hour",
            self.strategy_id,
            recent_count,
            self.max_restarts_per_hour,
        )

        # Reset heartbeat time so watchdog doesn't immediately re-trigger
        self._last_heartbeat_time = time.time()

        if self._on_restart is not None:
            try:
                result = self._on_restart(self.strategy_id)
                if asyncio.iscoroutine(result) or asyncio.isfuture(result):
                    await result
            except Exception:
                logger.exception(
                    "[%s] Error in restart callback", self.strategy_id
                )

    # ------------------------------------------------------------------
    # Private: payload
    # ------------------------------------------------------------------

    def _build_payload(self) -> dict:
        """Build the heartbeat payload."""
        now_ms = int(time.time() * 1000)

        payload: Dict[str, Any] = {
            "timestamp_ms": now_ms,
            "strategy_id": self.strategy_id,
            "active_positions_count": self._positions_count,
            "unrealized_pnl": self._unrealized_pnl,
            "last_data_timestamp_per_stream": dict(self._stream_timestamps),
            "memory_usage_mb": self._get_memory_usage_mb(),
            "uptime_seconds": round(self.get_uptime(), 2),
        }

        cpu = self._get_cpu_usage_pct()
        if cpu is not None:
            payload["cpu_usage_pct"] = cpu

        return payload

    @staticmethod
    def _get_memory_usage_mb() -> float:
        """Get current process memory usage in MB."""
        try:
            import resource
            # ru_maxrss is in bytes on Linux, kilobytes on macOS
            usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            import sys
            if sys.platform == "darwin":
                return round(usage / (1024 * 1024), 2)  # macOS: bytes -> MB
            else:
                return round(usage / 1024, 2)  # Linux: KB -> MB
        except ImportError:
            pass

        try:
            import psutil
            process = psutil.Process(os.getpid())
            return round(process.memory_info().rss / (1024 * 1024), 2)
        except ImportError:
            return 0.0

    @staticmethod
    def _get_cpu_usage_pct() -> Optional[float]:
        """Get CPU usage percentage if psutil is available."""
        try:
            import psutil
            return psutil.Process(os.getpid()).cpu_percent(interval=None)
        except (ImportError, Exception):
            return None
