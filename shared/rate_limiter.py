"""Weight-based rate limiter for Binance API.

Binance tracks request weight independently for spot and futures APIs.
This limiter enforces per-minute weight budgets with priority queuing,
auto-throttling, and a kill-switch bypass for safety-critical operations.
"""

import asyncio
import time
import logging
from collections import deque
from enum import IntEnum
from dataclasses import dataclass, field
from typing import Dict, Optional, Deque, Tuple

logger = logging.getLogger(__name__)


class Priority(IntEnum):
    """Request priority levels.

    CRITICAL — stop-loss, kill switch, forced exits.  Bypasses the queue.
    HIGH     — normal exit orders.
    NORMAL   — market data, account info, klines.
    """
    NORMAL = 0
    HIGH = 1
    CRITICAL = 2


class ApiType:
    SPOT = "spot"
    FUTURES = "futures"


@dataclass
class _WeightWindow:
    """Sliding-window weight tracker for a single API type."""

    budget: int
    burst: int
    _entries: Deque[Tuple[float, int]] = field(default_factory=deque)

    # ── public ──────────────────────────────────────────────────────────

    @property
    def used(self) -> int:
        """Total weight consumed in the current 60-second window."""
        self._evict()
        return sum(w for _, w in self._entries)

    @property
    def remaining(self) -> int:
        return max(0, self.budget - self.used)

    @property
    def utilization(self) -> float:
        """Fraction of the budget consumed (0.0 – 1.0+)."""
        if self.budget == 0:
            return 0.0
        return self.used / self.budget

    def record(self, weight: int) -> None:
        self._entries.append((time.monotonic(), weight))

    def seconds_until_free(self, weight_needed: int) -> float:
        """Return how many seconds to wait before *weight_needed* fits."""
        self._evict()
        now = time.monotonic()
        deficit = self.used + weight_needed - self.budget
        if deficit <= 0:
            return 0.0
        # Walk oldest entries until we free enough
        freed = 0
        for ts, w in self._entries:
            freed += w
            if freed >= deficit:
                # This entry expires at ts + 60
                return max(0.0, (ts + 60.0) - now + 0.05)
        # Worst case: wait for full window to roll over
        return 60.0

    def reset(self) -> None:
        """Force-clear the window (called on reconnect / manual reset)."""
        self._entries.clear()

    # ── internal ────────────────────────────────────────────────────────

    def _evict(self) -> None:
        cutoff = time.monotonic() - 60.0
        while self._entries and self._entries[0][0] < cutoff:
            self._entries.popleft()


@dataclass
class EndpointStats:
    """Per-endpoint cumulative stats."""
    total_weight: int = 0
    call_count: int = 0
    last_called: float = 0.0


class RateLimiter:
    """Async, weight-based, priority-aware rate limiter for Binance APIs.

    Usage::

        limiter = RateLimiter(budget=200, burst=400)

        # Normal data request (weight 1)
        await limiter.acquire(weight=1, priority=Priority.NORMAL, api_type="futures")

        # Critical stop-loss (weight 1, bypasses throttle queue)
        await limiter.acquire(weight=1, priority=Priority.CRITICAL, api_type="futures")

        # Kill-switch mode — unlimited, bypasses everything
        limiter.set_kill_switch(True)
        await limiter.acquire(weight=5, priority=Priority.CRITICAL, api_type="futures")
    """

    # Throttle threshold — start delaying when utilization exceeds this.
    THROTTLE_THRESHOLD = 0.80
    # Maximum delay injected by auto-throttle (seconds).
    MAX_THROTTLE_DELAY = 5.0

    def __init__(
        self,
        budget: int = 200,
        burst: int = 400,
        spot_budget: Optional[int] = None,
        spot_burst: Optional[int] = None,
    ):
        # Separate windows for each API
        self._windows: Dict[str, _WeightWindow] = {
            ApiType.FUTURES: _WeightWindow(budget=budget, burst=burst),
            ApiType.SPOT: _WeightWindow(
                budget=spot_budget or budget,
                burst=spot_burst or burst,
            ),
        }

        self._lock = asyncio.Lock()
        self._kill_switch: bool = False

        # Per-endpoint tracking
        self._endpoint_stats: Dict[str, EndpointStats] = {}

        # Counters
        self._total_acquired: int = 0
        self._total_throttled: int = 0
        self._total_waited_s: float = 0.0

    # ── configuration ───────────────────────────────────────────────────

    def set_kill_switch(self, enabled: bool) -> None:
        """Enable / disable kill-switch mode (unlimited, bypass everything)."""
        if enabled != self._kill_switch:
            logger.warning("Rate limiter kill switch %s", "ENGAGED" if enabled else "disengaged")
        self._kill_switch = enabled

    @property
    def kill_switch(self) -> bool:
        return self._kill_switch

    def update_budget(self, api_type: str, budget: int, burst: int) -> None:
        """Dynamically update weight budget (e.g. after reading X-MBX headers)."""
        if api_type in self._windows:
            self._windows[api_type].budget = budget
            self._windows[api_type].burst = burst
            logger.info("Rate limit budget updated: %s budget=%d burst=%d", api_type, budget, burst)

    # ── core acquire ────────────────────────────────────────────────────

    async def acquire(
        self,
        weight: int = 1,
        priority: Priority = Priority.NORMAL,
        api_type: str = ApiType.FUTURES,
        endpoint: str = "",
    ) -> None:
        """Wait until *weight* units can be consumed, then record them.

        - CRITICAL priority: bypass throttle delay, only hard-blocked at burst.
        - Kill-switch mode: bypass everything unconditionally.
        """
        # Kill switch — record for stats but never block.
        if self._kill_switch:
            self._record(weight, api_type, endpoint)
            return

        window = self._windows.get(api_type)
        if window is None:
            logger.error("Unknown api_type %r — allowing request", api_type)
            return

        async with self._lock:
            # ── Hard limit: block until weight fits under burst ceiling ──
            wait_s = self._hard_wait(window, weight)
            if wait_s > 0:
                logger.warning(
                    "Rate limit hard-wait %.2fs  api=%s  weight=%d  used=%d/%d",
                    wait_s, api_type, weight, window.used, window.budget,
                )
                self._total_throttled += 1
                self._total_waited_s += wait_s
                # Release lock while sleeping so CRITICAL requests can slip through.
                # We re-check after waking.

        # Sleep outside the lock for hard waits.
        if wait_s > 0:
            await asyncio.sleep(wait_s)
            # Re-acquire lock and re-check.
            async with self._lock:
                wait_s2 = self._hard_wait(window, weight)
                if wait_s2 > 0:
                    await asyncio.sleep(wait_s2)
                    self._total_waited_s += wait_s2

        # ── Soft throttle (skip for CRITICAL) ───────────────────────────
        if priority < Priority.CRITICAL:
            throttle = self._throttle_delay(window, weight)
            if throttle > 0:
                logger.debug(
                    "Auto-throttle %.3fs  api=%s  utilization=%.1f%%",
                    throttle, api_type, window.utilization * 100,
                )
                self._total_throttled += 1
                self._total_waited_s += throttle
                await asyncio.sleep(throttle)

        # ── Record ──────────────────────────────────────────────────────
        async with self._lock:
            self._record(weight, api_type, endpoint)

    # ── helpers ─────────────────────────────────────────────────────────

    def _hard_wait(self, window: _WeightWindow, weight: int) -> float:
        """Seconds to wait before *weight* fits under the burst ceiling."""
        # Use burst as the hard ceiling (the absolute max before Binance bans).
        if window.used + weight <= window.burst:
            return 0.0
        return window.seconds_until_free(weight)

    def _throttle_delay(self, window: _WeightWindow, weight: int) -> float:
        """Proportional backpressure when approaching the budget limit."""
        util = window.utilization
        if util < self.THROTTLE_THRESHOLD:
            return 0.0
        # Linear ramp: 0s at threshold, MAX_THROTTLE_DELAY at 100%.
        fraction = (util - self.THROTTLE_THRESHOLD) / (1.0 - self.THROTTLE_THRESHOLD)
        fraction = min(fraction, 1.0)
        return fraction * self.MAX_THROTTLE_DELAY

    def _record(self, weight: int, api_type: str, endpoint: str) -> None:
        window = self._windows.get(api_type)
        if window:
            window.record(weight)
        self._total_acquired += weight

        if endpoint:
            stats = self._endpoint_stats.setdefault(endpoint, EndpointStats())
            stats.total_weight += weight
            stats.call_count += 1
            stats.last_called = time.monotonic()

    # ── reset / stats ───────────────────────────────────────────────────

    def reset(self, api_type: Optional[str] = None) -> None:
        """Force-clear weight windows.  Called on reconnect or minute tick."""
        if api_type:
            if api_type in self._windows:
                self._windows[api_type].reset()
                logger.debug("Rate limiter reset: %s", api_type)
        else:
            for w in self._windows.values():
                w.reset()
            logger.debug("Rate limiter reset: all")

    def get_usage(self) -> Dict:
        """Return current usage snapshot for monitoring / dashboards."""
        result: Dict = {
            "kill_switch": self._kill_switch,
            "total_acquired": self._total_acquired,
            "total_throttled": self._total_throttled,
            "total_waited_s": round(self._total_waited_s, 3),
        }
        for api_type, window in self._windows.items():
            result[api_type] = {
                "used": window.used,
                "budget": window.budget,
                "burst": window.burst,
                "remaining": window.remaining,
                "utilization_pct": round(window.utilization * 100, 1),
            }
        return result

    def get_endpoint_stats(self) -> Dict[str, Dict]:
        """Return per-endpoint cumulative stats."""
        return {
            ep: {
                "total_weight": s.total_weight,
                "call_count": s.call_count,
            }
            for ep, s in self._endpoint_stats.items()
        }
