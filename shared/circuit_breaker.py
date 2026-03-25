"""Circuit breaker system for detecting anomalous market conditions.

Implements multiple independent breaker types that can halt or restrict
trading when unusual conditions are detected.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)
system_logger = logging.getLogger("system")


# ---------------------------------------------------------------------------
# Enums & data classes
# ---------------------------------------------------------------------------

class BreakerLevel(str, Enum):
    """Severity levels for spread-based breakers."""

    NORMAL = "normal"
    WARNING = "warning"       # halt new orders
    CRITICAL = "critical"     # close with LIMIT only


class HaltScope(str, Enum):
    """Scope of a halt: system-wide or per strategy."""

    SYSTEM = "system"
    STRATEGY = "strategy"


@dataclass
class BreakerTrip:
    """Record of a circuit breaker trip."""

    breaker_name: str
    timestamp: float
    details: Dict[str, Any]
    scope: HaltScope = HaltScope.SYSTEM
    strategy_id: Optional[str] = None
    cooldown_until: float = 0.0

    @property
    def is_active(self) -> bool:
        return time.time() < self.cooldown_until


@dataclass
class CooldownEntry:
    """Cooldown state for a strategy or system-wide halt."""

    until: float
    reason: str
    breaker_name: str


# ---------------------------------------------------------------------------
# Individual breaker implementations
# ---------------------------------------------------------------------------

class FlashCrashBreaker:
    """Triggers when any asset drops more than *threshold_pct* within
    *window_minutes* using 1-minute kline close data.

    Parameters
    ----------
    threshold_pct:
        Percentage drop to trigger (default 10.0 = 10%).
    window_minutes:
        Look-back window in minutes (default 5).
    """

    def __init__(self, threshold_pct: float = 10.0, window_minutes: int = 5) -> None:
        self.threshold_pct = threshold_pct
        self.window_minutes = window_minutes

    def check(self, symbol: str, prices_1m: List[float]) -> Tuple[bool, Dict[str, Any]]:
        """Check 1-minute close prices for a flash crash.

        Parameters
        ----------
        prices_1m:
            List of 1-minute close prices ordered chronologically.
            At least *window_minutes* + 1 entries are needed.

        Returns
        -------
        (tripped, details)
        """
        if len(prices_1m) < self.window_minutes + 1:
            return False, {}

        window = prices_1m[-(self.window_minutes + 1):]
        high_in_window = max(window[:-1])  # highest price before current candle
        current = window[-1]

        if high_in_window <= 0:
            return False, {}

        drop_pct = ((high_in_window - current) / high_in_window) * 100.0

        if drop_pct >= self.threshold_pct:
            details = {
                "symbol": symbol,
                "drop_pct": round(drop_pct, 2),
                "high_price": high_in_window,
                "current_price": current,
                "window_minutes": self.window_minutes,
                "threshold_pct": self.threshold_pct,
            }
            logger.critical(
                "FLASH CRASH detected: %s dropped %.2f%% in %d min (%.4f -> %.4f)",
                symbol, drop_pct, self.window_minutes, high_in_window, current,
            )
            return True, details

        return False, {}


class SpreadAnomalyBreaker:
    """Monitors bid-ask spread as a percentage of mid price.

    Two thresholds:
    - *warning_pct*: halt new orders (default 0.5%)
    - *critical_pct*: close only with LIMIT orders (default 1.0%)
    """

    def __init__(self, warning_pct: float = 0.5, critical_pct: float = 1.0) -> None:
        self.warning_pct = warning_pct
        self.critical_pct = critical_pct

    def check(self, symbol: str, bid: float, ask: float, mid: float) -> Tuple[bool, str, Dict[str, Any]]:
        """Check spread.

        Returns
        -------
        (tripped, level, details)
            *level* is ``"normal"``, ``"warning"``, or ``"critical"``.
        """
        if mid <= 0:
            return False, BreakerLevel.NORMAL, {}

        spread = ask - bid
        spread_pct = (spread / mid) * 100.0

        details = {
            "symbol": symbol,
            "bid": bid,
            "ask": ask,
            "mid": mid,
            "spread": spread,
            "spread_pct": round(spread_pct, 4),
        }

        if spread_pct >= self.critical_pct:
            logger.warning(
                "SPREAD CRITICAL: %s spread %.4f%% (bid=%.4f ask=%.4f)",
                symbol, spread_pct, bid, ask,
            )
            return True, BreakerLevel.CRITICAL, details

        if spread_pct >= self.warning_pct:
            logger.warning(
                "SPREAD WARNING: %s spread %.4f%% (bid=%.4f ask=%.4f)",
                symbol, spread_pct, bid, ask,
            )
            return True, BreakerLevel.WARNING, details

        return False, BreakerLevel.NORMAL, details


class ExchangeAnomalyBreaker:
    """Monitors exchange health signals.

    Trips when:
    - WebSocket latency exceeds *max_ws_latency_ms* (default 5000)
    - *consecutive_api_failures* consecutive API calls fail (default 3)
    - Balance mismatch exceeds *balance_mismatch_pct* (default 1.0%)
    """

    def __init__(
        self,
        max_ws_latency_ms: int = 5000,
        consecutive_api_failures: int = 3,
        balance_mismatch_pct: float = 1.0,
    ) -> None:
        self.max_ws_latency_ms = max_ws_latency_ms
        self.max_consecutive_failures = consecutive_api_failures
        self.balance_mismatch_pct = balance_mismatch_pct

        self._lock = threading.Lock()
        self._consecutive_failures: int = 0
        self._last_ws_latency_ms: int = 0
        self._last_balance_mismatch_pct: float = 0.0

    def record_api_failure(self) -> None:
        """Record a failed API call."""
        with self._lock:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self.max_consecutive_failures:
                logger.error(
                    "Exchange health: %d consecutive API failures",
                    self._consecutive_failures,
                )

    def record_api_success(self) -> None:
        """Record a successful API call — resets the failure counter."""
        with self._lock:
            self._consecutive_failures = 0

    def record_ws_latency(self, latency_ms: int) -> None:
        """Record the latest WebSocket round-trip latency."""
        with self._lock:
            self._last_ws_latency_ms = latency_ms

    def record_balance_mismatch(self, mismatch_pct: float) -> None:
        """Record the latest balance mismatch between expected and actual."""
        with self._lock:
            self._last_balance_mismatch_pct = mismatch_pct

    def check(self) -> Tuple[bool, List[str]]:
        """Check exchange health.

        Returns
        -------
        (healthy, issues)
            *healthy* is True when no issues are detected.
        """
        issues: List[str] = []
        with self._lock:
            if self._last_ws_latency_ms > self.max_ws_latency_ms:
                issues.append(
                    f"WS latency {self._last_ws_latency_ms}ms > {self.max_ws_latency_ms}ms"
                )

            if self._consecutive_failures >= self.max_consecutive_failures:
                issues.append(
                    f"{self._consecutive_failures} consecutive API failures "
                    f"(threshold: {self.max_consecutive_failures})"
                )

            if self._last_balance_mismatch_pct > self.balance_mismatch_pct:
                issues.append(
                    f"Balance mismatch {self._last_balance_mismatch_pct:.2f}% "
                    f"> {self.balance_mismatch_pct}%"
                )

        healthy = len(issues) == 0
        if not healthy:
            logger.warning("Exchange anomaly detected: %s", "; ".join(issues))
        return healthy, issues


class ConsecutiveLossBreaker:
    """Halts trading for a strategy after N consecutive losses for M hours.

    Parameters
    ----------
    max_consecutive:
        Number of consecutive losses to trigger halt (default 5).
    cooldown_hours:
        Duration of trading halt in hours (default 4).
    """

    def __init__(self, max_consecutive: int = 5, cooldown_hours: float = 4.0) -> None:
        self.max_consecutive = max_consecutive
        self.cooldown_seconds = cooldown_hours * 3600.0

    def check(self, strategy_id: str, count: int) -> Tuple[bool, Dict[str, Any]]:
        """Check if the consecutive-loss count trips this breaker.

        Returns
        -------
        (tripped, details)
        """
        if count >= self.max_consecutive:
            details = {
                "strategy_id": strategy_id,
                "consecutive_losses": count,
                "threshold": self.max_consecutive,
                "cooldown_hours": self.cooldown_seconds / 3600.0,
            }
            logger.warning(
                "ConsecutiveLossBreaker tripped: %s has %d losses (threshold %d)",
                strategy_id, count, self.max_consecutive,
            )
            return True, details
        return False, {}


# ---------------------------------------------------------------------------
# Main CircuitBreaker
# ---------------------------------------------------------------------------

class CircuitBreaker:
    """Aggregates all circuit breaker checks into a unified system.

    Parameters
    ----------
    config:
        Optional dict to override individual breaker defaults.  Keys:

        - ``flash_crash_pct`` (float): drop threshold (default 10.0)
        - ``flash_crash_minutes`` (int): window (default 5)
        - ``spread_warning_pct`` (float): default 0.5
        - ``spread_critical_pct`` (float): default 1.0
        - ``ws_latency_ms`` (int): default 5000
        - ``api_failures`` (int): default 3
        - ``balance_mismatch_pct`` (float): default 1.0
        - ``consecutive_losses`` (int): default 5
        - ``consecutive_loss_cooldown_hours`` (float): default 4
        - ``default_cooldown_seconds`` (float): system-wide default (default 1800)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}

        self._flash = FlashCrashBreaker(
            threshold_pct=cfg.get("flash_crash_pct", 10.0),
            window_minutes=cfg.get("flash_crash_minutes", 5),
        )
        self._spread = SpreadAnomalyBreaker(
            warning_pct=cfg.get("spread_warning_pct", 0.5),
            critical_pct=cfg.get("spread_critical_pct", 1.0),
        )
        self._exchange = ExchangeAnomalyBreaker(
            max_ws_latency_ms=cfg.get("ws_latency_ms", 5000),
            consecutive_api_failures=cfg.get("api_failures", 3),
            balance_mismatch_pct=cfg.get("balance_mismatch_pct", 1.0),
        )
        self._consec_loss = ConsecutiveLossBreaker(
            max_consecutive=cfg.get("consecutive_losses", 5),
            cooldown_hours=cfg.get("consecutive_loss_cooldown_hours", 4.0),
        )

        self._default_cooldown = cfg.get("default_cooldown_seconds", 1800.0)

        self._lock = threading.Lock()

        # Active cooldowns: key is strategy_id (or "__SYSTEM__")
        self._cooldowns: Dict[str, CooldownEntry] = {}

        # Trip history (most recent 100 trips)
        self._trip_history: List[BreakerTrip] = []
        self._max_history = 100

        logger.info("CircuitBreaker initialised with config: %s", cfg)

    # ======================================================================
    #  Price / spread / exchange checks
    # ======================================================================

    def check_price(
        self, symbol: str, prices_1m: List[float],
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Check for flash crash on 1-minute kline closes.

        Returns
        -------
        (tripped, breaker_name, details)
        """
        tripped, details = self._flash.check(symbol, prices_1m)
        if tripped:
            self._record_trip("FlashCrashBreaker", details)
            self.trigger_cooldown("__SYSTEM__", self._default_cooldown)
            return True, "FlashCrashBreaker", details
        return False, "", {}

    def check_spread(
        self, symbol: str, bid: float, ask: float, mid: float,
    ) -> Tuple[bool, str]:
        """Check bid-ask spread.

        Returns
        -------
        (tripped, level)
            *level* is ``"normal"``, ``"warning"``, or ``"critical"``.
        """
        tripped, level, details = self._spread.check(symbol, bid, ask, mid)
        if tripped:
            self._record_trip(
                "SpreadAnomalyBreaker",
                details,
            )
        return tripped, level

    def check_exchange_health(self) -> Tuple[bool, List[str]]:
        """Check exchange connectivity and consistency.

        Returns
        -------
        (healthy, issues)
        """
        healthy, issues = self._exchange.check()
        if not healthy:
            self._record_trip(
                "ExchangeAnomalyBreaker",
                {"issues": issues},
            )
            self.trigger_cooldown("__SYSTEM__", self._default_cooldown)
        return healthy, issues

    def check_consecutive_losses(
        self, strategy_id: str, count: int,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check consecutive losses for a specific strategy.

        Returns
        -------
        (tripped, details)
        """
        tripped, details = self._consec_loss.check(strategy_id, count)
        if tripped:
            self._record_trip(
                "ConsecutiveLossBreaker",
                details,
                scope=HaltScope.STRATEGY,
                strategy_id=strategy_id,
            )
            self.trigger_cooldown(strategy_id, self._consec_loss.cooldown_seconds)
        return tripped, details

    # ======================================================================
    #  Delegated recording methods (exchange anomaly)
    # ======================================================================

    def record_api_failure(self) -> None:
        """Record an API failure for exchange-health tracking."""
        self._exchange.record_api_failure()

    def record_api_success(self) -> None:
        """Record an API success for exchange-health tracking."""
        self._exchange.record_api_success()

    def record_ws_latency(self, latency_ms: int) -> None:
        """Record WebSocket latency for exchange-health tracking."""
        self._exchange.record_ws_latency(latency_ms)

    def record_balance_mismatch(self, mismatch_pct: float) -> None:
        """Record balance mismatch for exchange-health tracking."""
        self._exchange.record_balance_mismatch(mismatch_pct)

    # ======================================================================
    #  Halt / cooldown management
    # ======================================================================

    def is_halted(self, strategy_id: Optional[str] = None) -> bool:
        """Return True if trading is halted.

        Checks both system-wide and per-strategy cooldowns.
        """
        with self._lock:
            now = time.time()

            # System-wide halt
            system_cd = self._cooldowns.get("__SYSTEM__")
            if system_cd and now < system_cd.until:
                return True

            # Per-strategy halt
            if strategy_id:
                strat_cd = self._cooldowns.get(strategy_id)
                if strat_cd and now < strat_cd.until:
                    return True

        return False

    def get_cooldown_remaining(self, strategy_id: Optional[str] = None) -> float:
        """Return seconds remaining on the active cooldown (0 if not halted)."""
        with self._lock:
            now = time.time()
            remaining = 0.0

            system_cd = self._cooldowns.get("__SYSTEM__")
            if system_cd:
                remaining = max(remaining, system_cd.until - now)

            if strategy_id:
                strat_cd = self._cooldowns.get(strategy_id)
                if strat_cd:
                    remaining = max(remaining, strat_cd.until - now)

        return max(0.0, remaining)

    def trigger_cooldown(self, strategy_id: str, duration_seconds: float) -> None:
        """Activate a cooldown for *strategy_id* (or ``"__SYSTEM__"``).

        If an existing cooldown is longer, the longer one is kept.
        """
        with self._lock:
            now = time.time()
            new_until = now + duration_seconds
            existing = self._cooldowns.get(strategy_id)
            if existing and existing.until > new_until:
                return  # keep the longer cooldown

            self._cooldowns[strategy_id] = CooldownEntry(
                until=new_until,
                reason=f"cooldown triggered for {duration_seconds:.0f}s",
                breaker_name="manual",
            )
            scope = "SYSTEM" if strategy_id == "__SYSTEM__" else strategy_id
            system_logger.info(
                "CIRCUIT_BREAKER\tcooldown_set\tscope=%s\tduration=%.0fs",
                scope, duration_seconds,
            )

    def reset(self, strategy_id: Optional[str] = None) -> None:
        """Clear cooldown(s).

        If *strategy_id* is None, all cooldowns are cleared (system + all strategies).
        """
        with self._lock:
            if strategy_id is None:
                self._cooldowns.clear()
                logger.info("All circuit breaker cooldowns cleared")
            else:
                self._cooldowns.pop(strategy_id, None)
                self._cooldowns.pop("__SYSTEM__", None)
                logger.info("Circuit breaker cooldowns cleared for %s", strategy_id)

    # ======================================================================
    #  Trip history
    # ======================================================================

    def get_trip_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return the most recent breaker trips as dicts."""
        with self._lock:
            trips = self._trip_history[-limit:]
        return [
            {
                "breaker": t.breaker_name,
                "timestamp": t.timestamp,
                "details": t.details,
                "scope": t.scope.value,
                "strategy_id": t.strategy_id,
                "cooldown_active": t.is_active,
            }
            for t in reversed(trips)
        ]

    def _record_trip(
        self,
        breaker_name: str,
        details: Dict[str, Any],
        scope: HaltScope = HaltScope.SYSTEM,
        strategy_id: Optional[str] = None,
    ) -> None:
        trip = BreakerTrip(
            breaker_name=breaker_name,
            timestamp=time.time(),
            details=details,
            scope=scope,
            strategy_id=strategy_id,
        )
        with self._lock:
            self._trip_history.append(trip)
            if len(self._trip_history) > self._max_history:
                self._trip_history = self._trip_history[-self._max_history:]

        system_logger.info(
            "CIRCUIT_BREAKER\ttrip\tbreaker=%s\tscope=%s\tstrategy=%s\tdetails=%s",
            breaker_name, scope.value, strategy_id or "all", details,
        )
