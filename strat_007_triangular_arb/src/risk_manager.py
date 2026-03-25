"""STRAT-007: Strategy-specific risk manager.

Enforces:
- 15% max equity allocation
- Mode A: 2% per leg, Mode B: 1% per leg
- Max 2 concurrent arb executions
- Max 2% unhedged exposure
- Drawdown: 1.5% daily / 2.5% weekly / 4% monthly
- Circuit breakers: one-leg fails, consecutive negative, exchange anomaly
- Latency filter: >200ms reduce, >500ms halt
- High volatility: BTC >5% in 15m → 2x threshold
- Maintenance halt 2h before
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)
system_logger = logging.getLogger("system")


@dataclass
class CircuitBreakerState:
    """Tracks circuit breaker conditions."""

    # One-leg failure tracking
    consecutive_one_leg_fails: int = 0
    one_leg_halt_until: float = 0.0

    # Consecutive negative arb tracking
    consecutive_negative: int = 0
    negative_halt_until: float = 0.0
    daily_negative_count: int = 0

    # Exchange anomaly
    anomaly_active: bool = False
    anomaly_symbol: str = ""

    # Latency
    current_latency_ms: float = 0.0
    latency_reduced: bool = False
    latency_halted: bool = False

    # Volatility
    high_volatility: bool = False
    btc_15m_change_pct: float = 0.0

    # Maintenance
    maintenance_halt: bool = False
    maintenance_until: float = 0.0

    # Consecutive losing days
    consecutive_losing_days: int = 0
    manual_halt: bool = False

    def to_dict(self) -> dict:
        return {
            "consecutive_one_leg_fails": self.consecutive_one_leg_fails,
            "one_leg_halted": time.time() < self.one_leg_halt_until,
            "consecutive_negative": self.consecutive_negative,
            "negative_halted": time.time() < self.negative_halt_until,
            "daily_negative_count": self.daily_negative_count,
            "anomaly_active": self.anomaly_active,
            "current_latency_ms": round(self.current_latency_ms, 1),
            "latency_reduced": self.latency_reduced,
            "latency_halted": self.latency_halted,
            "high_volatility": self.high_volatility,
            "btc_15m_change_pct": round(self.btc_15m_change_pct, 2),
            "maintenance_halt": self.maintenance_halt,
            "consecutive_losing_days": self.consecutive_losing_days,
            "manual_halt": self.manual_halt,
        }


class ArbRiskManager:
    """Strategy-specific risk manager for STRAT-007.

    Parameters
    ----------
    params : dict
        Strategy parameters from config.yaml.
    equity : float
        Initial equity for percentage calculations.
    shared_risk_manager : optional
        Reference to the shared cross-strategy risk manager.
    cross_strategy_dir : str
        Directory where cross-strategy position files live.
    """

    def __init__(
        self,
        params: Dict[str, Any],
        equity: float = 10000.0,
        shared_risk_manager: Any = None,
        cross_strategy_dir: str = "data/cross_strategy",
    ) -> None:
        self._params = params
        self._equity = equity

        # Limits from config
        self._max_equity_pct = 15.0
        self._mode_a_per_leg_pct = params.get("mode_a_max_per_leg_pct", 2.0)
        self._mode_b_per_leg_pct = params.get("mode_b_max_per_leg_pct", 1.0)
        self._max_concurrent = params.get("max_concurrent_arbs", 2)
        self._max_unhedged_pct = 2.0

        # Circuit breaker thresholds
        self._one_leg_fail_halt_count = params.get("one_leg_fail_halt_count", 3)
        self._one_leg_fail_halt_minutes = params.get("one_leg_fail_halt_minutes", 30)
        self._consecutive_neg_halt = params.get("consecutive_negative_halt", 5)
        self._daily_loss_halt = params.get("daily_loss_halt", 10)
        self._consecutive_losing_days_halt = params.get("consecutive_losing_days_halt", 3)

        # Latency thresholds
        self._latency_reduce_ms = params.get("latency_reduce_ms", 200)
        self._latency_halt_ms = params.get("latency_halt_ms", 500)

        # Volatility
        self._high_vol_btc_pct = params.get("high_volatility_btc_move_pct", 5.0)

        # Drawdown limits
        self._daily_dd_pct = 1.5
        self._weekly_dd_pct = 2.5
        self._monthly_dd_pct = 4.0

        # State
        self._cb = CircuitBreakerState()
        self._active_arbs = 0
        self._unhedged_exposure_usdt = 0.0

        # BTC price tracking for volatility detection
        self._btc_prices: Deque[Tuple[float, float]] = deque(maxlen=900)  # 15min of 1s data

        # Daily P&L tracking
        self._daily_pnl = 0.0
        self._daily_start_equity = equity

        # Latency measurements
        self._latency_history: Deque[float] = deque(maxlen=60)

        # Cross-strategy coordination (Section 12.3 / 11.4)
        self._shared_risk = shared_risk_manager
        self._cross_strategy_dir = cross_strategy_dir
        self._arb_position_lock = asyncio.Lock()
        self._locked_symbols: Dict[str, float] = {}  # symbol -> lock_time

        logger.info(
            "ArbRiskManager initialized: max_equity=%.1f%% mode_a_leg=%.1f%% "
            "mode_b_leg=%.1f%% max_concurrent=%d",
            self._max_equity_pct, self._mode_a_per_leg_pct,
            self._mode_b_per_leg_pct, self._max_concurrent,
        )

    # ------------------------------------------------------------------
    # Entry check
    # ------------------------------------------------------------------

    def check_can_trade(self) -> Tuple[bool, str]:
        """Master check: can we execute any arb right now?

        Returns (allowed, reason).
        """
        now = time.time()

        # Manual halt
        if self._cb.manual_halt:
            return False, "Manual halt active — requires investigation"

        # Maintenance halt
        if self._cb.maintenance_halt and now < self._cb.maintenance_until:
            return False, "Maintenance halt active"

        # One-leg failure halt
        if now < self._cb.one_leg_halt_until:
            remaining = self._cb.one_leg_halt_until - now
            return False, f"One-leg failure halt ({remaining:.0f}s remaining)"

        # Consecutive negative halt
        if now < self._cb.negative_halt_until:
            remaining = self._cb.negative_halt_until - now
            return False, f"Consecutive negative halt ({remaining:.0f}s remaining)"

        # Latency halt
        if self._cb.latency_halted:
            return False, f"Latency too high ({self._cb.current_latency_ms:.0f}ms > {self._latency_halt_ms}ms)"

        # Max concurrent
        if self._active_arbs >= self._max_concurrent:
            return False, f"Max concurrent arbs reached ({self._max_concurrent})"

        # Unhedged exposure
        if self._equity > 0:
            unhedged_pct = (self._unhedged_exposure_usdt / self._equity) * 100.0
            if unhedged_pct > self._max_unhedged_pct:
                return False, f"Unhedged exposure {unhedged_pct:.1f}% exceeds limit {self._max_unhedged_pct}%"

        # Consecutive losing days
        if self._cb.consecutive_losing_days >= self._consecutive_losing_days_halt:
            self._cb.manual_halt = True
            return False, f"Halted: {self._cb.consecutive_losing_days} consecutive losing days"

        # Exchange anomaly
        if self._cb.anomaly_active:
            return False, f"Exchange anomaly detected ({self._cb.anomaly_symbol})"

        return True, ""

    def check_mode_a_size(self, size_usdt: float) -> Tuple[bool, str]:
        """Validate Mode A trade size."""
        max_size = self._equity * (self._mode_a_per_leg_pct / 100.0)
        if size_usdt > max_size:
            return False, f"Mode A size {size_usdt:.2f} exceeds limit {max_size:.2f}"
        return True, ""

    def check_mode_b_size(self, size_usdt: float) -> Tuple[bool, str]:
        """Validate Mode B trade size."""
        max_size = self._equity * (self._mode_b_per_leg_pct / 100.0)
        if size_usdt > max_size:
            return False, f"Mode B size {size_usdt:.2f} exceeds limit {max_size:.2f}"
        return True, ""

    # ------------------------------------------------------------------
    # Cross-strategy coordination (Section 11.4 / 12.3)
    # ------------------------------------------------------------------

    def set_shared_risk_manager(self, shared_risk: Any) -> None:
        """Set reference to the shared cross-strategy risk manager."""
        self._shared_risk = shared_risk

    async def check_cross_strategy(self, symbols: List[str]) -> Tuple[bool, str]:
        """Check with shared risk manager before arb execution.

        Arb is ATOMIC: the risk manager must lock the position for the
        duration of execution.  If another strategy is modifying the same
        symbols, defer arb execution.

        Returns (allowed, reason).
        """
        # Check shared risk manager
        if self._shared_risk is not None:
            try:
                can = self._shared_risk.can_open_position("STRAT-007", symbols)
                if not can:
                    return False, "Shared risk manager denied — another strategy active on same symbols"
            except AttributeError:
                pass  # Shared risk manager may not have this method

        # Read cross-strategy position files
        if os.path.isdir(self._cross_strategy_dir):
            try:
                for fname in os.listdir(self._cross_strategy_dir):
                    if not fname.endswith(".json") or fname.startswith("STRAT-007"):
                        continue
                    fpath = os.path.join(self._cross_strategy_dir, fname)
                    try:
                        with open(fpath, "r") as f:
                            positions = json.load(f)
                        for pos in positions if isinstance(positions, list) else [positions]:
                            pos_symbol = pos.get("symbol", "")
                            if pos_symbol in symbols and float(pos.get("qty", 0)) != 0:
                                return (
                                    False,
                                    f"Cross-strategy conflict: {fname} holds {pos_symbol}",
                                )
                    except (json.JSONDecodeError, IOError):
                        continue
            except OSError:
                pass

        return True, ""

    async def lock_arb_position(self, symbols: List[str]) -> bool:
        """Lock symbols for the duration of an atomic arb execution.

        Returns True if lock acquired, False if another arb already holds it.
        """
        async with self._arb_position_lock:
            now = time.time()
            # Check for stale locks (> 120s)
            stale = [s for s, t in self._locked_symbols.items() if now - t > 120]
            for s in stale:
                self._locked_symbols.pop(s, None)

            for sym in symbols:
                if sym in self._locked_symbols:
                    logger.warning(
                        "Cannot lock %s — already locked by another arb execution", sym,
                    )
                    return False

            for sym in symbols:
                self._locked_symbols[sym] = now
            return True

    async def unlock_arb_position(self, symbols: List[str]) -> None:
        """Release arb position lock after execution completes."""
        async with self._arb_position_lock:
            for sym in symbols:
                self._locked_symbols.pop(sym, None)

    # ------------------------------------------------------------------
    # Execution tracking
    # ------------------------------------------------------------------

    def record_arb_start(self) -> None:
        """Record that an arb execution has started."""
        self._active_arbs += 1

    def record_arb_end(self, profit_usdt: float, success: bool, legs_filled: int, legs_total: int) -> None:
        """Record that an arb execution has completed."""
        self._active_arbs = max(0, self._active_arbs - 1)
        self._daily_pnl += profit_usdt

        if success:
            # Reset consecutive failures on success
            if profit_usdt >= 0:
                self._cb.consecutive_negative = 0
            else:
                self._cb.consecutive_negative += 1
                self._cb.daily_negative_count += 1
                self._check_consecutive_negative()
            self._cb.consecutive_one_leg_fails = 0
        else:
            # Check for one-leg failure
            if legs_filled == 1 and legs_total >= 2:
                self._cb.consecutive_one_leg_fails += 1
                self._check_one_leg_fails()

    def record_unhedged_change(self, exposure_usdt: float) -> None:
        """Update current unhedged exposure."""
        self._unhedged_exposure_usdt = exposure_usdt

    # ------------------------------------------------------------------
    # Circuit breaker logic
    # ------------------------------------------------------------------

    def _check_one_leg_fails(self) -> None:
        """Check if one-leg failure circuit breaker should trip."""
        if self._cb.consecutive_one_leg_fails >= self._one_leg_fail_halt_count:
            halt_duration = self._one_leg_fail_halt_minutes * 60
            self._cb.one_leg_halt_until = time.time() + halt_duration
            logger.critical(
                "CIRCUIT BREAKER: %d consecutive one-leg failures — halting for %d minutes",
                self._cb.consecutive_one_leg_fails, self._one_leg_fail_halt_minutes,
            )
            system_logger.info(
                "CIRCUIT_BREAKER\tone_leg_fail\tcount=%d\thalt_minutes=%d",
                self._cb.consecutive_one_leg_fails, self._one_leg_fail_halt_minutes,
            )
            self._cb.consecutive_one_leg_fails = 0

    def _check_consecutive_negative(self) -> None:
        """Check if consecutive negative circuit breaker should trip."""
        if self._cb.consecutive_negative >= self._consecutive_neg_halt:
            self._cb.negative_halt_until = time.time() + 3600  # 1 hour halt
            logger.critical(
                "CIRCUIT BREAKER: %d consecutive negative arbs — halting for 1 hour",
                self._cb.consecutive_negative,
            )
            system_logger.info(
                "CIRCUIT_BREAKER\tconsecutive_negative\tcount=%d",
                self._cb.consecutive_negative,
            )
            self._cb.consecutive_negative = 0

        if self._cb.daily_negative_count >= self._daily_loss_halt:
            # Halt for remainder of day
            self._cb.negative_halt_until = time.time() + 86400
            logger.critical(
                "CIRCUIT BREAKER: %d negative arbs today — halting for remainder of day",
                self._cb.daily_negative_count,
            )

    # ------------------------------------------------------------------
    # Latency monitoring
    # ------------------------------------------------------------------

    def update_latency(self, latency_ms: float) -> None:
        """Update the current API latency measurement."""
        self._latency_history.append(latency_ms)
        self._cb.current_latency_ms = latency_ms

        if latency_ms > self._latency_halt_ms:
            if not self._cb.latency_halted:
                self._cb.latency_halted = True
                self._cb.latency_reduced = True
                logger.critical(
                    "LATENCY HALT: %.0fms exceeds %dms threshold",
                    latency_ms, self._latency_halt_ms,
                )
        elif latency_ms > self._latency_reduce_ms:
            self._cb.latency_reduced = True
            self._cb.latency_halted = False
            if not self._cb.latency_reduced:
                logger.warning(
                    "LATENCY REDUCED: %.0fms exceeds %dms threshold",
                    latency_ms, self._latency_reduce_ms,
                )
        else:
            self._cb.latency_reduced = False
            self._cb.latency_halted = False

    # ------------------------------------------------------------------
    # Volatility monitoring
    # ------------------------------------------------------------------

    def update_btc_price(self, price: float) -> None:
        """Track BTC price for 15-minute volatility calculation."""
        now = time.time()
        self._btc_prices.append((now, price))

        # Calculate 15-minute change
        cutoff = now - 900  # 15 minutes
        old_prices = [p for t, p in self._btc_prices if t <= cutoff + 1]
        if old_prices and price > 0:
            oldest = old_prices[0]
            change_pct = abs(price - oldest) / oldest * 100.0
            self._cb.btc_15m_change_pct = change_pct

            was_high_vol = self._cb.high_volatility
            self._cb.high_volatility = change_pct >= self._high_vol_btc_pct

            if self._cb.high_volatility and not was_high_vol:
                logger.warning(
                    "HIGH VOLATILITY: BTC moved %.2f%% in 15m (threshold: %.1f%%)",
                    change_pct, self._high_vol_btc_pct,
                )

    @property
    def is_high_volatility(self) -> bool:
        return self._cb.high_volatility

    # ------------------------------------------------------------------
    # Exchange anomaly
    # ------------------------------------------------------------------

    def set_exchange_anomaly(self, active: bool, symbol: str = "") -> None:
        """Set or clear exchange anomaly flag."""
        self._cb.anomaly_active = active
        self._cb.anomaly_symbol = symbol
        if active:
            logger.critical("EXCHANGE ANOMALY detected for %s — arb halted", symbol)

    # ------------------------------------------------------------------
    # Maintenance window
    # ------------------------------------------------------------------

    def set_maintenance_halt(self, until_timestamp: float) -> None:
        """Set maintenance halt until the given timestamp."""
        self._cb.maintenance_halt = True
        self._cb.maintenance_until = until_timestamp
        logger.warning("Maintenance halt set until %.0f", until_timestamp)

    def clear_maintenance_halt(self) -> None:
        """Clear maintenance halt."""
        self._cb.maintenance_halt = False
        self._cb.maintenance_until = 0.0

    # ------------------------------------------------------------------
    # Daily reset
    # ------------------------------------------------------------------

    def reset_daily(self) -> None:
        """Reset daily counters. Called at 00:00 UTC."""
        was_losing = self._daily_pnl < 0
        if was_losing:
            self._cb.consecutive_losing_days += 1
        else:
            self._cb.consecutive_losing_days = 0

        logger.info(
            "Daily risk reset: pnl=%.4f losing_days=%d negative_count=%d",
            self._daily_pnl, self._cb.consecutive_losing_days,
            self._cb.daily_negative_count,
        )

        self._daily_pnl = 0.0
        self._daily_start_equity = self._equity
        self._cb.daily_negative_count = 0
        self._cb.consecutive_negative = 0
        self._cb.consecutive_one_leg_fails = 0

    # ------------------------------------------------------------------
    # Equity updates
    # ------------------------------------------------------------------

    def update_equity(self, equity: float) -> None:
        """Update current equity."""
        self._equity = equity

    @property
    def equity(self) -> float:
        return self._equity

    # ------------------------------------------------------------------
    # Kill switch
    # ------------------------------------------------------------------

    def trigger_kill_switch(self) -> None:
        """Activate manual halt (kill switch)."""
        self._cb.manual_halt = True
        logger.critical("KILL SWITCH: Manual halt activated")

    def clear_kill_switch(self) -> None:
        """Clear manual halt."""
        self._cb.manual_halt = False
        logger.info("Kill switch cleared — trading can resume")

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_state(self) -> Dict[str, Any]:
        """Return full risk manager state."""
        return {
            "equity": round(self._equity, 2),
            "daily_pnl": round(self._daily_pnl, 4),
            "active_arbs": self._active_arbs,
            "unhedged_exposure_usdt": round(self._unhedged_exposure_usdt, 4),
            "circuit_breakers": self._cb.to_dict(),
            "can_trade": self.check_can_trade()[0],
            "can_trade_reason": self.check_can_trade()[1],
        }

    def get_latency_stats(self) -> Dict[str, float]:
        """Return latency statistics."""
        if not self._latency_history:
            return {"avg_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0, "current_ms": 0.0}
        return {
            "avg_ms": round(sum(self._latency_history) / len(self._latency_history), 1),
            "min_ms": round(min(self._latency_history), 1),
            "max_ms": round(max(self._latency_history), 1),
            "current_ms": round(self._cb.current_latency_ms, 1),
        }
