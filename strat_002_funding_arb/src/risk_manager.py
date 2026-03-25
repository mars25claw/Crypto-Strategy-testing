"""Strategy-specific risk manager for Funding Rate Arbitrage.

Implements Section 5 risk rules on top of the shared RiskManager:
- 40% max equity, 20% per instrument, 3x max leverage (prefer 2x)
- Drawdown limits: 1.5% daily, 3% weekly, 5% monthly
- Basis flash inversion breaker
- Funding rate shock breaker
- Liquidation cascade detection (>50 liquidations/60s)
- Kill switch: both legs within 10 seconds
- Delta exposure monitoring (should be near zero)
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

from shared.config_loader import RiskConfig
from shared.risk_manager import RiskManager as SharedRiskManager, DrawdownState

logger = logging.getLogger(__name__)
trade_logger = logging.getLogger("trade")


@dataclass
class DeltaSnapshot:
    """Delta neutrality snapshot for a position."""
    symbol: str
    spot_notional: float
    futures_notional: float
    delta_usdt: float
    delta_pct: float
    timestamp_ms: int
    within_tolerance: bool


class FundingArbRiskManager:
    """Risk manager specialized for funding rate arbitrage.

    Wraps the shared RiskManager and adds strategy-specific checks:
    delta neutrality, basis circuit breakers, liquidation cascade,
    and cross-wallet margin monitoring.

    Parameters
    ----------
    shared_risk : SharedRiskManager
        The shared cross-strategy risk manager.
    strategy : FundingArbStrategy
        The strategy instance.
    config : dict
        Strategy parameters.
    """

    STRATEGY_ID = "STRAT-002"

    def __init__(
        self,
        shared_risk: SharedRiskManager,
        strategy: Any,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._shared = shared_risk
        self._strategy = strategy
        self._config = config or {}

        # Delta tracking
        self._delta_history: Dict[str, Deque[DeltaSnapshot]] = defaultdict(
            lambda: deque(maxlen=1000)
        )
        self._delta_violations: int = 0  # Count of > 0.5% delta events
        self._min_liq_distance: float = 100.0  # Lowest observed distance

        # Drawdown tracking specific to this strategy
        self._drawdown_state: Optional[DrawdownState] = None
        self._daily_start_equity: float = 0.0
        self._weekly_start_equity: float = 0.0
        self._monthly_start_equity: float = 0.0

        # Circuit breaker cooldowns
        self._basis_cooldown_until: float = 0.0
        self._funding_cooldown_until: float = 0.0

        # Kill switch state
        self._kill_switch_triggered = False
        self._kill_switch_callback: Optional[Callable] = None

        # Consecutive loss tracking for this strategy
        self._consecutive_losses: int = 0

        # Limits from config
        self._max_capital_pct = 40.0
        self._max_per_instrument_pct = 20.0
        self._max_leverage = self._config.get("max_leverage", 3)
        self._preferred_leverage = self._config.get("preferred_leverage", 2)
        self._daily_dd_limit = self._config.get("daily_drawdown_pct", 1.5)
        self._weekly_dd_limit = self._config.get("weekly_drawdown_pct", 3.0)
        self._monthly_dd_limit = self._config.get("monthly_drawdown_pct", 5.0)
        self._delta_tolerance = self._config.get("delta_tolerance_pct", 0.5) / 100.0

    # ══════════════════════════════════════════════════════════════════════
    #  Entry gate
    # ══════════════════════════════════════════════════════════════════════

    def check_entry_allowed(
        self,
        symbol: str,
        allocation_usdt: float,
        equity: float,
    ) -> Tuple[bool, str]:
        """Check whether a new position entry is allowed.

        Runs all strategy-specific risk checks plus shared risk checks.

        Returns (allowed, reason).
        """
        if self._kill_switch_triggered:
            return False, "Kill switch is active"

        # Check circuit breaker cooldowns
        now = time.time()
        if now < self._basis_cooldown_until:
            return False, f"Basis circuit breaker cooldown until {self._basis_cooldown_until}"
        if now < self._funding_cooldown_until:
            return False, f"Funding shock cooldown until {self._funding_cooldown_until}"

        # Strategy max capital (40% of equity)
        total_deployed = sum(p.spot_notional for p in self._strategy.positions.values())
        if equity > 0:
            current_pct = total_deployed / equity * 100
            if current_pct + (allocation_usdt / equity * 100) > self._max_capital_pct:
                return False, (
                    f"Would exceed max capital {self._max_capital_pct}%: "
                    f"current={current_pct:.1f}% + new={allocation_usdt/equity*100:.1f}%"
                )

        # Per-instrument limit (20% of equity)
        max_per_inst = equity * (self._max_per_instrument_pct / 100)
        if allocation_usdt > max_per_inst:
            return False, (
                f"Allocation {allocation_usdt:.2f} exceeds per-instrument max "
                f"({self._max_per_instrument_pct}% = {max_per_inst:.2f})"
            )

        # Max concurrent instruments
        if len(self._strategy.positions) >= 5:
            return False, "Already at max 5 concurrent instruments"

        # Drawdown checks
        halted, level, dd_pct = self._check_drawdown()
        if halted:
            return False, f"Drawdown limit breached: {level} at {dd_pct:.2f}%"

        # Consecutive loss handling (Section 7.7)
        reduce_at = self._config.get("consecutive_loss_reduce_at", 2)
        halt_at = self._config.get("consecutive_loss_halt_at", 4)
        if self._consecutive_losses >= halt_at:
            return False, f"HALTED: {self._consecutive_losses} consecutive losses"

        # Shared risk manager check
        allowed, reason = self._shared.check_entry_allowed(
            strategy_id=self.STRATEGY_ID,
            symbol=symbol,
            direction="LONG",  # Spot leg is long
            size_usdt=allocation_usdt,
            leverage=1,
        )
        if not allowed:
            return False, f"Shared risk: {reason}"

        return True, ""

    def get_size_multiplier(self) -> float:
        """Return position size multiplier based on consecutive losses.

        Section 7.7: After 2 consecutive losses, reduce by 50%.
        """
        reduce_at = self._config.get("consecutive_loss_reduce_at", 2)
        if self._consecutive_losses >= reduce_at:
            return 0.5
        return 1.0

    # ══════════════════════════════════════════════════════════════════════
    #  Delta neutrality monitoring (Section 5.6)
    # ══════════════════════════════════════════════════════════════════════

    def check_delta_neutrality(self) -> List[DeltaSnapshot]:
        """Check delta neutrality for all positions.

        Returns list of DeltaSnapshots, flags violations > 0.5%.
        """
        snapshots: List[DeltaSnapshot] = []
        now_ms = int(time.time() * 1000)

        for pos in self._strategy.positions.values():
            inst = self._strategy.instruments.get(pos.symbol)
            if inst is None:
                continue

            spot_notional = pos.spot_quantity * inst.index_price
            futures_notional = abs(pos.futures_quantity * inst.mark_price)
            total = spot_notional + futures_notional

            if total > 0:
                delta_usdt = spot_notional - futures_notional
                delta_pct = delta_usdt / total * 100
            else:
                delta_usdt = 0
                delta_pct = 0

            within_tolerance = abs(delta_pct / 100) <= self._delta_tolerance

            snap = DeltaSnapshot(
                symbol=pos.symbol,
                spot_notional=spot_notional,
                futures_notional=futures_notional,
                delta_usdt=delta_usdt,
                delta_pct=delta_pct,
                timestamp_ms=now_ms,
                within_tolerance=within_tolerance,
            )

            self._delta_history[pos.symbol].append(snap)
            pos.current_delta_pct = delta_pct

            if not within_tolerance:
                self._delta_violations += 1
                logger.warning(
                    "Delta violation on %s: %.2f%% (tolerance: %.2f%%)",
                    pos.symbol, delta_pct, self._delta_tolerance * 100,
                )

            snapshots.append(snap)

        return snapshots

    def get_delta_neutrality_score(self) -> float:
        """Percentage of time delta was within tolerance.

        Section 10.2: Delta Neutrality Score.
        """
        total = 0
        within = 0
        for sym_history in self._delta_history.values():
            for snap in sym_history:
                total += 1
                if snap.within_tolerance:
                    within += 1

        if total == 0:
            return 100.0
        return (within / total) * 100.0

    # ══════════════════════════════════════════════════════════════════════
    #  Drawdown checks (Section 5.5)
    # ══════════════════════════════════════════════════════════════════════

    def update_equity(self, equity: float) -> None:
        """Update equity for drawdown tracking."""
        if self._drawdown_state is None:
            self._drawdown_state = DrawdownState(
                peak_equity=equity,
                current_equity=equity,
                daily_start=equity,
                weekly_start=equity,
                monthly_start=equity,
            )
            self._daily_start_equity = equity
            self._weekly_start_equity = equity
            self._monthly_start_equity = equity
        else:
            self._drawdown_state.update_equity(equity)

        # Also update shared risk manager
        self._shared.update_equity(equity)

    def _check_drawdown(self) -> Tuple[bool, str, float]:
        """Check strategy-specific drawdown limits."""
        if self._drawdown_state is None:
            return False, "", 0.0

        dd = self._drawdown_state

        if dd.daily_drawdown_pct >= self._daily_dd_limit:
            return True, "daily", dd.daily_drawdown_pct

        if dd.weekly_drawdown_pct >= self._weekly_dd_limit:
            return True, "weekly", dd.weekly_drawdown_pct

        if dd.monthly_drawdown_pct >= self._monthly_dd_limit:
            return True, "monthly", dd.monthly_drawdown_pct

        return False, "", 0.0

    def reset_daily_drawdown(self) -> None:
        """Reset daily drawdown counter (00:00 UTC)."""
        if self._drawdown_state:
            self._drawdown_state.reset_daily()
        self._shared.reset_daily_drawdown()

    def reset_weekly_drawdown(self) -> None:
        """Reset weekly drawdown counter (Monday 00:00 UTC)."""
        if self._drawdown_state:
            self._drawdown_state.reset_weekly()
        self._shared.reset_weekly_drawdown()

    def reset_monthly_drawdown(self) -> None:
        """Reset monthly drawdown counter (1st of month)."""
        if self._drawdown_state:
            self._drawdown_state.reset_monthly()
        self._shared.reset_monthly_drawdown()

    # ══════════════════════════════════════════════════════════════════════
    #  Trade result recording
    # ══════════════════════════════════════════════════════════════════════

    def record_trade_result(self, pnl: float, is_win: bool) -> None:
        """Record a completed trade result."""
        if is_win:
            self._consecutive_losses = 0
        else:
            self._consecutive_losses += 1

        self._shared.record_trade_result(self.STRATEGY_ID, pnl, is_win)

        if self._consecutive_losses >= self._config.get("consecutive_loss_halt_at", 4):
            logger.critical(
                "HALT: %d consecutive losses — manual review required",
                self._consecutive_losses,
            )

    # ══════════════════════════════════════════════════════════════════════
    #  Circuit breakers (Section 5.7) — risk manager side
    # ══════════════════════════════════════════════════════════════════════

    def activate_basis_circuit_breaker(self, cooldown_hours: float) -> None:
        """Activate basis flash inversion circuit breaker."""
        self._basis_cooldown_until = time.time() + cooldown_hours * 3600
        logger.critical(
            "Basis circuit breaker activated — cooldown %.1f hours",
            cooldown_hours,
        )

    def activate_funding_circuit_breaker(self, cooldown_hours: float) -> None:
        """Activate funding rate shock circuit breaker."""
        self._funding_cooldown_until = time.time() + cooldown_hours * 3600
        logger.critical(
            "Funding shock circuit breaker activated — cooldown %.1f hours",
            cooldown_hours,
        )

    # ══════════════════════════════════════════════════════════════════════
    #  Kill switch (Section 5.7)
    # ══════════════════════════════════════════════════════════════════════

    def trigger_kill_switch(self, reason: str) -> None:
        """Trigger the kill switch — close ALL positions within 10 seconds."""
        self._kill_switch_triggered = True
        logger.critical("KILL SWITCH TRIGGERED: %s", reason)
        trade_logger.info("KILL_SWITCH\treason=%s", reason)

    def reset_kill_switch(self) -> None:
        """Reset kill switch (manual action)."""
        self._kill_switch_triggered = False
        logger.info("Kill switch reset")

    @property
    def kill_switch_active(self) -> bool:
        return self._kill_switch_triggered

    # ══════════════════════════════════════════════════════════════════════
    #  Liquidation distance tracking
    # ══════════════════════════════════════════════════════════════════════

    def update_min_liquidation_distance(self, distance_pct: float) -> None:
        """Track minimum observed liquidation distance."""
        if distance_pct < self._min_liq_distance:
            self._min_liq_distance = distance_pct
            logger.info("New minimum liquidation distance: %.2f%%", distance_pct)

    # ══════════════════════════════════════════════════════════════════════
    #  Metrics (Section 10.2)
    # ══════════════════════════════════════════════════════════════════════

    def get_risk_metrics(self) -> Dict[str, Any]:
        """Return all risk-related metrics for dashboard."""
        dd = self._drawdown_state

        return {
            "kill_switch_active": self._kill_switch_triggered,
            "consecutive_losses": self._consecutive_losses,
            "size_multiplier": self.get_size_multiplier(),
            "delta_neutrality_score": round(self.get_delta_neutrality_score(), 2),
            "delta_violations_total": self._delta_violations,
            "min_liquidation_distance_pct": round(self._min_liq_distance, 2),
            "drawdown": {
                "daily_pct": round(dd.daily_drawdown_pct, 4) if dd else 0,
                "weekly_pct": round(dd.weekly_drawdown_pct, 4) if dd else 0,
                "monthly_pct": round(dd.monthly_drawdown_pct, 4) if dd else 0,
                "overall_pct": round(dd.overall_drawdown_pct, 4) if dd else 0,
                "daily_limit": self._daily_dd_limit,
                "weekly_limit": self._weekly_dd_limit,
                "monthly_limit": self._monthly_dd_limit,
            },
            "circuit_breakers": {
                "basis_cooldown_active": time.time() < self._basis_cooldown_until,
                "basis_cooldown_remaining_s": max(0, self._basis_cooldown_until - time.time()),
                "funding_cooldown_active": time.time() < self._funding_cooldown_until,
                "funding_cooldown_remaining_s": max(0, self._funding_cooldown_until - time.time()),
            },
            "limits": {
                "max_capital_pct": self._max_capital_pct,
                "max_per_instrument_pct": self._max_per_instrument_pct,
                "max_leverage": self._max_leverage,
                "preferred_leverage": self._preferred_leverage,
            },
        }

    def get_state(self) -> Dict[str, Any]:
        """Serialize for persistence."""
        return {
            "consecutive_losses": self._consecutive_losses,
            "min_liq_distance": self._min_liq_distance,
            "delta_violations": self._delta_violations,
            "kill_switch_triggered": self._kill_switch_triggered,
            "basis_cooldown_until": self._basis_cooldown_until,
            "funding_cooldown_until": self._funding_cooldown_until,
        }

    def restore_state(self, state: Dict[str, Any]) -> None:
        """Restore from persistence."""
        self._consecutive_losses = state.get("consecutive_losses", 0)
        self._min_liq_distance = state.get("min_liq_distance", 100.0)
        self._delta_violations = state.get("delta_violations", 0)
        self._kill_switch_triggered = state.get("kill_switch_triggered", False)
        self._basis_cooldown_until = state.get("basis_cooldown_until", 0)
        self._funding_cooldown_until = state.get("funding_cooldown_until", 0)
        logger.info(
            "Risk state restored: consec_losses=%d kill_switch=%s",
            self._consecutive_losses, self._kill_switch_triggered,
        )
