"""STRAT-005 Grid-Specific Risk Manager.

Extends shared risk management with grid-specific rules:
- 20% max equity, 10% per instrument (Section 5.1)
- Max 2x leverage (Section 6.2)
- 60% max inventory (Section 5.2)
- 3% equity unrealized loss halt (Section 5.3)
- 5% hard stop liquidate all (Section 5.4)
- Drawdown: 2%/4%/7% daily/weekly/monthly (Section 5.3)
- Circuit breakers: 8 consecutive levels, flash crash, volume spike (Section 5.5)
- Trend detection: ADX < 25 required (Section 7.1)
- STRAT-001 conflict check (Section 7.7)
- Whipsaw cooldown: 12 hours (Section 7.5)
- Consecutive loss: 3 cycles net loss -> 48 hour halt (Section 7.6)
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from shared.risk_manager import RiskManager, DrawdownState
from shared.cross_strategy import CrossStrategyManager

from .strategy import GridStrategy, InstrumentState, BreakoutDirection

logger = logging.getLogger(__name__)
trade_logger = logging.getLogger("trade")


class GridRiskManager:
    """Grid-specific risk layer on top of the shared RiskManager.

    Parameters
    ----------
    shared_risk : RiskManager
        The shared risk manager instance.
    strategy : GridStrategy
        The grid strategy logic instance.
    cross_strategy : CrossStrategyManager
        Cross-strategy position awareness.
    config : dict
        Strategy-specific configuration.
    """

    STRATEGY_ID = "STRAT-005"

    def __init__(
        self,
        shared_risk: RiskManager,
        strategy: GridStrategy,
        cross_strategy: CrossStrategyManager,
        config: dict,
    ) -> None:
        self._shared = shared_risk
        self._strategy = strategy
        self._cross = cross_strategy
        self._cfg = config

        # Capital limits
        self._max_capital_pct = config.get("max_capital_pct", 20.0)  # 20% of total equity
        self._max_per_instrument_pct = config.get("max_per_trade_pct", 10.0)  # 10% per instrument
        self._max_leverage = config.get("max_leverage", 2)
        self._preferred_leverage = config.get("preferred_leverage", 1)

        # Drawdown thresholds
        self._daily_dd_pct = config.get("daily_drawdown_pct", 2.0)
        self._weekly_dd_pct = config.get("weekly_drawdown_pct", 4.0)
        self._monthly_dd_pct = config.get("monthly_drawdown_pct", 7.0)

        # Grid-specific
        self._max_inventory_pct = config.get("max_inventory_pct", 60.0)
        self._unrealized_halt_pct = config.get("unrealized_loss_halt_pct", 3.0)
        self._hard_stop_pct = config.get("hard_stop_loss_pct", 5.0)

        # Circuit breaker tracking
        self._whipsaw_cooldown_hours = config.get("whipsaw_cooldown_hours", 12.0)
        self._consec_loss_halt_hours = config.get("consecutive_loss_halt_hours", 48.0)

        # Halt tracking per instrument
        self._halt_until: Dict[str, float] = {}  # symbol -> timestamp
        self._halt_reasons: Dict[str, str] = {}

        # Track total equity
        self._total_equity: float = 0.0

    # ======================================================================
    #  Equity
    # ======================================================================

    def update_equity(self, equity: float) -> None:
        """Update current total equity."""
        self._total_equity = equity
        self._shared.update_equity(equity)

    def get_equity(self) -> float:
        return self._total_equity

    # ======================================================================
    #  Capital Allocation
    # ======================================================================

    def calculate_capital_per_instrument(self, num_instruments: int) -> float:
        """Calculate capital allocation per instrument.

        Section 5.1: max 20% total equity, max 10% per instrument.
        """
        if self._total_equity <= 0 or num_instruments <= 0:
            return 0.0

        max_total = self._total_equity * (self._max_capital_pct / 100.0)
        max_per = self._total_equity * (self._max_per_instrument_pct / 100.0)
        per_instrument = max_total / num_instruments

        return min(per_instrument, max_per)

    def get_leverage(self) -> int:
        """Return the leverage to use. Section 6.2: prefer 1x, max 2x."""
        return self._preferred_leverage

    # ======================================================================
    #  Pre-Deployment Checks
    # ======================================================================

    def check_deployment_allowed(
        self,
        symbol: str,
        allocated_capital: float,
    ) -> Tuple[bool, str]:
        """Run all pre-deployment risk checks.

        Returns (allowed, reason).
        """
        # Check shared risk manager
        allowed, reason = self._shared.check_entry_allowed(
            strategy_id=self.STRATEGY_ID,
            symbol=symbol,
            direction="LONG",
            size_usdt=allocated_capital,
            leverage=self._preferred_leverage,
        )
        if not allowed:
            return False, f"Shared risk: {reason}"

        # Check drawdown
        halted, level, dd_pct = self._shared.check_drawdown()
        if halted:
            return False, f"Drawdown halt: {level} {dd_pct:.2f}%"

        # Check per-instrument halt
        if self.is_instrument_halted(symbol):
            reason = self._halt_reasons.get(symbol, "unknown")
            return False, f"Instrument halted: {reason}"

        # Check STRAT-001 conflict
        all_positions = self._cross.read_all_positions()
        cross_positions = []
        for sid, positions in all_positions.items():
            if sid != self.STRATEGY_ID:
                cross_positions.extend(positions)

        conflict, reason = self._strategy.check_strat001_conflict(symbol, cross_positions)
        if conflict:
            return False, reason

        # Check trend filter
        allowed, reason = self._strategy.check_trend_filter(symbol)
        if not allowed:
            return False, reason

        # Check volatility filter
        allowed, reason = self._strategy.check_volatility_filter(symbol)
        if not allowed:
            return False, reason

        return True, ""

    # ======================================================================
    #  Runtime Risk Checks
    # ======================================================================

    def check_runtime_risks(
        self,
        state: InstrumentState,
    ) -> Tuple[str, str]:
        """Run runtime risk checks on an active grid.

        Returns (action, reason) where action is one of:
        - "" (no action needed)
        - "cancel_buys" (Section 5.3)
        - "liquidate" (Section 5.4)
        - "halt_grid" (circuit breakers)
        - "halt_grid_trend" (trend detected)
        """
        symbol = state.symbol

        # Section 5.4: Hard stop — unrealized loss > 5% equity
        halted, reason = self._strategy.check_hard_stop(state, self._total_equity)
        if halted:
            return "liquidate", reason

        # Section 5.3: Unrealized loss > 3% equity — cancel buys
        halted, reason = self._strategy.check_unrealized_loss_halt(state, self._total_equity)
        if halted:
            return "cancel_buys", reason

        # Section 5.5: Consecutive levels circuit breaker
        halted, reason = self._strategy.check_consecutive_levels(state)
        if halted:
            self.halt_instrument(symbol, reason, 4 * 3600)  # 4 hours
            return "halt_grid", reason

        # Section 5.5: Flash crash
        halted, reason = self._strategy.check_flash_crash(symbol)
        if halted:
            return "cancel_buys", reason  # Cancel buys, keep inventory with hard stop

        # Section 5.5: Volume spike
        halted, reason = self._strategy.check_volume_spike(state)
        if halted:
            self.halt_instrument(symbol, reason, 3600)  # 1 hour
            return "halt_grid", reason

        # Section 5.2: Inventory exposure > 60%
        halted, reason = self._strategy.check_inventory_exposure(state)
        if halted:
            return "cancel_buys", reason

        # Section 7.1: Trend detection on active grid
        allowed, reason = self._strategy.check_trend_filter(symbol)
        if not allowed:
            return "halt_grid_trend", reason

        # Drawdown checks
        halted, level, dd_pct = self._shared.check_drawdown()
        if halted:
            return "halt_grid", f"Drawdown: {level} {dd_pct:.2f}%"

        return "", ""

    # ======================================================================
    #  Instrument Halt Management
    # ======================================================================

    def halt_instrument(self, symbol: str, reason: str, duration_s: float) -> None:
        """Halt an instrument for a specified duration."""
        self._halt_until[symbol] = time.time() + duration_s
        self._halt_reasons[symbol] = reason
        logger.warning(
            "[%s] HALTED for %.0fs: %s", symbol, duration_s, reason,
        )
        trade_logger.info("HALT\t%s\tduration=%.0fs\treason=%s", symbol, duration_s, reason)

    def is_instrument_halted(self, symbol: str) -> bool:
        """Check if an instrument is currently halted."""
        until = self._halt_until.get(symbol, 0)
        if until > time.time():
            return True
        # Clean up expired halts
        if symbol in self._halt_until:
            del self._halt_until[symbol]
            self._halt_reasons.pop(symbol, None)
        return False

    def halt_after_shutdown(self, symbol: str) -> None:
        """Section 7.5: Whipsaw protection — 12 hour cooldown after grid shutdown."""
        self.halt_instrument(
            symbol,
            "whipsaw_cooldown",
            self._whipsaw_cooldown_hours * 3600,
        )

    def halt_consecutive_losses(self, symbol: str) -> None:
        """Section 7.6: 3 consecutive net-loss cycles -> 48 hour halt."""
        self.halt_instrument(
            symbol,
            "consecutive_loss_halt",
            self._consec_loss_halt_hours * 3600,
        )

    # ======================================================================
    #  Position Reporting for Cross-Strategy
    # ======================================================================

    def report_positions(self, instruments: Dict[str, InstrumentState]) -> None:
        """Write current positions to the shared directory for cross-strategy awareness."""
        positions = []
        for symbol, state in instruments.items():
            if state.inventory_qty > 0:
                positions.append({
                    "symbol": symbol,
                    "direction": "LONG",
                    "size_usdt": state.inventory_qty * state.inventory_avg_cost,
                    "entry_price": state.inventory_avg_cost,
                    "strategy_id": self.STRATEGY_ID,
                    "timestamp_ms": int(time.time() * 1000),
                })

        self._cross.write_positions(positions)

    # ======================================================================
    #  Summary
    # ======================================================================

    def get_risk_summary(self, instruments: Dict[str, InstrumentState]) -> dict:
        """Get a summary of current risk state for the dashboard."""
        total_inventory_value = 0.0
        total_unrealized = 0.0

        for sym, state in instruments.items():
            if state.inventory_qty > 0:
                total_inventory_value += state.inventory_qty * state.inventory_avg_cost
                total_unrealized += state.unrealized_pnl

        equity = self._total_equity if self._total_equity > 0 else 1.0
        inventory_pct = total_inventory_value / equity * 100.0
        unrealized_pct = total_unrealized / equity * 100.0

        halted_instruments = {
            sym: self._halt_reasons.get(sym, "unknown")
            for sym in self._halt_until
            if self.is_instrument_halted(sym)
        }

        return {
            "total_equity": self._total_equity,
            "total_inventory_value": round(total_inventory_value, 2),
            "inventory_pct_of_equity": round(inventory_pct, 2),
            "total_unrealized_pnl": round(total_unrealized, 4),
            "unrealized_pct_of_equity": round(unrealized_pct, 4),
            "max_capital_pct": self._max_capital_pct,
            "max_per_instrument_pct": self._max_per_instrument_pct,
            "leverage": self._preferred_leverage,
            "halted_instruments": halted_instruments,
            "drawdown": self._shared.check_drawdown(),
        }
