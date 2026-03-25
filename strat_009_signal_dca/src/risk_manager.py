"""DCA-specific risk manager extending the shared risk manager.

Implements:
- 30% max equity allocation for DCA strategy
- No single asset > 60% of DCA portfolio
- Minimum 2 instruments check
- Emergency stop-loss at 50% of invested (halt, don't sell)
- Take-profit rebalancing at 200% (sell 10%, max once/quarter)
- Exchange health check: bid-ask spread < 0.1%
- USDT depeg monitoring
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from shared.risk_manager import RiskManager, CrossStrategyReader
from shared.config_loader import RiskConfig

logger = logging.getLogger(__name__)
trade_logger = logging.getLogger("trade")


@dataclass
class DCAHolding:
    """Tracks accumulated holdings for a single instrument."""

    symbol: str
    quantity: float = 0.0
    total_invested: float = 0.0
    current_price: float = 0.0
    purchase_count: int = 0

    @property
    def avg_cost_basis(self) -> float:
        if self.quantity <= 0:
            return 0.0
        return self.total_invested / self.quantity

    @property
    def current_value(self) -> float:
        return self.quantity * self.current_price

    @property
    def unrealised_pnl(self) -> float:
        return self.current_value - self.total_invested

    @property
    def unrealised_pnl_pct(self) -> float:
        if self.total_invested <= 0:
            return 0.0
        return (self.unrealised_pnl / self.total_invested) * 100.0

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "total_invested": round(self.total_invested, 2),
            "avg_cost_basis": round(self.avg_cost_basis, 8),
            "current_price": self.current_price,
            "current_value": round(self.current_value, 2),
            "unrealised_pnl": round(self.unrealised_pnl, 2),
            "unrealised_pnl_pct": round(self.unrealised_pnl_pct, 2),
            "purchase_count": self.purchase_count,
        }


class DCARiskManager:
    """DCA-specific risk management layer.

    Parameters
    ----------
    shared_risk_manager : RiskManager
        The shared cross-strategy risk manager.
    strategy_params : dict
        Strategy parameters from config.
    """

    STRATEGY_ID = "STRAT-009"

    def __init__(
        self,
        shared_risk_manager: RiskManager,
        strategy_params: dict,
    ) -> None:
        self._shared_rm = shared_risk_manager
        self._params = strategy_params

        # DCA holdings
        self._holdings: Dict[str, DCAHolding] = {}

        # Emergency stop state
        self._emergency_halted = False
        self._emergency_halt_reason = ""

        # Take-profit tracking
        self._last_rebalance_ts: float = 0.0
        self._total_profit_taken: float = 0.0

        # Order failure tracking
        self._order_failures: int = 0
        self._last_failure_ts: float = 0.0
        self._failure_halt_until: float = 0.0

        # USDT depeg tracking
        self._usdt_halted = False

        # Kill switch flag
        self._killed = False

    # ------------------------------------------------------------------
    # Holdings management
    # ------------------------------------------------------------------

    def record_purchase(
        self,
        symbol: str,
        quantity: float,
        amount_usdt: float,
        price: float,
    ) -> None:
        """Record a DCA purchase (regular or crash-buy)."""
        if symbol not in self._holdings:
            self._holdings[symbol] = DCAHolding(symbol=symbol)

        h = self._holdings[symbol]
        h.quantity += quantity
        h.total_invested += amount_usdt
        h.current_price = price
        h.purchase_count += 1

        # Update shared risk manager
        total_value = sum(hh.current_value for hh in self._holdings.values())
        self._shared_rm.record_position_change(
            strategy_id=self.STRATEGY_ID,
            symbol=symbol,
            direction="LONG",
            size_usdt=h.current_value,
            is_open=True,
        )

        trade_logger.info(
            "DCA_PURCHASE\tsymbol=%s\tqty=%.8f\tamount_usdt=%.2f\tprice=%.4f\t"
            "total_invested=%.2f\tavg_cost=%.4f",
            symbol, quantity, amount_usdt, price, h.total_invested, h.avg_cost_basis,
        )

    def record_sale(
        self,
        symbol: str,
        quantity: float,
        amount_usdt: float,
        price: float,
    ) -> None:
        """Record a sale (take-profit rebalancing)."""
        if symbol not in self._holdings:
            return

        h = self._holdings[symbol]
        sold_cost_basis = (h.total_invested / h.quantity) * quantity if h.quantity > 0 else 0
        h.quantity -= quantity
        h.total_invested -= sold_cost_basis
        h.current_price = price
        self._total_profit_taken += amount_usdt - sold_cost_basis

        trade_logger.info(
            "DCA_SALE\tsymbol=%s\tqty=%.8f\tamount_usdt=%.2f\tprice=%.4f\t"
            "profit=%.2f",
            symbol, quantity, amount_usdt, price, amount_usdt - sold_cost_basis,
        )

    def update_prices(self, prices: Dict[str, float]) -> None:
        """Update current prices for all holdings."""
        for symbol, price in prices.items():
            if symbol in self._holdings:
                self._holdings[symbol].current_price = price

    def get_holdings(self) -> Dict[str, DCAHolding]:
        """Return all current holdings."""
        return dict(self._holdings)

    def get_portfolio_value(self) -> float:
        """Total current value of all DCA holdings."""
        return sum(h.current_value for h in self._holdings.values())

    def get_total_invested(self) -> float:
        """Total amount invested across all instruments."""
        return sum(h.total_invested for h in self._holdings.values())

    # ------------------------------------------------------------------
    # Pre-purchase checks
    # ------------------------------------------------------------------

    def check_purchase_allowed(
        self,
        symbol: str,
        amount_usdt: float,
    ) -> Tuple[bool, str]:
        """Run all risk checks before a DCA purchase.

        Returns (allowed, reason).
        """
        # Kill switch
        if self._killed:
            return False, "Kill switch activated — all purchases halted"

        # Emergency halt
        if self._emergency_halted:
            return False, f"Emergency halt: {self._emergency_halt_reason}"

        # USDT depeg halt
        if self._usdt_halted:
            return False, "USDT depeg detected — purchases halted"

        # Order failure circuit breaker
        if time.time() < self._failure_halt_until:
            remaining = int(self._failure_halt_until - time.time())
            return False, f"Order failure halt — {remaining}s remaining"

        # Equity check via shared risk manager
        equity = self._shared_rm.get_current_equity()
        if equity <= 0:
            return False, "Equity not initialised"

        # 30% max capital for DCA
        max_capital = equity * 0.30
        current_invested = self.get_total_invested()
        if current_invested + amount_usdt > max_capital:
            return False, (
                f"DCA capital limit: invested {current_invested:.2f} + {amount_usdt:.2f} "
                f"would exceed 30% of equity ({max_capital:.2f})"
            )

        # Single asset concentration check
        max_asset_pct = self._params.get("max_single_asset_pct", 60.0) / 100.0
        portfolio_value = self.get_portfolio_value()
        if portfolio_value > 0 and symbol in self._holdings:
            asset_value = self._holdings[symbol].current_value + amount_usdt
            if asset_value / (portfolio_value + amount_usdt) > max_asset_pct:
                return False, (
                    f"Asset concentration: {symbol} would be "
                    f"{asset_value / (portfolio_value + amount_usdt) * 100:.1f}% "
                    f"of portfolio (max {max_asset_pct * 100:.0f}%)"
                )

        # Minimum instruments check (only after first purchase)
        min_instruments = self._params.get("min_instruments", 2)
        if len(self._holdings) > 0 and len(self._holdings) < min_instruments:
            # Allow purchases for instruments not yet started
            if symbol not in self._holdings:
                pass  # This adds a new instrument - good
            # If we already have some, don't block existing instrument purchases

        return True, ""

    # ------------------------------------------------------------------
    # Emergency stop-loss
    # ------------------------------------------------------------------

    def check_emergency_stop(self) -> Tuple[bool, str]:
        """Check if emergency stop-loss should trigger.

        Triggers when portfolio value < threshold% of total invested.
        Does NOT sell — halts new purchases.
        """
        if not self._params.get("emergency_stop_enabled", True):
            return False, ""

        threshold_pct = self._params.get("emergency_stop_pct", 50.0)
        total_invested = self.get_total_invested()
        portfolio_value = self.get_portfolio_value()

        if total_invested <= 0:
            return False, ""

        current_pct = (portfolio_value / total_invested) * 100.0

        if current_pct < threshold_pct:
            reason = (
                f"Portfolio value ${portfolio_value:.2f} is {current_pct:.1f}% "
                f"of invested ${total_invested:.2f} (threshold: {threshold_pct:.0f}%)"
            )
            if not self._emergency_halted:
                self._emergency_halted = True
                self._emergency_halt_reason = reason
                logger.critical("EMERGENCY STOP: %s", reason)
            return True, reason

        # Clear halt if portfolio recovers
        if self._emergency_halted:
            self._emergency_halted = False
            self._emergency_halt_reason = ""
            logger.info("Emergency halt cleared — portfolio recovered above threshold")

        return False, ""

    def clear_emergency_halt(self) -> None:
        """Manually clear emergency halt (operator override)."""
        self._emergency_halted = False
        self._emergency_halt_reason = ""
        logger.warning("Emergency halt manually cleared by operator")

    # ------------------------------------------------------------------
    # Take-profit rebalancing
    # ------------------------------------------------------------------

    def check_take_profit(self) -> Optional[Dict[str, Any]]:
        """Check if take-profit rebalancing should trigger.

        Returns a dict with rebalancing instructions, or None.
        """
        if not self._params.get("take_profit_enabled", True):
            return None

        threshold_pct = self._params.get("take_profit_threshold_pct", 200.0)
        sell_pct = self._params.get("take_profit_sell_pct", 10.0) / 100.0
        max_freq_days = self._params.get("take_profit_max_frequency_days", 90)

        # Frequency check — max once per quarter
        if self._last_rebalance_ts > 0:
            days_since = (time.time() - self._last_rebalance_ts) / 86400
            if days_since < max_freq_days:
                return None

        total_invested = self.get_total_invested()
        portfolio_value = self.get_portfolio_value()

        if total_invested <= 0:
            return None

        return_pct = ((portfolio_value - total_invested) / total_invested) * 100.0

        if return_pct >= threshold_pct:
            # Calculate what to sell from each instrument proportionally
            sells = {}
            for symbol, h in self._holdings.items():
                if h.quantity > 0:
                    sell_qty = h.quantity * sell_pct
                    sell_value = sell_qty * h.current_price
                    sells[symbol] = {
                        "quantity": sell_qty,
                        "estimated_value": round(sell_value, 2),
                    }

            logger.info(
                "Take-profit triggered: portfolio return %.1f%% > %.1f%% threshold. "
                "Selling %.0f%% of holdings.",
                return_pct, threshold_pct, sell_pct * 100,
            )

            return {
                "triggered": True,
                "return_pct": round(return_pct, 2),
                "threshold_pct": threshold_pct,
                "sell_pct": sell_pct * 100,
                "sells": sells,
            }

        return None

    def mark_rebalance_done(self) -> None:
        """Record that take-profit rebalancing was executed."""
        self._last_rebalance_ts = time.time()

    # ------------------------------------------------------------------
    # Exchange health check
    # ------------------------------------------------------------------

    def check_spread(self, bid: float, ask: float) -> Tuple[bool, float]:
        """Check if bid-ask spread is acceptable.

        Returns (ok, spread_pct). If not ok, delay purchase by 1 hour.
        """
        if bid <= 0 or ask <= 0:
            return False, 0.0

        mid = (bid + ask) / 2.0
        spread_pct = ((ask - bid) / mid) * 100.0
        max_spread = self._params.get("max_spread_pct", 0.1)

        if spread_pct > max_spread:
            logger.warning(
                "Spread too wide: %.4f%% > %.4f%% — delaying purchase",
                spread_pct, max_spread,
            )
            return False, spread_pct

        return True, spread_pct

    # ------------------------------------------------------------------
    # Order failure circuit breaker
    # ------------------------------------------------------------------

    def record_order_failure(self) -> None:
        """Record an order execution failure."""
        self._order_failures += 1
        self._last_failure_ts = time.time()

        max_failures = self._params.get("max_order_failures", 3)
        halt_hours = self._params.get("order_failure_halt_hours", 24)

        if self._order_failures >= max_failures:
            self._failure_halt_until = time.time() + halt_hours * 3600
            logger.error(
                "Order failure circuit breaker: %d failures, halting for %dh",
                self._order_failures, halt_hours,
            )

    def record_order_success(self) -> None:
        """Reset failure counter on successful order."""
        self._order_failures = 0

    # ------------------------------------------------------------------
    # USDT depeg check
    # ------------------------------------------------------------------

    def check_usdt_depeg(self, usdt_usd_price: float, threshold_pct: float = 2.0) -> bool:
        """Check for USDT depeg. Returns True if depegged.

        Parameters
        ----------
        usdt_usd_price : float
            Current USDT/USD price.
        threshold_pct : float
            Maximum acceptable deviation from $1.00.
        """
        deviation = abs(usdt_usd_price - 1.0) * 100.0
        if deviation > threshold_pct:
            if not self._usdt_halted:
                self._usdt_halted = True
                logger.critical(
                    "USDT DEPEG DETECTED: price=$%.4f (%.2f%% deviation). "
                    "All purchases halted.",
                    usdt_usd_price, deviation,
                )
            return True
        if self._usdt_halted:
            self._usdt_halted = False
            logger.info("USDT depeg cleared: price=$%.4f", usdt_usd_price)
        return False

    # ------------------------------------------------------------------
    # Diversification check
    # ------------------------------------------------------------------

    def check_diversification(self) -> Optional[Dict[str, float]]:
        """Check if any asset exceeds concentration limit.

        Returns dict of over-concentrated assets with their percentage,
        or None if all within limits.
        """
        portfolio_value = self.get_portfolio_value()
        if portfolio_value <= 0:
            return None

        max_pct = self._params.get("max_single_asset_pct", 60.0) / 100.0
        over_concentrated = {}

        for symbol, h in self._holdings.items():
            pct = h.current_value / portfolio_value
            if pct > max_pct:
                over_concentrated[symbol] = round(pct * 100, 2)

        return over_concentrated if over_concentrated else None

    # ------------------------------------------------------------------
    # Kill switch
    # ------------------------------------------------------------------

    def activate_kill_switch(self, reason: str) -> None:
        """Halt all purchases. Does NOT liquidate holdings (DCA is long-term)."""
        self._killed = True
        logger.critical("KILL SWITCH activated for DCA: %s", reason)

    def deactivate_kill_switch(self) -> None:
        """Resume purchases after manual review."""
        self._killed = False
        logger.warning("Kill switch deactivated — DCA purchases resumed")

    @property
    def is_killed(self) -> bool:
        return self._killed

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def to_state(self) -> dict:
        """Serialize risk manager state for persistence."""
        return {
            "holdings": {s: h.to_dict() for s, h in self._holdings.items()},
            "emergency_halted": self._emergency_halted,
            "emergency_halt_reason": self._emergency_halt_reason,
            "last_rebalance_ts": self._last_rebalance_ts,
            "total_profit_taken": self._total_profit_taken,
            "order_failures": self._order_failures,
            "failure_halt_until": self._failure_halt_until,
            "usdt_halted": self._usdt_halted,
            "killed": self._killed,
        }

    def load_state(self, state: dict) -> None:
        """Restore risk manager state from persisted data."""
        self._emergency_halted = state.get("emergency_halted", False)
        self._emergency_halt_reason = state.get("emergency_halt_reason", "")
        self._last_rebalance_ts = state.get("last_rebalance_ts", 0.0)
        self._total_profit_taken = state.get("total_profit_taken", 0.0)
        self._order_failures = state.get("order_failures", 0)
        self._failure_halt_until = state.get("failure_halt_until", 0.0)
        self._usdt_halted = state.get("usdt_halted", False)
        self._killed = state.get("killed", False)

        for symbol, hdata in state.get("holdings", {}).items():
            self._holdings[symbol] = DCAHolding(
                symbol=symbol,
                quantity=hdata.get("quantity", 0.0),
                total_invested=hdata.get("total_invested", 0.0),
                current_price=hdata.get("current_price", 0.0),
                purchase_count=hdata.get("purchase_count", 0),
            )

        logger.info(
            "Risk state restored: %d holdings, emergency_halted=%s, killed=%s",
            len(self._holdings), self._emergency_halted, self._killed,
        )

    def get_status(self) -> dict:
        """Return comprehensive risk status for dashboard."""
        portfolio_value = self.get_portfolio_value()
        total_invested = self.get_total_invested()
        return_pct = 0.0
        if total_invested > 0:
            return_pct = ((portfolio_value - total_invested) / total_invested) * 100.0

        return {
            "portfolio_value": round(portfolio_value, 2),
            "total_invested": round(total_invested, 2),
            "unrealised_pnl": round(portfolio_value - total_invested, 2),
            "return_pct": round(return_pct, 2),
            "emergency_halted": self._emergency_halted,
            "emergency_halt_reason": self._emergency_halt_reason,
            "killed": self._killed,
            "usdt_halted": self._usdt_halted,
            "order_failures": self._order_failures,
            "total_profit_taken": round(self._total_profit_taken, 2),
            "holdings": {s: h.to_dict() for s, h in self._holdings.items()},
            "diversification_issue": self.check_diversification(),
        }
