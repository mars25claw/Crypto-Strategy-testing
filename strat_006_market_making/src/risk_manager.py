"""Market Making Risk Manager for STRAT-006.

Enforces MM-specific risk rules on top of the shared RiskManager:
- 15% max equity, 7% per instrument, max 3 instruments
- Inventory limits: 2% warn, 3% critical, 4% emergency
- Drawdown: 1%/2%/4% (daily/weekly/monthly)
- Quote staleness > 60s -> health check
- Inventory blowout detection (LONG <-> SHORT in 1h)
- Daily PnL negative -> halt
- Volatility spike / large order / spread collapse withdrawal
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class InventoryState:
    """Tracks inventory state and phase for a single instrument."""
    symbol: str
    qty: float = 0.0
    avg_cost: float = 0.0
    notional: float = 0.0
    pct_of_equity: float = 0.0

    # Inventory direction history for blowout detection
    # List of (direction, timestamp): "LONG", "SHORT", "FLAT"
    direction_history: Deque[Tuple[str, float]] = field(
        default_factory=lambda: deque(maxlen=200)
    )

    # Track peak inventory levels with timestamps for blowout detection
    # (pct_of_max, timestamp) — records when inventory reaches warn/critical thresholds
    peak_inventory_events: Deque[Tuple[str, float, float]] = field(
        default_factory=lambda: deque(maxlen=100)
    )  # (direction "LONG"/"SHORT", pct_of_max, timestamp)

    # Liquidation phase tracking
    phase: int = 0           # 0=normal, 1=skew, 2=market_reduce, 3=emergency
    phase_entered_at: float = 0.0

    # Withdrawal state
    withdrawn: bool = False
    withdraw_reason: str = ""
    withdraw_until: float = 0.0

    @property
    def direction(self) -> str:
        if self.qty > 0:
            return "LONG"
        elif self.qty < 0:
            return "SHORT"
        return "FLAT"


@dataclass
class DailyPnL:
    """Tracks daily P&L for the halt-on-negative rule."""
    date: str = ""
    realized_pnl: float = 0.0
    fees_paid: float = 0.0
    net_pnl: float = 0.0
    fill_count: int = 0
    round_trips: int = 0


class MarketMakingRiskManager:
    """Risk manager specific to market making.

    Parameters
    ----------
    params : dict
        Strategy parameters from config.yaml.
    total_equity : float
        Current total account equity.
    """

    def __init__(self, params: dict, total_equity: float = 10000.0) -> None:
        self._params = params
        self._total_equity = total_equity

        # Thresholds
        self._max_capital_pct = 15.0         # 15% of equity
        self._max_per_instrument_pct = 7.0   # 7% per instrument
        self._max_instruments = 3
        self._max_leverage = 3

        # Inventory thresholds (as pct of total equity)
        self._inv_warn_pct = params.get("inventory_warn_pct", 2.0)
        self._inv_critical_pct = params.get("inventory_critical_pct", 3.0)
        self._inv_emergency_pct = params.get("inventory_emergency_pct", 4.0)
        self._inv_liquidation_delay_min = params.get("inventory_liquidation_delay_min", 30)
        self._inv_phase2_reduce_pct = params.get("inventory_phase2_reduce_pct", 50)
        self._inv_phase3_pause_hours = params.get("inventory_phase3_pause_hours", 1)

        # Drawdown
        self._daily_dd_pct = 1.0
        self._weekly_dd_pct = 2.0
        self._monthly_dd_pct = 4.0

        # Quote staleness
        self._quote_stale_seconds = params.get("quote_stale_seconds", 60)

        # Blowout detection
        self._blowout_window_hours = params.get("blowout_window_hours", 1)

        # Withdrawal timing
        self._vol_spike_withdraw_min = params.get("vol_spike_withdraw_minutes", 5)
        self._large_trade_withdraw_sec = params.get("large_trade_withdraw_seconds", 30)
        self._liquidation_withdraw_min = params.get("liquidation_withdraw_minutes", 10)

        # Maker fee tracking (Section 11.4)
        self._current_maker_fee_pct: float = params.get("maker_fee_pct", 0.02)  # 0.02%

        # Per-instrument state
        self._inventory: Dict[str, InventoryState] = {}

        # Daily PnL tracking
        self._daily_pnl = DailyPnL()
        self._daily_pnl_history: Deque[DailyPnL] = deque(maxlen=90)

        # Equity tracking
        self._peak_equity = total_equity
        self._daily_start_equity = total_equity
        self._weekly_start_equity = total_equity
        self._monthly_start_equity = total_equity

        # Global halt
        self._is_halted = False
        self._halt_reason = ""

        logger.info(
            "MarketMakingRiskManager initialized: equity=%.2f, "
            "inv_warn=%.1f%%, inv_crit=%.1f%%, inv_emg=%.1f%%, "
            "daily_dd=%.1f%%",
            total_equity, self._inv_warn_pct, self._inv_critical_pct,
            self._inv_emergency_pct, self._daily_dd_pct,
        )

    # ------------------------------------------------------------------
    # Equity management
    # ------------------------------------------------------------------

    def update_equity(self, equity: float) -> None:
        """Update current equity."""
        self._total_equity = equity
        if equity > self._peak_equity:
            self._peak_equity = equity

    def get_total_equity(self) -> float:
        return self._total_equity

    def get_allocated_capital(self) -> float:
        """Return total capital allocated for this strategy."""
        return self._total_equity * (self._max_capital_pct / 100.0)

    def get_per_instrument_capital(self) -> float:
        """Return capital allocated per instrument."""
        return self._total_equity * (self._max_per_instrument_pct / 100.0)

    def get_max_inventory_notional(self, symbol: str) -> float:
        """Return maximum allowed inventory notional for an instrument.

        Uses critical threshold as the hard limit.
        """
        return self._total_equity * (self._inv_critical_pct / 100.0)

    # ------------------------------------------------------------------
    # Inventory tracking
    # ------------------------------------------------------------------

    def update_inventory(self, symbol: str, qty: float, avg_cost: float,
                         mid_price: float) -> None:
        """Update inventory state for an instrument."""
        if symbol not in self._inventory:
            self._inventory[symbol] = InventoryState(symbol=symbol)

        inv = self._inventory[symbol]
        inv.qty = qty
        inv.avg_cost = avg_cost
        inv.notional = abs(qty * mid_price) if mid_price > 0 else 0.0
        inv.pct_of_equity = (inv.notional / self._total_equity * 100.0) if self._total_equity > 0 else 0.0

        # Track direction changes for blowout detection
        now = time.time()
        new_dir = inv.direction
        if not inv.direction_history or inv.direction_history[-1][0] != new_dir:
            inv.direction_history.append((new_dir, now))

        # Record peak inventory events for blowout detection
        # Track when inventory reaches warn level or above
        if inv.pct_of_equity >= self._inv_warn_pct and new_dir in ("LONG", "SHORT"):
            # Only record if this is a new peak or direction changed
            should_record = True
            if inv.peak_inventory_events:
                last_dir, last_pct, last_ts = inv.peak_inventory_events[-1]
                # Don't record if same direction and within 60s
                if last_dir == new_dir and now - last_ts < 60:
                    should_record = False
            if should_record:
                inv.peak_inventory_events.append((new_dir, inv.pct_of_equity, now))

    def check_inventory_limits(self, symbol: str) -> Tuple[str, str]:
        """Check inventory limits and determine required action.

        Returns
        -------
        (action, reason):
            action = "normal", "warn", "stop_accumulating", "emergency_reduce"
            reason = human-readable explanation
        """
        inv = self._inventory.get(symbol)
        if inv is None:
            return "normal", ""

        pct = inv.pct_of_equity

        if pct >= self._inv_emergency_pct:
            return "emergency_reduce", (
                f"Inventory {pct:.2f}% >= emergency {self._inv_emergency_pct}% — "
                f"market reduce to {self._inv_warn_pct}%"
            )

        if pct >= self._inv_critical_pct:
            return "stop_accumulating", (
                f"Inventory {pct:.2f}% >= critical {self._inv_critical_pct}% — "
                f"stop placing orders on accumulating side"
            )

        if pct >= self._inv_warn_pct:
            return "warn", (
                f"Inventory {pct:.2f}% >= warn {self._inv_warn_pct}% — "
                f"increase gamma to aggressive"
            )

        return "normal", ""

    def get_emergency_reduce_qty(self, symbol: str, mid_price: float) -> float:
        """Calculate quantity to reduce inventory to warn level.

        Called during emergency: market order to reduce to 2%.
        """
        inv = self._inventory.get(symbol)
        if inv is None or mid_price <= 0:
            return 0.0

        target_notional = self._total_equity * (self._inv_warn_pct / 100.0)
        current_notional = abs(inv.qty * mid_price)
        excess = current_notional - target_notional

        if excess <= 0:
            return 0.0

        return excess / mid_price

    def get_accumulating_side(self, symbol: str) -> Optional[str]:
        """Return the side that is accumulating inventory, or None."""
        inv = self._inventory.get(symbol)
        if inv is None:
            return None
        if inv.qty > 0:
            return "BUY"   # Long inventory, BUY is accumulating
        elif inv.qty < 0:
            return "SELL"  # Short inventory, SELL is accumulating
        return None

    # ------------------------------------------------------------------
    # Inventory liquidation phases
    # ------------------------------------------------------------------

    def check_liquidation_phase(self, symbol: str) -> Tuple[int, str]:
        """Check and advance liquidation phase if needed.

        Phase 0: Normal
        Phase 1: Skew (gamma increase, handled by strategy)
        Phase 2: 50% market order (after 30 min in phase 1)
        Phase 3: Close all + pause 1 hour
        """
        inv = self._inventory.get(symbol)
        if inv is None:
            return 0, ""

        action, reason = self.check_inventory_limits(symbol)

        if action == "normal":
            if inv.phase > 0:
                inv.phase = 0
                inv.phase_entered_at = 0
            return 0, ""

        now = time.time()

        if action == "warn":
            if inv.phase < 1:
                inv.phase = 1
                inv.phase_entered_at = now
            return 1, reason

        if action == "stop_accumulating":
            if inv.phase < 1:
                inv.phase = 1
                inv.phase_entered_at = now
            # Check if we've been in phase 1 long enough for phase 2
            if inv.phase == 1 and (now - inv.phase_entered_at) > self._inv_liquidation_delay_min * 60:
                inv.phase = 2
                inv.phase_entered_at = now
                return 2, f"Phase 2: market reduce {self._inv_phase2_reduce_pct}%"
            return inv.phase, reason

        if action == "emergency_reduce":
            inv.phase = 3
            inv.phase_entered_at = now
            return 3, f"Phase 3: emergency close all, pause {self._inv_phase3_pause_hours}h"

        return inv.phase, ""

    # ------------------------------------------------------------------
    # Drawdown checks
    # ------------------------------------------------------------------

    def check_drawdown(self) -> Tuple[bool, str, float]:
        """Check if any drawdown threshold is breached.

        Returns (halted, level, pct).
        """
        equity = self._total_equity

        # Daily
        if self._daily_start_equity > 0:
            daily_dd = (self._daily_start_equity - equity) / self._daily_start_equity * 100
            if daily_dd >= self._daily_dd_pct:
                return True, "daily", daily_dd

        # Weekly
        if self._weekly_start_equity > 0:
            weekly_dd = (self._weekly_start_equity - equity) / self._weekly_start_equity * 100
            if weekly_dd >= self._weekly_dd_pct:
                return True, "weekly", weekly_dd

        # Monthly
        if self._monthly_start_equity > 0:
            monthly_dd = (self._monthly_start_equity - equity) / self._monthly_start_equity * 100
            if monthly_dd >= self._monthly_dd_pct:
                return True, "monthly", monthly_dd

        return False, "", 0.0

    def reset_daily(self) -> None:
        """Reset daily drawdown and PnL tracking."""
        self._daily_start_equity = self._total_equity

        # Archive daily PnL
        if self._daily_pnl.date:
            self._daily_pnl_history.append(self._daily_pnl)

        from datetime import datetime, timezone
        self._daily_pnl = DailyPnL(
            date=datetime.now(timezone.utc).strftime("%Y-%m-%d")
        )

        # Check if daily PnL was negative -> halt
        if self._daily_pnl_history and self._daily_pnl_history[-1].net_pnl < 0:
            self._is_halted = True
            self._halt_reason = (
                f"Daily PnL negative ({self._daily_pnl_history[-1].net_pnl:.4f}) — "
                f"halt and investigate"
            )
            logger.warning("HALT: %s", self._halt_reason)

    def reset_weekly(self) -> None:
        self._weekly_start_equity = self._total_equity

    def reset_monthly(self) -> None:
        self._monthly_start_equity = self._total_equity

    # ------------------------------------------------------------------
    # Daily PnL tracking
    # ------------------------------------------------------------------

    def record_fill_pnl(self, realized_pnl: float, fee: float) -> None:
        """Record a fill's P&L contribution."""
        self._daily_pnl.realized_pnl += realized_pnl
        self._daily_pnl.fees_paid += fee
        self._daily_pnl.net_pnl = self._daily_pnl.realized_pnl - self._daily_pnl.fees_paid
        self._daily_pnl.fill_count += 1

    def record_round_trip(self, net_profit: float) -> None:
        """Record a completed round-trip."""
        self._daily_pnl.round_trips += 1

    def check_daily_pnl_halt(self) -> Tuple[bool, str]:
        """Check if daily PnL is sufficiently negative to halt."""
        if self._daily_pnl.net_pnl < 0 and self._daily_pnl.fill_count > 20:
            return True, f"Daily PnL is negative ({self._daily_pnl.net_pnl:.4f}) after {self._daily_pnl.fill_count} fills"
        return False, ""

    # ------------------------------------------------------------------
    # Quote staleness check
    # ------------------------------------------------------------------

    def check_quote_staleness(self, symbol: str, oldest_quote_age: Optional[float]) -> Tuple[bool, str]:
        """Check if quotes are stale (> 60 seconds without update).

        Returns (is_stale, message).
        """
        if oldest_quote_age is None:
            return False, ""

        if oldest_quote_age > self._quote_stale_seconds:
            return True, (
                f"Quote staleness: {symbol} oldest quote is {oldest_quote_age:.1f}s old "
                f"(threshold={self._quote_stale_seconds}s)"
            )
        return False, ""

    # ------------------------------------------------------------------
    # Inventory blowout detection
    # ------------------------------------------------------------------

    def check_inventory_blowout(self, symbol: str) -> Tuple[bool, str]:
        """Check if inventory swung from max LONG to max SHORT (or vice versa) within 1 hour.

        Detects if inventory was at/near the critical threshold on BOTH sides
        within the blowout window. This indicates the market is whipsawing
        through quotes too aggressively.

        If detected: halt strategy immediately, send CRITICAL alert.
        """
        inv = self._inventory.get(symbol)
        if inv is None:
            return False, ""

        now = time.time()
        window = self._blowout_window_hours * 3600

        # Check peak inventory events within the window
        recent_peaks = [
            (direction, pct, ts)
            for direction, pct, ts in inv.peak_inventory_events
            if now - ts < window
        ]

        # Check if we had significant inventory on BOTH sides within the window
        # "significant" = reached at least warn level (2% of equity)
        blowout_threshold_pct = self._inv_warn_pct  # Inventory was at warn+ level
        has_significant_long = any(
            d == "LONG" and pct >= blowout_threshold_pct
            for d, pct, _ in recent_peaks
        )
        has_significant_short = any(
            d == "SHORT" and pct >= blowout_threshold_pct
            for d, pct, _ in recent_peaks
        )

        if has_significant_long and has_significant_short:
            # Find the timestamps for reporting
            long_ts = max(ts for d, pct, ts in recent_peaks if d == "LONG" and pct >= blowout_threshold_pct)
            short_ts = max(ts for d, pct, ts in recent_peaks if d == "SHORT" and pct >= blowout_threshold_pct)
            swing_minutes = abs(long_ts - short_ts) / 60.0

            return True, (
                f"CRITICAL inventory blowout: {symbol} swung from max LONG to max SHORT "
                f"(or vice versa) within {swing_minutes:.0f}m — "
                f"halt strategy immediately"
            )

        # Also check basic direction history as fallback
        recent_dirs = [(d, t) for d, t in inv.direction_history if now - t < window]
        if len(recent_dirs) >= 2:
            dirs = [d for d, _ in recent_dirs]
            has_long = "LONG" in dirs
            has_short = "SHORT" in dirs
            if has_long and has_short:
                return True, (
                    f"Inventory blowout: {symbol} swung LONG<->SHORT within "
                    f"{self._blowout_window_hours}h — halt"
                )

        return False, ""

    # ------------------------------------------------------------------
    # Withdrawal management
    # ------------------------------------------------------------------

    def set_withdrawal(self, symbol: str, reason: str, duration_s: float) -> None:
        """Set quote withdrawal for an instrument."""
        inv = self._inventory.get(symbol)
        if inv is None:
            self._inventory[symbol] = InventoryState(symbol=symbol)
            inv = self._inventory[symbol]

        inv.withdrawn = True
        inv.withdraw_reason = reason
        inv.withdraw_until = time.time() + duration_s
        logger.warning("WITHDRAW: %s for %.0fs — %s", symbol, duration_s, reason)

    def check_withdrawal(self, symbol: str) -> Tuple[bool, str]:
        """Check if a symbol is currently withdrawn from quoting."""
        inv = self._inventory.get(symbol)
        if inv is None:
            return False, ""

        if not inv.withdrawn:
            return False, ""

        if time.time() >= inv.withdraw_until:
            inv.withdrawn = False
            inv.withdraw_reason = ""
            logger.info("RESUME: %s withdrawal expired", symbol)
            return False, ""

        return True, inv.withdraw_reason

    # ------------------------------------------------------------------
    # Global halt
    # ------------------------------------------------------------------

    def is_halted(self) -> Tuple[bool, str]:
        """Check if the strategy is globally halted."""
        if self._is_halted:
            return True, self._halt_reason

        # Check drawdown
        halted, level, pct = self.check_drawdown()
        if halted:
            self._is_halted = True
            self._halt_reason = f"{level} drawdown {pct:.2f}% exceeded limit"
            return True, self._halt_reason

        return False, ""

    def clear_halt(self) -> None:
        """Manually clear global halt (for operator override)."""
        self._is_halted = False
        self._halt_reason = ""
        logger.info("Global halt cleared manually")

    # ------------------------------------------------------------------
    # Maker fee change handling (Section 11.4)
    # ------------------------------------------------------------------

    def update_maker_fee(self, new_fee_pct: float) -> Tuple[bool, str, float]:
        """Handle maker fee tier change.

        Detects if the maker fee has changed and evaluates impact:
        - Recalculates minimum viable spread
        - If fee increase > 50%: halt strategy

        Parameters
        ----------
        new_fee_pct : float
            New maker fee as percentage (e.g. 0.02 for 0.02%).

        Returns
        -------
        (should_halt, reason, new_min_spread_pct):
            should_halt: True if fee increase > 50% -> halt
            reason: explanation
            new_min_spread_pct: recalculated minimum spread percentage
        """
        current_fee = self._current_maker_fee_pct
        if current_fee <= 0:
            self._current_maker_fee_pct = new_fee_pct
            new_min = 2.0 * new_fee_pct + 0.005
            return False, "", new_min

        if abs(new_fee_pct - current_fee) < 1e-6:
            # No change
            new_min = 2.0 * new_fee_pct + 0.005
            return False, "", new_min

        # Calculate percentage increase
        fee_change_pct = (new_fee_pct - current_fee) / current_fee * 100.0
        old_fee = current_fee
        self._current_maker_fee_pct = new_fee_pct

        # Recalculate minimum spread: 2 * maker_fee + 0.005%
        new_min_spread = 2.0 * new_fee_pct + 0.005

        logger.info(
            "Maker fee changed: %.4f%% -> %.4f%% (%.1f%% change), "
            "new min spread: %.4f%%",
            old_fee, new_fee_pct, fee_change_pct, new_min_spread,
        )

        if fee_change_pct > 50.0:
            reason = (
                f"Maker fee increased >50%: {old_fee:.4f}% -> {new_fee_pct:.4f}% "
                f"({fee_change_pct:.1f}% increase) — halt and re-evaluate"
            )
            self._is_halted = True
            self._halt_reason = reason
            logger.warning("HALT: %s", reason)
            return True, reason, new_min_spread

        return False, f"Fee updated: {old_fee:.4f}% -> {new_fee_pct:.4f}%", new_min_spread

    def get_current_maker_fee(self) -> float:
        """Return current maker fee percentage."""
        return self._current_maker_fee_pct

    def get_min_viable_spread_pct(self) -> float:
        """Return current minimum viable spread as percentage.

        min_spread = 2 * maker_fee + 0.005%
        """
        return 2.0 * self._current_maker_fee_pct + 0.005

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_metrics(self) -> Dict[str, Any]:
        """Return risk manager metrics."""
        return {
            "total_equity": self._total_equity,
            "allocated_capital": self.get_allocated_capital(),
            "per_instrument_capital": self.get_per_instrument_capital(),
            "is_halted": self._is_halted,
            "halt_reason": self._halt_reason,
            "daily_pnl": {
                "date": self._daily_pnl.date,
                "realized_pnl": self._daily_pnl.realized_pnl,
                "fees_paid": self._daily_pnl.fees_paid,
                "net_pnl": self._daily_pnl.net_pnl,
                "fill_count": self._daily_pnl.fill_count,
                "round_trips": self._daily_pnl.round_trips,
            },
            "drawdown": {
                "daily": (self._daily_start_equity - self._total_equity) / max(1, self._daily_start_equity) * 100,
                "weekly": (self._weekly_start_equity - self._total_equity) / max(1, self._weekly_start_equity) * 100,
                "monthly": (self._monthly_start_equity - self._total_equity) / max(1, self._monthly_start_equity) * 100,
            },
            "inventory": {
                sym: {
                    "qty": inv.qty,
                    "avg_cost": inv.avg_cost,
                    "notional": inv.notional,
                    "pct_of_equity": inv.pct_of_equity,
                    "direction": inv.direction,
                    "phase": inv.phase,
                    "withdrawn": inv.withdrawn,
                    "withdraw_reason": inv.withdraw_reason,
                }
                for sym, inv in self._inventory.items()
            },
        }

    def get_state_for_persistence(self) -> Dict[str, Any]:
        """Return state for persistence."""
        return {
            "total_equity": self._total_equity,
            "peak_equity": self._peak_equity,
            "daily_start_equity": self._daily_start_equity,
            "weekly_start_equity": self._weekly_start_equity,
            "monthly_start_equity": self._monthly_start_equity,
            "is_halted": self._is_halted,
            "halt_reason": self._halt_reason,
            "daily_pnl": {
                "date": self._daily_pnl.date,
                "realized_pnl": self._daily_pnl.realized_pnl,
                "fees_paid": self._daily_pnl.fees_paid,
                "net_pnl": self._daily_pnl.net_pnl,
                "fill_count": self._daily_pnl.fill_count,
                "round_trips": self._daily_pnl.round_trips,
            },
            "inventory": {
                sym: {
                    "qty": inv.qty,
                    "avg_cost": inv.avg_cost,
                    "phase": inv.phase,
                }
                for sym, inv in self._inventory.items()
            },
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Restore state from persistence."""
        if not state:
            return
        self._total_equity = state.get("total_equity", self._total_equity)
        self._peak_equity = state.get("peak_equity", self._peak_equity)
        self._daily_start_equity = state.get("daily_start_equity", self._daily_start_equity)
        self._weekly_start_equity = state.get("weekly_start_equity", self._weekly_start_equity)
        self._monthly_start_equity = state.get("monthly_start_equity", self._monthly_start_equity)
        self._is_halted = state.get("is_halted", False)
        self._halt_reason = state.get("halt_reason", "")

        pnl_data = state.get("daily_pnl", {})
        if pnl_data:
            self._daily_pnl = DailyPnL(
                date=pnl_data.get("date", ""),
                realized_pnl=pnl_data.get("realized_pnl", 0.0),
                fees_paid=pnl_data.get("fees_paid", 0.0),
                net_pnl=pnl_data.get("net_pnl", 0.0),
                fill_count=pnl_data.get("fill_count", 0),
                round_trips=pnl_data.get("round_trips", 0),
            )

        for sym, inv_data in state.get("inventory", {}).items():
            if sym not in self._inventory:
                self._inventory[sym] = InventoryState(symbol=sym)
            self._inventory[sym].qty = inv_data.get("qty", 0.0)
            self._inventory[sym].avg_cost = inv_data.get("avg_cost", 0.0)
            self._inventory[sym].phase = inv_data.get("phase", 0)

        logger.info("Risk state restored: equity=%.2f, halted=%s", self._total_equity, self._is_halted)
