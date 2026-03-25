"""Sub-Strategy A: Covered Call Writing.

Holds spot BTC/ETH and sells OTM calls against the position via synthetic
implementation on Binance (TAKE_PROFIT_MARKET at the strike price).

7-day cycles with auto-rolling. Premium calculated via Black-Scholes.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from src.black_scholes import call_price, price_option, BSResult, delta as bs_delta

logger = logging.getLogger(__name__)
trade_logger = logging.getLogger("trade")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CoveredCallCycle:
    """Tracks one covered call cycle (entry to expiration or exercise)."""
    cycle_id: str
    symbol: str
    # Spot leg
    spot_entry_price: float
    spot_quantity: float
    # Synthetic call sold
    strike_price: float
    otm_pct: float
    premium: float          # BS-calculated theoretical premium
    iv_at_entry: float      # IV when position was opened (%)
    rv_at_entry: float      # RV when position was opened (%)
    # Timing
    opened_at: float        # timestamp
    expiration_at: float    # timestamp (opened_at + cycle_days * 86400)
    cycle_days: int = 7
    # State
    status: str = "open"    # "open", "exercised", "expired", "early_exit"
    # Binance order tracking
    tp_order_id: Optional[int] = None
    # PnL
    realized_pnl: float = 0.0
    hedge_costs: float = 0.0
    fees_paid: float = 0.0

    # Cycle PnL attribution
    premium_income: float = 0.0       # Premium from selling the call
    underlying_pnl: float = 0.0       # PnL from underlying price movement
    net_cycle_pnl: float = 0.0        # Net after all costs

    # Exercise tracking
    delta_at_entry: float = 0.0       # BS delta at entry (expected exercise probability)

    @property
    def time_remaining_days(self) -> float:
        remaining = self.expiration_at - time.time()
        return max(0.0, remaining / 86400.0)

    @property
    def time_elapsed_pct(self) -> float:
        total = self.expiration_at - self.opened_at
        if total <= 0:
            return 1.0
        elapsed = time.time() - self.opened_at
        return min(1.0, elapsed / total)

    def to_dict(self) -> dict:
        return {
            "cycle_id": self.cycle_id,
            "symbol": self.symbol,
            "spot_entry": self.spot_entry_price,
            "spot_qty": self.spot_quantity,
            "strike": self.strike_price,
            "otm_pct": round(self.otm_pct * 100, 2),
            "premium": round(self.premium, 4),
            "iv_at_entry": round(self.iv_at_entry, 1),
            "rv_at_entry": round(self.rv_at_entry, 1),
            "opened_at": self.opened_at,
            "expiration_at": self.expiration_at,
            "remaining_days": round(self.time_remaining_days, 2),
            "elapsed_pct": round(self.time_elapsed_pct * 100, 1),
            "status": self.status,
            "realized_pnl": round(self.realized_pnl, 4),
            "hedge_costs": round(self.hedge_costs, 4),
            "fees_paid": round(self.fees_paid, 4),
            "delta_at_entry": round(self.delta_at_entry, 4),
            "attribution": {
                "premium_income": round(self.premium_income, 4),
                "underlying_pnl": round(self.underlying_pnl, 4),
                "hedge_costs": round(self.hedge_costs, 4),
                "net_cycle_pnl": round(self.net_cycle_pnl, 4),
            },
        }


# ---------------------------------------------------------------------------
# CoveredCallManager
# ---------------------------------------------------------------------------

class CoveredCallManager:
    """Manages covered call positions for STRAT-008 Sub-Strategy A.

    In Synthetic Mode (Binance only):
    - Hold spot BTC/ETH
    - Place TAKE_PROFIT_MARKET sell order at the strike price
    - This simulates the call being "exercised" if price reaches strike
    - Premium is tracked theoretically using Black-Scholes

    Parameters
    ----------
    config : dict
        Strategy parameters.
    """

    def __init__(self, config: dict) -> None:
        self._config = config
        self._risk_free_rate = config.get("risk_free_rate", 0.05)
        self._otm_default = config.get("cc_otm_pct_default", 0.05)
        self._otm_high_iv = config.get("cc_otm_pct_high_iv", 0.08)
        self._iv_high_threshold = config.get("cc_iv_high_threshold", 80.0)
        self._max_allocation_pct = config.get("cc_max_allocation_pct", 15.0)
        self._cycle_days = config.get("option_cycle_days", 7)
        self._adx_strong = config.get("adx_strong_trend", 30)

        # Active cycles: cycle_id -> CoveredCallCycle
        self._active_cycles: Dict[str, CoveredCallCycle] = {}
        # Completed cycles for performance tracking
        self._completed_cycles: List[CoveredCallCycle] = []
        # Cycle counter
        self._cycle_counter = 0
        # Consecutive losses
        self._consecutive_losses = 0

    # ------------------------------------------------------------------
    # Entry logic
    # ------------------------------------------------------------------

    def should_enter(
        self,
        symbol: str,
        iv: float,
        rv_7d: float,
        iv_rv_ratio: float,
        current_price: float,
        ema_20: float,
        ema_50: float,
        daily_atr: float,
        adx_value: float,
        adx_plus_di: float,
    ) -> tuple[bool, str]:
        """Evaluate whether to enter a new covered call cycle.

        Conditions (Section 3.2):
        - Hold spot BTC/ETH (or open one at market)
        - IV/RV ratio > 1.2
        - Daily trend is neutral to slightly bullish
          (EMA20 > EMA50, but price not > 1 ATR above EMA20)
        - NOT in strong uptrend (ADX > 30 with bullish direction)

        Returns
        -------
        (should_enter, reason)
        """
        # Check if already have active cycle for this symbol
        active = self._get_active_cycle(symbol)
        if active:
            return False, f"Active cycle exists: {active.cycle_id}"

        # Check consecutive losses
        if self._consecutive_losses >= 3:
            return False, f"Halted: {self._consecutive_losses} consecutive losses"

        # IV/RV ratio check
        if iv_rv_ratio < 1.2:
            return False, f"IV/RV ratio {iv_rv_ratio:.2f} < 1.2 threshold"

        # Trend filter
        if ema_20 <= 0 or ema_50 <= 0:
            return False, "Insufficient EMA data"

        # Strong uptrend filter: don't write calls when opportunity cost is high
        if adx_value > self._adx_strong and adx_plus_di > 0:
            return False, (
                f"Strong uptrend (ADX={adx_value:.1f}): "
                "opportunity cost of capped upside too high"
            )

        # Neutral to slightly bullish check
        if ema_20 < ema_50:
            # Bearish trend - still OK for covered calls (premium is protection)
            pass

        # Price not more than 1 ATR above EMA20 (avoiding extended moves)
        if daily_atr > 0 and current_price > ema_20 + daily_atr:
            return False, (
                f"Price {current_price:.2f} > EMA20+ATR "
                f"({ema_20:.2f}+{daily_atr:.2f}): too extended"
            )

        return True, "Conditions met for covered call entry"

    def calculate_entry(
        self,
        symbol: str,
        current_price: float,
        spot_quantity: float,
        iv: float,
        rv_7d: float,
    ) -> CoveredCallCycle:
        """Calculate covered call parameters and create a cycle.

        Parameters
        ----------
        symbol : str
            Trading pair (e.g. "BTCUSDT").
        current_price : float
            Current spot price.
        spot_quantity : float
            Spot holding quantity.
        iv : float
            Current IV in percentage (e.g. 65.0).
        rv_7d : float
            7-day realized volatility in percentage.

        Returns
        -------
        CoveredCallCycle ready for execution.
        """
        # Determine OTM percentage based on IV level
        if iv > self._iv_high_threshold:
            otm_pct = self._otm_high_iv
        else:
            otm_pct = self._otm_default

        # Calculate strike
        strike = current_price * (1.0 + otm_pct)

        # Calculate theoretical premium via Black-Scholes
        T = self._cycle_days / 365.0
        sigma = iv / 100.0  # Convert to decimal
        premium_per_unit = call_price(
            S=current_price,
            K=strike,
            T=T,
            r=self._risk_free_rate,
            sigma=sigma,
        )
        total_premium = premium_per_unit * spot_quantity

        # Compute delta at entry for exercise rate tracking
        entry_delta = bs_delta(
            S=current_price, K=strike, T=T, r=self._risk_free_rate,
            sigma=sigma, option_type="call",
        )

        # Create cycle
        self._cycle_counter += 1
        now = time.time()
        cycle = CoveredCallCycle(
            cycle_id=f"CC-{symbol}-{self._cycle_counter:04d}",
            symbol=symbol,
            spot_entry_price=current_price,
            spot_quantity=spot_quantity,
            strike_price=strike,
            otm_pct=otm_pct,
            premium=total_premium,
            iv_at_entry=iv,
            rv_at_entry=rv_7d,
            opened_at=now,
            expiration_at=now + self._cycle_days * 86400,
            cycle_days=self._cycle_days,
            premium_income=total_premium,
            delta_at_entry=entry_delta,
        )

        logger.info(
            "Covered call calculated: %s strike=%.2f (%.1f%% OTM) "
            "premium=%.4f IV=%.1f%% T=%dd",
            cycle.cycle_id, strike, otm_pct * 100,
            total_premium, iv, self._cycle_days,
        )

        return cycle

    def activate_cycle(
        self, cycle: CoveredCallCycle, tp_order_id: Optional[int] = None
    ) -> None:
        """Register an active cycle after orders are placed."""
        cycle.tp_order_id = tp_order_id
        self._active_cycles[cycle.cycle_id] = cycle

        trade_logger.info(
            "CC_OPEN\t%s\t%s\tstrike=%.2f\tpremium=%.4f\t"
            "iv=%.1f\totm=%.1f%%\texpiry=%.0f",
            cycle.cycle_id, cycle.symbol, cycle.strike_price,
            cycle.premium, cycle.iv_at_entry,
            cycle.otm_pct * 100, cycle.expiration_at,
        )

    # ------------------------------------------------------------------
    # Exit logic (Section 4.1)
    # ------------------------------------------------------------------

    def check_exits(
        self,
        symbol: str,
        current_price: float,
        daily_atr: float,
        current_iv: float,
        rv_7d: float,
    ) -> List[dict]:
        """Check all active cycles for exit conditions.

        Returns list of exit actions to take.
        """
        actions: List[dict] = []
        cycle = self._get_active_cycle(symbol)
        if not cycle:
            return actions

        now = time.time()

        # 1. Strike reached (simulated exercise)
        if current_price >= cycle.strike_price:
            actions.append({
                "action": "exercise",
                "cycle_id": cycle.cycle_id,
                "reason": (
                    f"Price {current_price:.2f} reached strike "
                    f"{cycle.strike_price:.2f}"
                ),
            })
            return actions

        # 2. Expiration (7 days passed)
        if now >= cycle.expiration_at:
            actions.append({
                "action": "expire",
                "cycle_id": cycle.cycle_id,
                "reason": "Cycle expired — premium kept",
            })
            return actions

        # 3. Early exit: underlying drops > 1 daily ATR
        if daily_atr > 0:
            drop = cycle.spot_entry_price - current_price
            if drop > daily_atr:
                actions.append({
                    "action": "early_exit_drop",
                    "cycle_id": cycle.cycle_id,
                    "reason": (
                        f"Drop {drop:.2f} > ATR {daily_atr:.2f}: "
                        "closing spot to protect downside"
                    ),
                })
                return actions

        # 4. IV crush: IV drops below RV_7d
        if current_iv > 0 and current_iv < rv_7d:
            actions.append({
                "action": "early_exit_iv_crush",
                "cycle_id": cycle.cycle_id,
                "reason": (
                    f"IV crush: IV {current_iv:.1f}% < RV_7d {rv_7d:.1f}%: "
                    "remaining time value minimal"
                ),
            })
            return actions

        return actions

    def close_cycle(
        self,
        cycle_id: str,
        exit_type: str,
        exit_price: float,
        fees: float = 0.0,
    ) -> Optional[CoveredCallCycle]:
        """Close a cycle and compute PnL.

        Parameters
        ----------
        cycle_id : str
            The cycle to close.
        exit_type : str
            "exercised", "expired", "early_exit"
        exit_price : float
            Current or exit price.
        fees : float
            Total fees paid during the cycle.

        Returns
        -------
        The completed cycle, or None if not found.
        """
        cycle = self._active_cycles.pop(cycle_id, None)
        if not cycle:
            logger.warning("Cycle %s not found for closing", cycle_id)
            return None

        cycle.status = exit_type
        cycle.fees_paid = fees

        if exit_type == "exercised":
            # Spot sold at strike price
            spot_pnl = (cycle.strike_price - cycle.spot_entry_price) * cycle.spot_quantity
            cycle.realized_pnl = spot_pnl + cycle.premium - fees
            cycle.underlying_pnl = spot_pnl
            cycle.premium_income = cycle.premium
        elif exit_type == "expired":
            # Premium kept, spot still held (unrealized spot PnL not counted)
            cycle.realized_pnl = cycle.premium - fees
            cycle.underlying_pnl = 0.0
            cycle.premium_income = cycle.premium
        else:
            # Early exit
            spot_pnl = (exit_price - cycle.spot_entry_price) * cycle.spot_quantity
            # Partial premium: proportional to time elapsed
            kept_premium = cycle.premium * cycle.time_elapsed_pct
            cycle.realized_pnl = spot_pnl + kept_premium - fees
            cycle.underlying_pnl = spot_pnl
            cycle.premium_income = kept_premium

        cycle.net_cycle_pnl = cycle.realized_pnl

        # Track consecutive losses
        if cycle.realized_pnl < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

        self._completed_cycles.append(cycle)

        trade_logger.info(
            "CC_CLOSE\t%s\t%s\texit=%s\tpnl=%.4f\tpremium=%.4f\tfees=%.4f",
            cycle.cycle_id, cycle.symbol, exit_type,
            cycle.realized_pnl, cycle.premium, fees,
        )

        return cycle

    # ------------------------------------------------------------------
    # Auto-rolling
    # ------------------------------------------------------------------

    def should_roll(
        self,
        symbol: str,
        iv_rv_ratio: float,
    ) -> bool:
        """Check if a new cycle should be started after expiration.

        Section 8.3: If IV/RV ratio still favorable, enter new cycle within 1h.
        """
        active = self._get_active_cycle(symbol)
        if active:
            return False

        if iv_rv_ratio < 1.2:
            return False

        if self._consecutive_losses >= 3:
            return False

        return True

    # ------------------------------------------------------------------
    # Current theoretical value
    # ------------------------------------------------------------------

    def get_current_premium_value(
        self,
        cycle_id: str,
        current_price: float,
        current_iv: float,
    ) -> float:
        """Calculate the current theoretical value of the sold call.

        Used for mark-to-market and PnL tracking.
        """
        cycle = self._active_cycles.get(cycle_id)
        if not cycle:
            return 0.0

        T = cycle.time_remaining_days / 365.0
        if T <= 0:
            return max(current_price - cycle.strike_price, 0.0) * cycle.spot_quantity

        sigma = current_iv / 100.0
        current_call_value = call_price(
            S=current_price,
            K=cycle.strike_price,
            T=T,
            r=self._risk_free_rate,
            sigma=sigma,
        )
        return current_call_value * cycle.spot_quantity

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_active_cycle(self, symbol: str) -> Optional[CoveredCallCycle]:
        """Get the active cycle for a symbol."""
        for cycle in self._active_cycles.values():
            if cycle.symbol == symbol and cycle.status == "open":
                return cycle
        return None

    def get_active_cycles(self) -> Dict[str, CoveredCallCycle]:
        return dict(self._active_cycles)

    def get_completed_cycles(self) -> List[CoveredCallCycle]:
        return list(self._completed_cycles)

    @property
    def consecutive_losses(self) -> int:
        return self._consecutive_losses

    def get_metrics(self) -> dict:
        """Return performance metrics for covered calls."""
        completed = self._completed_cycles
        if not completed:
            return {
                "total_cycles": 0,
                "active_cycles": len(self._active_cycles),
                "total_premium": 0.0,
                "total_pnl": 0.0,
                "exercise_rate": 0.0,
                "expected_exercise_rate": 0.0,
                "avg_premium_per_cycle": 0.0,
            }

        total_pnl = sum(c.realized_pnl for c in completed)
        total_premium = sum(c.premium for c in completed)
        total_hedge_costs = sum(c.hedge_costs for c in completed)
        exercised = sum(1 for c in completed if c.status == "exercised")
        wins = sum(1 for c in completed if c.realized_pnl > 0)

        # Exercise rate tracking: actual vs expected (from delta at entry)
        actual_exercise_rate = exercised / len(completed) * 100.0
        # Expected exercise rate = average delta at entry (delta ~ probability of ITM)
        deltas = [c.delta_at_entry for c in completed if c.delta_at_entry > 0]
        expected_exercise_rate = (
            sum(deltas) / len(deltas) * 100.0 if deltas else 0.0
        )

        return {
            "total_cycles": len(completed),
            "active_cycles": len(self._active_cycles),
            "total_premium": round(total_premium, 4),
            "total_pnl": round(total_pnl, 4),
            "total_fees": round(sum(c.fees_paid for c in completed), 4),
            "total_hedge_costs": round(total_hedge_costs, 4),
            "exercise_rate": round(actual_exercise_rate, 1),
            "expected_exercise_rate": round(expected_exercise_rate, 1),
            "exercise_rate_vs_expected": round(
                actual_exercise_rate - expected_exercise_rate, 1
            ),
            "win_rate": round(wins / len(completed) * 100, 1),
            "avg_premium_per_cycle": round(total_premium / len(completed), 4),
            "consecutive_losses": self._consecutive_losses,
            # Rolling attribution
            "rolling_attribution": {
                "total_premium_income": round(
                    sum(c.premium_income for c in completed), 4,
                ),
                "total_underlying_pnl": round(
                    sum(c.underlying_pnl for c in completed), 4,
                ),
                "total_hedge_costs": round(total_hedge_costs, 4),
                "net_yield": round(
                    total_premium - total_hedge_costs, 4,
                ),
            },
        }

    def get_state(self) -> dict:
        """Serialize state for persistence."""
        return {
            "active_cycles": {
                k: v.to_dict() for k, v in self._active_cycles.items()
            },
            "completed_count": len(self._completed_cycles),
            "cycle_counter": self._cycle_counter,
            "consecutive_losses": self._consecutive_losses,
        }
