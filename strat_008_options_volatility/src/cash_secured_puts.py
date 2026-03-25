"""Sub-Strategy B: Cash-Secured Put Selling.

Sells OTM puts with USDT collateral via synthetic implementation on Binance
(LIMIT BUY at the strike price). 7-day cycles with auto-rolling.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.black_scholes import put_price, price_option, delta as bs_delta

logger = logging.getLogger(__name__)
trade_logger = logging.getLogger("trade")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CashSecuredPutCycle:
    """Tracks one cash-secured put cycle."""
    cycle_id: str
    symbol: str
    # Put parameters
    strike_price: float
    otm_pct: float
    premium: float         # BS-calculated theoretical premium
    iv_at_entry: float
    rv_at_entry: float
    # Collateral
    reserved_usdt: float   # Strike * Quantity
    quantity: float         # How much asset we'd buy if exercised
    spot_price_at_entry: float
    # Timing
    opened_at: float
    expiration_at: float
    cycle_days: int = 7
    # State
    status: str = "open"   # "open", "exercised", "expired", "early_exit"
    # Binance order
    limit_buy_order_id: Optional[int] = None
    # PnL
    realized_pnl: float = 0.0
    fees_paid: float = 0.0

    # Cycle PnL attribution
    premium_income: float = 0.0       # Premium from selling the put
    underlying_pnl: float = 0.0       # PnL from underlying if exercised
    hedge_costs: float = 0.0
    net_cycle_pnl: float = 0.0

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
            "strike": self.strike_price,
            "otm_pct": round(self.otm_pct * 100, 2),
            "premium": round(self.premium, 4),
            "iv_at_entry": round(self.iv_at_entry, 1),
            "rv_at_entry": round(self.rv_at_entry, 1),
            "reserved_usdt": round(self.reserved_usdt, 2),
            "quantity": self.quantity,
            "spot_at_entry": round(self.spot_price_at_entry, 2),
            "opened_at": self.opened_at,
            "expiration_at": self.expiration_at,
            "remaining_days": round(self.time_remaining_days, 2),
            "elapsed_pct": round(self.time_elapsed_pct * 100, 1),
            "status": self.status,
            "realized_pnl": round(self.realized_pnl, 4),
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
# CashSecuredPutManager
# ---------------------------------------------------------------------------

class CashSecuredPutManager:
    """Manages cash-secured put positions for STRAT-008 Sub-Strategy B.

    In Synthetic Mode (Binance only):
    - Place LIMIT BUY at the strike price
    - Reserve USDT = Strike x Quantity (fully cash-secured)
    - If price drops to strike, buy fills (put "exercised")
    - Premium tracked theoretically via Black-Scholes

    Parameters
    ----------
    config : dict
        Strategy parameters.
    """

    def __init__(self, config: dict) -> None:
        self._config = config
        self._risk_free_rate = config.get("risk_free_rate", 0.05)
        self._otm_default = config.get("csp_otm_pct_default", 0.05)
        self._otm_min = config.get("csp_otm_pct_min", 0.03)
        self._max_allocation_pct = config.get("csp_max_allocation_pct", 10.0)
        self._cycle_days = config.get("option_cycle_days", 7)
        self._rsi_floor = config.get("csp_rsi_floor", 40)
        self._crash_7d_pct = config.get("crash_7d_pct", 10.0)
        self._crash_rsi_floor = config.get("crash_rsi_floor", 30)

        # Active cycles
        self._active_cycles: Dict[str, CashSecuredPutCycle] = {}
        self._completed_cycles: List[CashSecuredPutCycle] = []
        self._cycle_counter = 0
        self._consecutive_losses = 0
        # Total USDT currently reserved across all active cycles
        self._total_reserved_usdt = 0.0

    # ------------------------------------------------------------------
    # Entry logic (Section 3.3)
    # ------------------------------------------------------------------

    def should_enter(
        self,
        symbol: str,
        iv_rv_ratio: float,
        rsi_14_daily: float,
        price_7d_change_pct: float,
        available_usdt: float,
        equity: float,
    ) -> tuple[bool, str]:
        """Evaluate whether to enter a new cash-secured put cycle.

        Conditions:
        - Holding USDT (not deployed in other strategies)
        - IV/RV ratio > 1.2
        - Asset in uptrend or neutral (not in a crash)
        - RSI(14) daily > 40
        - Price not dropped > 10% in 7 days
        - RSI(14) daily > 30 (not oversold crash)

        Returns
        -------
        (should_enter, reason)
        """
        # Check for existing cycle
        active = self._get_active_cycle(symbol)
        if active:
            return False, f"Active cycle exists: {active.cycle_id}"

        if self._consecutive_losses >= 3:
            return False, f"Halted: {self._consecutive_losses} consecutive losses"

        # IV/RV ratio
        if iv_rv_ratio < 1.2:
            return False, f"IV/RV ratio {iv_rv_ratio:.2f} < 1.2"

        # RSI check
        if rsi_14_daily < self._rsi_floor:
            return False, f"RSI {rsi_14_daily:.1f} < floor {self._rsi_floor}"

        # Crash filter: 7-day drop
        if price_7d_change_pct < -self._crash_7d_pct:
            return False, (
                f"7-day drop {price_7d_change_pct:.1f}% exceeds "
                f"-{self._crash_7d_pct}% crash filter"
            )

        # RSI crash floor
        if rsi_14_daily < self._crash_rsi_floor:
            return False, f"RSI {rsi_14_daily:.1f} < crash floor {self._crash_rsi_floor}"

        # Available USDT check
        max_alloc = equity * (self._max_allocation_pct / 100.0)
        remaining_alloc = max_alloc - self._total_reserved_usdt
        if remaining_alloc <= 0 or available_usdt <= 0:
            return False, "Insufficient available USDT"

        return True, "Conditions met for cash-secured put entry"

    def calculate_entry(
        self,
        symbol: str,
        current_price: float,
        available_usdt: float,
        equity: float,
        iv: float,
        rv_7d: float,
    ) -> CashSecuredPutCycle:
        """Calculate cash-secured put parameters.

        Parameters
        ----------
        symbol : str
            Trading pair.
        current_price : float
            Current spot price.
        available_usdt : float
            Available USDT for collateral.
        equity : float
            Total equity.
        iv : float
            Current IV in percentage.
        rv_7d : float
            7-day RV in percentage.

        Returns
        -------
        CashSecuredPutCycle ready for execution.
        """
        # Determine OTM percentage
        otm_pct = self._otm_default
        # Adjust OTM based on IV level: higher IV -> wider OTM
        if iv > 80:
            otm_pct = self._otm_default  # 5%
        elif iv < 50:
            otm_pct = self._otm_min  # 3%

        # Calculate strike
        strike = current_price * (1.0 - otm_pct)

        # Calculate quantity: min of available USDT and max allocation
        max_alloc = equity * (self._max_allocation_pct / 100.0)
        remaining_alloc = max_alloc - self._total_reserved_usdt
        usable_usdt = min(available_usdt, remaining_alloc)

        quantity = usable_usdt / strike if strike > 0 else 0.0
        reserved = strike * quantity

        # Calculate theoretical premium via Black-Scholes
        T = self._cycle_days / 365.0
        sigma = iv / 100.0
        premium_per_unit = put_price(
            S=current_price,
            K=strike,
            T=T,
            r=self._risk_free_rate,
            sigma=sigma,
        )
        total_premium = premium_per_unit * quantity

        # Compute delta at entry for exercise rate tracking
        # For puts, delta is negative; use abs for probability interpretation
        entry_delta = abs(bs_delta(
            S=current_price, K=strike, T=T, r=self._risk_free_rate,
            sigma=sigma, option_type="put",
        ))

        # Create cycle
        self._cycle_counter += 1
        now = time.time()
        cycle = CashSecuredPutCycle(
            cycle_id=f"CSP-{symbol}-{self._cycle_counter:04d}",
            symbol=symbol,
            strike_price=strike,
            otm_pct=otm_pct,
            premium=total_premium,
            iv_at_entry=iv,
            rv_at_entry=rv_7d,
            reserved_usdt=reserved,
            quantity=quantity,
            spot_price_at_entry=current_price,
            opened_at=now,
            expiration_at=now + self._cycle_days * 86400,
            cycle_days=self._cycle_days,
            premium_income=total_premium,
            delta_at_entry=entry_delta,
        )

        logger.info(
            "CSP calculated: %s strike=%.2f (%.1f%% OTM) premium=%.4f "
            "qty=%.6f reserved=$%.2f IV=%.1f%%",
            cycle.cycle_id, strike, otm_pct * 100, total_premium,
            quantity, reserved, iv,
        )

        return cycle

    def activate_cycle(
        self, cycle: CashSecuredPutCycle, order_id: Optional[int] = None
    ) -> None:
        """Register active cycle after limit buy order is placed."""
        cycle.limit_buy_order_id = order_id
        self._active_cycles[cycle.cycle_id] = cycle
        self._total_reserved_usdt += cycle.reserved_usdt

        trade_logger.info(
            "CSP_OPEN\t%s\t%s\tstrike=%.2f\tpremium=%.4f\t"
            "qty=%.6f\treserved=$%.2f\tiv=%.1f\texpiry=%.0f",
            cycle.cycle_id, cycle.symbol, cycle.strike_price,
            cycle.premium, cycle.quantity, cycle.reserved_usdt,
            cycle.iv_at_entry, cycle.expiration_at,
        )

    # ------------------------------------------------------------------
    # Exit logic (Section 4.2)
    # ------------------------------------------------------------------

    def check_exits(
        self,
        symbol: str,
        current_price: float,
        daily_atr: float,
    ) -> List[dict]:
        """Check all active cycles for exit conditions.

        Returns list of exit actions.
        """
        actions: List[dict] = []
        cycle = self._get_active_cycle(symbol)
        if not cycle:
            return actions

        now = time.time()

        # 1. Strike reached (put "exercised"): buy order fills
        if current_price <= cycle.strike_price:
            actions.append({
                "action": "exercise",
                "cycle_id": cycle.cycle_id,
                "reason": (
                    f"Price {current_price:.2f} reached strike "
                    f"{cycle.strike_price:.2f} — buy fills"
                ),
            })
            return actions

        # 2. Expiration
        if now >= cycle.expiration_at:
            actions.append({
                "action": "expire",
                "cycle_id": cycle.cycle_id,
                "reason": "Cycle expired — premium kept, cancel buy order",
            })
            return actions

        # 3. Early exit: crash > 2 ATR approaching strike
        if daily_atr > 0:
            drop = cycle.spot_price_at_entry - current_price
            if drop > 2 * daily_atr:
                distance_to_strike = current_price - cycle.strike_price
                if distance_to_strike < daily_atr:
                    actions.append({
                        "action": "early_exit_crash",
                        "cycle_id": cycle.cycle_id,
                        "reason": (
                            f"Crash: drop {drop:.2f} > 2*ATR {2*daily_atr:.2f}, "
                            f"approaching strike (dist={distance_to_strike:.2f})"
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
    ) -> Optional[CashSecuredPutCycle]:
        """Close a cycle and compute PnL."""
        cycle = self._active_cycles.pop(cycle_id, None)
        if not cycle:
            logger.warning("CSP cycle %s not found for closing", cycle_id)
            return None

        cycle.status = exit_type
        cycle.fees_paid = fees
        self._total_reserved_usdt -= cycle.reserved_usdt

        if exit_type == "exercised":
            # Asset purchased at strike. Effective cost = strike - premium
            effective_cost = cycle.strike_price - (cycle.premium / cycle.quantity if cycle.quantity > 0 else 0)
            if exit_price > 0:
                unrealized = (exit_price - effective_cost) * cycle.quantity
            else:
                unrealized = 0.0
            cycle.realized_pnl = cycle.premium - fees
            cycle.premium_income = cycle.premium
            cycle.underlying_pnl = unrealized
        elif exit_type == "expired":
            # Premium kept in full
            cycle.realized_pnl = cycle.premium - fees
            cycle.premium_income = cycle.premium
            cycle.underlying_pnl = 0.0
        else:
            # Early exit: partial premium
            kept_premium = cycle.premium * cycle.time_elapsed_pct
            cycle.realized_pnl = kept_premium - fees
            cycle.premium_income = kept_premium
            cycle.underlying_pnl = 0.0

        cycle.net_cycle_pnl = cycle.realized_pnl

        if cycle.realized_pnl < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

        self._completed_cycles.append(cycle)

        trade_logger.info(
            "CSP_CLOSE\t%s\t%s\texit=%s\tpnl=%.4f\tpremium=%.4f\tfees=%.4f",
            cycle.cycle_id, cycle.symbol, exit_type,
            cycle.realized_pnl, cycle.premium, fees,
        )

        return cycle

    # ------------------------------------------------------------------
    # Auto-rolling
    # ------------------------------------------------------------------

    def should_roll(self, symbol: str, iv_rv_ratio: float) -> bool:
        """Check if a new cycle should start after expiration."""
        active = self._get_active_cycle(symbol)
        if active:
            return False
        if iv_rv_ratio < 1.2:
            return False
        if self._consecutive_losses >= 3:
            return False
        return True

    # ------------------------------------------------------------------
    # Current value
    # ------------------------------------------------------------------

    def get_current_premium_value(
        self, cycle_id: str, current_price: float, current_iv: float,
    ) -> float:
        """Current theoretical value of the sold put."""
        cycle = self._active_cycles.get(cycle_id)
        if not cycle:
            return 0.0

        T = cycle.time_remaining_days / 365.0
        if T <= 0:
            return max(cycle.strike_price - current_price, 0.0) * cycle.quantity

        sigma = current_iv / 100.0
        current_put_value = put_price(
            S=current_price,
            K=cycle.strike_price,
            T=T,
            r=self._risk_free_rate,
            sigma=sigma,
        )
        return current_put_value * cycle.quantity

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_active_cycle(self, symbol: str) -> Optional[CashSecuredPutCycle]:
        for cycle in self._active_cycles.values():
            if cycle.symbol == symbol and cycle.status == "open":
                return cycle
        return None

    def get_active_cycles(self) -> Dict[str, CashSecuredPutCycle]:
        return dict(self._active_cycles)

    def get_completed_cycles(self) -> List[CashSecuredPutCycle]:
        return list(self._completed_cycles)

    @property
    def total_reserved_usdt(self) -> float:
        return self._total_reserved_usdt

    @property
    def consecutive_losses(self) -> int:
        return self._consecutive_losses

    def get_metrics(self) -> dict:
        completed = self._completed_cycles
        if not completed:
            return {
                "total_cycles": 0,
                "active_cycles": len(self._active_cycles),
                "total_premium": 0.0,
                "total_pnl": 0.0,
                "exercise_rate": 0.0,
                "expected_exercise_rate": 0.0,
                "reserved_usdt": self._total_reserved_usdt,
            }

        total_pnl = sum(c.realized_pnl for c in completed)
        total_premium = sum(c.premium for c in completed)
        total_hedge_costs = sum(c.hedge_costs for c in completed)
        exercised = sum(1 for c in completed if c.status == "exercised")
        wins = sum(1 for c in completed if c.realized_pnl > 0)

        # Exercise rate tracking: actual vs expected
        actual_exercise_rate = exercised / len(completed) * 100.0
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
            "reserved_usdt": round(self._total_reserved_usdt, 2),
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
                "net_yield": round(total_premium - total_hedge_costs, 4),
            },
        }

    def get_state(self) -> dict:
        return {
            "active_cycles": {
                k: v.to_dict() for k, v in self._active_cycles.items()
            },
            "completed_count": len(self._completed_cycles),
            "cycle_counter": self._cycle_counter,
            "consecutive_losses": self._consecutive_losses,
            "total_reserved_usdt": self._total_reserved_usdt,
        }
