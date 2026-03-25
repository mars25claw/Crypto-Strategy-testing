"""Sub-Strategy C: Delta-Neutral Volatility Selling (Deribit Required).

Sells ATM straddles or OTM strangles on Deribit, delta-hedges on Binance
perpetual futures. Profits from theta decay when RV < IV.

Includes:
- Pre-emptive hedging and velocity-based anticipation for delta lag mitigation
- Hedge queue with priority execution
- Per-cycle PnL attribution (premium, hedge cost, greek-level breakdown)
- Deribit connectivity loss monitoring with automatic hedge closure
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

from src.black_scholes import OptionGreeks, aggregate_greeks

logger = logging.getLogger(__name__)
trade_logger = logging.getLogger("trade")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DeltaNeutralPosition:
    """Tracks a delta-neutral vol-selling position."""
    position_id: str
    symbol: str              # Base asset: "BTC" or "ETH"
    binance_symbol: str      # "BTCUSDT" or "ETHUSDT"
    # Option legs (Deribit)
    strategy_type: str       # "straddle" or "strangle"
    call_instrument: str     # Deribit instrument name
    put_instrument: str
    call_strike: float
    put_strike: float
    contracts_sold: float    # Number of contracts per leg
    # Premium collected
    call_premium: float
    put_premium: float
    total_premium: float     # call + put premium in USD
    # Pricing at entry
    index_price_at_entry: float
    iv_at_entry: float
    rv_at_entry: float
    # Greeks at entry
    initial_delta: float
    initial_gamma: float
    initial_theta: float
    initial_vega: float
    # Current Greeks (updated via polling)
    current_delta: float = 0.0
    current_gamma: float = 0.0
    current_theta: float = 0.0
    current_vega: float = 0.0
    # Binance hedge
    hedge_quantity: float = 0.0     # Positive = long futures, negative = short
    hedge_avg_price: float = 0.0
    # Timing
    opened_at: float = 0.0
    expiration_at: float = 0.0
    cycle_days: int = 7
    # State
    status: str = "open"    # "open", "closed_time", "closed_profit", "closed_stop", "closed_circuit"
    # PnL tracking
    option_pnl: float = 0.0
    hedge_pnl: float = 0.0
    hedge_fees: float = 0.0
    rebalance_count: int = 0
    max_unrealized_loss: float = 0.0

    # Cycle PnL Attribution
    premium_collected: float = 0.0       # Same as total_premium but explicit
    hedge_cost_total: float = 0.0        # Cumulative hedge trading costs
    underlying_movement_pnl: float = 0.0 # PnL from underlying price change
    delta_pnl: float = 0.0              # PnL attributable to delta exposure
    gamma_pnl: float = 0.0              # PnL attributable to gamma
    theta_pnl: float = 0.0              # PnL from theta decay (positive for sellers)
    vega_pnl: float = 0.0               # PnL from IV changes

    # Price snapshots for attribution
    last_attribution_price: float = 0.0
    last_attribution_iv: float = 0.0
    last_attribution_time: float = 0.0

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

    @property
    def net_pnl(self) -> float:
        return self.option_pnl + self.hedge_pnl - self.hedge_fees

    @property
    def max_theoretical_profit(self) -> float:
        """Max profit = total premium collected."""
        return self.total_premium

    def to_dict(self) -> dict:
        return {
            "position_id": self.position_id,
            "symbol": self.symbol,
            "strategy_type": self.strategy_type,
            "call_instrument": self.call_instrument,
            "put_instrument": self.put_instrument,
            "call_strike": self.call_strike,
            "put_strike": self.put_strike,
            "contracts": self.contracts_sold,
            "total_premium": round(self.total_premium, 4),
            "iv_at_entry": round(self.iv_at_entry, 1),
            "current_delta": round(self.current_delta, 4),
            "current_gamma": round(self.current_gamma, 6),
            "current_theta": round(self.current_theta, 4),
            "current_vega": round(self.current_vega, 4),
            "hedge_qty": round(self.hedge_quantity, 6),
            "remaining_days": round(self.time_remaining_days, 2),
            "elapsed_pct": round(self.time_elapsed_pct * 100, 1),
            "status": self.status,
            "option_pnl": round(self.option_pnl, 4),
            "hedge_pnl": round(self.hedge_pnl, 4),
            "hedge_fees": round(self.hedge_fees, 4),
            "net_pnl": round(self.net_pnl, 4),
            "rebalance_count": self.rebalance_count,
            # PnL attribution
            "attribution": {
                "premium_collected": round(self.premium_collected, 4),
                "hedge_cost": round(self.hedge_cost_total, 4),
                "underlying_movement": round(self.underlying_movement_pnl, 4),
                "delta_pnl": round(self.delta_pnl, 4),
                "gamma_pnl": round(self.gamma_pnl, 4),
                "theta_pnl": round(self.theta_pnl, 4),
                "vega_pnl": round(self.vega_pnl, 4),
                "net_pnl": round(self.net_pnl, 4),
            },
        }


# ---------------------------------------------------------------------------
# DeltaNeutralManager
# ---------------------------------------------------------------------------

class DeltaNeutralManager:
    """Manages delta-neutral vol-selling positions for Sub-Strategy C.

    Requires active Deribit connection. Sells options on Deribit and
    hedges delta on Binance perpetual futures.

    Parameters
    ----------
    config : dict
        Strategy parameters.
    """

    def __init__(self, config: dict) -> None:
        self._config = config
        self._iv_rv_threshold = config.get("dn_iv_rv_threshold", 1.4)
        self._delta_rebal_threshold = config.get("dn_delta_rebal_threshold", 0.10)
        self._delta_target = config.get("dn_delta_target", 0.03)
        self._max_rebalances_per_day = config.get("dn_max_rebalances_per_day", 20)
        self._time_exit_pct = config.get("dn_time_exit_pct", 0.50)
        self._profit_target_pct = config.get("dn_profit_target_pct", 0.60)
        self._stop_loss_multiple = config.get("dn_stop_loss_multiple", 1.0)
        self._max_allocation_pct = config.get("dn_max_allocation_pct", 15.0)
        self._cycle_days = config.get("option_cycle_days", 7)

        # Greek limits
        self._max_delta_per_10k = config.get("max_net_delta_per_10k", 0.15)
        self._max_gamma_per_premium = config.get("max_gamma_per_premium", -0.005)
        self._max_vega_per_iv_point = config.get("max_vega_per_iv_point", -500.0)

        # Active positions
        self._positions: Dict[str, DeltaNeutralPosition] = {}
        self._completed: List[DeltaNeutralPosition] = []
        self._position_counter = 0

        # Daily rebalance counter (reset daily)
        self._daily_rebalances: Dict[str, int] = {}
        self._last_rebal_reset_day: str = ""

        # --- Delta Hedge Lag Mitigation ---
        # Pre-emptive threshold: 80% of rebalance threshold
        self._preemptive_delta_pct = config.get("dn_preemptive_delta_pct", 0.80)
        self._preemptive_threshold = self._delta_rebal_threshold * self._preemptive_delta_pct

        # Delta velocity tracking: position_id -> deque of (timestamp, net_delta)
        self._delta_history: Dict[str, Deque[Tuple[float, float]]] = {}
        self._delta_velocity_window = 60.0  # Track delta over last 60 seconds

        # Hedge execution timing tracking
        self._hedge_execution_times: Deque[float] = deque(maxlen=100)
        self._hedge_buffer_extra = 0.0  # Extra buffer from slow executions

        # Hedge queue with priority
        self._hedge_queue: List[dict] = []  # priority-sorted hedge requests

        # --- Deribit Connectivity Loss Monitoring ---
        self._deribit_last_seen: float = time.time()
        self._deribit_connected: bool = True
        self._deribit_disconnect_time: Optional[float] = None
        self._connectivity_loss_threshold_s = config.get(
            "deribit_connectivity_loss_seconds", 300,  # 5 minutes
        )
        self._hedges_closed_for_disconnect: Dict[str, float] = {}

        # --- Cross-cycle tracking ---
        self._total_premium_all_cycles: float = 0.0
        self._total_hedge_cost_all_cycles: float = 0.0
        self._total_net_yield_all_cycles: float = 0.0

    # ------------------------------------------------------------------
    # Entry logic (Section 3.4)
    # ------------------------------------------------------------------

    def should_enter(
        self,
        symbol: str,
        iv_rv_ratio: float,
        deribit_connected: bool,
        deribit_balance_ok: bool,
        has_upcoming_event: bool = False,
    ) -> tuple[bool, str]:
        """Evaluate whether to enter a delta-neutral position.

        Conditions:
        - IV/RV ratio > 1.4
        - No major events within expiration window
        - Sufficient Deribit account balance
        - Deribit connection active

        Returns
        -------
        (should_enter, reason)
        """
        if not deribit_connected:
            return False, "Deribit not connected (required for delta-neutral)"

        active = self._get_active_position(symbol)
        if active:
            return False, f"Active position exists: {active.position_id}"

        if iv_rv_ratio < self._iv_rv_threshold:
            return False, f"IV/RV ratio {iv_rv_ratio:.2f} < {self._iv_rv_threshold} threshold"

        if has_upcoming_event:
            return False, "Major event within expiration window"

        if not deribit_balance_ok:
            return False, "Insufficient Deribit balance"

        return True, "Conditions met for delta-neutral entry"

    def create_position(
        self,
        symbol: str,
        binance_symbol: str,
        strategy_type: str,  # "straddle" or "strangle"
        call_instrument: str,
        put_instrument: str,
        call_strike: float,
        put_strike: float,
        contracts: float,
        call_premium: float,
        put_premium: float,
        index_price: float,
        iv: float,
        rv_7d: float,
        initial_greeks: OptionGreeks,
    ) -> DeltaNeutralPosition:
        """Create a delta-neutral position after options are sold on Deribit."""
        self._position_counter += 1
        now = time.time()

        pos = DeltaNeutralPosition(
            position_id=f"DN-{symbol}-{self._position_counter:04d}",
            symbol=symbol,
            binance_symbol=binance_symbol,
            strategy_type=strategy_type,
            call_instrument=call_instrument,
            put_instrument=put_instrument,
            call_strike=call_strike,
            put_strike=put_strike,
            contracts_sold=contracts,
            call_premium=call_premium,
            put_premium=put_premium,
            total_premium=call_premium + put_premium,
            index_price_at_entry=index_price,
            iv_at_entry=iv,
            rv_at_entry=rv_7d,
            initial_delta=initial_greeks.delta,
            initial_gamma=initial_greeks.gamma,
            initial_theta=initial_greeks.theta,
            initial_vega=initial_greeks.vega,
            current_delta=initial_greeks.delta,
            current_gamma=initial_greeks.gamma,
            current_theta=initial_greeks.theta,
            current_vega=initial_greeks.vega,
            opened_at=now,
            expiration_at=now + self._cycle_days * 86400,
            cycle_days=self._cycle_days,
        )

        # Initialize attribution fields
        pos.premium_collected = pos.total_premium
        pos.last_attribution_price = index_price
        pos.last_attribution_iv = iv
        pos.last_attribution_time = now

        self._positions[pos.position_id] = pos

        # Init delta history for velocity tracking
        self._delta_history[pos.position_id] = deque(maxlen=120)

        trade_logger.info(
            "DN_OPEN\t%s\t%s\ttype=%s\tcall=%s\tput=%s\t"
            "premium=%.4f\tdelta=%.4f\tiv=%.1f",
            pos.position_id, symbol, strategy_type,
            call_instrument, put_instrument,
            pos.total_premium, initial_greeks.delta, iv,
        )

        return pos

    # ------------------------------------------------------------------
    # Delta hedging (Section 3.4 Delta Rebalancing)
    # ------------------------------------------------------------------

    def calculate_hedge_needed(
        self, position_id: str
    ) -> Optional[Tuple[float, str]]:
        """Calculate the hedge adjustment needed with pre-emptive hedging.

        Includes:
        - Standard rebalance when delta exceeds threshold
        - Pre-emptive hedging when delta approaches 80% of limit
        - Velocity-based anticipation: if delta is moving fast, start earlier
        - Extra buffer from slow execution history

        Returns
        -------
        (quantity_adjustment, reason) or None if no adjustment needed.
            Positive quantity = need to go more long on Binance
            Negative quantity = need to go more short on Binance
        """
        pos = self._positions.get(position_id)
        if not pos or pos.status != "open":
            return None

        # Check daily rebalance limit
        self._check_rebal_reset()
        daily_count = self._daily_rebalances.get(pos.symbol, 0)
        if daily_count >= self._max_rebalances_per_day:
            logger.warning(
                "Max daily rebalances (%d) reached for %s",
                self._max_rebalances_per_day, pos.symbol,
            )
            return None

        # Net delta = option delta + hedge delta
        net_delta = pos.current_delta + pos.hedge_quantity

        # Track delta history for velocity calculation
        now = time.time()
        history = self._delta_history.get(position_id)
        if history is not None:
            history.append((now, net_delta))

        # Calculate delta velocity (delta change per second)
        delta_velocity = self._calculate_delta_velocity(position_id)

        # Effective threshold: adjust for execution buffer from slow hedges
        effective_threshold = max(
            self._delta_target + 0.01,
            self._delta_rebal_threshold - self._hedge_buffer_extra,
        )

        # Velocity-based anticipation:
        # If delta is moving fast toward the threshold, start rebalancing earlier
        anticipation_offset = 0.0
        if abs(delta_velocity) > 0.001:  # meaningful velocity
            # Project delta forward by 5 seconds
            projected_delta = abs(net_delta) + abs(delta_velocity) * 5.0
            if projected_delta > effective_threshold:
                anticipation_offset = abs(delta_velocity) * 5.0
                logger.info(
                    "Velocity anticipation for %s: velocity=%.6f/s, "
                    "projected_delta=%.4f, offset=%.4f",
                    position_id, delta_velocity, projected_delta,
                    anticipation_offset,
                )

        # Decide whether to hedge
        trigger_threshold = effective_threshold - anticipation_offset
        needs_standard = abs(net_delta) > effective_threshold
        needs_preemptive = (
            abs(net_delta) > self._preemptive_threshold
            and abs(delta_velocity) > 0.002  # delta moving fast
        )
        needs_velocity = abs(net_delta) > trigger_threshold and anticipation_offset > 0

        if not (needs_standard or needs_preemptive or needs_velocity):
            return None

        # Target: bring net delta within +/- target
        if net_delta > 0:
            target = -self._delta_target
        else:
            target = self._delta_target

        adjustment = target - net_delta

        # Determine reason
        if needs_standard:
            reason_type = "standard"
        elif needs_velocity:
            reason_type = "velocity_anticipation"
        else:
            reason_type = "preemptive"

        reason = (
            f"Delta rebalance ({reason_type}): net_delta={net_delta:.4f} "
            f"(option={pos.current_delta:.4f} hedge={pos.hedge_quantity:.4f}) "
            f"adjustment={adjustment:.4f} "
            f"velocity={delta_velocity:.6f}/s buffer={self._hedge_buffer_extra:.4f}"
        )

        # Add to hedge queue with priority
        priority = 0 if needs_standard else (1 if needs_velocity else 2)
        self._enqueue_hedge(position_id, adjustment, priority, reason)

        return adjustment, reason

    def record_hedge_fill(
        self,
        position_id: str,
        quantity: float,
        fill_price: float,
        fees: float,
    ) -> None:
        """Record a hedge fill on Binance."""
        pos = self._positions.get(position_id)
        if not pos:
            return

        pos.hedge_quantity += quantity
        pos.hedge_fees += fees
        pos.rebalance_count += 1

        self._daily_rebalances[pos.symbol] = (
            self._daily_rebalances.get(pos.symbol, 0) + 1
        )

        # Update hedge average price
        if abs(pos.hedge_quantity) > 0:
            pos.hedge_avg_price = fill_price  # Simplified

        trade_logger.info(
            "DN_HEDGE\t%s\tqty=%.6f\tprice=%.2f\tfees=%.4f\t"
            "net_hedge=%.6f\trebal=%d",
            position_id, quantity, fill_price, fees,
            pos.hedge_quantity, pos.rebalance_count,
        )

    def update_greeks(
        self,
        position_id: str,
        delta: float,
        gamma: float,
        theta: float,
        vega: float,
    ) -> None:
        """Update the current Greeks from Deribit polling."""
        pos = self._positions.get(position_id)
        if not pos:
            return

        pos.current_delta = delta
        pos.current_gamma = gamma
        pos.current_theta = theta
        pos.current_vega = vega

    # ------------------------------------------------------------------
    # Exit logic (Section 4.3)
    # ------------------------------------------------------------------

    def check_exits(
        self,
        position_id: str,
        current_option_value: float,
    ) -> Optional[dict]:
        """Check exit conditions for a delta-neutral position.

        Parameters
        ----------
        position_id : str
            Position to check.
        current_option_value : float
            Current mark value of the sold options (positive = we owe this).

        Returns
        -------
        Exit action dict or None.
        """
        pos = self._positions.get(position_id)
        if not pos or pos.status != "open":
            return None

        # Unrealized PnL: premium collected - current value of options
        unrealized_option_pnl = pos.total_premium - current_option_value
        pos.option_pnl = unrealized_option_pnl

        # Track max loss
        if unrealized_option_pnl < 0:
            pos.max_unrealized_loss = max(
                pos.max_unrealized_loss, abs(unrealized_option_pnl)
            )

        # 1. Time-based exit: close 50% at 50% time elapsed
        if pos.time_elapsed_pct >= self._time_exit_pct:
            return {
                "action": "time_exit",
                "position_id": position_id,
                "close_pct": 0.50,
                "reason": (
                    f"Time exit: {pos.time_elapsed_pct*100:.0f}% elapsed "
                    f"(threshold: {self._time_exit_pct*100:.0f}%)"
                ),
            }

        # 2. Profit target: close at 60% of max theoretical profit
        profit_pct = unrealized_option_pnl / pos.total_premium if pos.total_premium > 0 else 0
        if profit_pct >= self._profit_target_pct:
            return {
                "action": "profit_exit",
                "position_id": position_id,
                "close_pct": 1.0,
                "reason": (
                    f"Profit target: {profit_pct*100:.1f}% of max "
                    f"(threshold: {self._profit_target_pct*100:.0f}%)"
                ),
            }

        # 3. Stop loss: unrealized loss > 100% of premium
        loss_multiple = abs(unrealized_option_pnl) / pos.total_premium if pos.total_premium > 0 else 0
        if unrealized_option_pnl < 0 and loss_multiple >= self._stop_loss_multiple:
            return {
                "action": "stop_exit",
                "position_id": position_id,
                "close_pct": 1.0,
                "reason": (
                    f"Stop loss: loss {loss_multiple:.1f}x premium "
                    f"(threshold: {self._stop_loss_multiple:.1f}x)"
                ),
            }

        return None

    def close_position(
        self,
        position_id: str,
        exit_type: str,
        option_close_pnl: float,
        hedge_close_pnl: float,
        fees: float,
    ) -> Optional[DeltaNeutralPosition]:
        """Close a position and record final PnL."""
        pos = self._positions.pop(position_id, None)
        if not pos:
            return None

        pos.status = exit_type
        pos.option_pnl = option_close_pnl
        pos.hedge_pnl = hedge_close_pnl
        pos.hedge_fees += fees

        self._completed.append(pos)

        trade_logger.info(
            "DN_CLOSE\t%s\texit=%s\toption_pnl=%.4f\t"
            "hedge_pnl=%.4f\tnet=%.4f\trebalances=%d",
            position_id, exit_type, option_close_pnl,
            hedge_close_pnl, pos.net_pnl, pos.rebalance_count,
        )

        return pos

    # ------------------------------------------------------------------
    # Greek limits check (Section 5.3)
    # ------------------------------------------------------------------

    def check_greek_limits(
        self, position_id: str, notional: float
    ) -> List[str]:
        """Check if any Greek limits are breached.

        Returns list of breach descriptions (empty if all OK).
        """
        pos = self._positions.get(position_id)
        if not pos:
            return []

        breaches: List[str] = []
        per_10k = notional / 10000.0 if notional > 0 else 1.0

        # Delta limit: +/-0.15 per $10k notional
        net_delta = abs(pos.current_delta + pos.hedge_quantity)
        delta_limit = self._max_delta_per_10k * per_10k
        if net_delta > delta_limit:
            breaches.append(
                f"Delta breach: {net_delta:.4f} > limit {delta_limit:.4f}"
            )

        # Gamma limit: -0.005 per dollar of premium sold
        gamma_limit = abs(self._max_gamma_per_premium * pos.total_premium)
        if abs(pos.current_gamma) > gamma_limit and gamma_limit > 0:
            breaches.append(
                f"Gamma breach: {abs(pos.current_gamma):.6f} > limit {gamma_limit:.6f}"
            )

        # Vega limit: -$500 per 1% IV
        if pos.current_vega < self._max_vega_per_iv_point:
            breaches.append(
                f"Vega breach: {pos.current_vega:.2f} < limit {self._max_vega_per_iv_point:.2f}"
            )

        return breaches

    # ------------------------------------------------------------------
    # Event risk management (Section 5.5)
    # ------------------------------------------------------------------

    def tighten_delta_for_event(self, position_id: str) -> Optional[float]:
        """Tighten delta tolerance to +/-0.05 before events.

        Returns hedge adjustment needed, or None.
        """
        pos = self._positions.get(position_id)
        if not pos or pos.status != "open":
            return None

        net_delta = pos.current_delta + pos.hedge_quantity
        tightened_threshold = 0.05

        if abs(net_delta) > tightened_threshold:
            adjustment = -net_delta
            logger.info(
                "Tightening delta for event: %s net_delta=%.4f adj=%.4f",
                position_id, net_delta, adjustment,
            )
            return adjustment

        return None

    # ------------------------------------------------------------------
    # Delta Hedge Lag Mitigation
    # ------------------------------------------------------------------

    def _calculate_delta_velocity(self, position_id: str) -> float:
        """Calculate the rate of change of net delta (delta per second).

        Uses the delta history buffer to compute a linear regression slope
        over the velocity window.
        """
        history = self._delta_history.get(position_id)
        if not history or len(history) < 2:
            return 0.0

        now = time.time()
        cutoff = now - self._delta_velocity_window
        recent = [(t, d) for t, d in history if t >= cutoff]

        if len(recent) < 2:
            return 0.0

        # Simple linear regression: velocity = delta_change / time_change
        t_first, d_first = recent[0]
        t_last, d_last = recent[-1]
        dt = t_last - t_first

        if dt < 1.0:
            return 0.0

        return (d_last - d_first) / dt

    def _enqueue_hedge(
        self, position_id: str, adjustment: float, priority: int, reason: str,
    ) -> None:
        """Add a hedge request to the priority queue.

        Priority: 0 = highest (standard breach), 1 = velocity, 2 = preemptive.
        """
        # Remove any existing request for this position
        self._hedge_queue = [
            h for h in self._hedge_queue if h["position_id"] != position_id
        ]
        self._hedge_queue.append({
            "position_id": position_id,
            "adjustment": adjustment,
            "priority": priority,
            "reason": reason,
            "enqueued_at": time.time(),
        })
        # Sort by priority (lower = higher priority)
        self._hedge_queue.sort(key=lambda h: h["priority"])

    def get_next_hedge(self) -> Optional[dict]:
        """Pop the highest-priority hedge request from the queue."""
        if self._hedge_queue:
            return self._hedge_queue.pop(0)
        return None

    def get_hedge_queue_size(self) -> int:
        """Return the number of pending hedge requests."""
        return len(self._hedge_queue)

    def record_hedge_execution_time(self, execution_ms: float) -> None:
        """Record hedge execution latency and adjust buffer if slow.

        If execution takes > 500ms, log WARNING and add extra buffer to
        future rebalance thresholds.
        """
        self._hedge_execution_times.append(execution_ms)

        if execution_ms > 500.0:
            # Add extra buffer proportional to how slow it was
            extra = min(0.02, (execution_ms - 500.0) / 10000.0)
            self._hedge_buffer_extra = min(
                0.05,  # Cap at 0.05 additional buffer
                self._hedge_buffer_extra + extra,
            )
            logger.warning(
                "Hedge execution slow: %.0fms > 500ms. "
                "Extra buffer now: %.4f",
                execution_ms, self._hedge_buffer_extra,
            )
        else:
            # Decay the buffer slowly when executions are fast
            self._hedge_buffer_extra = max(
                0.0, self._hedge_buffer_extra - 0.001,
            )

    # ------------------------------------------------------------------
    # PnL Attribution (per-cycle Greek-level breakdown)
    # ------------------------------------------------------------------

    def update_pnl_attribution(
        self,
        position_id: str,
        current_price: float,
        current_iv: float,
    ) -> None:
        """Update per-cycle PnL attribution based on Greek decomposition.

        Breaks down PnL into:
        - delta_pnl: delta * dS (price change)
        - gamma_pnl: 0.5 * gamma * dS^2
        - theta_pnl: theta * dt (time decay)
        - vega_pnl: vega * d(IV) (IV change)

        Parameters
        ----------
        position_id : str
            Position to update.
        current_price : float
            Current underlying price.
        current_iv : float
            Current implied volatility (percentage).
        """
        pos = self._positions.get(position_id)
        if not pos or pos.status != "open":
            return

        now = time.time()

        # Skip if first call (need previous snapshot)
        if pos.last_attribution_time == 0 or pos.last_attribution_price == 0:
            pos.last_attribution_price = current_price
            pos.last_attribution_iv = current_iv
            pos.last_attribution_time = now
            return

        # Compute changes since last attribution
        dS = current_price - pos.last_attribution_price
        dt_days = (now - pos.last_attribution_time) / 86400.0
        dIV = current_iv - pos.last_attribution_iv

        # Greek-level PnL decomposition (for short options, negate)
        # pos.current_delta etc. are for the sold options position
        delta_contribution = pos.current_delta * dS
        gamma_contribution = 0.5 * pos.current_gamma * dS * dS
        theta_contribution = pos.current_theta * dt_days  # theta is per-day
        vega_contribution = pos.current_vega * dIV  # vega is per 1% IV

        # Accumulate
        pos.delta_pnl += delta_contribution
        pos.gamma_pnl += gamma_contribution
        pos.theta_pnl += theta_contribution
        pos.vega_pnl += vega_contribution
        pos.underlying_movement_pnl += dS * pos.contracts_sold
        pos.hedge_cost_total = pos.hedge_fees

        # Update snapshots
        pos.last_attribution_price = current_price
        pos.last_attribution_iv = current_iv
        pos.last_attribution_time = now

    def get_cycle_attribution(self, position_id: str) -> Optional[dict]:
        """Get the full PnL attribution for a position.

        Returns per-cycle breakdown: premium collected, hedge cost,
        underlying movement, and Greek-level PnL.
        """
        pos = self._positions.get(position_id)
        if not pos:
            # Check completed
            for p in self._completed:
                if p.position_id == position_id:
                    pos = p
                    break
        if not pos:
            return None

        return {
            "position_id": pos.position_id,
            "premium_collected": round(pos.premium_collected, 4),
            "hedge_cost": round(pos.hedge_cost_total, 4),
            "underlying_movement": round(pos.underlying_movement_pnl, 4),
            "delta_pnl": round(pos.delta_pnl, 4),
            "gamma_pnl": round(pos.gamma_pnl, 4),
            "theta_pnl": round(pos.theta_pnl, 4),
            "vega_pnl": round(pos.vega_pnl, 4),
            "net_pnl": round(pos.net_pnl, 4),
        }

    def get_rolling_attribution(self) -> dict:
        """Get rolling attribution across all completed cycles.

        Returns cumulative: premium income vs hedge costs vs Greek-level PnL.
        """
        all_positions = list(self._completed) + [
            p for p in self._positions.values() if p.status == "open"
        ]

        if not all_positions:
            return {
                "total_premium": 0.0,
                "total_hedge_cost": 0.0,
                "total_delta_pnl": 0.0,
                "total_gamma_pnl": 0.0,
                "total_theta_pnl": 0.0,
                "total_vega_pnl": 0.0,
                "net_yield": 0.0,
            }

        total_premium = sum(p.premium_collected for p in all_positions)
        total_hedge = sum(p.hedge_cost_total for p in all_positions)
        total_delta = sum(p.delta_pnl for p in all_positions)
        total_gamma = sum(p.gamma_pnl for p in all_positions)
        total_theta = sum(p.theta_pnl for p in all_positions)
        total_vega = sum(p.vega_pnl for p in all_positions)

        return {
            "total_premium": round(total_premium, 4),
            "total_hedge_cost": round(total_hedge, 4),
            "total_delta_pnl": round(total_delta, 4),
            "total_gamma_pnl": round(total_gamma, 4),
            "total_theta_pnl": round(total_theta, 4),
            "total_vega_pnl": round(total_vega, 4),
            "net_yield": round(total_premium - total_hedge, 4),
        }

    # ------------------------------------------------------------------
    # Deribit Connectivity Loss Monitoring (Section 11.5)
    # ------------------------------------------------------------------

    def update_deribit_heartbeat(self) -> None:
        """Record that Deribit is still reachable. Call on every successful response."""
        self._deribit_last_seen = time.time()
        if not self._deribit_connected:
            self._deribit_connected = True
            self._deribit_disconnect_time = None
            logger.info("Deribit connection restored")

    def check_deribit_connectivity(self) -> Optional[dict]:
        """Check if Deribit connection has been lost for > 5 minutes.

        If lost > 5 minutes:
        - Returns action dict to close all Binance hedges
        - Logs CRITICAL alert
        - The caller should close Binance hedge positions

        Returns
        -------
        dict with action details, or None if connection is OK.
        """
        now = time.time()
        seconds_since_last = now - self._deribit_last_seen

        if seconds_since_last <= self._connectivity_loss_threshold_s:
            # Connection OK or within tolerance
            if not self._deribit_connected:
                self._deribit_connected = True
                self._deribit_disconnect_time = None
            return None

        # Connection lost
        if self._deribit_connected:
            self._deribit_connected = False
            self._deribit_disconnect_time = now
            logger.warning(
                "Deribit connection lost (last seen %.0fs ago)",
                seconds_since_last,
            )

        # Check if threshold exceeded
        if self._deribit_disconnect_time is not None:
            disconnect_duration = now - self._deribit_disconnect_time
            if disconnect_duration >= self._connectivity_loss_threshold_s:
                # Threshold exceeded: close all Binance hedges
                positions_to_close = []
                for pos_id, pos in self._positions.items():
                    if pos.status == "open" and abs(pos.hedge_quantity) > 0:
                        if pos_id not in self._hedges_closed_for_disconnect:
                            positions_to_close.append(pos_id)
                            self._hedges_closed_for_disconnect[pos_id] = now

                if positions_to_close:
                    logger.critical(
                        "DERIBIT CONNECTIVITY LOSS > %ds: "
                        "Closing Binance hedges for %d positions. "
                        "Positions convert to UNHEDGED vol sale. "
                        "CRITICAL RISK ALERT.",
                        self._connectivity_loss_threshold_s,
                        len(positions_to_close),
                    )

                    return {
                        "action": "close_binance_hedges",
                        "reason": (
                            f"Deribit unreachable for "
                            f"{disconnect_duration:.0f}s (>{self._connectivity_loss_threshold_s}s)"
                        ),
                        "positions": positions_to_close,
                        "alert_level": "CRITICAL",
                    }

        return None

    def on_deribit_reconnected(self) -> List[str]:
        """Handle Deribit reconnection: resume hedging for affected positions.

        Returns list of position IDs that need hedge re-establishment.
        """
        self._deribit_connected = True
        self._deribit_disconnect_time = None
        self._deribit_last_seen = time.time()

        positions_to_rehedge = list(self._hedges_closed_for_disconnect.keys())
        self._hedges_closed_for_disconnect.clear()

        if positions_to_rehedge:
            logger.info(
                "Deribit reconnected. Resuming hedging for %d positions: %s",
                len(positions_to_rehedge), positions_to_rehedge,
            )

        return positions_to_rehedge

    @property
    def deribit_connected(self) -> bool:
        return self._deribit_connected

    def get_connectivity_status(self) -> dict:
        """Return Deribit connectivity status."""
        now = time.time()
        return {
            "connected": self._deribit_connected,
            "last_seen_seconds_ago": round(now - self._deribit_last_seen, 1),
            "disconnect_duration": (
                round(now - self._deribit_disconnect_time, 1)
                if self._deribit_disconnect_time else 0
            ),
            "hedges_closed_for_disconnect": len(self._hedges_closed_for_disconnect),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_active_position(self, symbol: str) -> Optional[DeltaNeutralPosition]:
        for pos in self._positions.values():
            if pos.symbol == symbol and pos.status == "open":
                return pos
        return None

    def _check_rebal_reset(self) -> None:
        """Reset daily rebalance counters if the day has changed."""
        import datetime
        today = datetime.datetime.utcnow().strftime("%Y-%m-%d")
        if today != self._last_rebal_reset_day:
            self._daily_rebalances.clear()
            self._last_rebal_reset_day = today

    def get_active_positions(self) -> Dict[str, DeltaNeutralPosition]:
        return dict(self._positions)

    def get_completed_positions(self) -> List[DeltaNeutralPosition]:
        return list(self._completed)

    def get_aggregate_greeks(self) -> OptionGreeks:
        """Aggregate Greeks across all active delta-neutral positions."""
        agg = OptionGreeks()
        for pos in self._positions.values():
            if pos.status != "open":
                continue
            agg.delta += pos.current_delta + pos.hedge_quantity
            agg.gamma += pos.current_gamma
            agg.theta += pos.current_theta
            agg.vega += pos.current_vega
        return agg

    def get_metrics(self) -> dict:
        completed = self._completed
        active = [p for p in self._positions.values() if p.status == "open"]

        if not completed:
            return {
                "total_trades": 0,
                "active_positions": len(active),
                "total_premium": sum(p.total_premium for p in active),
                "total_rebalances": sum(p.rebalance_count for p in active),
            }

        total_net = sum(p.net_pnl for p in completed)
        total_premium = sum(p.total_premium for p in completed)
        total_hedge_costs = sum(p.hedge_fees for p in completed)
        total_rebalances = sum(p.rebalance_count for p in completed)
        wins = sum(1 for p in completed if p.net_pnl > 0)

        return {
            "total_trades": len(completed),
            "active_positions": len(active),
            "total_premium_collected": round(total_premium, 4),
            "total_net_pnl": round(total_net, 4),
            "total_hedge_costs": round(total_hedge_costs, 4),
            "hedge_cost_ratio": round(
                total_hedge_costs / total_premium * 100 if total_premium > 0 else 0, 1
            ),
            "win_rate": round(wins / len(completed) * 100, 1) if completed else 0,
            "total_rebalances": total_rebalances,
            "avg_rebalances_per_trade": round(
                total_rebalances / len(completed), 1
            ) if completed else 0,
            # Rolling attribution
            "rolling_attribution": self.get_rolling_attribution(),
            # Connectivity
            "deribit_connectivity": self.get_connectivity_status(),
        }

    def get_state(self) -> dict:
        return {
            "positions": {
                k: v.to_dict() for k, v in self._positions.items()
            },
            "completed_count": len(self._completed),
            "position_counter": self._position_counter,
            "hedge_buffer_extra": self._hedge_buffer_extra,
            "deribit_connectivity": self.get_connectivity_status(),
        }
