"""Comprehensive performance metrics for STRAT-009 Signal-Enhanced DCA.

Implements:
- Section 10.1: Standard DCA metrics (Total Return, Avg Cost Basis, etc.)
- Section 10.2: Strategy-specific metrics (Signal vs Vanilla DCA, Crash-Buy
  Performance, Signal Multiplier Distribution, Cost Basis Improvement,
  Unrealised PnL, Monthly Spending Tracker, Units per Dollar)
- Section 10.3: Dimensional breakdowns (by instrument, by signal range,
  by time period) and dashboard integration
- Vanilla DCA comparison engine
- Go-live criteria evaluation (Section 9.3)
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Vanilla DCA Tracker
# ---------------------------------------------------------------------------

@dataclass
class VanillaDCASnapshot:
    """Tracks what a fixed-amount DCA would have achieved for comparison."""

    symbol: str
    total_invested: float = 0.0
    total_units: float = 0.0

    @property
    def avg_cost_basis(self) -> float:
        if self.total_units <= 0:
            return 0.0
        return self.total_invested / self.total_units

    def record_hypothetical_purchase(self, base_amount: float, price: float) -> None:
        """Record a hypothetical fixed-amount purchase at the same time as
        the real signal-enhanced purchase."""
        if price <= 0 or base_amount <= 0:
            return
        qty = base_amount / price
        self.total_invested += base_amount
        self.total_units += qty

    def current_value(self, price: float) -> float:
        return self.total_units * price

    def return_pct(self, price: float) -> float:
        if self.total_invested <= 0:
            return 0.0
        return ((self.current_value(price) - self.total_invested) / self.total_invested) * 100.0

    def to_dict(self, current_price: float) -> dict:
        return {
            "symbol": self.symbol,
            "total_invested": round(self.total_invested, 2),
            "total_units": self.total_units,
            "avg_cost_basis": round(self.avg_cost_basis, 8),
            "current_value": round(self.current_value(current_price), 2),
            "return_pct": round(self.return_pct(current_price), 2),
        }


class VanillaDCAComparison:
    """Compare signal-enhanced DCA performance against fixed-amount vanilla DCA.

    At each scheduled DCA purchase, records what a vanilla DCA would have
    bought (fixed base amount at the same price). This allows direct comparison.
    """

    def __init__(self, base_amounts: Dict[str, float]) -> None:
        self._base_amounts = base_amounts
        self._snapshots: Dict[str, VanillaDCASnapshot] = {
            sym: VanillaDCASnapshot(symbol=sym) for sym in base_amounts
        }

    def record_purchase(self, symbol: str, price: float) -> None:
        """Record a vanilla purchase at the same time as the real one."""
        base = self._base_amounts.get(symbol, 0.0)
        snap = self._snapshots.get(symbol)
        if snap is None:
            snap = VanillaDCASnapshot(symbol=symbol)
            self._snapshots[symbol] = snap
        snap.record_hypothetical_purchase(base, price)

    def get_comparison(
        self,
        enhanced_invested: Dict[str, float],
        enhanced_units: Dict[str, float],
        current_prices: Dict[str, float],
    ) -> Dict[str, Any]:
        """Generate full comparison between signal-enhanced and vanilla DCA.

        Returns per-instrument and aggregate comparison metrics including:
        - Vanilla cost basis, units, return %
        - Cost basis improvement %
        - Units improvement %
        - Return improvement %
        """
        per_instrument: Dict[str, Dict[str, Any]] = {}
        total_vanilla_invested = 0.0
        total_vanilla_value = 0.0
        total_enhanced_invested = 0.0
        total_enhanced_value = 0.0

        for symbol, snap in self._snapshots.items():
            price = current_prices.get(symbol, 0.0)
            vanilla = snap.to_dict(price)

            enh_invested = enhanced_invested.get(symbol, 0.0)
            enh_units = enhanced_units.get(symbol, 0.0)
            enh_cost_basis = enh_invested / enh_units if enh_units > 0 else 0.0
            enh_value = enh_units * price
            enh_return = ((enh_value - enh_invested) / enh_invested * 100.0) if enh_invested > 0 else 0.0

            # Improvements
            cost_basis_improvement = 0.0
            if snap.avg_cost_basis > 0 and enh_cost_basis > 0:
                cost_basis_improvement = ((snap.avg_cost_basis - enh_cost_basis) / snap.avg_cost_basis) * 100.0

            units_improvement = 0.0
            if snap.total_units > 0:
                units_improvement = ((enh_units - snap.total_units) / snap.total_units) * 100.0

            return_improvement = enh_return - vanilla["return_pct"]

            per_instrument[symbol] = {
                "vanilla": vanilla,
                "enhanced": {
                    "total_invested": round(enh_invested, 2),
                    "total_units": enh_units,
                    "avg_cost_basis": round(enh_cost_basis, 8),
                    "current_value": round(enh_value, 2),
                    "return_pct": round(enh_return, 2),
                },
                "cost_basis_improvement_pct": round(cost_basis_improvement, 2),
                "units_improvement_pct": round(units_improvement, 2),
                "return_improvement_pct": round(return_improvement, 2),
            }

            total_vanilla_invested += snap.total_invested
            total_vanilla_value += snap.current_value(price)
            total_enhanced_invested += enh_invested
            total_enhanced_value += enh_value

        # Aggregate
        agg_vanilla_return = (
            ((total_vanilla_value - total_vanilla_invested) / total_vanilla_invested * 100.0)
            if total_vanilla_invested > 0
            else 0.0
        )
        agg_enhanced_return = (
            ((total_enhanced_value - total_enhanced_invested) / total_enhanced_invested * 100.0)
            if total_enhanced_invested > 0
            else 0.0
        )

        return {
            "per_instrument": per_instrument,
            "aggregate": {
                "vanilla_invested": round(total_vanilla_invested, 2),
                "vanilla_value": round(total_vanilla_value, 2),
                "vanilla_return_pct": round(agg_vanilla_return, 2),
                "enhanced_invested": round(total_enhanced_invested, 2),
                "enhanced_value": round(total_enhanced_value, 2),
                "enhanced_return_pct": round(agg_enhanced_return, 2),
                "return_improvement_pct": round(agg_enhanced_return - agg_vanilla_return, 2),
            },
        }

    def to_state(self) -> dict:
        return {
            sym: {"total_invested": s.total_invested, "total_units": s.total_units}
            for sym, s in self._snapshots.items()
        }

    def load_state(self, state: dict) -> None:
        for sym, data in state.items():
            if sym not in self._snapshots:
                self._snapshots[sym] = VanillaDCASnapshot(symbol=sym)
            self._snapshots[sym].total_invested = data.get("total_invested", 0.0)
            self._snapshots[sym].total_units = data.get("total_units", 0.0)


# ---------------------------------------------------------------------------
# Section 10.2 Strategy Metrics
# ---------------------------------------------------------------------------

class DCAStrategyMetrics:
    """Comprehensive metrics tracker for Section 10.2 and 10.3.

    Tracks and computes:
    - Total Return %
    - Avg Cost Basis per instrument
    - Signal-Enhanced vs Vanilla DCA comparison
    - Units Accumulated per Dollar Spent
    - Cost Basis Improvement
    - Crash-Buy Performance vs Regular DCA
    - Signal Multiplier Distribution
    - Unrealised PnL per instrument
    - Monthly spending tracker
    """

    def __init__(self, instruments: List[str], base_amounts: Dict[str, float]) -> None:
        self._instruments = instruments
        self._vanilla = VanillaDCAComparison(base_amounts)

        # Monthly spending tracker: {YYYY-MM: {symbol: amount_usdt}}
        self._monthly_spending: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

        # Crash-buy separate tracking
        self._crash_purchases: List[dict] = []
        self._regular_purchases: List[dict] = []

        # Multiplier history for distribution
        self._multiplier_history: List[float] = []

    def record_purchase(
        self,
        symbol: str,
        amount_usdt: float,
        quantity: float,
        price: float,
        multiplier: float,
        is_crash_buy: bool,
        crash_level: int = 0,
    ) -> None:
        """Record a completed purchase for metrics tracking."""
        now = datetime.now(timezone.utc)
        month_key = now.strftime("%Y-%m")

        # Monthly tracker
        self._monthly_spending[month_key][symbol] += amount_usdt

        # Vanilla DCA: record hypothetical fixed purchase at same price
        if not is_crash_buy:
            self._vanilla.record_purchase(symbol, price)

        # Multiplier
        self._multiplier_history.append(multiplier)

        # Categorise
        record = {
            "symbol": symbol,
            "timestamp": now.timestamp(),
            "amount_usdt": amount_usdt,
            "quantity": quantity,
            "price": price,
            "multiplier": multiplier,
            "is_crash_buy": is_crash_buy,
            "crash_level": crash_level,
        }
        if is_crash_buy:
            self._crash_purchases.append(record)
        else:
            self._regular_purchases.append(record)

    def get_full_metrics(
        self,
        holdings: Dict[str, Any],
        current_prices: Dict[str, float],
        total_invested: float,
        portfolio_value: float,
    ) -> Dict[str, Any]:
        """Compute all Section 10.2 and 10.3 metrics.

        Parameters
        ----------
        holdings : dict
            Current holdings per instrument (from risk manager).
        current_prices : dict
            Current prices per symbol.
        total_invested : float
            Total amount invested across all instruments.
        portfolio_value : float
            Current total portfolio value.
        """
        # Total Return %
        total_return_pct = 0.0
        if total_invested > 0:
            total_return_pct = ((portfolio_value - total_invested) / total_invested) * 100.0

        # Avg Cost Basis per instrument
        cost_basis_per_instrument = {}
        for sym, h in holdings.items():
            qty = h.get("quantity", h.quantity if hasattr(h, "quantity") else 0)
            invested = h.get("total_invested", h.total_invested if hasattr(h, "total_invested") else 0)
            price = current_prices.get(sym, 0.0)
            avg_cost = invested / qty if qty > 0 else 0.0
            cost_basis_per_instrument[sym] = {
                "avg_cost_basis": round(avg_cost, 8),
                "current_price": round(price, 4),
                "vs_cost_pct": round(((price / avg_cost - 1) * 100), 2) if avg_cost > 0 else 0.0,
            }

        # Signal-Enhanced vs Vanilla DCA comparison
        enhanced_invested = {}
        enhanced_units = {}
        for sym, h in holdings.items():
            enhanced_invested[sym] = h.get("total_invested", h.total_invested if hasattr(h, "total_invested") else 0)
            enhanced_units[sym] = h.get("quantity", h.quantity if hasattr(h, "quantity") else 0)

        vanilla_comparison = self._vanilla.get_comparison(enhanced_invested, enhanced_units, current_prices)

        # Units Accumulated per Dollar Spent
        units_per_dollar = {}
        for sym, h in holdings.items():
            invested = h.get("total_invested", h.total_invested if hasattr(h, "total_invested") else 0)
            qty = h.get("quantity", h.quantity if hasattr(h, "quantity") else 0)
            units_per_dollar[sym] = round(qty / invested, 10) if invested > 0 else 0.0

        # Cost Basis Improvement (from vanilla comparison)
        cost_basis_improvement = vanilla_comparison.get("aggregate", {}).get("return_improvement_pct", 0.0)

        # Crash-Buy Performance vs Regular DCA
        crash_perf = self._compute_crash_performance(current_prices, holdings)

        # Signal Multiplier Distribution
        mult_dist = self._compute_multiplier_distribution()

        # Unrealised PnL per instrument
        unrealised_pnl = {}
        for sym, h in holdings.items():
            invested = h.get("total_invested", h.total_invested if hasattr(h, "total_invested") else 0)
            qty = h.get("quantity", h.quantity if hasattr(h, "quantity") else 0)
            price = current_prices.get(sym, 0.0)
            value = qty * price
            unrealised_pnl[sym] = {
                "pnl": round(value - invested, 2),
                "pnl_pct": round(((value - invested) / invested * 100), 2) if invested > 0 else 0.0,
            }

        # Monthly spending tracker
        monthly_spending = self._get_monthly_spending_summary()

        # Section 10.3: Dimensional breakdowns
        dimensional = self._compute_dimensional_breakdowns(holdings, current_prices)

        return {
            # Section 10.1
            "total_invested": round(total_invested, 2),
            "portfolio_value": round(portfolio_value, 2),
            "total_return_pct": round(total_return_pct, 2),

            # Section 10.2
            "avg_cost_basis_per_instrument": cost_basis_per_instrument,
            "signal_vs_vanilla_comparison": vanilla_comparison,
            "units_per_dollar_spent": units_per_dollar,
            "cost_basis_improvement_pct": round(cost_basis_improvement, 2),
            "crash_buy_performance": crash_perf,
            "signal_multiplier_distribution": mult_dist,
            "unrealised_pnl_per_instrument": unrealised_pnl,
            "monthly_spending_tracker": monthly_spending,

            # Section 10.3
            "dimensional_breakdowns": dimensional,
        }

    def _compute_crash_performance(
        self,
        current_prices: Dict[str, float],
        holdings: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compute crash-buy performance metrics."""
        if not self._crash_purchases:
            return {"count": 0, "total_invested": 0.0, "current_value": 0.0, "return_pct": 0.0}

        total_invested = sum(p["amount_usdt"] for p in self._crash_purchases)
        total_value = 0.0
        by_level = defaultdict(lambda: {"count": 0, "invested": 0.0, "value": 0.0})

        for p in self._crash_purchases:
            price = current_prices.get(p["symbol"], 0.0)
            value = p["quantity"] * price
            total_value += value
            level = p.get("crash_level", 0)
            by_level[level]["count"] += 1
            by_level[level]["invested"] += p["amount_usdt"]
            by_level[level]["value"] += value

        return_pct = ((total_value - total_invested) / total_invested * 100.0) if total_invested > 0 else 0.0

        # Regular DCA performance for comparison
        regular_invested = sum(p["amount_usdt"] for p in self._regular_purchases)
        regular_value = sum(
            p["quantity"] * current_prices.get(p["symbol"], 0.0)
            for p in self._regular_purchases
        )
        regular_return = (
            ((regular_value - regular_invested) / regular_invested * 100.0)
            if regular_invested > 0
            else 0.0
        )

        return {
            "count": len(self._crash_purchases),
            "total_invested": round(total_invested, 2),
            "current_value": round(total_value, 2),
            "return_pct": round(return_pct, 2),
            "vs_regular_dca_return_pct": round(return_pct - regular_return, 2),
            "by_level": {
                str(k): {
                    "count": v["count"],
                    "invested": round(v["invested"], 2),
                    "value": round(v["value"], 2),
                    "return_pct": round(
                        ((v["value"] - v["invested"]) / v["invested"] * 100.0)
                        if v["invested"] > 0
                        else 0.0,
                        2,
                    ),
                }
                for k, v in sorted(by_level.items())
            },
        }

    def _compute_multiplier_distribution(self) -> Dict[str, Any]:
        """Compute signal multiplier distribution histogram."""
        if not self._multiplier_history:
            return {"count": 0}

        arr = np.array(self._multiplier_history)
        # Histogram buckets: 0.25-0.5, 0.5-0.75, 0.75-1.0, 1.0-1.25, 1.25-1.5, 1.5-2.0, 2.0-3.0
        edges = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
        hist, _ = np.histogram(arr, bins=edges)
        labels = [f"{edges[i]:.2f}-{edges[i+1]:.2f}" for i in range(len(edges) - 1)]

        return {
            "count": len(arr),
            "mean": round(float(np.mean(arr)), 4),
            "median": round(float(np.median(arr)), 4),
            "std": round(float(np.std(arr)), 4),
            "min": round(float(np.min(arr)), 4),
            "max": round(float(np.max(arr)), 4),
            "histogram": dict(zip(labels, [int(h) for h in hist])),
        }

    def _get_monthly_spending_summary(self) -> Dict[str, Any]:
        """Return monthly spending summary for budget utilisation gauge."""
        summary = {}
        for month_key, instruments in sorted(self._monthly_spending.items()):
            total = sum(instruments.values())
            summary[month_key] = {
                "total": round(total, 2),
                "per_instrument": {k: round(v, 2) for k, v in instruments.items()},
            }
        return summary

    def _compute_dimensional_breakdowns(
        self,
        holdings: Dict[str, Any],
        current_prices: Dict[str, float],
    ) -> Dict[str, Any]:
        """Section 10.3 dimensional breakdowns.

        Breakdowns by:
        - instrument
        - signal regime (high fear / neutral / greed)
        - crash vs regular
        - time period (weekly / monthly)
        """
        # By instrument
        by_instrument = {}
        all_purchases = self._regular_purchases + self._crash_purchases
        for sym in self._instruments:
            sym_purchases = [p for p in all_purchases if p["symbol"] == sym]
            if not sym_purchases:
                continue
            invested = sum(p["amount_usdt"] for p in sym_purchases)
            units = sum(p["quantity"] for p in sym_purchases)
            price = current_prices.get(sym, 0.0)
            value = units * price
            by_instrument[sym] = {
                "purchases": len(sym_purchases),
                "invested": round(invested, 2),
                "units": units,
                "current_value": round(value, 2),
                "return_pct": round(((value - invested) / invested * 100.0) if invested > 0 else 0.0, 2),
                "avg_multiplier": round(
                    np.mean([p["multiplier"] for p in sym_purchases]), 4
                ),
            }

        # Asset allocation
        portfolio_value = sum(
            (h.quantity if hasattr(h, "quantity") else h.get("quantity", 0))
            * current_prices.get(sym, 0.0)
            for sym, h in holdings.items()
        )
        allocation = {}
        if portfolio_value > 0:
            for sym, h in holdings.items():
                qty = h.quantity if hasattr(h, "quantity") else h.get("quantity", 0)
                val = qty * current_prices.get(sym, 0.0)
                allocation[sym] = round((val / portfolio_value) * 100, 2)

        # Historical cost basis data for chart overlay
        cost_basis_history = []
        running_invested: Dict[str, float] = defaultdict(float)
        running_units: Dict[str, float] = defaultdict(float)
        for p in sorted(all_purchases, key=lambda x: x["timestamp"]):
            running_invested[p["symbol"]] += p["amount_usdt"]
            running_units[p["symbol"]] += p["quantity"]
            total_inv = sum(running_invested.values())
            total_u = sum(running_units.values())
            avg_cb = total_inv / total_u if total_u > 0 else 0.0
            cost_basis_history.append({
                "timestamp": p["timestamp"],
                "avg_cost_basis": round(avg_cb, 4),
                "total_invested": round(total_inv, 2),
            })

        return {
            "by_instrument": by_instrument,
            "asset_allocation": allocation,
            "cost_basis_history": cost_basis_history[-100:],  # Last 100 for chart
        }

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def to_state(self) -> dict:
        return {
            "vanilla": self._vanilla.to_state(),
            "monthly_spending": dict(self._monthly_spending),
            "crash_purchases": self._crash_purchases[-500:],
            "regular_purchases": self._regular_purchases[-500:],
            "multiplier_history": self._multiplier_history[-1000:],
        }

    def load_state(self, state: dict) -> None:
        vanilla_state = state.get("vanilla", {})
        if vanilla_state:
            self._vanilla.load_state(vanilla_state)
        self._monthly_spending = defaultdict(
            lambda: defaultdict(float),
            {k: defaultdict(float, v) for k, v in state.get("monthly_spending", {}).items()},
        )
        self._crash_purchases = state.get("crash_purchases", [])
        self._regular_purchases = state.get("regular_purchases", [])
        self._multiplier_history = state.get("multiplier_history", [])


# ---------------------------------------------------------------------------
# Go-Live Criteria (Section 9.3)
# ---------------------------------------------------------------------------

class GoLiveCriteria:
    """Evaluate go-live criteria for the Signal-Enhanced DCA strategy.

    Section 9.3 (adapted for 30-day minimum paper):
    - 30-day paper trading without missed intervals
    - Net positive PnL (portfolio value > total invested)
    - Max drawdown < 8%
    - Consistent signal calculation (no errors)
    """

    def __init__(self, min_days: int = 30) -> None:
        self._min_days = min_days
        self._start_time: float = 0.0
        self._missed_intervals: int = 0
        self._signal_errors: int = 0
        self._peak_value: float = 0.0
        self._max_drawdown_pct: float = 0.0

    def start(self) -> None:
        """Mark the start of paper trading evaluation."""
        self._start_time = time.time()

    def record_dca_execution(self, scheduled: bool, missed: bool) -> None:
        """Record a DCA execution (or missed one)."""
        if missed:
            self._missed_intervals += 1

    def record_signal_error(self) -> None:
        self._signal_errors += 1

    def update_portfolio_value(self, value: float) -> None:
        """Update peak and drawdown tracking."""
        if value > self._peak_value:
            self._peak_value = value
        if self._peak_value > 0:
            dd = ((self._peak_value - value) / self._peak_value) * 100.0
            if dd > self._max_drawdown_pct:
                self._max_drawdown_pct = dd

    def evaluate(
        self,
        total_invested: float,
        portfolio_value: float,
        vanilla_return: float,
        enhanced_return: float,
        crash_buy_executed: bool,
    ) -> Dict[str, Any]:
        """Evaluate all go-live criteria.

        Returns dict with each criterion, its status, and overall pass/fail.
        """
        elapsed_days = (time.time() - self._start_time) / 86400 if self._start_time > 0 else 0

        criteria = {
            "paper_days": {
                "required": self._min_days,
                "actual": round(elapsed_days, 1),
                "pass": elapsed_days >= self._min_days,
            },
            "net_positive": {
                "required": True,
                "actual": portfolio_value > total_invested,
                "pass": portfolio_value > total_invested,
                "return_pct": round(
                    ((portfolio_value - total_invested) / total_invested * 100.0)
                    if total_invested > 0
                    else 0.0,
                    2,
                ),
            },
            "max_drawdown_under_8pct": {
                "required": 8.0,
                "actual": round(self._max_drawdown_pct, 2),
                "pass": self._max_drawdown_pct < 8.0,
            },
            "consistent_signal_calculation": {
                "required": 0,
                "actual": self._signal_errors,
                "pass": self._signal_errors == 0,
            },
            "no_missed_intervals": {
                "required": 0,
                "actual": self._missed_intervals,
                "pass": self._missed_intervals == 0,
            },
            "signal_beats_vanilla": {
                "required": True,
                "actual": enhanced_return > vanilla_return,
                "pass": enhanced_return > vanilla_return,
                "improvement_pct": round(enhanced_return - vanilla_return, 2),
            },
            "crash_buy_tested": {
                "required": True,
                "actual": crash_buy_executed,
                "pass": crash_buy_executed,
            },
        }

        all_pass = all(c["pass"] for c in criteria.values())

        return {
            "criteria": criteria,
            "overall_pass": all_pass,
            "ready_for_live": all_pass,
            "evaluation_time": datetime.now(timezone.utc).isoformat(),
        }

    def to_state(self) -> dict:
        return {
            "start_time": self._start_time,
            "missed_intervals": self._missed_intervals,
            "signal_errors": self._signal_errors,
            "peak_value": self._peak_value,
            "max_drawdown_pct": self._max_drawdown_pct,
        }

    def load_state(self, state: dict) -> None:
        self._start_time = state.get("start_time", 0.0)
        self._missed_intervals = state.get("missed_intervals", 0)
        self._signal_errors = state.get("signal_errors", 0)
        self._peak_value = state.get("peak_value", 0.0)
        self._max_drawdown_pct = state.get("max_drawdown_pct", 0.0)
