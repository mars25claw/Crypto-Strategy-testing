"""Section 10.2 & 10.3 Strategy-Specific Metrics for STRAT-005 Grid Trading.

Metrics tracked:
- Grid Cycle Count / Avg Profit Per Cycle
- Grid Utilization Rate (% of levels that triggered)
- Inventory Exposure Time (avg hours holding inventory)
- Max Concurrent Inventory (levels filled simultaneously)
- Range Containment Rate (% time price within grid)
- Breakout Count (upside / downside)
- Redeployment Frequency
- Upside vs Downside Breakout PnL
- Fee Efficiency (net profit / gross fees)
- Order Placement Latency

Dimensional breakdowns by:
- Grid type (geometric/arithmetic)
- Volatility regime (low/medium/high)
- Market regime (trending/ranging)
- Instrument

Go-live criteria:
- 45-day paper trading
- Positive total PnL
- 60% profitable days
- Inventory DD < 5%
- Cycle completion > 50%
"""

from __future__ import annotations

import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
#  Dimensional Breakdown Engine (Section 10.3)
# ══════════════════════════════════════════════════════════════════════

class DimensionalBreakdown:
    """Multi-dimensional performance breakdown for grid trading."""

    def __init__(self) -> None:
        self._buckets: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(
            lambda: defaultdict(lambda: {
                "cycles": 0,
                "cycle_profit": 0.0,
                "total_fees": 0.0,
                "breakouts_up": 0,
                "breakouts_down": 0,
                "redeployments": 0,
                "holding_hours": 0.0,
            })
        )

    def record(
        self,
        dimensions: Dict[str, str],
        cycle_profit: float = 0.0,
        fees: float = 0.0,
        breakout_up: bool = False,
        breakout_down: bool = False,
        is_redeployment: bool = False,
        holding_hours: float = 0.0,
    ) -> None:
        for dim_name, bucket_name in dimensions.items():
            bucket = self._buckets[dim_name][bucket_name]
            bucket["cycles"] += 1
            bucket["cycle_profit"] += cycle_profit
            bucket["total_fees"] += fees
            if breakout_up:
                bucket["breakouts_up"] += 1
            if breakout_down:
                bucket["breakouts_down"] += 1
            if is_redeployment:
                bucket["redeployments"] += 1
            bucket["holding_hours"] += holding_hours

    def get_breakdown(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        result: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for dim_name, buckets in self._buckets.items():
            dim_result: Dict[str, Dict[str, Any]] = {}
            for bucket_name, data in buckets.items():
                cycles = data["cycles"]
                dim_result[bucket_name] = {
                    "cycles": int(cycles),
                    "total_cycle_profit": round(data["cycle_profit"], 6),
                    "avg_cycle_profit": round(data["cycle_profit"] / cycles, 6) if cycles > 0 else 0.0,
                    "total_fees": round(data["total_fees"], 6),
                    "breakouts_up": int(data["breakouts_up"]),
                    "breakouts_down": int(data["breakouts_down"]),
                    "redeployments": int(data["redeployments"]),
                    "avg_holding_hours": round(data["holding_hours"] / cycles, 2) if cycles > 0 else 0.0,
                }
            result[dim_name] = dim_result
        return result

    def get_state(self) -> Dict[str, Any]:
        return {
            "buckets": {
                dim: {bucket: dict(data) for bucket, data in buckets.items()}
                for dim, buckets in self._buckets.items()
            }
        }

    def restore_state(self, state: Dict[str, Any]) -> None:
        for dim, buckets in state.get("buckets", {}).items():
            for bucket, data in buckets.items():
                self._buckets[dim][bucket] = data


# ══════════════════════════════════════════════════════════════════════
#  Go-Live Criteria Checker
# ══════════════════════════════════════════════════════════════════════

@dataclass
class GoLiveCriterion:
    name: str
    description: str
    threshold: float
    current_value: float = 0.0
    passed: bool = False
    unit: str = ""


class GoLiveCriteriaChecker:
    """STRAT-005 go-live criteria:
    - 45-day paper trading
    - Positive total PnL
    - 60% profitable days
    - Inventory DD < 5%
    - Cycle completion > 50%
    """

    def evaluate(
        self,
        trading_days: int,
        total_pnl: float,
        profitable_days_pct: float,
        inventory_max_dd_pct: float,
        cycle_completion_pct: float,
    ) -> List[GoLiveCriterion]:
        return [
            GoLiveCriterion(
                name="paper_trading_days",
                description="Minimum 45-day paper trading period",
                threshold=45,
                current_value=trading_days,
                passed=trading_days >= 45,
                unit="days",
            ),
            GoLiveCriterion(
                name="positive_pnl",
                description="Positive total PnL",
                threshold=0.0,
                current_value=round(total_pnl, 2),
                passed=total_pnl > 0,
                unit="USDT",
            ),
            GoLiveCriterion(
                name="profitable_days",
                description="60% profitable days",
                threshold=60.0,
                current_value=round(profitable_days_pct, 2),
                passed=profitable_days_pct >= 60.0,
                unit="%",
            ),
            GoLiveCriterion(
                name="inventory_drawdown",
                description="Inventory drawdown < 5%",
                threshold=5.0,
                current_value=round(inventory_max_dd_pct, 2),
                passed=inventory_max_dd_pct < 5.0,
                unit="%",
            ),
            GoLiveCriterion(
                name="cycle_completion",
                description="Cycle completion > 50%",
                threshold=50.0,
                current_value=round(cycle_completion_pct, 2),
                passed=cycle_completion_pct > 50.0,
                unit="%",
            ),
        ]

    def all_passed(self, criteria: List[GoLiveCriterion]) -> bool:
        return all(c.passed for c in criteria)

    def to_dict(self, criteria: List[GoLiveCriterion]) -> Dict[str, Any]:
        return {
            "all_passed": self.all_passed(criteria),
            "criteria": [
                {
                    "name": c.name,
                    "description": c.description,
                    "threshold": c.threshold,
                    "current_value": c.current_value,
                    "passed": c.passed,
                    "unit": c.unit,
                }
                for c in criteria
            ],
        }


# ══════════════════════════════════════════════════════════════════════
#  Strategy Metrics Aggregator (Section 10.2)
# ══════════════════════════════════════════════════════════════════════

class GridMetrics:
    """Aggregates all Section 10.2 strategy-specific metrics for STRAT-005."""

    def __init__(
        self,
        strategy: Any = None,
        grid_manager: Any = None,
        risk_manager: Any = None,
    ) -> None:
        self._strategy = strategy
        self._grid_manager = grid_manager
        self._risk = risk_manager

        self.dimensional = DimensionalBreakdown()
        self.go_live = GoLiveCriteriaChecker()

        self._start_time: float = time.time()

        # Cycle tracking
        self._total_cycles: int = 0
        self._total_cycle_profit: float = 0.0
        self._total_fees: float = 0.0

        # Grid utilization
        self._levels_triggered: int = 0
        self._levels_total: int = 0

        # Inventory exposure
        self._inventory_exposure_hours: List[float] = []
        self._max_concurrent_inventory: int = 0

        # Range containment
        self._containment_checks: int = 0
        self._containment_within: int = 0

        # Breakouts
        self._breakout_up_count: int = 0
        self._breakout_down_count: int = 0
        self._breakout_up_pnl: float = 0.0
        self._breakout_down_pnl: float = 0.0

        # Redeployments
        self._redeployment_count: int = 0

        # Order latency
        self._order_latencies_ms: List[float] = []

        # Daily PnL tracking
        self._daily_pnl: Dict[str, float] = defaultdict(float)

    # ──────────────────────────────────────────────────────────────────
    #  Event recording
    # ──────────────────────────────────────────────────────────────────

    def record_cycle(
        self,
        symbol: str,
        profit: float,
        fees: float,
        grid_type: str = "geometric",
        volatility_regime: str = "medium",
        market_regime: str = "ranging",
    ) -> None:
        self._total_cycles += 1
        self._total_cycle_profit += profit
        self._total_fees += fees

        # Daily tracking
        from datetime import datetime, timezone
        day_key = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self._daily_pnl[day_key] += profit - fees

        dimensions = {
            "grid_type": grid_type,
            "volatility_regime": volatility_regime,
            "market_regime": market_regime,
            "instrument": symbol,
        }
        self.dimensional.record(
            dimensions=dimensions,
            cycle_profit=profit,
            fees=fees,
        )

    def record_breakout(self, direction: str, pnl: float) -> None:
        if direction == "upside":
            self._breakout_up_count += 1
            self._breakout_up_pnl += pnl
        elif direction == "downside":
            self._breakout_down_count += 1
            self._breakout_down_pnl += pnl

    def record_redeployment(self) -> None:
        self._redeployment_count += 1

    def record_containment_check(self, within_range: bool) -> None:
        self._containment_checks += 1
        if within_range:
            self._containment_within += 1

    def record_grid_utilization(self, triggered: int, total: int) -> None:
        self._levels_triggered += triggered
        self._levels_total += total

    def record_inventory_exposure(self, hours: float, concurrent_levels: int) -> None:
        self._inventory_exposure_hours.append(hours)
        if concurrent_levels > self._max_concurrent_inventory:
            self._max_concurrent_inventory = concurrent_levels

    def record_order_latency(self, latency_ms: float) -> None:
        self._order_latencies_ms.append(latency_ms)
        if len(self._order_latencies_ms) > 5000:
            self._order_latencies_ms = self._order_latencies_ms[-5000:]

    # ──────────────────────────────────────────────────────────────────
    #  Metrics computation
    # ──────────────────────────────────────────────────────────────────

    def get_all_metrics(self) -> Dict[str, Any]:
        elapsed_days = (time.time() - self._start_time) / 86400.0

        # Grid Cycle Count / Avg Profit Per Cycle
        avg_cycle_profit = (
            self._total_cycle_profit / self._total_cycles
            if self._total_cycles > 0 else 0.0
        )

        # Grid Utilization Rate
        grid_utilization = (
            self._levels_triggered / self._levels_total * 100
            if self._levels_total > 0 else 0.0
        )

        # Inventory Exposure Time
        avg_inventory_exposure = (
            sum(self._inventory_exposure_hours) / len(self._inventory_exposure_hours)
            if self._inventory_exposure_hours else 0.0
        )

        # Range Containment Rate
        containment_rate = (
            self._containment_within / self._containment_checks * 100
            if self._containment_checks > 0 else 100.0
        )

        # Fee Efficiency
        net_profit = self._total_cycle_profit - self._total_fees
        fee_efficiency = (
            net_profit / self._total_fees
            if self._total_fees > 0 else 0.0
        )

        # Order Placement Latency
        avg_latency = (
            sum(self._order_latencies_ms) / len(self._order_latencies_ms)
            if self._order_latencies_ms else 0.0
        )
        max_latency = max(self._order_latencies_ms) if self._order_latencies_ms else 0.0
        p99_latency = 0.0
        if self._order_latencies_ms:
            sorted_lat = sorted(self._order_latencies_ms)
            idx = int(len(sorted_lat) * 0.99)
            p99_latency = sorted_lat[min(idx, len(sorted_lat) - 1)]

        # Profitable days
        profitable_days = sum(1 for v in self._daily_pnl.values() if v > 0)
        total_days = len(self._daily_pnl) or 1
        profitable_days_pct = profitable_days / total_days * 100

        # Cycle completion rate
        cycle_completion = grid_utilization  # Approximate

        return {
            "grid_cycle_count": self._total_cycles,
            "avg_profit_per_cycle": round(avg_cycle_profit, 6),
            "total_cycle_profit": round(self._total_cycle_profit, 6),
            "total_fees": round(self._total_fees, 6),
            "net_profit": round(net_profit, 6),
            "grid_utilization_rate_pct": round(grid_utilization, 2),
            "avg_inventory_exposure_hours": round(avg_inventory_exposure, 2),
            "max_concurrent_inventory": self._max_concurrent_inventory,
            "range_containment_rate_pct": round(containment_rate, 2),
            "breakout_count_up": self._breakout_up_count,
            "breakout_count_down": self._breakout_down_count,
            "breakout_pnl_up": round(self._breakout_up_pnl, 6),
            "breakout_pnl_down": round(self._breakout_down_pnl, 6),
            "redeployment_frequency": self._redeployment_count,
            "fee_efficiency": round(fee_efficiency, 2),
            "order_latency_avg_ms": round(avg_latency, 2),
            "order_latency_max_ms": round(max_latency, 2),
            "order_latency_p99_ms": round(p99_latency, 2),
            "profitable_days_pct": round(profitable_days_pct, 2),
            "elapsed_days": round(elapsed_days, 1),
            "dimensional_breakdowns": self.dimensional.get_breakdown(),
        }

    def get_go_live_status(self) -> Dict[str, Any]:
        metrics = self.get_all_metrics()

        # Inventory DD — approximate from daily PnL
        inventory_dd = 0.0  # Would need position-level tracking
        if self._risk:
            risk_data = self._risk.get_risk_status() if hasattr(self._risk, 'get_risk_status') else {}
            inventory_dd = risk_data.get("max_inventory_drawdown_pct", 0.0)

        criteria = self.go_live.evaluate(
            trading_days=int(metrics["elapsed_days"]),
            total_pnl=metrics["net_profit"],
            profitable_days_pct=metrics["profitable_days_pct"],
            inventory_max_dd_pct=inventory_dd,
            cycle_completion_pct=metrics["grid_utilization_rate_pct"],
        )
        return self.go_live.to_dict(criteria)

    # ──────────────────────────────────────────────────────────────────
    #  Persistence
    # ──────────────────────────────────────────────────────────────────

    def get_state(self) -> Dict[str, Any]:
        return {
            "start_time": self._start_time,
            "total_cycles": self._total_cycles,
            "total_cycle_profit": self._total_cycle_profit,
            "total_fees": self._total_fees,
            "levels_triggered": self._levels_triggered,
            "levels_total": self._levels_total,
            "max_concurrent_inventory": self._max_concurrent_inventory,
            "containment_checks": self._containment_checks,
            "containment_within": self._containment_within,
            "breakout_up_count": self._breakout_up_count,
            "breakout_down_count": self._breakout_down_count,
            "breakout_up_pnl": self._breakout_up_pnl,
            "breakout_down_pnl": self._breakout_down_pnl,
            "redeployment_count": self._redeployment_count,
            "daily_pnl": dict(self._daily_pnl),
            "dimensional": self.dimensional.get_state(),
        }

    def restore_state(self, state: Dict[str, Any]) -> None:
        self._start_time = state.get("start_time", self._start_time)
        self._total_cycles = state.get("total_cycles", 0)
        self._total_cycle_profit = state.get("total_cycle_profit", 0.0)
        self._total_fees = state.get("total_fees", 0.0)
        self._levels_triggered = state.get("levels_triggered", 0)
        self._levels_total = state.get("levels_total", 0)
        self._max_concurrent_inventory = state.get("max_concurrent_inventory", 0)
        self._containment_checks = state.get("containment_checks", 0)
        self._containment_within = state.get("containment_within", 0)
        self._breakout_up_count = state.get("breakout_up_count", 0)
        self._breakout_down_count = state.get("breakout_down_count", 0)
        self._breakout_up_pnl = state.get("breakout_up_pnl", 0.0)
        self._breakout_down_pnl = state.get("breakout_down_pnl", 0.0)
        self._redeployment_count = state.get("redeployment_count", 0)
        self._daily_pnl = defaultdict(float, state.get("daily_pnl", {}))
        dim_state = state.get("dimensional")
        if dim_state:
            self.dimensional.restore_state(dim_state)
        logger.info("GridMetrics state restored: %d cycles tracked", self._total_cycles)
