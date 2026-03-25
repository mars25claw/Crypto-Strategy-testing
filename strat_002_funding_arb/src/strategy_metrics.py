"""Section 10.2 & 10.3 Strategy-Specific Metrics for STRAT-002 Funding Rate Arbitrage.

Metrics tracked:
- Cumulative Funding Income
- Average Funding Rate Captured
- Annualized Yield
- Basis Capture Efficiency
- Delta Neutrality Score (% time within +/-0.5%)
- Funding Win/Loss ratio
- Cost-to-Income Ratio
- Liquidation Distance Minimum
- Wallet Rebalancing Count
- Average Holding Duration
- Yield Per Instrument

Dimensional breakdowns by:
- Funding rate regime (high/medium/low)
- Basis regime (premium/par/discount)
- Volatility regime (high/medium/low)

Go-live criteria per Section 10.4:
- 30-day paper trading
- Yield > 8% annualized
- Max drawdown < 3%
- Delta within +/-1% for > 95% of time
- 98% uptime
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
#  Dimensional Breakdown Engine (Section 10.3)
# ══════════════════════════════════════════════════════════════════════

class DimensionalBreakdown:
    """Multi-dimensional performance breakdown engine.

    Tracks trade outcomes broken down by configurable dimensions
    such as funding rate regime, basis regime, and volatility regime.
    """

    def __init__(self) -> None:
        # dimension_name -> bucket_name -> {trades, wins, pnl, funding_income}
        self._buckets: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(
            lambda: defaultdict(lambda: {
                "trades": 0,
                "wins": 0,
                "pnl": 0.0,
                "funding_income": 0.0,
                "total_fees": 0.0,
                "holding_hours": 0.0,
            })
        )

    def record(
        self,
        dimensions: Dict[str, str],
        pnl: float,
        is_win: bool,
        funding_income: float = 0.0,
        fees: float = 0.0,
        holding_hours: float = 0.0,
    ) -> None:
        """Record a trade outcome across all provided dimensions."""
        for dim_name, bucket_name in dimensions.items():
            bucket = self._buckets[dim_name][bucket_name]
            bucket["trades"] += 1
            if is_win:
                bucket["wins"] += 1
            bucket["pnl"] += pnl
            bucket["funding_income"] += funding_income
            bucket["total_fees"] += fees
            bucket["holding_hours"] += holding_hours

    def get_breakdown(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Return all dimensional breakdowns."""
        result: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for dim_name, buckets in self._buckets.items():
            dim_result: Dict[str, Dict[str, Any]] = {}
            for bucket_name, data in buckets.items():
                trades = data["trades"]
                dim_result[bucket_name] = {
                    "trades": int(trades),
                    "wins": int(data["wins"]),
                    "win_rate": round(data["wins"] / trades * 100, 2) if trades > 0 else 0.0,
                    "total_pnl": round(data["pnl"], 6),
                    "avg_pnl": round(data["pnl"] / trades, 6) if trades > 0 else 0.0,
                    "total_funding_income": round(data["funding_income"], 6),
                    "total_fees": round(data["total_fees"], 6),
                    "avg_holding_hours": round(data["holding_hours"] / trades, 2) if trades > 0 else 0.0,
                }
            result[dim_name] = dim_result
        return result

    def get_state(self) -> Dict[str, Any]:
        """Serialize for persistence."""
        return {
            "buckets": {
                dim: {bucket: dict(data) for bucket, data in buckets.items()}
                for dim, buckets in self._buckets.items()
            }
        }

    def restore_state(self, state: Dict[str, Any]) -> None:
        """Restore from persistence."""
        for dim, buckets in state.get("buckets", {}).items():
            for bucket, data in buckets.items():
                self._buckets[dim][bucket] = data


# ══════════════════════════════════════════════════════════════════════
#  Go-Live Criteria Checker (Section 10.4)
# ══════════════════════════════════════════════════════════════════════

@dataclass
class GoLiveCriterion:
    """A single go-live criterion with pass/fail tracking."""
    name: str
    description: str
    threshold: float
    current_value: float = 0.0
    passed: bool = False
    unit: str = ""


class GoLiveCriteriaChecker:
    """Evaluates whether the strategy meets go-live requirements.

    STRAT-002 criteria:
    - 30-day paper trading period
    - Yield > 8% annualized
    - Max drawdown < 3%
    - Delta within +/-1% for > 95% of time
    - 98% uptime
    """

    def __init__(self) -> None:
        self._start_time: float = time.time()
        self._uptime_checks: int = 0
        self._uptime_passes: int = 0

    def evaluate(
        self,
        trading_days: int,
        annualized_yield_pct: float,
        max_drawdown_pct: float,
        delta_within_1pct_time_pct: float,
        uptime_pct: float,
    ) -> List[GoLiveCriterion]:
        """Evaluate all go-live criteria and return status."""
        criteria = [
            GoLiveCriterion(
                name="paper_trading_days",
                description="Minimum 30-day paper trading period",
                threshold=30,
                current_value=trading_days,
                passed=trading_days >= 30,
                unit="days",
            ),
            GoLiveCriterion(
                name="annualized_yield",
                description="Yield > 8% annualized",
                threshold=8.0,
                current_value=round(annualized_yield_pct, 2),
                passed=annualized_yield_pct > 8.0,
                unit="%",
            ),
            GoLiveCriterion(
                name="max_drawdown",
                description="Max drawdown < 3%",
                threshold=3.0,
                current_value=round(max_drawdown_pct, 2),
                passed=max_drawdown_pct < 3.0,
                unit="%",
            ),
            GoLiveCriterion(
                name="delta_neutrality",
                description="Delta within +/-1% for > 95% of time",
                threshold=95.0,
                current_value=round(delta_within_1pct_time_pct, 2),
                passed=delta_within_1pct_time_pct > 95.0,
                unit="%",
            ),
            GoLiveCriterion(
                name="uptime",
                description="98% uptime",
                threshold=98.0,
                current_value=round(uptime_pct, 2),
                passed=uptime_pct >= 98.0,
                unit="%",
            ),
        ]
        return criteria

    def all_passed(self, criteria: List[GoLiveCriterion]) -> bool:
        """Return True if all criteria are met."""
        return all(c.passed for c in criteria)

    def to_dict(self, criteria: List[GoLiveCriterion]) -> Dict[str, Any]:
        """Return criteria as a serializable dict."""
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

class FundingArbMetrics:
    """Aggregates all Section 10.2 strategy-specific metrics for STRAT-002.

    Parameters
    ----------
    strategy : FundingArbStrategy
        The strategy instance.
    risk_manager : FundingArbRiskManager
        The risk manager.
    funding_tracker : FundingTracker
        Funding income tracker.
    wallet_manager : WalletManager
        Cross-wallet manager.
    """

    def __init__(
        self,
        strategy: Any,
        risk_manager: Any,
        funding_tracker: Any,
        wallet_manager: Any,
    ) -> None:
        self._strategy = strategy
        self._risk = risk_manager
        self._funding = funding_tracker
        self._wallet = wallet_manager

        # Dimensional breakdowns
        self.dimensional = DimensionalBreakdown()

        # Go-live checker
        self.go_live = GoLiveCriteriaChecker()

        # Internal tracking
        self._start_time: float = time.time()
        self._trade_count: int = 0
        self._total_funding_income: float = 0.0
        self._total_fees: float = 0.0
        self._yield_per_instrument: Dict[str, float] = defaultdict(float)
        self._holding_durations: List[float] = []
        self._uptime_checks: int = 0
        self._uptime_up: int = 0

    # ──────────────────────────────────────────────────────────────────
    #  Regime classification helpers
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def classify_funding_rate_regime(rate: float) -> str:
        """Classify funding rate into a regime bucket."""
        if rate >= 0.001:       # >= 0.1%
            return "high"
        elif rate >= 0.0003:    # >= 0.03%
            return "medium"
        else:
            return "low"

    @staticmethod
    def classify_basis_regime(basis_pct: float) -> str:
        """Classify basis spread into a regime bucket."""
        if basis_pct > 0.05:
            return "premium"
        elif basis_pct >= -0.01:
            return "par"
        else:
            return "discount"

    @staticmethod
    def classify_volatility_regime(atr_pct: float) -> str:
        """Classify volatility (ATR as % of price) into a regime bucket."""
        if atr_pct > 3.0:
            return "high"
        elif atr_pct > 1.5:
            return "medium"
        else:
            return "low"

    # ──────────────────────────────────────────────────────────────────
    #  Trade recording
    # ──────────────────────────────────────────────────────────────────

    def record_trade(
        self,
        symbol: str,
        pnl: float,
        is_win: bool,
        funding_income: float,
        fees: float,
        holding_hours: float,
        funding_rate_at_entry: float,
        basis_pct_at_entry: float,
        volatility_pct: float = 0.0,
    ) -> None:
        """Record a completed trade with dimensional breakdown."""
        self._trade_count += 1
        self._total_funding_income += funding_income
        self._total_fees += fees
        self._yield_per_instrument[symbol] += funding_income
        self._holding_durations.append(holding_hours)

        dimensions = {
            "funding_rate_regime": self.classify_funding_rate_regime(funding_rate_at_entry),
            "basis_regime": self.classify_basis_regime(basis_pct_at_entry),
            "volatility_regime": self.classify_volatility_regime(volatility_pct),
        }

        self.dimensional.record(
            dimensions=dimensions,
            pnl=pnl,
            is_win=is_win,
            funding_income=funding_income,
            fees=fees,
            holding_hours=holding_hours,
        )

    def record_uptime_check(self, is_up: bool) -> None:
        """Record an uptime check result."""
        self._uptime_checks += 1
        if is_up:
            self._uptime_up += 1

    # ──────────────────────────────────────────────────────────────────
    #  Metrics computation
    # ──────────────────────────────────────────────────────────────────

    def get_all_metrics(self) -> Dict[str, Any]:
        """Compute and return all Section 10.2 metrics."""
        # Get data from component trackers
        funding_metrics = self._funding.get_funding_metrics() if self._funding else {}
        risk_metrics = self._risk.get_risk_metrics() if self._risk else {}
        wallet_metrics = self._wallet.get_wallet_metrics() if self._wallet else {}

        # Cumulative Funding Income
        cumulative_income = funding_metrics.get("total_funding_income", 0.0)

        # Average Funding Rate Captured
        avg_rate_captured = funding_metrics.get("average_funding_rate_captured", 0.0)

        # Annualized Yield
        elapsed_days = (time.time() - self._start_time) / 86400.0
        equity = wallet_metrics.get("total_equity", 0)
        if equity > 0 and elapsed_days > 0:
            annualized_yield = (cumulative_income / equity) * (365.0 / elapsed_days) * 100
        else:
            annualized_yield = 0.0

        # Basis Capture Efficiency
        # (actual basis captured vs theoretical maximum)
        basis_efficiency = self._calc_basis_capture_efficiency()

        # Delta Neutrality Score
        delta_score = risk_metrics.get("delta_neutrality_score", 100.0)

        # Funding Win/Loss
        funding_win_loss = funding_metrics.get("funding_win_loss_pct", 0.0)

        # Cost-to-Income Ratio
        cost_to_income = 0.0
        if cumulative_income > 0:
            cost_to_income = (self._total_fees / cumulative_income) * 100

        # Liquidation Distance Minimum
        min_liq_distance = risk_metrics.get("min_liquidation_distance_pct", 100.0)

        # Wallet Rebalancing Count
        rebalance_count = wallet_metrics.get("rebalance_count", 0)

        # Average Holding Duration
        avg_holding = 0.0
        if self._holding_durations:
            avg_holding = sum(self._holding_durations) / len(self._holding_durations) / 24.0  # days

        # Yield Per Instrument
        yield_per_inst = dict(self._yield_per_instrument)

        # Uptime
        uptime_pct = (self._uptime_up / self._uptime_checks * 100) if self._uptime_checks > 0 else 100.0

        return {
            "cumulative_funding_income": round(cumulative_income, 6),
            "avg_funding_rate_captured": round(avg_rate_captured, 6),
            "annualized_yield_pct": round(annualized_yield, 2),
            "basis_capture_efficiency_pct": round(basis_efficiency, 2),
            "delta_neutrality_score_pct": round(delta_score, 2),
            "funding_win_loss_pct": round(funding_win_loss, 2),
            "cost_to_income_ratio_pct": round(cost_to_income, 2),
            "liquidation_distance_min_pct": round(min_liq_distance, 2),
            "wallet_rebalancing_count": rebalance_count,
            "avg_holding_duration_days": round(avg_holding, 2),
            "yield_per_instrument": yield_per_inst,
            "trade_count": self._trade_count,
            "uptime_pct": round(uptime_pct, 2),
            "elapsed_days": round(elapsed_days, 1),
            # Section 10.3 dimensional breakdowns
            "dimensional_breakdowns": self.dimensional.get_breakdown(),
        }

    def get_go_live_status(self) -> Dict[str, Any]:
        """Evaluate and return go-live criteria status."""
        metrics = self.get_all_metrics()
        risk_metrics = self._risk.get_risk_metrics() if self._risk else {}

        elapsed_days = metrics["elapsed_days"]
        annualized_yield = metrics["annualized_yield_pct"]
        max_dd = risk_metrics.get("drawdown", {}).get("overall_pct", 0.0)
        delta_score = metrics["delta_neutrality_score_pct"]
        uptime = metrics["uptime_pct"]

        criteria = self.go_live.evaluate(
            trading_days=int(elapsed_days),
            annualized_yield_pct=annualized_yield,
            max_drawdown_pct=max_dd,
            delta_within_1pct_time_pct=delta_score,
            uptime_pct=uptime,
        )

        return self.go_live.to_dict(criteria)

    def _calc_basis_capture_efficiency(self) -> float:
        """Ratio of actual basis captured to intended basis at entry."""
        total_actual = 0.0
        total_intended = 0.0
        for pos in self._strategy.positions.values():
            total_actual += pos.entry_basis_pct
            total_intended += pos.intended_basis_pct if pos.intended_basis_pct > 0 else pos.entry_basis_pct
        if total_intended <= 0:
            return 100.0
        return (total_actual / total_intended) * 100.0

    # ──────────────────────────────────────────────────────────────────
    #  Persistence
    # ──────────────────────────────────────────────────────────────────

    def get_state(self) -> Dict[str, Any]:
        """Serialize for persistence."""
        return {
            "start_time": self._start_time,
            "trade_count": self._trade_count,
            "total_funding_income": self._total_funding_income,
            "total_fees": self._total_fees,
            "yield_per_instrument": dict(self._yield_per_instrument),
            "holding_durations": self._holding_durations[-500:],
            "uptime_checks": self._uptime_checks,
            "uptime_up": self._uptime_up,
            "dimensional": self.dimensional.get_state(),
        }

    def restore_state(self, state: Dict[str, Any]) -> None:
        """Restore from persistence."""
        self._start_time = state.get("start_time", self._start_time)
        self._trade_count = state.get("trade_count", 0)
        self._total_funding_income = state.get("total_funding_income", 0.0)
        self._total_fees = state.get("total_fees", 0.0)
        self._yield_per_instrument = defaultdict(float, state.get("yield_per_instrument", {}))
        self._holding_durations = state.get("holding_durations", [])
        self._uptime_checks = state.get("uptime_checks", 0)
        self._uptime_up = state.get("uptime_up", 0)
        dim_state = state.get("dimensional")
        if dim_state:
            self.dimensional.restore_state(dim_state)
        logger.info("FundingArbMetrics state restored: %d trades tracked", self._trade_count)
