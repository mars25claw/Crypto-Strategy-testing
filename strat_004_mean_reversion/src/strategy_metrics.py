"""Section 10.2 & 10.3 Strategy-Specific Metrics for STRAT-004 Mean Reversion.

Metrics tracked:
- Regime Classification Accuracy
- Average Reversion Captured %
- Signal Agreement Rate (2-of-3 vs 3-of-3)
- Time-to-Reversion (actual vs predicted)
- Overshoot Capture (Tranche 3 frequency)
- Regime vs Performance breakdown
- Complement Analysis (correlation with STRAT-001)

Dimensional breakdowns by:
- Regime type (ranging/trending)
- Signal strength (2-of-3 vs 3-of-3)
- Instrument
- Entry type (BB/Z-score/RSI dominant)

Go-live criteria:
- 60-day paper trading
- 40+ trades
- Win rate > 50%
- Profit Factor > 1.2
- Max drawdown < 10%
- Sharpe > 0.7
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
    """Multi-dimensional performance breakdown for mean reversion."""

    def __init__(self) -> None:
        self._buckets: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(
            lambda: defaultdict(lambda: {
                "trades": 0,
                "wins": 0,
                "pnl": 0.0,
                "total_fees": 0.0,
                "holding_hours": 0.0,
                "tranche_3_hits": 0,
            })
        )

    def record(
        self,
        dimensions: Dict[str, str],
        pnl: float,
        is_win: bool,
        fees: float = 0.0,
        holding_hours: float = 0.0,
        tranche_3_hit: bool = False,
    ) -> None:
        for dim_name, bucket_name in dimensions.items():
            bucket = self._buckets[dim_name][bucket_name]
            bucket["trades"] += 1
            if is_win:
                bucket["wins"] += 1
            bucket["pnl"] += pnl
            bucket["total_fees"] += fees
            bucket["holding_hours"] += holding_hours
            if tranche_3_hit:
                bucket["tranche_3_hits"] += 1

    def get_breakdown(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
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
                    "total_fees": round(data["total_fees"], 6),
                    "avg_holding_hours": round(data["holding_hours"] / trades, 2) if trades > 0 else 0.0,
                    "tranche_3_frequency": round(data["tranche_3_hits"] / trades * 100, 2) if trades > 0 else 0.0,
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
    """STRAT-004 go-live criteria:
    - 60-day paper trading
    - 40+ trades
    - Win rate > 50%
    - Profit Factor > 1.2
    - Max drawdown < 10%
    - Sharpe > 0.7
    """

    def evaluate(
        self,
        trading_days: int,
        trade_count: int,
        win_rate_pct: float,
        profit_factor: float,
        max_drawdown_pct: float,
        sharpe_ratio: float,
    ) -> List[GoLiveCriterion]:
        return [
            GoLiveCriterion(
                name="paper_trading_days",
                description="Minimum 60-day paper trading period",
                threshold=60,
                current_value=trading_days,
                passed=trading_days >= 60,
                unit="days",
            ),
            GoLiveCriterion(
                name="trade_count",
                description="Minimum 40 completed trades",
                threshold=40,
                current_value=trade_count,
                passed=trade_count >= 40,
                unit="trades",
            ),
            GoLiveCriterion(
                name="win_rate",
                description="Win rate > 50%",
                threshold=50.0,
                current_value=round(win_rate_pct, 2),
                passed=win_rate_pct > 50.0,
                unit="%",
            ),
            GoLiveCriterion(
                name="profit_factor",
                description="Profit Factor > 1.2",
                threshold=1.2,
                current_value=round(profit_factor, 2),
                passed=profit_factor > 1.2,
                unit="x",
            ),
            GoLiveCriterion(
                name="max_drawdown",
                description="Max drawdown < 10%",
                threshold=10.0,
                current_value=round(max_drawdown_pct, 2),
                passed=max_drawdown_pct < 10.0,
                unit="%",
            ),
            GoLiveCriterion(
                name="sharpe_ratio",
                description="Sharpe ratio > 0.7",
                threshold=0.7,
                current_value=round(sharpe_ratio, 2),
                passed=sharpe_ratio > 0.7,
                unit="",
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

class MeanReversionMetrics:
    """Aggregates all Section 10.2 strategy-specific metrics for STRAT-004."""

    def __init__(
        self,
        strategy: Any = None,
        risk_manager: Any = None,
        regime_classifier: Any = None,
    ) -> None:
        self._strategy = strategy
        self._risk = risk_manager
        self._regime = regime_classifier

        self.dimensional = DimensionalBreakdown()
        self.go_live = GoLiveCriteriaChecker()

        self._start_time: float = time.time()
        self._trade_count: int = 0
        self._win_count: int = 0
        self._gross_profit: float = 0.0
        self._gross_loss: float = 0.0
        self._trade_pnls: List[float] = []

        # Signal tracking
        self._signal_2_of_3_count: int = 0
        self._signal_3_of_3_count: int = 0
        self._signal_2_of_3_wins: int = 0
        self._signal_3_of_3_wins: int = 0

        # Reversion tracking
        self._reversion_captured_pcts: List[float] = []
        self._time_to_reversion_hours: List[float] = []
        self._tranche_3_hits: int = 0

        # Regime tracking
        self._regime_at_entry: List[str] = []
        self._regime_correct: int = 0
        self._regime_total: int = 0

        # Complement analysis
        self._strat001_correlation_samples: List[float] = []

    # ──────────────────────────────────────────────────────────────────
    #  Trade recording
    # ──────────────────────────────────────────────────────────────────

    def record_trade(
        self,
        symbol: str,
        pnl: float,
        is_win: bool,
        signal_count: int,
        regime: str,
        holding_hours: float,
        reversion_captured_pct: float,
        tranche_3_hit: bool = False,
        fees: float = 0.0,
        regime_was_correct: bool = True,
    ) -> None:
        self._trade_count += 1
        if is_win:
            self._win_count += 1
        if pnl > 0:
            self._gross_profit += pnl
        else:
            self._gross_loss += abs(pnl)
        self._trade_pnls.append(pnl)

        # Signal agreement
        if signal_count >= 3:
            self._signal_3_of_3_count += 1
            if is_win:
                self._signal_3_of_3_wins += 1
        elif signal_count >= 2:
            self._signal_2_of_3_count += 1
            if is_win:
                self._signal_2_of_3_wins += 1

        self._reversion_captured_pcts.append(reversion_captured_pct)
        self._time_to_reversion_hours.append(holding_hours)
        if tranche_3_hit:
            self._tranche_3_hits += 1

        self._regime_at_entry.append(regime)
        self._regime_total += 1
        if regime_was_correct:
            self._regime_correct += 1

        dimensions = {
            "regime_type": regime,
            "signal_strength": f"{signal_count}_of_3",
            "instrument": symbol,
        }

        self.dimensional.record(
            dimensions=dimensions,
            pnl=pnl,
            is_win=is_win,
            fees=fees,
            holding_hours=holding_hours,
            tranche_3_hit=tranche_3_hit,
        )

    # ──────────────────────────────────────────────────────────────────
    #  Metrics computation
    # ──────────────────────────────────────────────────────────────────

    def get_all_metrics(self) -> Dict[str, Any]:
        elapsed_days = (time.time() - self._start_time) / 86400.0

        # Regime Classification Accuracy
        regime_accuracy = (
            self._regime_correct / self._regime_total * 100
            if self._regime_total > 0 else 0.0
        )

        # Average Reversion Captured %
        avg_reversion = (
            sum(self._reversion_captured_pcts) / len(self._reversion_captured_pcts)
            if self._reversion_captured_pcts else 0.0
        )

        # Signal Agreement Rate
        total_signals = self._signal_2_of_3_count + self._signal_3_of_3_count
        signal_2_of_3_rate = (
            self._signal_2_of_3_count / total_signals * 100
            if total_signals > 0 else 0.0
        )
        signal_3_of_3_rate = (
            self._signal_3_of_3_count / total_signals * 100
            if total_signals > 0 else 0.0
        )
        signal_2_of_3_win_rate = (
            self._signal_2_of_3_wins / self._signal_2_of_3_count * 100
            if self._signal_2_of_3_count > 0 else 0.0
        )
        signal_3_of_3_win_rate = (
            self._signal_3_of_3_wins / self._signal_3_of_3_count * 100
            if self._signal_3_of_3_count > 0 else 0.0
        )

        # Time-to-Reversion
        avg_time_to_reversion = (
            sum(self._time_to_reversion_hours) / len(self._time_to_reversion_hours)
            if self._time_to_reversion_hours else 0.0
        )

        # Overshoot Capture (Tranche 3 frequency)
        tranche_3_freq = (
            self._tranche_3_hits / self._trade_count * 100
            if self._trade_count > 0 else 0.0
        )

        # Win Rate and Profit Factor
        win_rate = (self._win_count / self._trade_count * 100) if self._trade_count > 0 else 0.0
        profit_factor = self._gross_profit / self._gross_loss if self._gross_loss > 0 else float("inf")

        # Sharpe Ratio
        sharpe = self._calc_sharpe()

        return {
            "regime_classification_accuracy_pct": round(regime_accuracy, 2),
            "avg_reversion_captured_pct": round(avg_reversion, 2),
            "signal_agreement_rate": {
                "2_of_3_rate_pct": round(signal_2_of_3_rate, 2),
                "3_of_3_rate_pct": round(signal_3_of_3_rate, 2),
                "2_of_3_win_rate_pct": round(signal_2_of_3_win_rate, 2),
                "3_of_3_win_rate_pct": round(signal_3_of_3_win_rate, 2),
            },
            "avg_time_to_reversion_hours": round(avg_time_to_reversion, 2),
            "tranche_3_frequency_pct": round(tranche_3_freq, 2),
            "trade_count": self._trade_count,
            "win_count": self._win_count,
            "win_rate_pct": round(win_rate, 2),
            "profit_factor": round(profit_factor, 2),
            "sharpe_ratio": round(sharpe, 2),
            "gross_profit": round(self._gross_profit, 6),
            "gross_loss": round(self._gross_loss, 6),
            "elapsed_days": round(elapsed_days, 1),
            "dimensional_breakdowns": self.dimensional.get_breakdown(),
        }

    def get_go_live_status(self) -> Dict[str, Any]:
        metrics = self.get_all_metrics()
        risk_status = self._risk.get_risk_status() if self._risk else {}

        max_dd = risk_status.get("max_drawdown_pct", 0.0)

        criteria = self.go_live.evaluate(
            trading_days=int(metrics["elapsed_days"]),
            trade_count=self._trade_count,
            win_rate_pct=metrics["win_rate_pct"],
            profit_factor=metrics["profit_factor"],
            max_drawdown_pct=max_dd,
            sharpe_ratio=metrics["sharpe_ratio"],
        )
        return self.go_live.to_dict(criteria)

    def _calc_sharpe(self) -> float:
        if len(self._trade_pnls) < 2:
            return 0.0
        import numpy as np
        arr = np.array(self._trade_pnls)
        mean_return = np.mean(arr)
        std_return = np.std(arr, ddof=1)
        if std_return == 0:
            return 0.0
        annualized = mean_return * math.sqrt(min(250, len(self._trade_pnls)))
        return annualized / std_return

    # ──────────────────────────────────────────────────────────────────
    #  Persistence
    # ──────────────────────────────────────────────────────────────────

    def get_state(self) -> Dict[str, Any]:
        return {
            "start_time": self._start_time,
            "trade_count": self._trade_count,
            "win_count": self._win_count,
            "gross_profit": self._gross_profit,
            "gross_loss": self._gross_loss,
            "trade_pnls": self._trade_pnls[-500:],
            "signal_2_of_3_count": self._signal_2_of_3_count,
            "signal_3_of_3_count": self._signal_3_of_3_count,
            "signal_2_of_3_wins": self._signal_2_of_3_wins,
            "signal_3_of_3_wins": self._signal_3_of_3_wins,
            "reversion_captured_pcts": self._reversion_captured_pcts[-500:],
            "time_to_reversion_hours": self._time_to_reversion_hours[-500:],
            "tranche_3_hits": self._tranche_3_hits,
            "regime_correct": self._regime_correct,
            "regime_total": self._regime_total,
            "dimensional": self.dimensional.get_state(),
        }

    def restore_state(self, state: Dict[str, Any]) -> None:
        self._start_time = state.get("start_time", self._start_time)
        self._trade_count = state.get("trade_count", 0)
        self._win_count = state.get("win_count", 0)
        self._gross_profit = state.get("gross_profit", 0.0)
        self._gross_loss = state.get("gross_loss", 0.0)
        self._trade_pnls = state.get("trade_pnls", [])
        self._signal_2_of_3_count = state.get("signal_2_of_3_count", 0)
        self._signal_3_of_3_count = state.get("signal_3_of_3_count", 0)
        self._signal_2_of_3_wins = state.get("signal_2_of_3_wins", 0)
        self._signal_3_of_3_wins = state.get("signal_3_of_3_wins", 0)
        self._reversion_captured_pcts = state.get("reversion_captured_pcts", [])
        self._time_to_reversion_hours = state.get("time_to_reversion_hours", [])
        self._tranche_3_hits = state.get("tranche_3_hits", 0)
        self._regime_correct = state.get("regime_correct", 0)
        self._regime_total = state.get("regime_total", 0)
        dim_state = state.get("dimensional")
        if dim_state:
            self.dimensional.restore_state(dim_state)
        logger.info("MeanReversionMetrics state restored: %d trades tracked", self._trade_count)
