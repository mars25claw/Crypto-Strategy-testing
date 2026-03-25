"""Section 10.2 & 10.3 Strategy-Specific Metrics for STRAT-003 Stat Arb Pairs.

Metrics tracked:
- Z-Score Distribution (histogram of Z at entry/exit)
- Mean Reversion Capture Rate (% of theoretical mean-reversion captured)
- Half-Life Accuracy (actual vs predicted half-life)
- Cointegration Stability Score (% of time pairs remain cointegrated)
- Beta Drift (actual vs original hedge ratio drift)
- Pair-Level Attribution (PnL per pair)
- Net Directional Exposure over Time
- Stop Loss Frequency per Pair
- Z-Score at Entry Distribution

Dimensional breakdowns by:
- Z-score regime (shallow/medium/deep)
- Volatility regime (low/medium/high)
- Pair tier (1/2/3)
- Market regime (trending/ranging)

Go-live criteria:
- 60-day paper trading
- 30+ trades
- Profit Factor > 1.3
- Win rate > 50%
- Sharpe > 1.0
- Max drawdown < 8%
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
    """Multi-dimensional performance breakdown engine for pairs trading."""

    def __init__(self) -> None:
        self._buckets: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(
            lambda: defaultdict(lambda: {
                "trades": 0,
                "wins": 0,
                "pnl": 0.0,
                "total_fees": 0.0,
                "holding_hours": 0.0,
                "stop_losses": 0,
            })
        )

    def record(
        self,
        dimensions: Dict[str, str],
        pnl: float,
        is_win: bool,
        fees: float = 0.0,
        holding_hours: float = 0.0,
        is_stop_loss: bool = False,
    ) -> None:
        for dim_name, bucket_name in dimensions.items():
            bucket = self._buckets[dim_name][bucket_name]
            bucket["trades"] += 1
            if is_win:
                bucket["wins"] += 1
            bucket["pnl"] += pnl
            bucket["total_fees"] += fees
            bucket["holding_hours"] += holding_hours
            if is_stop_loss:
                bucket["stop_losses"] += 1

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
                    "stop_losses": int(data["stop_losses"]),
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
    """STRAT-003 go-live criteria:
    - 60-day paper trading
    - 30+ trades
    - Profit Factor > 1.3
    - Win rate > 50%
    - Sharpe > 1.0
    - Max drawdown < 8%
    """

    def evaluate(
        self,
        trading_days: int,
        trade_count: int,
        profit_factor: float,
        win_rate_pct: float,
        sharpe_ratio: float,
        max_drawdown_pct: float,
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
                description="Minimum 30 completed trades",
                threshold=30,
                current_value=trade_count,
                passed=trade_count >= 30,
                unit="trades",
            ),
            GoLiveCriterion(
                name="profit_factor",
                description="Profit Factor > 1.3",
                threshold=1.3,
                current_value=round(profit_factor, 2),
                passed=profit_factor > 1.3,
                unit="x",
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
                name="sharpe_ratio",
                description="Sharpe ratio > 1.0",
                threshold=1.0,
                current_value=round(sharpe_ratio, 2),
                passed=sharpe_ratio > 1.0,
                unit="",
            ),
            GoLiveCriterion(
                name="max_drawdown",
                description="Max drawdown < 8%",
                threshold=8.0,
                current_value=round(max_drawdown_pct, 2),
                passed=max_drawdown_pct < 8.0,
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

class PairsMetrics:
    """Aggregates all Section 10.2 strategy-specific metrics for STRAT-003."""

    def __init__(
        self,
        strategy: Any,
        coint_engine: Any,
        risk_manager: Any,
        perf_tracker: Any,
    ) -> None:
        self._strategy = strategy
        self._coint = coint_engine
        self._risk = risk_manager
        self._perf = perf_tracker

        # Dimensional breakdowns
        self.dimensional = DimensionalBreakdown()
        self.go_live = GoLiveCriteriaChecker()

        # Internal tracking
        self._start_time: float = time.time()
        self._z_scores_at_entry: List[float] = []
        self._z_scores_at_exit: List[float] = []
        self._half_life_predicted: List[float] = []
        self._half_life_actual: List[float] = []
        self._pair_pnl: Dict[str, float] = defaultdict(float)
        self._pair_trades: Dict[str, int] = defaultdict(int)
        self._pair_stops: Dict[str, int] = defaultdict(int)
        self._directional_exposure_history: List[Dict[str, float]] = []
        self._trade_count: int = 0
        self._win_count: int = 0
        self._gross_profit: float = 0.0
        self._gross_loss: float = 0.0
        self._trade_pnls: List[float] = []

    # ──────────────────────────────────────────────────────────────────
    #  Regime classification
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def classify_z_score_regime(z: float) -> str:
        abs_z = abs(z)
        if abs_z >= 3.0:
            return "deep"
        elif abs_z >= 2.0:
            return "medium"
        else:
            return "shallow"

    @staticmethod
    def classify_volatility_regime(spread_std: float) -> str:
        if spread_std > 0.05:
            return "high"
        elif spread_std > 0.02:
            return "medium"
        else:
            return "low"

    @staticmethod
    def classify_pair_tier(asset_a: str, asset_b: str) -> str:
        tier1 = {
            frozenset({"BTCUSDT", "ETHUSDT"}),
            frozenset({"ETHUSDT", "SOLUSDT"}),
            frozenset({"BTCUSDT", "BNBUSDT"}),
        }
        tier2_assets = {"SOLUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT",
                        "MATICUSDT", "ARBUSDT", "DOGEUSDT", "SHIBUSDT"}

        pair = frozenset({asset_a, asset_b})
        if pair in tier1:
            return "tier_1"
        if asset_a in tier2_assets and asset_b in tier2_assets:
            return "tier_2"
        return "tier_3"

    # ──────────────────────────────────────────────────────────────────
    #  Trade recording
    # ──────────────────────────────────────────────────────────────────

    def record_trade(
        self,
        asset_a: str,
        asset_b: str,
        pnl: float,
        is_win: bool,
        z_at_entry: float,
        z_at_exit: float,
        predicted_half_life: float,
        actual_holding_hours: float,
        spread_std: float,
        fees: float = 0.0,
        is_stop_loss: bool = False,
        market_regime: str = "unknown",
    ) -> None:
        self._trade_count += 1
        if is_win:
            self._win_count += 1
        if pnl > 0:
            self._gross_profit += pnl
        else:
            self._gross_loss += abs(pnl)
        self._trade_pnls.append(pnl)

        pair_key = f"{asset_a}/{asset_b}"
        self._pair_pnl[pair_key] += pnl
        self._pair_trades[pair_key] += 1
        if is_stop_loss:
            self._pair_stops[pair_key] += 1

        self._z_scores_at_entry.append(z_at_entry)
        self._z_scores_at_exit.append(z_at_exit)
        self._half_life_predicted.append(predicted_half_life)
        self._half_life_actual.append(actual_holding_hours / 24.0)

        dimensions = {
            "z_score_regime": self.classify_z_score_regime(z_at_entry),
            "volatility_regime": self.classify_volatility_regime(spread_std),
            "pair_tier": self.classify_pair_tier(asset_a, asset_b),
            "market_regime": market_regime,
        }

        self.dimensional.record(
            dimensions=dimensions,
            pnl=pnl,
            is_win=is_win,
            fees=fees,
            holding_hours=actual_holding_hours,
            is_stop_loss=is_stop_loss,
        )

    def record_directional_exposure(self, long_pct: float, short_pct: float, net_pct: float) -> None:
        self._directional_exposure_history.append({
            "timestamp": time.time(),
            "long_pct": long_pct,
            "short_pct": short_pct,
            "net_pct": net_pct,
        })
        # Keep last 2000 snapshots
        if len(self._directional_exposure_history) > 2000:
            self._directional_exposure_history = self._directional_exposure_history[-2000:]

    # ──────────────────────────────────────────────────────────────────
    #  Metrics computation
    # ──────────────────────────────────────────────────────────────────

    def get_all_metrics(self) -> Dict[str, Any]:
        elapsed_days = (time.time() - self._start_time) / 86400.0

        # Z-Score Distribution
        z_entry_dist = self._compute_histogram(self._z_scores_at_entry)

        # Mean Reversion Capture Rate
        mean_rev_capture = self._calc_mean_reversion_capture()

        # Half-Life Accuracy
        half_life_accuracy = self._calc_half_life_accuracy()

        # Cointegration Stability Score
        coint_stability = self._calc_coint_stability()

        # Beta Drift
        beta_drift = self._calc_beta_drift()

        # Pair-Level Attribution
        pair_attribution = {
            pair: {
                "pnl": round(pnl, 6),
                "trades": self._pair_trades.get(pair, 0),
                "stop_losses": self._pair_stops.get(pair, 0),
            }
            for pair, pnl in self._pair_pnl.items()
        }

        # Profit Factor
        profit_factor = self._gross_profit / self._gross_loss if self._gross_loss > 0 else float("inf")

        # Win Rate
        win_rate = (self._win_count / self._trade_count * 100) if self._trade_count > 0 else 0.0

        # Sharpe Ratio
        sharpe = self._calc_sharpe()

        return {
            "z_score_entry_distribution": z_entry_dist,
            "z_score_exit_distribution": self._compute_histogram(self._z_scores_at_exit),
            "mean_reversion_capture_rate_pct": round(mean_rev_capture, 2),
            "half_life_accuracy_pct": round(half_life_accuracy, 2),
            "cointegration_stability_score_pct": round(coint_stability, 2),
            "beta_drift_avg": round(beta_drift, 4),
            "pair_level_attribution": pair_attribution,
            "net_directional_exposure_history": self._directional_exposure_history[-50:],
            "stop_loss_frequency_per_pair": {
                pair: stops for pair, stops in self._pair_stops.items()
            },
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
        perf_metrics = self._perf.get_metrics() if self._perf else {}

        elapsed_days = int(metrics["elapsed_days"])
        max_dd = perf_metrics.get("max_drawdown_pct", 0.0)

        criteria = self.go_live.evaluate(
            trading_days=elapsed_days,
            trade_count=self._trade_count,
            profit_factor=metrics["profit_factor"],
            win_rate_pct=metrics["win_rate_pct"],
            sharpe_ratio=metrics["sharpe_ratio"],
            max_drawdown_pct=max_dd,
        )
        return self.go_live.to_dict(criteria)

    def _compute_histogram(self, values: List[float], bins: int = 10) -> Dict[str, int]:
        if not values:
            return {}
        min_val = min(values)
        max_val = max(values)
        if min_val == max_val:
            return {f"{min_val:.2f}": len(values)}
        bin_width = (max_val - min_val) / bins
        hist: Dict[str, int] = {}
        for v in values:
            idx = min(int((v - min_val) / bin_width), bins - 1)
            lo = min_val + idx * bin_width
            hi = lo + bin_width
            key = f"{lo:.2f}_to_{hi:.2f}"
            hist[key] = hist.get(key, 0) + 1
        return hist

    def _calc_mean_reversion_capture(self) -> float:
        if not self._z_scores_at_entry or not self._z_scores_at_exit:
            return 0.0
        total_possible = 0.0
        total_captured = 0.0
        for z_entry, z_exit in zip(self._z_scores_at_entry, self._z_scores_at_exit):
            possible = abs(z_entry)
            captured = abs(z_entry) - abs(z_exit)
            total_possible += possible
            total_captured += max(0, captured)
        return (total_captured / total_possible * 100) if total_possible > 0 else 0.0

    def _calc_half_life_accuracy(self) -> float:
        if not self._half_life_predicted or not self._half_life_actual:
            return 0.0
        errors = []
        for pred, actual in zip(self._half_life_predicted, self._half_life_actual):
            if pred > 0:
                errors.append(1.0 - abs(actual - pred) / pred)
        if not errors:
            return 0.0
        return max(0, sum(errors) / len(errors) * 100)

    def _calc_coint_stability(self) -> float:
        if self._coint is None:
            return 0.0
        qualified = self._coint.qualified_pairs
        if not qualified:
            return 0.0
        stable = sum(1 for p in qualified if p.adf_p_value < 0.05)
        return stable / len(qualified) * 100

    def _calc_beta_drift(self) -> float:
        if self._coint is None:
            return 0.0
        qualified = self._coint.qualified_pairs
        if not qualified:
            return 0.0
        # Average absolute drift from 1.0 (no drift)
        drifts = [abs(p.hedge_ratio - 1.0) for p in qualified]
        return sum(drifts) / len(drifts)

    def _calc_sharpe(self) -> float:
        if len(self._trade_pnls) < 2:
            return 0.0
        import numpy as np
        arr = np.array(self._trade_pnls)
        mean_return = np.mean(arr)
        std_return = np.std(arr, ddof=1)
        if std_return == 0:
            return 0.0
        # Annualize: assume ~250 trades/year
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
            "z_scores_at_entry": self._z_scores_at_entry[-500:],
            "z_scores_at_exit": self._z_scores_at_exit[-500:],
            "half_life_predicted": self._half_life_predicted[-500:],
            "half_life_actual": self._half_life_actual[-500:],
            "pair_pnl": dict(self._pair_pnl),
            "pair_trades": dict(self._pair_trades),
            "pair_stops": dict(self._pair_stops),
            "dimensional": self.dimensional.get_state(),
        }

    def restore_state(self, state: Dict[str, Any]) -> None:
        self._start_time = state.get("start_time", self._start_time)
        self._trade_count = state.get("trade_count", 0)
        self._win_count = state.get("win_count", 0)
        self._gross_profit = state.get("gross_profit", 0.0)
        self._gross_loss = state.get("gross_loss", 0.0)
        self._trade_pnls = state.get("trade_pnls", [])
        self._z_scores_at_entry = state.get("z_scores_at_entry", [])
        self._z_scores_at_exit = state.get("z_scores_at_exit", [])
        self._half_life_predicted = state.get("half_life_predicted", [])
        self._half_life_actual = state.get("half_life_actual", [])
        self._pair_pnl = defaultdict(float, state.get("pair_pnl", {}))
        self._pair_trades = defaultdict(int, state.get("pair_trades", {}))
        self._pair_stops = defaultdict(int, state.get("pair_stops", {}))
        dim_state = state.get("dimensional")
        if dim_state:
            self.dimensional.restore_state(dim_state)
        logger.info("PairsMetrics state restored: %d trades tracked", self._trade_count)
