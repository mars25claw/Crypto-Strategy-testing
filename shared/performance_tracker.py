"""
Performance Metrics Calculator — STRAT-001 Section 10.1.

Standard performance tracking for ALL bots: trade recording, equity curves,
drawdown analysis, Sharpe/Sortino/Calmar ratios, and multi-dimensional breakdowns.
"""

from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class TradeRecord:
    """Internal representation of a completed trade."""
    trade_id: str
    symbol: str
    side: str  # "LONG" or "SHORT"
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    fees: float
    entry_time_ms: int
    exit_time_ms: int
    duration_ms: int
    initial_risk: Optional[float] = None  # Distance to stop-loss in USDT
    regime: Optional[str] = None  # "trending" / "ranging" / None


class PerformanceTracker:
    """
    Computes and maintains all Section 10.1 performance metrics.

    Args:
        strategy_id: Unique identifier for the strategy.
        risk_free_rate: Annual risk-free rate (decimal, e.g. 0.05 = 5%).
    """

    # Breakeven threshold: a trade with |pnl_pct| <= this is breakeven
    _BREAKEVEN_THRESHOLD_PCT = 0.05

    def __init__(self, strategy_id: str, risk_free_rate: float = 0.05) -> None:
        self.strategy_id = strategy_id
        self.risk_free_rate = risk_free_rate

        self._trades: List[TradeRecord] = []
        self._unrealized_pnl: float = 0.0

        # Equity tracking
        self._equity_snapshots: List[Tuple[int, float]] = []
        self._peak_equity: float = 0.0
        self._current_equity: float = 0.0

        # Drawdown tracking
        self._max_dd_pct: float = 0.0
        self._max_dd_start_ms: int = 0
        self._max_dd_end_ms: int = 0
        self._max_dd_recovery_ms: Optional[int] = None
        self._dd_start_ms: int = 0  # Current drawdown start

        # Streak tracking
        self._current_win_streak: int = 0
        self._current_loss_streak: int = 0
        self._best_win_streak: int = 0
        self._worst_loss_streak: int = 0

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_trade(self, trade_data: dict) -> None:
        """
        Record a completed trade.

        Expected keys in trade_data:
            trade_id, symbol, side ("LONG"/"SHORT"), entry_price, exit_price,
            quantity, pnl, pnl_pct, fees, entry_time_ms, exit_time_ms.
        Optional:
            initial_risk, regime.
        """
        entry_ms = int(trade_data["entry_time_ms"])
        exit_ms = int(trade_data["exit_time_ms"])
        record = TradeRecord(
            trade_id=str(trade_data["trade_id"]),
            symbol=str(trade_data["symbol"]),
            side=str(trade_data.get("side", "LONG")).upper(),
            entry_price=float(trade_data["entry_price"]),
            exit_price=float(trade_data["exit_price"]),
            quantity=float(trade_data["quantity"]),
            pnl=float(trade_data["pnl"]),
            pnl_pct=float(trade_data["pnl_pct"]),
            fees=float(trade_data["fees"]),
            entry_time_ms=entry_ms,
            exit_time_ms=exit_ms,
            duration_ms=exit_ms - entry_ms,
            initial_risk=trade_data.get("initial_risk"),
            regime=trade_data.get("regime"),
        )
        self._trades.append(record)
        self._update_equity_from_trade(record)
        self._update_streaks(record)

    def update_unrealized_pnl(self, unrealized: float) -> None:
        """Update the current unrealized PnL for metrics."""
        self._unrealized_pnl = unrealized

    def get_trade_count(self) -> int:
        """Return total number of recorded trades."""
        return len(self._trades)

    def get_recent_trades(self, limit: int = 50) -> list:
        """Return the most recent trades as dicts."""
        trades = self._trades[-limit:]
        return [self._trade_to_dict(t) for t in trades]

    # ------------------------------------------------------------------
    # Core metrics
    # ------------------------------------------------------------------

    def get_metrics(self) -> dict:
        """
        Compute and return ALL Section 10.1 performance metrics.

        Returns a dict with all metric keys documented in the spec.
        """
        trades = self._trades
        if not trades:
            return self._empty_metrics()

        wins = [t for t in trades if t.pnl_pct > self._BREAKEVEN_THRESHOLD_PCT]
        losses = [t for t in trades if t.pnl_pct < -self._BREAKEVEN_THRESHOLD_PCT]
        breakevens = [
            t
            for t in trades
            if abs(t.pnl_pct) <= self._BREAKEVEN_THRESHOLD_PCT
        ]

        n = len(trades)
        win_rate = len(wins) / n if n else 0.0
        loss_rate = len(losses) / n if n else 0.0
        breakeven_rate = len(breakevens) / n if n else 0.0

        total_fees = sum(t.fees for t in trades)
        realized_pnl = sum(t.pnl for t in trades)
        total_pnl = realized_pnl + self._unrealized_pnl  # Net of fees already in pnl

        avg_win_usdt = _safe_mean([t.pnl for t in wins])
        avg_loss_usdt = _safe_mean([t.pnl for t in losses])
        avg_win_pct = _safe_mean([t.pnl_pct for t in wins])
        avg_loss_pct = _safe_mean([t.pnl_pct for t in losses])

        sum_wins = sum(t.pnl for t in wins)
        sum_losses = abs(sum(t.pnl for t in losses))
        profit_factor = sum_wins / sum_losses if sum_losses > 0 else float("inf")

        # Sharpe & Sortino
        daily_returns = self._compute_daily_returns()
        sharpe_all = self._sharpe(daily_returns)
        sharpe_30d = self._sharpe(daily_returns[-30:]) if len(daily_returns) >= 2 else 0.0
        sortino = self._sortino(daily_returns)

        # Drawdown
        dd = self._compute_max_drawdown()

        # Calmar
        total_days = self._total_trading_days()
        annualized_return = (total_pnl / self._starting_equity_estimate()) * (365.25 / max(total_days, 1))
        calmar = annualized_return / dd["max_drawdown_pct"] if dd["max_drawdown_pct"] > 0 else float("inf")

        # Trade duration
        durations = [t.duration_ms for t in trades]
        avg_duration_ms = _safe_mean(durations)

        # Expectancy
        expectancy = (win_rate * avg_win_usdt) - (loss_rate * abs(avg_loss_usdt))

        # R-multiple distribution
        r_multiples = self._r_multiple_distribution()

        # Trade frequency
        freq = self._trade_frequency()

        # Equity curve with drawdown
        equity_curve = self._equity_curve_with_drawdown()

        return {
            "strategy_id": self.strategy_id,
            "total_pnl": round(total_pnl, 4),
            "realized_pnl": round(realized_pnl, 4),
            "unrealized_pnl": round(self._unrealized_pnl, 4),
            "total_fees": round(total_fees, 4),
            "trade_count": n,
            "win_rate": round(win_rate, 6),
            "loss_rate": round(loss_rate, 6),
            "breakeven_rate": round(breakeven_rate, 6),
            "avg_win": {"usdt": round(avg_win_usdt, 4), "pct": round(avg_win_pct, 4)},
            "avg_loss": {"usdt": round(avg_loss_usdt, 4), "pct": round(avg_loss_pct, 4)},
            "profit_factor": round(profit_factor, 4) if profit_factor != float("inf") else None,
            "sharpe_ratio": {
                "all_time": round(sharpe_all, 4),
                "rolling_30d": round(sharpe_30d, 4),
            },
            "sortino_ratio": round(sortino, 4),
            "max_drawdown": {
                "pct": round(dd["max_drawdown_pct"], 4),
                "start_ms": dd["start_ms"],
                "end_ms": dd["end_ms"],
                "recovery_ms": dd["recovery_ms"],
            },
            "calmar_ratio": round(calmar, 4) if calmar != float("inf") else None,
            "avg_trade_duration": {
                "ms": round(avg_duration_ms),
                "human": _ms_to_human(avg_duration_ms),
            },
            "expectancy": round(expectancy, 4),
            "r_multiple_distribution": r_multiples,
            "trade_frequency": freq,
            "consecutive_win_streak": {
                "current": self._current_win_streak,
                "all_time": self._best_win_streak,
            },
            "consecutive_loss_streak": {
                "current": self._current_loss_streak,
                "all_time": self._worst_loss_streak,
            },
            "equity_curve": equity_curve,
        }

    # ------------------------------------------------------------------
    # Breakdown by dimension
    # ------------------------------------------------------------------

    def get_breakdown(self, dimension: str) -> dict:
        """
        Return metrics broken down by the given dimension.

        Supported dimensions:
            "asset"        — per symbol
            "time_of_day"  — 4-hour UTC blocks (00-04, 04-08, ..., 20-24)
            "day_of_week"  — Mon through Sun
            "direction"    — long vs short
            "month"        — by calendar month (YYYY-MM)
            "regime"       — by market regime label

        Returns:
            Dict mapping each bucket to a sub-metrics dict.
        """
        grouper = {
            "asset": self._group_by_asset,
            "time_of_day": self._group_by_time_of_day,
            "day_of_week": self._group_by_day_of_week,
            "direction": self._group_by_direction,
            "month": self._group_by_month,
            "regime": self._group_by_regime,
        }.get(dimension)

        if grouper is None:
            raise ValueError(
                f"Unknown dimension '{dimension}'. "
                f"Supported: asset, time_of_day, day_of_week, direction, month, regime"
            )

        groups: Dict[str, List[TradeRecord]] = grouper()
        result: Dict[str, dict] = {}
        for key, group_trades in sorted(groups.items()):
            result[key] = self._metrics_for_trades(group_trades)
        return result

    # ------------------------------------------------------------------
    # Private: equity & drawdown
    # ------------------------------------------------------------------

    def _update_equity_from_trade(self, trade: TradeRecord) -> None:
        """Update equity tracking after a trade is recorded."""
        self._current_equity += trade.pnl
        ts = trade.exit_time_ms

        if self._current_equity > self._peak_equity:
            self._peak_equity = self._current_equity
            # Recovered from drawdown
            if self._max_dd_recovery_ms is None and self._max_dd_pct > 0:
                self._max_dd_recovery_ms = ts

        # Calculate current drawdown
        if self._peak_equity > 0:
            dd_pct = (self._peak_equity - self._current_equity) / self._peak_equity * 100.0
        else:
            dd_pct = 0.0

        if dd_pct > self._max_dd_pct:
            self._max_dd_pct = dd_pct
            self._max_dd_end_ms = ts
            if self._dd_start_ms == 0:
                self._dd_start_ms = ts
            self._max_dd_start_ms = self._dd_start_ms
            self._max_dd_recovery_ms = None  # Not recovered yet

        if dd_pct == 0 and self._dd_start_ms > 0:
            self._dd_start_ms = 0  # Reset for next drawdown period

        self._equity_snapshots.append((ts, self._current_equity))

    def _compute_max_drawdown(self) -> dict:
        return {
            "max_drawdown_pct": self._max_dd_pct,
            "start_ms": self._max_dd_start_ms,
            "end_ms": self._max_dd_end_ms,
            "recovery_ms": self._max_dd_recovery_ms,
        }

    def _starting_equity_estimate(self) -> float:
        """Estimate starting equity from first snapshot or default."""
        if self._equity_snapshots:
            first_eq = self._equity_snapshots[0][1] - self._trades[0].pnl
            return max(first_eq, 1.0)
        return 10000.0

    def _total_trading_days(self) -> float:
        if len(self._trades) < 2:
            return 1.0
        first_ms = self._trades[0].entry_time_ms
        last_ms = self._trades[-1].exit_time_ms
        return max((last_ms - first_ms) / 86_400_000, 1.0)

    def _equity_curve_with_drawdown(self) -> List[dict]:
        """Build equity curve entries with drawdown percentage."""
        if not self._equity_snapshots:
            return []

        curve = []
        peak = 0.0
        for ts, eq in self._equity_snapshots:
            if eq > peak:
                peak = eq
            dd_pct = ((peak - eq) / peak * 100.0) if peak > 0 else 0.0
            curve.append({
                "timestamp_ms": ts,
                "equity": round(eq, 4),
                "drawdown_pct": round(dd_pct, 4),
            })
        return curve

    # ------------------------------------------------------------------
    # Private: returns & ratios
    # ------------------------------------------------------------------

    def _compute_daily_returns(self) -> List[float]:
        """Aggregate trade PnLs into daily returns (pct of running equity)."""
        if not self._trades:
            return []

        daily: Dict[int, float] = defaultdict(float)
        for t in self._trades:
            day_key = t.exit_time_ms // 86_400_000
            daily[day_key] += t.pnl

        if not daily:
            return []

        equity = self._starting_equity_estimate()
        returns = []
        for day_key in sorted(daily):
            pnl = daily[day_key]
            ret = pnl / equity if equity > 0 else 0.0
            returns.append(ret)
            equity += pnl

        return returns

    def _sharpe(self, daily_returns: List[float]) -> float:
        """Annualized Sharpe ratio from daily return series."""
        if len(daily_returns) < 2:
            return 0.0
        mean_ret = sum(daily_returns) / len(daily_returns)
        std_ret = _std(daily_returns)
        if std_ret == 0:
            return 0.0
        daily_rf = self.risk_free_rate / 365.25
        return ((mean_ret - daily_rf) / std_ret) * math.sqrt(365.25)

    def _sortino(self, daily_returns: Optional[List[float]] = None) -> float:
        """Annualized Sortino ratio (downside deviation only)."""
        if daily_returns is None:
            daily_returns = self._compute_daily_returns()
        if len(daily_returns) < 2:
            return 0.0
        mean_ret = sum(daily_returns) / len(daily_returns)
        daily_rf = self.risk_free_rate / 365.25
        downside = [min(r - daily_rf, 0.0) for r in daily_returns]
        downside_dev = math.sqrt(sum(d ** 2 for d in downside) / len(downside))
        if downside_dev == 0:
            return 0.0
        return ((mean_ret - daily_rf) / downside_dev) * math.sqrt(365.25)

    # ------------------------------------------------------------------
    # Private: streaks
    # ------------------------------------------------------------------

    def _update_streaks(self, trade: TradeRecord) -> None:
        if trade.pnl_pct > self._BREAKEVEN_THRESHOLD_PCT:
            self._current_win_streak += 1
            self._current_loss_streak = 0
            self._best_win_streak = max(self._best_win_streak, self._current_win_streak)
        elif trade.pnl_pct < -self._BREAKEVEN_THRESHOLD_PCT:
            self._current_loss_streak += 1
            self._current_win_streak = 0
            self._worst_loss_streak = max(self._worst_loss_streak, self._current_loss_streak)
        else:
            # Breakeven resets neither streak
            pass

    # ------------------------------------------------------------------
    # Private: R-multiple distribution
    # ------------------------------------------------------------------

    def _r_multiple_distribution(self) -> dict:
        """
        Histogram of outcomes in units of initial risk (R).

        Buckets: < -2R, -2R to -1R, -1R to 0, 0 to 1R, 1R to 2R, 2R to 3R, > 3R.
        """
        buckets = {
            "< -2R": 0, "-2R to -1R": 0, "-1R to 0": 0,
            "0 to 1R": 0, "1R to 2R": 0, "2R to 3R": 0, "> 3R": 0,
        }
        r_values: List[float] = []

        for t in self._trades:
            if t.initial_risk and t.initial_risk > 0:
                r = t.pnl / t.initial_risk
            else:
                # Fallback: use 1% of notional as initial risk proxy
                notional = t.entry_price * t.quantity
                risk = notional * 0.01
                r = t.pnl / risk if risk > 0 else 0.0

            r_values.append(r)

            if r < -2:
                buckets["< -2R"] += 1
            elif r < -1:
                buckets["-2R to -1R"] += 1
            elif r < 0:
                buckets["-1R to 0"] += 1
            elif r < 1:
                buckets["0 to 1R"] += 1
            elif r < 2:
                buckets["1R to 2R"] += 1
            elif r < 3:
                buckets["2R to 3R"] += 1
            else:
                buckets["> 3R"] += 1

        return {
            "histogram": buckets,
            "mean_r": round(_safe_mean(r_values), 4) if r_values else 0.0,
            "median_r": round(_median(r_values), 4) if r_values else 0.0,
        }

    # ------------------------------------------------------------------
    # Private: trade frequency
    # ------------------------------------------------------------------

    def _trade_frequency(self) -> dict:
        """Trades per hour, day, and week."""
        n = len(self._trades)
        if n < 2:
            return {"per_hour": 0.0, "per_day": 0.0, "per_week": 0.0}

        span_ms = self._trades[-1].exit_time_ms - self._trades[0].entry_time_ms
        span_hours = max(span_ms / 3_600_000, 1.0)

        per_hour = n / span_hours
        return {
            "per_hour": round(per_hour, 4),
            "per_day": round(per_hour * 24, 4),
            "per_week": round(per_hour * 168, 4),
        }

    # ------------------------------------------------------------------
    # Private: grouping for breakdowns
    # ------------------------------------------------------------------

    def _group_by_asset(self) -> Dict[str, List[TradeRecord]]:
        groups: Dict[str, List[TradeRecord]] = defaultdict(list)
        for t in self._trades:
            groups[t.symbol].append(t)
        return groups

    def _group_by_time_of_day(self) -> Dict[str, List[TradeRecord]]:
        """4-hour UTC blocks."""
        blocks = ["00-04", "04-08", "08-12", "12-16", "16-20", "20-24"]
        groups: Dict[str, List[TradeRecord]] = defaultdict(list)
        for t in self._trades:
            dt = datetime.fromtimestamp(t.entry_time_ms / 1000, tz=timezone.utc)
            block_idx = dt.hour // 4
            groups[blocks[block_idx]].append(t)
        return groups

    def _group_by_day_of_week(self) -> Dict[str, List[TradeRecord]]:
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        groups: Dict[str, List[TradeRecord]] = defaultdict(list)
        for t in self._trades:
            dt = datetime.fromtimestamp(t.entry_time_ms / 1000, tz=timezone.utc)
            groups[day_names[dt.weekday()]].append(t)
        return groups

    def _group_by_direction(self) -> Dict[str, List[TradeRecord]]:
        groups: Dict[str, List[TradeRecord]] = defaultdict(list)
        for t in self._trades:
            groups[t.side.lower()].append(t)
        return groups

    def _group_by_month(self) -> Dict[str, List[TradeRecord]]:
        groups: Dict[str, List[TradeRecord]] = defaultdict(list)
        for t in self._trades:
            dt = datetime.fromtimestamp(t.entry_time_ms / 1000, tz=timezone.utc)
            key = dt.strftime("%Y-%m")
            groups[key].append(t)
        return groups

    def _group_by_regime(self) -> Dict[str, List[TradeRecord]]:
        groups: Dict[str, List[TradeRecord]] = defaultdict(list)
        for t in self._trades:
            key = t.regime if t.regime else "unknown"
            groups[key].append(t)
        return groups

    # ------------------------------------------------------------------
    # Private: metrics for a subset of trades
    # ------------------------------------------------------------------

    def _metrics_for_trades(self, trades: List[TradeRecord]) -> dict:
        """Compute summary metrics for a subset of trades."""
        if not trades:
            return {"trade_count": 0}

        n = len(trades)
        wins = [t for t in trades if t.pnl_pct > self._BREAKEVEN_THRESHOLD_PCT]
        losses = [t for t in trades if t.pnl_pct < -self._BREAKEVEN_THRESHOLD_PCT]

        total_pnl = sum(t.pnl for t in trades)
        win_rate = len(wins) / n if n else 0.0
        avg_win = _safe_mean([t.pnl for t in wins])
        avg_loss = _safe_mean([t.pnl for t in losses])

        sum_wins = sum(t.pnl for t in wins)
        sum_losses = abs(sum(t.pnl for t in losses))
        pf = sum_wins / sum_losses if sum_losses > 0 else None

        return {
            "trade_count": n,
            "total_pnl": round(total_pnl, 4),
            "win_rate": round(win_rate, 6),
            "avg_win": round(avg_win, 4),
            "avg_loss": round(avg_loss, 4),
            "profit_factor": round(pf, 4) if pf is not None else None,
            "total_fees": round(sum(t.fees for t in trades), 4),
            "avg_duration_ms": round(_safe_mean([t.duration_ms for t in trades])),
        }

    # ------------------------------------------------------------------
    # Private: helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _trade_to_dict(t: TradeRecord) -> dict:
        return {
            "trade_id": t.trade_id,
            "symbol": t.symbol,
            "side": t.side,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "quantity": t.quantity,
            "pnl": t.pnl,
            "pnl_pct": t.pnl_pct,
            "fees": t.fees,
            "entry_time_ms": t.entry_time_ms,
            "exit_time_ms": t.exit_time_ms,
            "duration_ms": t.duration_ms,
            "initial_risk": t.initial_risk,
            "regime": t.regime,
        }

    def _empty_metrics(self) -> dict:
        return {
            "strategy_id": self.strategy_id,
            "total_pnl": 0.0,
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "total_fees": 0.0,
            "trade_count": 0,
            "win_rate": 0.0,
            "loss_rate": 0.0,
            "breakeven_rate": 0.0,
            "avg_win": {"usdt": 0.0, "pct": 0.0},
            "avg_loss": {"usdt": 0.0, "pct": 0.0},
            "profit_factor": None,
            "sharpe_ratio": {"all_time": 0.0, "rolling_30d": 0.0},
            "sortino_ratio": 0.0,
            "max_drawdown": {"pct": 0.0, "start_ms": 0, "end_ms": 0, "recovery_ms": None},
            "calmar_ratio": None,
            "avg_trade_duration": {"ms": 0, "human": "0s"},
            "expectancy": 0.0,
            "r_multiple_distribution": {"histogram": {}, "mean_r": 0.0, "median_r": 0.0},
            "trade_frequency": {"per_hour": 0.0, "per_day": 0.0, "per_week": 0.0},
            "consecutive_win_streak": {"current": 0, "all_time": 0},
            "consecutive_loss_streak": {"current": 0, "all_time": 0},
            "equity_curve": [],
        }


# ======================================================================
# Section 10.3 — Dimensional Breakdowns
# ======================================================================

class DimensionalBreakdown:
    """
    Tracks ALL trade metrics broken down by multiple dimensions.

    Dimensions:
        asset        — per symbol (BTC, ETH, SOL, etc.)
        time_of_day  — 4-hour UTC blocks (00-04, 04-08, 08-12, 12-16, 16-20, 20-24)
        day_of_week  — Mon through Sun
        regime       — trending (H > 0.55), ranging (H < 0.45), neutral
        direction    — long vs short
        month        — Jan through Dec

    Each bucket tracks: win_rate, avg_win, avg_loss, profit_factor,
    total_pnl, trade_count, sharpe_ratio.
    """

    _TIME_BLOCKS = ["00-04", "04-08", "08-12", "12-16", "16-20", "20-24"]
    _DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    _MONTH_NAMES = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]

    # Hurst exponent thresholds for regime classification
    _TRENDING_THRESHOLD = 0.55
    _RANGING_THRESHOLD = 0.45

    # Breakeven threshold consistent with PerformanceTracker
    _BREAKEVEN_THRESHOLD_PCT = 0.05

    def __init__(self) -> None:
        # Each dimension maps bucket_name -> list of trade dicts
        self._data: Dict[str, Dict[str, List[dict]]] = {
            "asset": defaultdict(list),
            "time_of_day": defaultdict(list),
            "day_of_week": defaultdict(list),
            "regime": defaultdict(list),
            "direction": defaultdict(list),
            "month": defaultdict(list),
        }

    def record_trade_dimensional(self, trade_record: dict) -> None:
        """
        Extract dimensions from a trade record and store it in all buckets.

        Expected keys:
            symbol, side ("LONG"/"SHORT"), pnl, pnl_pct, fees,
            entry_time_ms, exit_time_ms, entry_price, exit_price, quantity.
        Optional:
            hurst_exponent (float) — used for regime classification.
            regime (str) — explicit regime label; used if hurst_exponent absent.
        """
        # --- Asset ---
        symbol = str(trade_record.get("symbol", "UNKNOWN"))
        self._data["asset"][symbol].append(trade_record)

        # --- Time of day (entry time, 4-hour UTC block) ---
        entry_ms = int(trade_record.get("entry_time_ms", 0))
        if entry_ms > 0:
            dt = datetime.fromtimestamp(entry_ms / 1000, tz=timezone.utc)
            block_idx = dt.hour // 4
            self._data["time_of_day"][self._TIME_BLOCKS[block_idx]].append(trade_record)

            # --- Day of week ---
            self._data["day_of_week"][self._DAY_NAMES[dt.weekday()]].append(trade_record)

            # --- Month ---
            self._data["month"][self._MONTH_NAMES[dt.month - 1]].append(trade_record)
        else:
            self._data["time_of_day"]["unknown"].append(trade_record)
            self._data["day_of_week"]["unknown"].append(trade_record)
            self._data["month"]["unknown"].append(trade_record)

        # --- Regime (trending / ranging / neutral) ---
        regime = self._classify_regime(trade_record)
        self._data["regime"][regime].append(trade_record)

        # --- Direction ---
        side = str(trade_record.get("side", "UNKNOWN")).lower()
        self._data["direction"][side].append(trade_record)

    def get_breakdown(self, dimension: str) -> dict:
        """
        Return all metrics for a single dimension.

        Args:
            dimension: One of asset, time_of_day, day_of_week, regime,
                       direction, month.

        Returns:
            Dict mapping each bucket name to its metrics dict containing:
            win_rate, avg_win, avg_loss, profit_factor, total_pnl,
            trade_count, sharpe_ratio.
        """
        if dimension not in self._data:
            raise ValueError(
                f"Unknown dimension '{dimension}'. "
                f"Supported: {', '.join(sorted(self._data.keys()))}"
            )

        result: Dict[str, dict] = {}
        for bucket, trades in sorted(self._data[dimension].items()):
            result[bucket] = self._compute_bucket_metrics(trades)
        return result

    def get_full_breakdown(self) -> dict:
        """
        Return metrics for ALL dimensions.

        Returns:
            Dict mapping each dimension name to its breakdown dict
            (as returned by get_breakdown).
        """
        return {dim: self.get_breakdown(dim) for dim in sorted(self._data.keys())}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _classify_regime(self, trade_record: dict) -> str:
        """Classify regime from hurst_exponent or explicit regime field."""
        hurst = trade_record.get("hurst_exponent")
        if hurst is not None:
            try:
                h = float(hurst)
                if h > self._TRENDING_THRESHOLD:
                    return "trending"
                elif h < self._RANGING_THRESHOLD:
                    return "ranging"
                else:
                    return "neutral"
            except (TypeError, ValueError):
                pass

        # Fall back to explicit regime label
        regime = trade_record.get("regime")
        if regime and isinstance(regime, str):
            regime_lower = regime.lower()
            if regime_lower in ("trending", "ranging", "neutral"):
                return regime_lower
        return "neutral"

    def _compute_bucket_metrics(self, trades: List[dict]) -> dict:
        """
        Compute the standard 7 metrics for a bucket of trades.

        Returns dict with: win_rate, avg_win, avg_loss, profit_factor,
        total_pnl, trade_count, sharpe_ratio.
        """
        if not trades:
            return {
                "win_rate": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "profit_factor": None,
                "total_pnl": 0.0,
                "trade_count": 0,
                "sharpe_ratio": 0.0,
            }

        n = len(trades)
        pnls = [float(t.get("pnl", 0.0)) for t in trades]
        pnl_pcts = [float(t.get("pnl_pct", 0.0)) for t in trades]

        wins = [p for p, pct in zip(pnls, pnl_pcts) if pct > self._BREAKEVEN_THRESHOLD_PCT]
        losses = [p for p, pct in zip(pnls, pnl_pcts) if pct < -self._BREAKEVEN_THRESHOLD_PCT]

        win_rate = len(wins) / n if n else 0.0
        avg_win = _safe_mean(wins)
        avg_loss = _safe_mean(losses)

        sum_wins = sum(w for w in wins if w > 0)
        sum_losses = abs(sum(lo for lo in losses if lo < 0))
        profit_factor = (sum_wins / sum_losses) if sum_losses > 0 else None

        total_pnl = sum(pnls)

        # Sharpe: annualized from per-trade returns
        sharpe = self._bucket_sharpe(pnl_pcts)

        return {
            "win_rate": round(win_rate, 6),
            "avg_win": round(avg_win, 4),
            "avg_loss": round(avg_loss, 4),
            "profit_factor": round(profit_factor, 4) if profit_factor is not None else None,
            "total_pnl": round(total_pnl, 4),
            "trade_count": n,
            "sharpe_ratio": round(sharpe, 4),
        }

    @staticmethod
    def _bucket_sharpe(pnl_pcts: List[float], annualization_factor: float = 365.25) -> float:
        """
        Simple Sharpe estimate from a list of per-trade pnl percentages.

        Uses trade-level returns; annualises assuming one trade per day as
        a rough proxy when exact timestamps are not aggregated.
        """
        if len(pnl_pcts) < 2:
            return 0.0
        mean_ret = sum(pnl_pcts) / len(pnl_pcts)
        std_ret = _std(pnl_pcts)
        if std_ret == 0:
            return 0.0
        return (mean_ret / std_ret) * math.sqrt(annualization_factor)


# ======================================================================
# Go-Live Criteria Enforcement
# ======================================================================

class GoLiveCriteriaChecker:
    """
    Validates whether a strategy meets ALL go-live criteria before
    transitioning from paper trading to live trading.

    Constructor args:
        tracker:            PerformanceTracker instance to pull metrics from.
        min_days:           Minimum number of paper trading days.
        min_trades:         Minimum number of completed trades.
        min_win_rate:       Minimum win rate (decimal, e.g. 0.55 = 55%).
        min_profit_factor:  Minimum profit factor.
        max_drawdown:       Maximum drawdown percentage allowed.
        min_sharpe:         Minimum all-time Sharpe ratio.
        max_single_loss_pct: Maximum single-trade loss as pct of equity.
        min_uptime_pct:     Minimum bot uptime percentage.
        custom_criteria:    Dict of {name: callable(metrics) -> bool} for
                            any additional strategy-specific checks.
    """

    def __init__(
        self,
        tracker: PerformanceTracker,
        min_days: float = 30,
        min_trades: int = 100,
        min_win_rate: float = 0.55,
        min_profit_factor: float = 1.5,
        max_drawdown: float = 15.0,
        min_sharpe: float = 1.5,
        max_single_loss_pct: float = 2.0,
        min_uptime_pct: float = 95.0,
        custom_criteria: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._tracker = tracker
        self.min_days = min_days
        self.min_trades = min_trades
        self.min_win_rate = min_win_rate
        self.min_profit_factor = min_profit_factor
        self.max_drawdown = max_drawdown
        self.min_sharpe = min_sharpe
        self.max_single_loss_pct = max_single_loss_pct
        self.min_uptime_pct = min_uptime_pct
        self.custom_criteria: Dict[str, Any] = custom_criteria or {}

        # Paper trading tracking
        self._paper_start_ms: Optional[int] = None
        self._uptime_checks_total: int = 0
        self._uptime_checks_up: int = 0

    # ------------------------------------------------------------------
    # Public: tracking helpers
    # ------------------------------------------------------------------

    def set_paper_start(self, timestamp_ms: Optional[int] = None) -> None:
        """Record when paper trading started."""
        self._paper_start_ms = timestamp_ms or int(time.time() * 1000)

    def record_uptime_check(self, is_up: bool) -> None:
        """Record an uptime heartbeat (call periodically)."""
        self._uptime_checks_total += 1
        if is_up:
            self._uptime_checks_up += 1

    # ------------------------------------------------------------------
    # Public: evaluation
    # ------------------------------------------------------------------

    def check(self) -> Tuple[bool, dict]:
        """
        Check ALL go-live criteria against current performance.

        Returns:
            (ready, results) where ready is True only if ALL criteria pass,
            and results is a dict mapping each criterion name to a dict of
            {passed: bool, current: value, required: value}.
        """
        metrics = self._tracker.get_metrics()
        results: Dict[str, dict] = {}

        # 1. Minimum days of paper trading
        days_elapsed = self._days_elapsed()
        results["min_days"] = {
            "passed": days_elapsed >= self.min_days,
            "current": round(days_elapsed, 2),
            "required": self.min_days,
        }

        # 2. Minimum trade count
        trade_count = metrics.get("trade_count", 0)
        results["min_trades"] = {
            "passed": trade_count >= self.min_trades,
            "current": trade_count,
            "required": self.min_trades,
        }

        # 3. Minimum win rate
        current_wr = metrics.get("win_rate", 0.0)
        results["min_win_rate"] = {
            "passed": current_wr >= self.min_win_rate,
            "current": round(current_wr, 6),
            "required": self.min_win_rate,
        }

        # 4. Minimum profit factor
        current_pf = metrics.get("profit_factor")
        pf_passed = current_pf is not None and current_pf >= self.min_profit_factor
        results["min_profit_factor"] = {
            "passed": pf_passed,
            "current": current_pf,
            "required": self.min_profit_factor,
        }

        # 5. Maximum drawdown
        current_dd = metrics.get("max_drawdown", {}).get("pct", 0.0)
        results["max_drawdown"] = {
            "passed": current_dd <= self.max_drawdown,
            "current": round(current_dd, 4),
            "required": self.max_drawdown,
        }

        # 6. Minimum Sharpe ratio
        current_sharpe = metrics.get("sharpe_ratio", {}).get("all_time", 0.0)
        results["min_sharpe"] = {
            "passed": current_sharpe >= self.min_sharpe,
            "current": round(current_sharpe, 4),
            "required": self.min_sharpe,
        }

        # 7. Maximum single-trade loss
        max_single_loss = self._max_single_loss_pct()
        results["max_single_loss_pct"] = {
            "passed": max_single_loss <= self.max_single_loss_pct,
            "current": round(max_single_loss, 4),
            "required": self.max_single_loss_pct,
        }

        # 8. Minimum uptime
        current_uptime = self._current_uptime_pct()
        results["min_uptime_pct"] = {
            "passed": current_uptime >= self.min_uptime_pct,
            "current": round(current_uptime, 4),
            "required": self.min_uptime_pct,
        }

        # 9. Custom criteria
        for name, check_fn in self.custom_criteria.items():
            try:
                passed = bool(check_fn(metrics))
            except Exception:
                passed = False
            results[f"custom_{name}"] = {
                "passed": passed,
                "current": "evaluated",
                "required": "pass",
            }

        ready = all(r["passed"] for r in results.values())
        return ready, results

    def get_status(self) -> dict:
        """
        Return current vs required for each criterion (convenience wrapper).

        Returns:
            Dict mapping criterion name to {current, required, passed}.
        """
        _, results = self.check()
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _days_elapsed(self) -> float:
        """Days since paper trading started."""
        if self._paper_start_ms is None:
            # Try to infer from first trade
            trades = self._tracker._trades
            if trades:
                first_ms = trades[0].entry_time_ms
                now_ms = int(time.time() * 1000)
                return (now_ms - first_ms) / 86_400_000
            return 0.0
        now_ms = int(time.time() * 1000)
        return (now_ms - self._paper_start_ms) / 86_400_000

    def _max_single_loss_pct(self) -> float:
        """Return the largest single-trade loss as a percentage."""
        trades = self._tracker._trades
        if not trades:
            return 0.0
        return max(abs(t.pnl_pct) for t in trades if t.pnl_pct < 0) if any(
            t.pnl_pct < 0 for t in trades
        ) else 0.0

    def _current_uptime_pct(self) -> float:
        """Current uptime percentage from heartbeat checks."""
        if self._uptime_checks_total == 0:
            return 100.0  # No checks yet — assume up
        return (self._uptime_checks_up / self._uptime_checks_total) * 100.0


# ======================================================================
# Strategy-Specific Metrics Base
# ======================================================================

class StrategyMetrics(ABC):
    """
    Abstract base class for strategy-specific metrics that individual
    bot implementations extend.

    Provides built-in tracking for:
        - R-multiple distribution
        - Holding period distribution
        - Slippage analysis (entry, exit, total as % of gross profits)

    Subclasses MUST override record_trade() and get_metrics().
    """

    def __init__(self) -> None:
        # R-multiple tracking
        self._r_multiples: List[float] = []

        # Holding period tracking (duration in ms)
        self._holding_periods_ms: List[int] = []

        # Slippage tracking
        self._entry_slippages: List[float] = []  # USDT
        self._exit_slippages: List[float] = []   # USDT
        self._gross_profits: float = 0.0

    # ------------------------------------------------------------------
    # Abstract methods — subclasses must implement
    # ------------------------------------------------------------------

    @abstractmethod
    def record_trade(self, trade: dict) -> None:
        """
        Record a trade with strategy-specific processing.

        Subclasses should call super().record_trade(trade) to populate
        the built-in R-multiple, holding period, and slippage tracking,
        then add their own strategy-specific logic.
        """
        self._record_r_multiple(trade)
        self._record_holding_period(trade)
        self._record_slippage(trade)

    @abstractmethod
    def get_metrics(self) -> dict:
        """
        Return strategy-specific metrics.

        Subclasses should call super().get_metrics() to get the base
        metrics dict, then merge in their own strategy-specific fields.
        """
        return {
            "r_multiple_distribution": self._get_r_multiple_stats(),
            "holding_period_distribution": self._get_holding_period_stats(),
            "slippage_analysis": self._get_slippage_analysis(),
        }

    # ------------------------------------------------------------------
    # Built-in: R-multiple distribution
    # ------------------------------------------------------------------

    def _record_r_multiple(self, trade: dict) -> None:
        """Extract and store the R-multiple for a trade."""
        pnl = float(trade.get("pnl", 0.0))
        initial_risk = trade.get("initial_risk")

        if initial_risk and float(initial_risk) > 0:
            r = pnl / float(initial_risk)
        else:
            # Fallback: 1% of notional as risk proxy
            entry_price = float(trade.get("entry_price", 0.0))
            quantity = float(trade.get("quantity", 0.0))
            notional = entry_price * quantity
            risk = notional * 0.01
            r = pnl / risk if risk > 0 else 0.0

        self._r_multiples.append(r)

    def _get_r_multiple_stats(self) -> dict:
        """Return R-multiple distribution statistics."""
        if not self._r_multiples:
            return {
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "histogram": {},
                "count": 0,
            }

        buckets = {
            "< -2R": 0, "-2R to -1R": 0, "-1R to 0": 0,
            "0 to 1R": 0, "1R to 2R": 0, "2R to 3R": 0, "> 3R": 0,
        }
        for r in self._r_multiples:
            if r < -2:
                buckets["< -2R"] += 1
            elif r < -1:
                buckets["-2R to -1R"] += 1
            elif r < 0:
                buckets["-1R to 0"] += 1
            elif r < 1:
                buckets["0 to 1R"] += 1
            elif r < 2:
                buckets["1R to 2R"] += 1
            elif r < 3:
                buckets["2R to 3R"] += 1
            else:
                buckets["> 3R"] += 1

        return {
            "mean": round(_safe_mean(self._r_multiples), 4),
            "median": round(_median(self._r_multiples), 4),
            "std": round(_std(self._r_multiples), 4),
            "min": round(min(self._r_multiples), 4),
            "max": round(max(self._r_multiples), 4),
            "histogram": buckets,
            "count": len(self._r_multiples),
        }

    # ------------------------------------------------------------------
    # Built-in: Holding period distribution
    # ------------------------------------------------------------------

    def _record_holding_period(self, trade: dict) -> None:
        """Extract and store the holding period for a trade."""
        entry_ms = int(trade.get("entry_time_ms", 0))
        exit_ms = int(trade.get("exit_time_ms", 0))
        if entry_ms > 0 and exit_ms > entry_ms:
            self._holding_periods_ms.append(exit_ms - entry_ms)

    def _get_holding_period_stats(self) -> dict:
        """Return holding period distribution statistics."""
        if not self._holding_periods_ms:
            return {
                "mean_ms": 0,
                "median_ms": 0,
                "min_ms": 0,
                "max_ms": 0,
                "mean_human": "0s",
                "median_human": "0s",
                "histogram": {},
                "count": 0,
            }

        periods = self._holding_periods_ms
        mean_ms = _safe_mean([float(p) for p in periods])
        median_ms = _median([float(p) for p in periods])

        # Histogram buckets
        buckets = {
            "< 1m": 0,
            "1m-5m": 0,
            "5m-15m": 0,
            "15m-1h": 0,
            "1h-4h": 0,
            "4h-24h": 0,
            "1d-7d": 0,
            "> 7d": 0,
        }
        for p in periods:
            minutes = p / 60_000
            if minutes < 1:
                buckets["< 1m"] += 1
            elif minutes < 5:
                buckets["1m-5m"] += 1
            elif minutes < 15:
                buckets["5m-15m"] += 1
            elif minutes < 60:
                buckets["15m-1h"] += 1
            elif minutes < 240:
                buckets["1h-4h"] += 1
            elif minutes < 1440:
                buckets["4h-24h"] += 1
            elif minutes < 10080:
                buckets["1d-7d"] += 1
            else:
                buckets["> 7d"] += 1

        return {
            "mean_ms": round(mean_ms),
            "median_ms": round(median_ms),
            "min_ms": min(periods),
            "max_ms": max(periods),
            "mean_human": _ms_to_human(mean_ms),
            "median_human": _ms_to_human(median_ms),
            "histogram": buckets,
            "count": len(periods),
        }

    # ------------------------------------------------------------------
    # Built-in: Slippage analysis
    # ------------------------------------------------------------------

    def _record_slippage(self, trade: dict) -> None:
        """
        Extract and store slippage data for a trade.

        Expected optional keys:
            expected_entry_price — price the strategy intended to enter at
            expected_exit_price  — price the strategy intended to exit at
            entry_price          — actual entry price
            exit_price           — actual exit price
            quantity             — trade quantity
            pnl                  — trade pnl (used for gross profits)
        """
        quantity = float(trade.get("quantity", 0.0))
        pnl = float(trade.get("pnl", 0.0))

        # Track gross profits (winning trades only)
        if pnl > 0:
            self._gross_profits += pnl

        # Entry slippage
        expected_entry = trade.get("expected_entry_price")
        actual_entry = trade.get("entry_price")
        if expected_entry is not None and actual_entry is not None:
            entry_slip = abs(float(actual_entry) - float(expected_entry)) * quantity
            self._entry_slippages.append(entry_slip)

        # Exit slippage
        expected_exit = trade.get("expected_exit_price")
        actual_exit = trade.get("exit_price")
        if expected_exit is not None and actual_exit is not None:
            exit_slip = abs(float(actual_exit) - float(expected_exit)) * quantity
            self._exit_slippages.append(exit_slip)

    def _get_slippage_analysis(self) -> dict:
        """
        Return slippage analysis: entry, exit, and total slippage
        both in absolute USDT and as a percentage of gross profits.
        """
        total_entry = sum(self._entry_slippages)
        total_exit = sum(self._exit_slippages)
        total_slippage = total_entry + total_exit

        gp = self._gross_profits if self._gross_profits > 0 else 1.0  # Avoid div/0

        return {
            "entry_slippage_usdt": round(total_entry, 4),
            "exit_slippage_usdt": round(total_exit, 4),
            "total_slippage_usdt": round(total_slippage, 4),
            "entry_slippage_pct_of_gross": round((total_entry / gp) * 100, 4),
            "exit_slippage_pct_of_gross": round((total_exit / gp) * 100, 4),
            "total_slippage_pct_of_gross": round((total_slippage / gp) * 100, 4),
            "avg_entry_slippage": round(
                _safe_mean(self._entry_slippages), 4
            ),
            "avg_exit_slippage": round(
                _safe_mean(self._exit_slippages), 4
            ),
            "entry_slippage_count": len(self._entry_slippages),
            "exit_slippage_count": len(self._exit_slippages),
        }


# ------------------------------------------------------------------
# Module-level utility functions
# ------------------------------------------------------------------

def _safe_mean(values: List[float]) -> float:
    """Mean that returns 0.0 for empty lists."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def _std(values: List[float]) -> float:
    """Population standard deviation."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(variance)


def _median(values: List[float]) -> float:
    """Median of a list."""
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 0:
        return (s[mid - 1] + s[mid]) / 2
    return s[mid]


def _ms_to_human(ms: float) -> str:
    """Convert milliseconds to human-readable duration string."""
    if ms <= 0:
        return "0s"
    total_seconds = ms / 1000
    days = int(total_seconds // 86400)
    hours = int((total_seconds % 86400) // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)

    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if seconds > 0 or not parts:
        parts.append(f"{seconds}s")
    return " ".join(parts)
