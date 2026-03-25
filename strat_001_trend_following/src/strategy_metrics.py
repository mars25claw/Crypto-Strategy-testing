"""STRAT-001 Strategy-Specific Metrics — Section 10.2.

Extends the shared StrategyMetrics ABC with trend-following-specific KPIs:
  - Trend Capture Efficiency: total trend move vs captured move per trade
  - Average R-Multiple: mean R achieved (target > 2.0R)
  - Scaling Effectiveness: avg PnL with scale-ins vs without
  - Timeframe Agreement Rate: % of 4h signals confirmed by daily TF
  - ADX-Stratified Performance: win_rate and profit_factor for ADX 25-30, 30-40, 40+
  - Volume-Stratified Performance: metrics for normal vs high volume entries
  - Holding Period Distribution: histogram of trade durations
  - Slippage Analysis: avg entry/exit slippage, total slippage as % of gross profits
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional

from shared.performance_tracker import StrategyMetrics

logger = logging.getLogger(__name__)


class TrendStrategyMetrics(StrategyMetrics):
    """Section 10.2 strategy-specific metrics for Trend Following.

    Tracks all trade-level data needed for the eight required KPIs,
    extending the base StrategyMetrics which provides R-multiple
    distribution, holding period distribution, and slippage analysis.
    """

    def __init__(self) -> None:
        super().__init__()

        # Trend Capture Efficiency: per-trade (captured_move, total_trend_move)
        self._trend_captures: List[Dict[str, float]] = []

        # Scaling effectiveness
        self._scaled_trades: List[Dict[str, Any]] = []
        self._unscaled_trades: List[Dict[str, Any]] = []

        # Timeframe agreement tracking
        self._total_4h_signals: int = 0
        self._daily_confirmed_signals: int = 0

        # ADX-stratified: bucket -> list of trades
        self._adx_buckets: Dict[str, List[Dict[str, Any]]] = {
            "25-30": [],
            "30-40": [],
            "40+": [],
        }

        # Volume-stratified: bucket -> list of trades
        self._volume_buckets: Dict[str, List[Dict[str, Any]]] = {
            "normal": [],     # volume_ratio < 1.5x
            "high": [],       # volume_ratio >= 1.5x
        }

    # ------------------------------------------------------------------
    # StrategyMetrics interface
    # ------------------------------------------------------------------

    def record_trade(self, trade: dict) -> None:
        """Record a completed trade with strategy-specific processing.

        Expected additional keys beyond base StrategyMetrics:
            adx_at_entry (float): ADX value when trade was opened
            volume_ratio_at_entry (float): volume ratio at entry
            scale_in_count (int): number of scale-ins executed
            trend_total_move (float): total trend move from entry signal to reversal
            trend_captured_move (float): move actually captured by the trade
            daily_confirmed (bool): whether daily TF confirmed the 4h signal
            entry_slippage (float): entry slippage in USDT
            exit_slippage (float): exit slippage in USDT
        """
        # Call base class to populate R-multiple, holding period, slippage
        super().record_trade(trade)

        pnl = float(trade.get("pnl", trade.get("realized_pnl", 0)))

        # --- Trend Capture Efficiency ---
        total_move = trade.get("trend_total_move", 0)
        captured_move = trade.get("trend_captured_move", 0)
        if total_move and total_move > 0:
            self._trend_captures.append({
                "total_move": float(total_move),
                "captured_move": float(captured_move),
                "efficiency_pct": float(captured_move) / float(total_move) * 100.0,
            })

        # --- Scaling Effectiveness ---
        scale_in_count = trade.get("scale_in_count", 0)
        trade_summary = {"pnl": pnl, "pnl_pct": float(trade.get("pnl_pct", 0))}
        if scale_in_count and int(scale_in_count) > 0:
            self._scaled_trades.append(trade_summary)
        else:
            self._unscaled_trades.append(trade_summary)

        # --- Timeframe Agreement ---
        self._total_4h_signals += 1
        if trade.get("daily_confirmed", True):
            self._daily_confirmed_signals += 1

        # --- ADX-Stratified ---
        adx_val = float(trade.get("adx_at_entry", 30))
        if 25 <= adx_val < 30:
            self._adx_buckets["25-30"].append(trade)
        elif 30 <= adx_val < 40:
            self._adx_buckets["30-40"].append(trade)
        elif adx_val >= 40:
            self._adx_buckets["40+"].append(trade)

        # --- Volume-Stratified ---
        vol_ratio = float(trade.get("volume_ratio_at_entry", 1.0))
        if vol_ratio >= 1.5:
            self._volume_buckets["high"].append(trade)
        else:
            self._volume_buckets["normal"].append(trade)

    def get_metrics(self) -> dict:
        """Return all Section 10.2 strategy-specific metrics."""
        base = super().get_metrics()

        base.update({
            "trend_capture_efficiency": self._get_trend_capture_efficiency(),
            "avg_r_multiple": self._get_avg_r_multiple(),
            "scaling_effectiveness": self._get_scaling_effectiveness(),
            "tf_agreement_rate": self._get_tf_agreement_rate(),
            "adx_stratified": self._get_adx_stratified(),
            "volume_stratified": self._get_volume_stratified(),
        })

        return base

    # ------------------------------------------------------------------
    # Trend Capture Efficiency
    # ------------------------------------------------------------------

    def _get_trend_capture_efficiency(self) -> dict:
        """Average % of total trend move captured per trade."""
        if not self._trend_captures:
            return {
                "avg_efficiency_pct": 0.0,
                "total_trades_with_data": 0,
                "best_pct": 0.0,
                "worst_pct": 0.0,
            }

        efficiencies = [t["efficiency_pct"] for t in self._trend_captures]
        return {
            "avg_efficiency_pct": round(sum(efficiencies) / len(efficiencies), 2),
            "total_trades_with_data": len(efficiencies),
            "best_pct": round(max(efficiencies), 2),
            "worst_pct": round(min(efficiencies), 2),
        }

    # ------------------------------------------------------------------
    # Average R-Multiple
    # ------------------------------------------------------------------

    def _get_avg_r_multiple(self) -> dict:
        """Mean R-multiple achieved (target > 2.0R)."""
        if not self._r_multiples:
            return {"mean_r": 0.0, "target": 2.0, "meets_target": False, "count": 0}

        mean_r = sum(self._r_multiples) / len(self._r_multiples)
        return {
            "mean_r": round(mean_r, 4),
            "target": 2.0,
            "meets_target": mean_r >= 2.0,
            "count": len(self._r_multiples),
        }

    # ------------------------------------------------------------------
    # Scaling Effectiveness
    # ------------------------------------------------------------------

    def _get_scaling_effectiveness(self) -> dict:
        """Compare avg PnL of trades with scale-ins vs without."""
        def _avg_pnl(trades: list) -> float:
            if not trades:
                return 0.0
            return sum(t["pnl"] for t in trades) / len(trades)

        scaled_avg = _avg_pnl(self._scaled_trades)
        unscaled_avg = _avg_pnl(self._unscaled_trades)
        improvement = 0.0
        if unscaled_avg != 0:
            improvement = ((scaled_avg - unscaled_avg) / abs(unscaled_avg)) * 100.0

        return {
            "with_scale_ins": {
                "count": len(self._scaled_trades),
                "avg_pnl": round(scaled_avg, 4),
            },
            "without_scale_ins": {
                "count": len(self._unscaled_trades),
                "avg_pnl": round(unscaled_avg, 4),
            },
            "improvement_pct": round(improvement, 2),
        }

    # ------------------------------------------------------------------
    # Timeframe Agreement Rate
    # ------------------------------------------------------------------

    def _get_tf_agreement_rate(self) -> dict:
        """Percentage of 4h signals confirmed by daily timeframe."""
        if self._total_4h_signals == 0:
            return {"rate_pct": 100.0, "confirmed": 0, "total": 0}

        rate = (self._daily_confirmed_signals / self._total_4h_signals) * 100.0
        return {
            "rate_pct": round(rate, 2),
            "confirmed": self._daily_confirmed_signals,
            "total": self._total_4h_signals,
        }

    # ------------------------------------------------------------------
    # ADX-Stratified Performance
    # ------------------------------------------------------------------

    def _get_adx_stratified(self) -> dict:
        """Win rate and profit factor for ADX 25-30, 30-40, 40+."""
        result = {}
        for bucket, trades in self._adx_buckets.items():
            result[bucket] = self._compute_stratified_metrics(trades)
        return result

    # ------------------------------------------------------------------
    # Volume-Stratified Performance
    # ------------------------------------------------------------------

    def _get_volume_stratified(self) -> dict:
        """Metrics for normal vs high volume entries."""
        result = {}
        for bucket, trades in self._volume_buckets.items():
            result[bucket] = self._compute_stratified_metrics(trades)
        return result

    # ------------------------------------------------------------------
    # Shared stratified metrics computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_stratified_metrics(trades: List[Dict[str, Any]]) -> dict:
        """Compute win_rate, profit_factor, avg_pnl, count for a bucket of trades."""
        if not trades:
            return {
                "count": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "avg_pnl": 0.0,
                "total_pnl": 0.0,
            }

        wins = []
        losses = []
        for t in trades:
            pnl = float(t.get("pnl", t.get("realized_pnl", 0)))
            if pnl > 0:
                wins.append(pnl)
            elif pnl < 0:
                losses.append(abs(pnl))

        total_wins = sum(wins)
        total_losses = sum(losses)
        win_rate = len(wins) / len(trades) * 100.0
        profit_factor = total_wins / total_losses if total_losses > 0 else (
            float("inf") if total_wins > 0 else 0.0
        )
        total_pnl = sum(
            float(t.get("pnl", t.get("realized_pnl", 0))) for t in trades
        )

        return {
            "count": len(trades),
            "win_rate": round(win_rate, 2),
            "profit_factor": round(profit_factor, 4) if profit_factor != float("inf") else "inf",
            "avg_pnl": round(total_pnl / len(trades), 4),
            "total_pnl": round(total_pnl, 4),
        }

    # ------------------------------------------------------------------
    # Signal tracking helpers (called from main bot)
    # ------------------------------------------------------------------

    def record_signal(self, daily_confirmed: bool) -> None:
        """Record a 4h signal for timeframe agreement tracking.

        Called when a signal is generated, before a trade is opened.
        """
        self._total_4h_signals += 1
        if daily_confirmed:
            self._daily_confirmed_signals += 1
