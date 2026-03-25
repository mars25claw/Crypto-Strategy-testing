"""STRAT-001 Dashboard — Section 12.5.

Extends the shared dashboard with strategy-specific endpoints:
  - EMA crossover chart data
  - ADX gauge
  - Multi-timeframe indicator panel
  - Section 10.2 strategy-specific metrics
  - Pending signals view
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, Dict, List, Optional

from shared.dashboard_base import DashboardBase

from . import STRATEGY_ID, STRATEGY_NAME

logger = logging.getLogger(__name__)


class TrendDashboard:
    """Dashboard wrapper for STRAT-001.

    Registers strategy-specific data providers with the shared DashboardBase
    and provides additional API endpoints for trend following metrics.

    Parameters
    ----------
    bot : object
        The main bot instance, providing access to all sub-components.
    port : int
        Dashboard port (default 8081).
    host : str
        Dashboard host (default 0.0.0.0).
    """

    def __init__(self, bot: Any, port: int = 8081, host: str = "0.0.0.0") -> None:
        self.bot = bot
        self._base = DashboardBase(
            strategy_id=STRATEGY_ID,
            strategy_name=STRATEGY_NAME,
            port=port,
            host=host,
            template_dir=str(bot.base_path / "templates"),
        )

        # Register standard data providers
        self._base.set_data_providers(
            positions_fn=self._get_positions,
            trades_fn=self._get_trades,
            metrics_fn=self._get_metrics,
            equity_fn=self._get_equity_curve,
            alerts_fn=self._get_alerts,
            config_fn=self._get_config,
            kill_fn=self._kill_switch,
        )

        # Register additional routes
        self._register_routes()

    def _register_routes(self) -> None:
        """Add strategy-specific API routes."""
        app = self._base.app

        @app.get("/api/indicators/{symbol}")
        async def get_indicators(symbol: str):
            """Get current indicator values for all timeframes."""
            return self.bot.strategy.get_indicator_state().get(symbol, {})

        @app.get("/api/signals")
        async def get_signals():
            """Get pending entry signals."""
            return self.bot.strategy.get_pending_signals_state()

        @app.get("/api/ema-chart/{symbol}")
        async def get_ema_chart(symbol: str):
            """Get EMA crossover chart data for a symbol."""
            return self._build_ema_chart_data(symbol)

        @app.get("/api/adx-gauge/{symbol}")
        async def get_adx_gauge(symbol: str):
            """Get ADX gauge data for a symbol."""
            return self._build_adx_gauge_data(symbol)

        @app.get("/api/multi-tf/{symbol}")
        async def get_multi_tf(symbol: str):
            """Get multi-timeframe alignment data."""
            return self._build_multi_tf_data(symbol)

        @app.get("/api/strat-metrics")
        async def get_strategy_metrics():
            """Get Section 10.2 strategy-specific metrics."""
            if self.bot.strat_metrics:
                return self.bot.strat_metrics.get_metrics()
            return self._build_strategy_metrics()

        @app.get("/api/dimensional-breakdowns")
        async def get_dimensional_breakdowns():
            """Get Section 10.3 dimensional breakdowns (all dimensions)."""
            if self.bot.dimensional:
                return self.bot.dimensional.get_full_breakdown()
            return {}

        @app.get("/api/dimensional-breakdowns/{dimension}")
        async def get_dimensional_breakdown(dimension: str):
            """Get Section 10.3 dimensional breakdown for a specific dimension."""
            if self.bot.dimensional:
                try:
                    return self.bot.dimensional.get_breakdown(dimension)
                except ValueError as e:
                    return {"error": str(e)}
            return {}

        @app.get("/api/go-live-status")
        async def get_go_live_status():
            """Get go-live criteria checker status (Section 9.4)."""
            if self.bot.go_live_checker:
                ready, results = self.bot.go_live_checker.check()
                return {"ready": ready, "criteria": results}
            return {"ready": False, "criteria": {}, "note": "Not in paper mode"}

        @app.get("/api/risk-state")
        async def get_risk_state():
            """Get risk manager state."""
            return self.bot.risk_mgr.get_state()

        @app.get("/api/filter-state")
        async def get_filter_state():
            """Get filter engine state."""
            return self.bot.filters.get_state()

        @app.get("/api/paper-criteria")
        async def get_paper_criteria():
            """Get paper trading go-live criteria status."""
            return self._build_paper_criteria()

    # ======================================================================
    # Data providers
    # ======================================================================

    def _get_positions(self) -> list:
        return self.bot.exit_mgr.get_positions_state()

    def _get_trades(self, limit: int = 50) -> list:
        trades = self.bot.exit_mgr.get_closed_trades()
        return trades[-limit:]

    def _get_metrics(self) -> dict:
        if self.bot.perf_tracker:
            return self.bot.perf_tracker.get_summary()
        return {}

    def _get_equity_curve(self) -> list:
        if self.bot.perf_tracker:
            return [
                {"timestamp": ts, "equity": eq}
                for ts, eq in self.bot.perf_tracker._equity_snapshots[-500:]
            ]
        return []

    def _get_alerts(self) -> list:
        if self.bot.alerter:
            return self.bot.alerter.get_recent(limit=50)
        return []

    def _get_config(self) -> dict:
        return dict(self.bot.config.strategy_params)

    async def _kill_switch(self, reason: str) -> dict:
        await self.bot.emergency_shutdown(reason)
        return {"status": "killed", "reason": reason}

    # ======================================================================
    # Chart data builders
    # ======================================================================

    def _build_ema_chart_data(self, symbol: str) -> dict:
        """Build EMA crossover chart data for the dashboard."""
        buf = self.bot.strategy.buffers.get(symbol, {}).get("4h")
        if not buf or len(buf) < 50:
            return {"timestamps": [], "closes": [], "ema20": [], "ema50": [], "ema200": []}

        from shared.indicators import ema as calc_ema

        closes = buf.get_closes()
        timestamps = buf.get_timestamps().tolist()

        ema20 = calc_ema(closes, 20).tolist()
        ema50 = calc_ema(closes, 50).tolist()
        ema200 = calc_ema(closes, 200).tolist()

        # Last 100 candles for chart
        n = min(100, len(closes))
        return {
            "timestamps": timestamps[-n:],
            "closes": closes[-n:].tolist(),
            "ema20": ema20[-n:],
            "ema50": ema50[-n:],
            "ema200": ema200[-n:],
            "symbol": symbol,
        }

    def _build_adx_gauge_data(self, symbol: str) -> dict:
        """Build ADX gauge data."""
        snap = self.bot.strategy.compute_indicators(symbol, "4h")
        return {
            "adx": snap.adx_value if snap else 0,
            "plus_di": snap.plus_di if snap else 0,
            "minus_di": snap.minus_di if snap else 0,
            "symbol": symbol,
            "threshold_weak": 20,
            "threshold_strong": 25,
            "threshold_very_strong": 40,
        }

    def _build_multi_tf_data(self, symbol: str) -> dict:
        """Build multi-timeframe alignment panel data."""
        result = {"symbol": symbol, "timeframes": {}}

        for tf in ("15m", "4h", "1d"):
            snap = self.bot.strategy.compute_indicators(symbol, tf)
            if snap:
                result["timeframes"][tf] = {
                    "ema_20": snap.ema_20,
                    "ema_50": snap.ema_50,
                    "ema_200": snap.ema_200,
                    "rsi": snap.rsi_value,
                    "macd_histogram": snap.macd_histogram,
                    "adx": snap.adx_value,
                    "atr": snap.atr_value,
                    "volume_ratio": snap.volume_ratio,
                }
                # Determine bias for this TF
                if snap.ema_20 > snap.ema_50 > snap.ema_200:
                    result["timeframes"][tf]["bias"] = "BULLISH"
                elif snap.ema_20 < snap.ema_50 < snap.ema_200:
                    result["timeframes"][tf]["bias"] = "BEARISH"
                else:
                    result["timeframes"][tf]["bias"] = "MIXED"
            else:
                result["timeframes"][tf] = {"bias": "NO_DATA"}

        return result

    # ======================================================================
    # Section 10.2 strategy-specific metrics
    # ======================================================================

    def _build_strategy_metrics(self) -> dict:
        """Build Section 10.2 strategy-specific performance metrics."""
        trades = self.bot.exit_mgr.get_closed_trades()
        if not trades:
            return {
                "trend_capture_efficiency": 0,
                "avg_r_multiple": 0,
                "scaling_effectiveness": {},
                "tf_agreement_rate": 0,
                "adx_stratified": {},
                "volume_stratified": {},
                "holding_period_distribution": {},
                "slippage_analysis": {},
            }

        # Trend Capture Efficiency
        r_multiples = [t.get("r_multiple", 0) for t in trades]
        avg_r = sum(r_multiples) / len(r_multiples) if r_multiples else 0

        # Scaling Effectiveness
        scaled_trades = [t for t in trades if t.get("scale_in_count", 0) > 0]
        unscaled_trades = [t for t in trades if t.get("scale_in_count", 0) == 0]
        scaling_eff = {
            "with_scale_ins": {
                "count": len(scaled_trades),
                "avg_pnl": sum(t.get("realized_pnl", 0) for t in scaled_trades) / max(1, len(scaled_trades)),
            },
            "without_scale_ins": {
                "count": len(unscaled_trades),
                "avg_pnl": sum(t.get("realized_pnl", 0) for t in unscaled_trades) / max(1, len(unscaled_trades)),
            },
        }

        # ADX-Stratified performance
        adx_buckets = {"25-30": [], "30-40": [], "40+": []}
        for t in trades:
            # We'd need indicator_values_at_entry to stratify properly
            adx_val = t.get("adx_at_entry", 30)
            if 25 <= adx_val < 30:
                adx_buckets["25-30"].append(t)
            elif 30 <= adx_val < 40:
                adx_buckets["30-40"].append(t)
            elif adx_val >= 40:
                adx_buckets["40+"].append(t)

        adx_stratified = {}
        for bucket, bucket_trades in adx_buckets.items():
            if bucket_trades:
                wins = [t for t in bucket_trades if t.get("realized_pnl", 0) > 0]
                adx_stratified[bucket] = {
                    "count": len(bucket_trades),
                    "win_rate": len(wins) / len(bucket_trades) * 100,
                    "avg_pnl": sum(t.get("realized_pnl", 0) for t in bucket_trades) / len(bucket_trades),
                }
            else:
                adx_stratified[bucket] = {"count": 0, "win_rate": 0, "avg_pnl": 0}

        # Holding Period Distribution (buckets: <1d, 1-3d, 3-7d, 7-10d, >10d)
        holding_dist = {"<1d": 0, "1-3d": 0, "3-7d": 0, "7-10d": 0, ">10d": 0}
        for t in trades:
            duration_h = t.get("duration_ms", 0) / (3600 * 1000)
            if duration_h < 24:
                holding_dist["<1d"] += 1
            elif duration_h < 72:
                holding_dist["1-3d"] += 1
            elif duration_h < 168:
                holding_dist["3-7d"] += 1
            elif duration_h < 240:
                holding_dist["7-10d"] += 1
            else:
                holding_dist[">10d"] += 1

        # Slippage Analysis (placeholder — requires entry/exit slippage logs)
        slippage_analysis = {
            "avg_entry_slippage_bps": 0,
            "avg_exit_slippage_bps": 0,
            "total_slippage_cost_pct": 0,
        }

        return {
            "trend_capture_efficiency": avg_r,
            "avg_r_multiple": avg_r,
            "scaling_effectiveness": scaling_eff,
            "tf_agreement_rate": 100.0,  # All entries require TF agreement
            "adx_stratified": adx_stratified,
            "volume_stratified": {},  # Would require volume_ratio_at_entry
            "holding_period_distribution": holding_dist,
            "slippage_analysis": slippage_analysis,
        }

    # ======================================================================
    # Paper trading criteria (Section 9.4)
    # ======================================================================

    def _build_paper_criteria(self) -> dict:
        """Build paper trading go-live criteria status."""
        cfg = self.bot.config.strategy_params
        metrics = self._get_metrics()
        trades = self.bot.exit_mgr.get_closed_trades()

        total_trades = len(trades)
        win_rate = metrics.get("win_rate", 0)
        profit_factor = metrics.get("profit_factor", 0)
        max_dd = metrics.get("max_drawdown_pct", 0)
        sharpe = metrics.get("sharpe_ratio", 0)
        net_pnl = sum(t.get("realized_pnl", 0) for t in trades)

        # Check single trade loss
        worst_trade_pct = 0
        equity = self.bot.risk_mgr.shared.get_current_equity()
        if equity > 0:
            for t in trades:
                pnl_pct = abs(t.get("realized_pnl", 0)) / equity * 100
                if t.get("realized_pnl", 0) < 0 and pnl_pct > worst_trade_pct:
                    worst_trade_pct = pnl_pct

        criteria = {
            "min_days": {
                "required": cfg.get("paper_min_days", 60),
                "current": 0,  # Would need startup timestamp
                "met": False,
            },
            "min_trades": {
                "required": cfg.get("paper_min_trades", 50),
                "current": total_trades,
                "met": total_trades >= cfg.get("paper_min_trades", 50),
            },
            "positive_pnl": {
                "required": "> 0",
                "current": net_pnl,
                "met": net_pnl > 0,
            },
            "win_rate": {
                "required": cfg.get("paper_min_win_rate", 35.0),
                "current": win_rate,
                "met": win_rate >= cfg.get("paper_min_win_rate", 35.0),
            },
            "profit_factor": {
                "required": cfg.get("paper_min_profit_factor", 1.3),
                "current": profit_factor,
                "met": profit_factor >= cfg.get("paper_min_profit_factor", 1.3),
            },
            "max_drawdown": {
                "required": f"< {cfg.get('paper_max_drawdown', 12.0)}%",
                "current": max_dd,
                "met": max_dd < cfg.get("paper_max_drawdown", 12.0),
            },
            "sharpe_ratio": {
                "required": cfg.get("paper_min_sharpe", 0.8),
                "current": sharpe,
                "met": sharpe >= cfg.get("paper_min_sharpe", 0.8),
            },
            "max_single_loss": {
                "required": f"< {cfg.get('paper_max_single_loss_pct', 2.5)}%",
                "current": worst_trade_pct,
                "met": worst_trade_pct < cfg.get("paper_max_single_loss_pct", 2.5),
            },
        }

        all_met = all(c["met"] for c in criteria.values())
        return {"criteria": criteria, "all_met": all_met, "ready_for_live": all_met}

    # ======================================================================
    # Lifecycle
    # ======================================================================

    async def start(self) -> None:
        """Start the dashboard server."""
        await self._base.start()
        logger.info("STRAT-001 dashboard started on port %d", self._base._port)

    async def stop(self) -> None:
        """Stop the dashboard server."""
        await self._base.stop()
