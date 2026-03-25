"""STRAT-003 Dashboard: Z-score gauges, spread charts, cointegration table, pair PnL.

Extends DashboardBase with pairs-specific endpoints and a custom template
that displays:
- Z-score gauges for all active pairs
- Historical spread charts
- Cointegration test results table
- Pair-level PnL attribution
- Exposure breakdown (long/short/net by asset)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from shared.dashboard_base import DashboardBase
from shared.performance_tracker import PerformanceTracker

from src.cointegration import CointegrationEngine
from src.risk_manager import PairsRiskManager
from src.strategy import PairsStrategy
from src.strategy_metrics import PairsMetrics

logger = logging.getLogger(__name__)


class PairsDashboard(DashboardBase):
    """Pairs-specific dashboard extending the shared base.

    Adds API endpoints for cointegration data, spread history, and Z-score
    gauges that the pairs-specific template consumes.

    Parameters
    ----------
    strategy : PairsStrategy
        The running strategy engine.
    coint_engine : CointegrationEngine
        The cointegration testing engine.
    pairs_risk : PairsRiskManager
        The pairs-specific risk manager.
    perf_tracker : PerformanceTracker
        Standard performance metrics tracker.
    """

    def __init__(
        self,
        strategy_id: str,
        strategy_name: str,
        host: str = "0.0.0.0",
        port: int = 8083,
        strategy: Optional[PairsStrategy] = None,
        coint_engine: Optional[CointegrationEngine] = None,
        pairs_risk: Optional[PairsRiskManager] = None,
        perf_tracker: Optional[PerformanceTracker] = None,
        strategy_metrics: Optional[PairsMetrics] = None,
    ) -> None:
        # Use our custom template directory
        template_dir = str(Path(__file__).parent.parent / "templates")

        super().__init__(
            strategy_id=strategy_id,
            strategy_name=strategy_name,
            host=host,
            port=port,
            template_dir=template_dir,
        )

        self._strategy = strategy
        self._coint = coint_engine
        self._pairs_risk = pairs_risk
        self._perf = perf_tracker
        self._strategy_metrics = strategy_metrics

        # Register data providers
        self.set_data_providers(
            positions_fn=self._get_positions,
            trades_fn=self._get_trades,
            metrics_fn=self._get_metrics,
            equity_fn=self._get_equity_curve,
            alerts_fn=self._get_alerts,
            config_fn=self._get_config,
        )

        # Register additional endpoints
        self._register_pairs_endpoints()

    def _register_pairs_endpoints(self) -> None:
        """Add pairs-specific API routes."""
        from fastapi import Request
        from fastapi.responses import JSONResponse

        @self.app.get("/api/cointegration")
        async def api_cointegration():
            """Return cointegration test results for all pairs."""
            if self._coint is None:
                return JSONResponse([])

            pairs = []
            for p in self._coint.qualified_pairs:
                pairs.append({
                    "asset_a": p.asset_a,
                    "asset_b": p.asset_b,
                    "hedge_ratio": round(p.hedge_ratio, 4),
                    "adf_p_value": round(p.adf_p_value, 6),
                    "half_life_days": round(p.half_life_days, 1),
                    "hurst_exponent": round(p.hurst_exponent, 4),
                    "correlation_stability": round(p.correlation_stability, 1),
                    "recent_correlation": round(p.recent_correlation, 4),
                    "is_marginal": p.is_marginal,
                    "is_preferred": p.is_preferred,
                    "rank_score": round(p.rank_score, 2),
                    "consecutive_stops": p.consecutive_stops,
                    "post_stop_entry_z": p.post_stop_entry_z,
                })
            return JSONResponse(pairs)

        @self.app.get("/api/spreads")
        async def api_spreads():
            """Return current spread / Z-score data for all active pairs."""
            if self._strategy is None:
                return JSONResponse([])
            return JSONResponse(self._strategy.get_spread_states())

        @self.app.get("/api/z-scores")
        async def api_z_scores():
            """Return Z-score gauges for dashboard."""
            if self._strategy is None:
                return JSONResponse([])

            gauges = []
            for state_data in self._strategy.get_spread_states():
                z = state_data.get("current_z", 0)
                pair = state_data.get("pair", "")
                # Determine zone
                if abs(z) < 0.5:
                    zone = "neutral"
                elif abs(z) < 2.0:
                    zone = "watch"
                elif abs(z) < 3.5:
                    zone = "active"
                else:
                    zone = "danger"

                gauges.append({
                    "pair": pair,
                    "z_score": round(z, 3),
                    "zone": zone,
                    "asset_a": state_data.get("asset_a", ""),
                    "asset_b": state_data.get("asset_b", ""),
                })
            return JSONResponse(gauges)

        @self.app.get("/api/exposure")
        async def api_exposure():
            """Return exposure breakdown."""
            if self._pairs_risk is None:
                return JSONResponse({})
            return JSONResponse(self._pairs_risk.get_exposure_summary())

        @self.app.get("/api/strategy-metrics")
        async def api_strategy_metrics():
            """Return Section 10.2 strategy-specific metrics."""
            if self._strategy is None:
                return JSONResponse({})
            return JSONResponse(self._strategy.get_strategy_specific_metrics())

        @self.app.get("/api/pair-pnl")
        async def api_pair_pnl():
            """Return pair-level PnL attribution."""
            if self._strategy is None:
                return JSONResponse({})
            metrics = self._strategy.get_strategy_specific_metrics()
            return JSONResponse(metrics.get("pair_level_attribution", {}))

    # ======================================================================
    #  Data provider callbacks
    # ======================================================================

    def _get_positions(self) -> List[Dict[str, Any]]:
        if self._strategy is None:
            return []
        return self._strategy.get_positions()

    def _get_trades(self, limit: int = 50) -> List[Dict[str, Any]]:
        if self._strategy is None:
            return []
        return self._strategy.get_recent_trades(limit)

    def _get_metrics(self) -> Dict[str, Any]:
        result = {}
        if self._perf is not None:
            result = self._perf.get_metrics()
        if self._strategy is not None:
            result["strategy_specific"] = self._strategy.get_strategy_specific_metrics()
        if self._pairs_risk is not None:
            result["exposure"] = self._pairs_risk.get_exposure_summary()
        # Section 10.2 + 10.3 strategy-specific metrics
        if self._strategy_metrics is not None:
            result["section_10_2_metrics"] = self._strategy_metrics.get_all_metrics()
            result["dimensional_breakdowns"] = result["section_10_2_metrics"].get("dimensional_breakdowns", {})
            result["go_live_criteria"] = self._strategy_metrics.get_go_live_status()
        return result

    def _get_equity_curve(self) -> List[Dict[str, Any]]:
        if self._perf is None:
            return []
        metrics = self._perf.get_metrics()
        return metrics.get("equity_curve", [])

    def _get_alerts(self) -> List[Dict[str, Any]]:
        if self._strategy is None:
            return []
        return self._strategy.get_alerts()

    def _get_config(self) -> Dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "strategy_name": self.strategy_name,
        }
