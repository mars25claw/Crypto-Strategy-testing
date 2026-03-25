"""STRAT-004 Mean Reversion Dashboard.

Extends the shared DashboardBase with mean-reversion-specific views:
- Regime status per instrument (Hurst / ADX / state)
- Bollinger Band chart data
- Z-score gauge
- Signal agreement display (which signals triggered, confirmation status)
- Anti-trend blacklist display
- Tranche progress visualization
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from shared.dashboard_base import DashboardBase
from src.strategy_metrics import MeanReversionMetrics

logger = logging.getLogger(__name__)


class MeanReversionDashboard(DashboardBase):
    """Real-time dashboard for STRAT-004 Mean Reversion.

    Adds strategy-specific API endpoints and WebSocket data fields
    on top of the shared dashboard infrastructure.

    Parameters
    ----------
    strategy :
        MeanReversionStrategy instance.
    regime_classifier :
        RegimeClassifier instance.
    risk_manager :
        MeanReversionRiskManager instance.
    host : str
        Bind address.
    port : int
        Bind port.
    """

    def __init__(
        self,
        strategy: Any = None,
        regime_classifier: Any = None,
        risk_manager: Any = None,
        host: str = "0.0.0.0",
        port: int = 8084,
        strategy_metrics: Optional[MeanReversionMetrics] = None,
    ) -> None:
        # Use the local templates directory
        template_dir = str(Path(__file__).parent.parent / "templates")

        super().__init__(
            strategy_id="STRAT-004",
            strategy_name="Mean Reversion",
            host=host,
            port=port,
            template_dir=template_dir,
        )

        self._strategy = strategy
        self._regime = regime_classifier
        self._risk_mgr = risk_manager
        self._strategy_metrics = strategy_metrics

        # Wire up data providers
        self.set_data_providers(
            positions_fn=self._get_positions,
            trades_fn=self._get_trades,
            metrics_fn=self._get_metrics,
            equity_fn=self._get_equity_curve,
            alerts_fn=self._get_alerts,
            config_fn=self._get_config,
        )

        # Add strategy-specific routes
        self._add_strategy_routes()

    # ------------------------------------------------------------------
    # Data providers
    # ------------------------------------------------------------------

    def _get_positions(self) -> List[dict]:
        if self._strategy is None:
            return []
        return self._strategy.get_positions()

    def _get_trades(self, limit: int = 50) -> List[dict]:
        # Trade history is maintained in risk manager
        if self._risk_mgr is None:
            return []
        status = self._risk_mgr.get_risk_status()
        return [{
            "trade_count": status.get("trade_count", 0),
            "wins": status.get("win_count", 0),
            "losses": status.get("loss_count", 0),
        }]

    def _get_metrics(self) -> dict:
        metrics = {}

        if self._strategy:
            metrics.update(self._strategy.get_metrics())

        if self._risk_mgr:
            risk_status = self._risk_mgr.get_risk_status()
            metrics["risk"] = risk_status
            metrics["consecutive_losses"] = risk_status.get("consecutive_losses", 0)
            metrics["is_halted"] = risk_status.get("is_halted", False)
            metrics["blacklist"] = risk_status.get("blacklist", {})

        if self._regime:
            metrics["regimes"] = self._regime.get_all_regimes()
            tradeable = self._regime.get_tradeable_instruments()
            metrics["tradeable_instruments"] = tradeable
            metrics["tradeable_count"] = len(tradeable)

        # Section 10.2 + 10.3 strategy-specific metrics
        if self._strategy_metrics:
            metrics["section_10_2_metrics"] = self._strategy_metrics.get_all_metrics()
            metrics["dimensional_breakdowns"] = metrics["section_10_2_metrics"].get("dimensional_breakdowns", {})
            metrics["go_live_criteria"] = self._strategy_metrics.get_go_live_status()

        return metrics

    def _get_equity_curve(self) -> List[dict]:
        # From paper trading engine if available
        if self._strategy and self._strategy._paper:
            curve = self._strategy._paper.get_equity_curve()
            return [{"timestamp": ts, "equity": eq} for ts, eq in curve]
        return []

    def _get_alerts(self) -> List[dict]:
        alerts = []

        # Pending signals
        if self._strategy:
            for sig in self._strategy.get_pending_signals():
                alerts.append({
                    "type": "signal",
                    "level": "info",
                    "message": (
                        f"Signal {sig['direction']} {sig['symbol']}: "
                        f"{sig['signal_count']}/3 ({sig['strength_pct']}%)"
                    ),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })

        # Risk alerts
        if self._risk_mgr:
            status = self._risk_mgr.get_risk_status()
            if status.get("is_halted"):
                alerts.append({
                    "type": "risk",
                    "level": "critical",
                    "message": "Strategy HALTED — circuit breaker or consecutive losses",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
            if status.get("drawdown_halted"):
                alerts.append({
                    "type": "risk",
                    "level": "critical",
                    "message": f"Drawdown halt: {status['drawdown_level']} {status['drawdown_pct']}%",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
            for sym, expiry in status.get("blacklist", {}).items():
                alerts.append({
                    "type": "blacklist",
                    "level": "warning",
                    "message": f"{sym} blacklisted until {expiry}",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })

        return alerts

    def _get_config(self) -> dict:
        if self._strategy:
            return self._strategy._config
        return {}

    # ------------------------------------------------------------------
    # Strategy-specific API routes
    # ------------------------------------------------------------------

    def _add_strategy_routes(self) -> None:
        """Add STRAT-004-specific API endpoints to the FastAPI app."""
        from fastapi import Query
        from fastapi.responses import JSONResponse

        app = self.app

        @app.get("/api/regimes")
        async def api_regimes():
            """Return current regime classification for all instruments."""
            if self._regime is None:
                return JSONResponse({})
            return JSONResponse(self._regime.get_all_regimes())

        @app.get("/api/regime_history")
        async def api_regime_history():
            """Return regime classification history."""
            if self._regime is None:
                return JSONResponse([])
            return JSONResponse(self._regime.get_classification_history())

        @app.get("/api/signals")
        async def api_signals():
            """Return pending and recent signals."""
            if self._strategy is None:
                return JSONResponse({"pending": [], "metrics": {}})
            return JSONResponse({
                "pending": self._strategy.get_pending_signals(),
                "metrics": {
                    "generated": self._strategy._signals_generated,
                    "confirmed": self._strategy._signals_confirmed,
                    "expired": self._strategy._signals_expired,
                },
            })

        @app.get("/api/indicators/{symbol}")
        async def api_indicators(symbol: str):
            """Return current indicator values for a symbol."""
            if self._strategy is None:
                return JSONResponse({})
            return JSONResponse(self._strategy.get_indicator_data(symbol))

        @app.get("/api/risk_status")
        async def api_risk_status():
            """Return detailed risk status."""
            if self._risk_mgr is None:
                return JSONResponse({})
            return JSONResponse(self._risk_mgr.get_risk_status())

        @app.get("/api/blacklist")
        async def api_blacklist():
            """Return current anti-trend blacklist."""
            if self._risk_mgr is None:
                return JSONResponse({})
            return JSONResponse(self._risk_mgr.get_blacklist())

    # ------------------------------------------------------------------
    # Enhanced WebSocket payload
    # ------------------------------------------------------------------

    async def _build_ws_payload(self) -> Dict[str, Any]:
        """Override to add mean-reversion-specific data to WS push."""
        base = await super()._build_ws_payload()

        # Add regime data
        if self._regime:
            base["regimes"] = self._regime.get_all_regimes()
            base["tradeable_count"] = len(self._regime.get_tradeable_instruments())

        # Add signal data
        if self._strategy:
            base["pending_signals"] = self._strategy.get_pending_signals()
            base["signal_metrics"] = {
                "generated": self._strategy._signals_generated,
                "confirmed": self._strategy._signals_confirmed,
                "expired": self._strategy._signals_expired,
            }

            # Add indicator snapshots for active instruments
            indicators = {}
            if self._regime:
                for sym in self._regime.get_tradeable_instruments()[:4]:
                    ind = self._strategy.get_indicator_data(sym)
                    if ind:
                        indicators[sym] = ind
            base["indicators"] = indicators

        # Add risk data
        if self._risk_mgr:
            base["risk_status"] = self._risk_mgr.get_risk_status()

        return base
