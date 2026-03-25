"""DCA-specific dashboard extending the shared base dashboard.

Adds DCA-specific pages and API endpoints:
- Portfolio pie chart data
- Cost basis per instrument
- Signal values and multiplier
- Next DCA countdown
- Monthly budget tracker
- Crash-buy cooldowns
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from shared.dashboard_base import DashboardBase, _safe_json

logger = logging.getLogger(__name__)


class DCADashboard(DashboardBase):
    """DCA-specific dashboard extending the shared base.

    Adds DCA-specific data providers and API endpoints for:
    - Portfolio breakdown / pie chart
    - Signal values
    - DCA schedule countdown
    - Budget tracking
    - Crash-buy cooldowns
    - Purchase history
    """

    def __init__(
        self,
        strategy_id: str = "STRAT-009",
        strategy_name: str = "Signal-Enhanced DCA",
        host: str = "0.0.0.0",
        port: int = 8080,
        template_dir: Optional[str] = None,
    ) -> None:
        if template_dir is None:
            template_dir = str(Path(__file__).parent.parent / "templates")

        super().__init__(
            strategy_id=strategy_id,
            strategy_name=strategy_name,
            host=host,
            port=port,
            template_dir=template_dir,
        )

        # DCA-specific data providers
        self._budget_fn: Optional[Callable] = None
        self._countdown_fn: Optional[Callable] = None
        self._signals_fn: Optional[Callable] = None
        self._purchases_fn: Optional[Callable] = None
        self._crash_cooldowns_fn: Optional[Callable] = None
        self._risk_status_fn: Optional[Callable] = None
        self._strategy_metrics_fn: Optional[Callable] = None
        self._go_live_fn: Optional[Callable] = None
        self._vanilla_comparison_fn: Optional[Callable] = None

    def set_dca_providers(
        self,
        budget_fn: Optional[Callable] = None,
        countdown_fn: Optional[Callable] = None,
        signals_fn: Optional[Callable] = None,
        purchases_fn: Optional[Callable] = None,
        crash_cooldowns_fn: Optional[Callable] = None,
        risk_status_fn: Optional[Callable] = None,
    ) -> None:
        """Register DCA-specific data provider callbacks.

        Parameters
        ----------
        budget_fn : callable
            ``() -> dict``  Monthly budget summary.
        countdown_fn : callable
            ``() -> dict``  Next DCA countdown info.
        signals_fn : callable
            ``() -> dict``  Current signal values per instrument.
        purchases_fn : callable
            ``(limit: int) -> list[dict]``  Recent purchase history.
        crash_cooldowns_fn : callable
            ``() -> dict``  Crash-buy cooldown status.
        risk_status_fn : callable
            ``() -> dict``  Risk manager status.
        """
        if budget_fn is not None:
            self._budget_fn = budget_fn
        if countdown_fn is not None:
            self._countdown_fn = countdown_fn
        if signals_fn is not None:
            self._signals_fn = signals_fn
        if purchases_fn is not None:
            self._purchases_fn = purchases_fn
        if crash_cooldowns_fn is not None:
            self._crash_cooldowns_fn = crash_cooldowns_fn
        if risk_status_fn is not None:
            self._risk_status_fn = risk_status_fn

    def set_metrics_providers(
        self,
        strategy_metrics_fn: Optional[Callable] = None,
        go_live_fn: Optional[Callable] = None,
        vanilla_comparison_fn: Optional[Callable] = None,
    ) -> None:
        """Register metrics-specific data provider callbacks."""
        if strategy_metrics_fn is not None:
            self._strategy_metrics_fn = strategy_metrics_fn
        if go_live_fn is not None:
            self._go_live_fn = go_live_fn
        if vanilla_comparison_fn is not None:
            self._vanilla_comparison_fn = vanilla_comparison_fn

    def _create_app(self):
        """Create the FastAPI app with DCA-specific endpoints."""
        app = super()._create_app()
        templates = Jinja2Templates(directory=self._template_dir)

        # ---- DCA-specific REST API ----

        @app.get("/api/budget")
        async def api_budget():
            data = await self._call_provider(self._budget_fn)
            return JSONResponse(_safe_json(data) if data else {})

        @app.get("/api/countdown")
        async def api_countdown():
            data = await self._call_provider(self._countdown_fn)
            return JSONResponse(_safe_json(data) if data else {})

        @app.get("/api/signals")
        async def api_signals():
            data = await self._call_provider(self._signals_fn)
            return JSONResponse(_safe_json(data) if data else {})

        @app.get("/api/purchases")
        async def api_purchases():
            data = await self._call_provider(self._purchases_fn, 50)
            return JSONResponse(_safe_json(data) if data else [])

        @app.get("/api/crash-cooldowns")
        async def api_crash_cooldowns():
            data = await self._call_provider(self._crash_cooldowns_fn)
            return JSONResponse(_safe_json(data) if data else {})

        @app.get("/api/risk-status")
        async def api_risk_status():
            data = await self._call_provider(self._risk_status_fn)
            return JSONResponse(_safe_json(data) if data else {})

        @app.get("/api/strategy-metrics")
        async def api_strategy_metrics():
            """Full Section 10.2 and 10.3 metrics."""
            data = await self._call_provider(self._strategy_metrics_fn)
            return JSONResponse(_safe_json(data) if data else {})

        @app.get("/api/go-live")
        async def api_go_live():
            """Go-live criteria evaluation."""
            data = await self._call_provider(self._go_live_fn)
            return JSONResponse(_safe_json(data) if data else {})

        @app.get("/api/vanilla-comparison")
        async def api_vanilla_comparison():
            """Signal-enhanced vs vanilla DCA comparison."""
            data = await self._call_provider(self._vanilla_comparison_fn)
            return JSONResponse(_safe_json(data) if data else {})

        @app.get("/api/portfolio")
        async def api_portfolio():
            """Portfolio breakdown for pie chart."""
            metrics = await self._call_provider(self._metrics_fn) or {}
            allocation = metrics.get("asset_allocation", {})
            holdings = metrics.get("holdings", {})
            return JSONResponse(_safe_json({
                "allocation": allocation,
                "holdings": holdings,
                "total_value": metrics.get("portfolio_value", 0),
                "total_invested": metrics.get("total_invested", 0),
                "total_return_pct": metrics.get("total_return_pct", 0),
            }))

        return app

    async def _build_ws_payload(self) -> Dict[str, Any]:
        """Override WS payload to include DCA-specific data."""
        base = await super()._build_ws_payload()

        # Add DCA data
        budget = await self._call_provider(self._budget_fn) or {}
        countdown = await self._call_provider(self._countdown_fn) or {}
        risk_status = await self._call_provider(self._risk_status_fn) or {}

        base["budget"] = budget
        base["countdown"] = countdown
        base["risk_status"] = risk_status

        return base
