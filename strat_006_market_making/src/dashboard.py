"""Dashboard for STRAT-006 Market Making.

Extends DashboardBase with market-making-specific views:
- Real-time quote positions on the order book
- Inventory gauge
- Spread capture metrics
- Adverse selection indicator
- Fill rate tracking
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from shared.dashboard_base import DashboardBase

logger = logging.getLogger(__name__)


class MarketMakingDashboard(DashboardBase):
    """Dashboard for the STRAT-006 Market Making bot.

    Extends the shared DashboardBase with strategy-specific data providers.

    Parameters
    ----------
    host : str
        Bind address.
    port : int
        Bind port.
    template_dir : str, optional
        Override the templates directory.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8086,
        template_dir: Optional[str] = None,
    ) -> None:
        super().__init__(
            strategy_id="STRAT-006",
            strategy_name="Market Making (Avellaneda-Stoikov)",
            host=host,
            port=port,
            template_dir=template_dir,
        )

        # Additional data providers specific to market making
        self._quotes_fn: Optional[Callable] = None
        self._model_state_fn: Optional[Callable] = None
        self._adverse_fn: Optional[Callable] = None
        self._risk_fn: Optional[Callable] = None
        self._fills_fn: Optional[Callable] = None
        self._round_trips_fn: Optional[Callable] = None
        self._strategy_metrics_fn: Optional[Callable] = None
        self._event_filter_fn: Optional[Callable] = None
        self._go_live_fn: Optional[Callable] = None

        # Register additional API routes
        self._register_mm_routes()

    def set_mm_providers(
        self,
        quotes_fn: Optional[Callable] = None,
        model_state_fn: Optional[Callable] = None,
        adverse_fn: Optional[Callable] = None,
        risk_fn: Optional[Callable] = None,
        fills_fn: Optional[Callable] = None,
        round_trips_fn: Optional[Callable] = None,
        strategy_metrics_fn: Optional[Callable] = None,
        event_filter_fn: Optional[Callable] = None,
        go_live_fn: Optional[Callable] = None,
    ) -> None:
        """Register market-making-specific data providers.

        Parameters
        ----------
        quotes_fn : callable
            () -> dict  Active quotes with prices and sizes.
        model_state_fn : callable
            () -> dict  Current model parameters per instrument.
        adverse_fn : callable
            () -> dict  Adverse selection metrics.
        risk_fn : callable
            () -> dict  Risk manager state (inventory, drawdown).
        fills_fn : callable
            (limit: int) -> list  Recent fills.
        round_trips_fn : callable
            (limit: int) -> list  Recent round-trips.
        strategy_metrics_fn : callable
            () -> dict  Strategy-specific metrics (Section 10.2).
        event_filter_fn : callable
            () -> dict  Event filter metrics.
        go_live_fn : callable
            () -> dict  Go-live criteria status.
        """
        if quotes_fn is not None:
            self._quotes_fn = quotes_fn
        if model_state_fn is not None:
            self._model_state_fn = model_state_fn
        if adverse_fn is not None:
            self._adverse_fn = adverse_fn
        if risk_fn is not None:
            self._risk_fn = risk_fn
        if fills_fn is not None:
            self._fills_fn = fills_fn
        if round_trips_fn is not None:
            self._round_trips_fn = round_trips_fn
        if strategy_metrics_fn is not None:
            self._strategy_metrics_fn = strategy_metrics_fn
        if event_filter_fn is not None:
            self._event_filter_fn = event_filter_fn
        if go_live_fn is not None:
            self._go_live_fn = go_live_fn

    def _register_mm_routes(self) -> None:
        """Register market-making-specific API routes."""
        from fastapi.responses import JSONResponse

        @self.app.get("/api/quotes")
        async def api_quotes():
            data = await self._call_provider(self._quotes_fn)
            return JSONResponse(data or {})

        @self.app.get("/api/model")
        async def api_model():
            data = await self._call_provider(self._model_state_fn)
            return JSONResponse(data or {})

        @self.app.get("/api/adverse")
        async def api_adverse():
            data = await self._call_provider(self._adverse_fn)
            return JSONResponse(data or {})

        @self.app.get("/api/risk")
        async def api_risk():
            data = await self._call_provider(self._risk_fn)
            return JSONResponse(data or {})

        @self.app.get("/api/fills")
        async def api_fills():
            data = await self._call_provider(self._fills_fn, 100)
            return JSONResponse(data or [])

        @self.app.get("/api/roundtrips")
        async def api_roundtrips():
            data = await self._call_provider(self._round_trips_fn, 50)
            return JSONResponse(data or [])

        @self.app.get("/api/strategy_metrics")
        async def api_strategy_metrics():
            data = await self._call_provider(self._strategy_metrics_fn)
            return JSONResponse(data or {})

        @self.app.get("/api/events")
        async def api_events():
            data = await self._call_provider(self._event_filter_fn)
            return JSONResponse(data or {})

        @self.app.get("/api/go_live")
        async def api_go_live():
            data = await self._call_provider(self._go_live_fn)
            return JSONResponse(data or {})

    async def _build_ws_payload(self) -> Dict[str, Any]:
        """Override to include MM-specific data in WebSocket push."""
        base = await super()._build_ws_payload()

        # Add MM-specific data
        quotes = await self._call_provider(self._quotes_fn) or {}
        model = await self._call_provider(self._model_state_fn) or {}
        adverse = await self._call_provider(self._adverse_fn) or {}
        risk = await self._call_provider(self._risk_fn) or {}

        base.update({
            "quotes": quotes,
            "model": model,
            "adverse_selection": adverse,
            "risk": risk,
        })

        return base
