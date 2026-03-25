"""STRAT-005 Grid Trading Dashboard.

Extends the shared DashboardBase with grid-specific visualization data:
- Grid level display (filled/pending/empty)
- Cycle counter and profit ticker
- Range boundary display with breakout warnings
- Inventory heat map data
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional

from shared.dashboard_base import DashboardBase

from .strategy import GridSide, InstrumentState
from .strategy_metrics import GridMetrics

logger = logging.getLogger(__name__)


class GridDashboard:
    """Grid-specific dashboard wrapper around DashboardBase.

    Parameters
    ----------
    host : str
        Bind address.
    port : int
        Bind port.
    template_dir : str
        Path to the templates directory.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8085,
        template_dir: str = "templates",
        strategy_metrics: Optional[GridMetrics] = None,
    ) -> None:
        self._base = DashboardBase(
            strategy_id="STRAT-005",
            strategy_name="Grid Trading",
            host=host,
            port=port,
            template_dir=template_dir,
        )

        # Grid-specific data providers
        self._grid_data_fn: Optional[Callable] = None
        self._risk_fn: Optional[Callable] = None
        self._strategy_metrics = strategy_metrics

    def set_providers(
        self,
        positions_fn: Optional[Callable] = None,
        trades_fn: Optional[Callable] = None,
        metrics_fn: Optional[Callable] = None,
        equity_fn: Optional[Callable] = None,
        alerts_fn: Optional[Callable] = None,
        config_fn: Optional[Callable] = None,
        kill_fn: Optional[Callable] = None,
        grid_data_fn: Optional[Callable] = None,
        risk_fn: Optional[Callable] = None,
    ) -> None:
        """Register all data provider callbacks."""
        self._base.set_data_providers(
            positions_fn=positions_fn,
            trades_fn=trades_fn,
            metrics_fn=metrics_fn,
            equity_fn=equity_fn,
            alerts_fn=alerts_fn,
            config_fn=config_fn,
            kill_fn=kill_fn,
        )
        self._grid_data_fn = grid_data_fn
        self._risk_fn = risk_fn

        # Add grid-specific API endpoints
        self._register_grid_endpoints()

    def _register_grid_endpoints(self) -> None:
        """Add grid-specific REST endpoints to the FastAPI app."""
        from fastapi import Request
        from fastapi.responses import JSONResponse

        app = self._base.app

        @app.get("/api/grid")
        async def api_grid():
            """Return grid visualization data for all instruments."""
            if self._grid_data_fn:
                data = self._grid_data_fn()
                return JSONResponse(data)
            return JSONResponse({})

        @app.get("/api/risk")
        async def api_risk():
            """Return risk summary."""
            if self._risk_fn:
                data = self._risk_fn()
                return JSONResponse(data)
            return JSONResponse({})

        @app.get("/api/strategy-metrics")
        async def api_strategy_metrics():
            """Return Section 10.2 strategy-specific metrics."""
            if self._strategy_metrics:
                return JSONResponse(self._strategy_metrics.get_all_metrics())
            return JSONResponse({})

        @app.get("/api/go-live")
        async def api_go_live():
            """Return go-live criteria status."""
            if self._strategy_metrics:
                return JSONResponse(self._strategy_metrics.get_go_live_status())
            return JSONResponse({})

        @app.get("/api/dimensional-breakdowns")
        async def api_dimensional_breakdowns():
            """Return Section 10.3 dimensional breakdowns."""
            if self._strategy_metrics:
                metrics = self._strategy_metrics.get_all_metrics()
                return JSONResponse(metrics.get("dimensional_breakdowns", {}))
            return JSONResponse({})

    async def start(self) -> None:
        """Start the dashboard server."""
        await self._base.start()

    def stop(self) -> None:
        """Stop the dashboard server."""
        self._base.stop()

    # ======================================================================
    #  Grid visualization data builders
    # ======================================================================

    @staticmethod
    def build_grid_data(instruments: Dict[str, InstrumentState]) -> dict:
        """Build grid visualization data for all instruments.

        Returns a dict that the dashboard template can render:
        - Per-instrument: levels with status (filled/active/empty), boundaries, metrics
        """
        result = {}

        for symbol, state in instruments.items():
            if not state.grid_params:
                continue

            levels_data = []
            for level in state.grid_params.levels:
                status = "empty"
                if level.filled:
                    status = "filled"
                elif level.active:
                    status = "active"

                levels_data.append({
                    "index": level.index,
                    "price": level.price,
                    "side": level.side.value if level.side else None,
                    "status": status,
                    "order_id": level.order_id,
                    "quantity": level.quantity,
                    "fill_price": level.fill_price,
                    "fill_time_ms": level.fill_time_ms,
                })

            result[symbol] = {
                "active": state.active,
                "halted": state.halted,
                "halt_reason": state.halt_reason,
                "upper_boundary": state.grid_params.upper_boundary,
                "lower_boundary": state.grid_params.lower_boundary,
                "center_price": state.grid_params.center_price,
                "current_price": state.current_price,
                "mark_price": state.mark_price,
                "num_levels": state.grid_params.num_levels,
                "grid_type": state.grid_params.grid_type.value,
                "grid_spacing_pct": round(state.grid_params.grid_spacing_pct, 4),
                "range_width_pct": round(state.grid_params.range_width_pct, 2),
                "levels": levels_data,
                "inventory": {
                    "quantity": state.inventory_qty,
                    "avg_cost": round(state.inventory_avg_cost, 4),
                    "levels_filled": state.inventory_levels_filled,
                    "unrealized_pnl": round(state.unrealized_pnl, 4),
                },
                "cycles": {
                    "total": state.total_cycles,
                    "total_profit": round(state.total_cycle_profit, 6),
                    "total_fees": round(state.total_fees, 6),
                    "realized_profit": round(state.realized_profit, 6),
                },
                "breakout": {
                    "direction": state.breakout_direction.value,
                    "consecutive_above": state.consecutive_closes_above,
                    "consecutive_below": state.consecutive_closes_below,
                },
                "circuit_breakers": {
                    "consecutive_buy_fills": state.consecutive_buy_fills,
                    "consecutive_sell_fills": state.consecutive_sell_fills,
                },
            }

        return result

    @staticmethod
    def build_positions_data(instruments: Dict[str, InstrumentState]) -> List[dict]:
        """Build position data for the standard dashboard positions panel."""
        positions = []
        for symbol, state in instruments.items():
            if state.inventory_qty > 0:
                positions.append({
                    "symbol": symbol,
                    "side": "LONG",
                    "quantity": state.inventory_qty,
                    "entry_price": round(state.inventory_avg_cost, 4),
                    "mark_price": round(state.mark_price, 4),
                    "unrealized_pnl": round(state.unrealized_pnl, 4),
                    "levels_filled": state.inventory_levels_filled,
                    "strategy_id": "STRAT-005",
                })
        return positions
