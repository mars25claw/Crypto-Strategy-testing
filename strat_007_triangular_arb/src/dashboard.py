"""STRAT-007: Real-time arbitrage dashboard.

Extends the shared DashboardBase with strategy-specific displays:
- Live opportunity scanner stream
- Execution log with success/failure indicators
- Win rate and net profit gauges
- Latency histogram
- Wallet balance tracker
- Mode A vs Mode B attribution
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from shared.dashboard_base import DashboardBase

logger = logging.getLogger(__name__)


class ArbDashboard(DashboardBase):
    """Dashboard for the Triangular Arbitrage strategy.

    Extends the base dashboard with arb-specific data providers.

    Parameters
    ----------
    host : str
        Bind address.
    port : int
        Bind port.
    template_dir : str, optional
        Path to templates directory. Defaults to the strategy's templates/.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8087,
        template_dir: Optional[str] = None,
    ) -> None:
        if template_dir is None:
            template_dir = str(Path(__file__).resolve().parent.parent / "templates")

        super().__init__(
            strategy_id="STRAT-007",
            strategy_name="Cross-Exchange Triangular Arbitrage",
            host=host,
            port=port,
            template_dir=template_dir,
        )

        # Strategy-specific data providers
        self._scanner_stats_fn: Optional[Callable] = None
        self._execution_stats_fn: Optional[Callable] = None
        self._risk_state_fn: Optional[Callable] = None
        self._wallet_stats_fn: Optional[Callable] = None
        self._opportunity_stream_fn: Optional[Callable] = None
        self._execution_history_fn: Optional[Callable] = None
        self._strategy_stats_fn: Optional[Callable] = None
        self._strategy_metrics_fn: Optional[Callable] = None
        self._go_live_fn: Optional[Callable] = None
        self._latency_dist_fn: Optional[Callable] = None
        self._partial_triangle_fn: Optional[Callable] = None

    def set_arb_providers(
        self,
        scanner_stats_fn: Optional[Callable] = None,
        execution_stats_fn: Optional[Callable] = None,
        risk_state_fn: Optional[Callable] = None,
        wallet_stats_fn: Optional[Callable] = None,
        opportunity_stream_fn: Optional[Callable] = None,
        execution_history_fn: Optional[Callable] = None,
        strategy_stats_fn: Optional[Callable] = None,
        strategy_metrics_fn: Optional[Callable] = None,
        go_live_fn: Optional[Callable] = None,
        latency_dist_fn: Optional[Callable] = None,
        partial_triangle_fn: Optional[Callable] = None,
    ) -> None:
        """Register arb-specific data provider callbacks."""
        self._scanner_stats_fn = scanner_stats_fn
        self._execution_stats_fn = execution_stats_fn
        self._risk_state_fn = risk_state_fn
        self._wallet_stats_fn = wallet_stats_fn
        self._opportunity_stream_fn = opportunity_stream_fn
        self._execution_history_fn = execution_history_fn
        self._strategy_stats_fn = strategy_stats_fn
        self._strategy_metrics_fn = strategy_metrics_fn
        self._go_live_fn = go_live_fn
        self._latency_dist_fn = latency_dist_fn
        self._partial_triangle_fn = partial_triangle_fn

    async def _build_ws_payload(self) -> Dict[str, Any]:
        """Override to add arb-specific data to the WebSocket payload."""
        base = await super()._build_ws_payload()

        # Inject arb-specific data
        scanner_stats = await self._call_provider(self._scanner_stats_fn) or {}
        execution_stats = await self._call_provider(self._execution_stats_fn) or {}
        risk_state = await self._call_provider(self._risk_state_fn) or {}
        wallet_stats = await self._call_provider(self._wallet_stats_fn) or {}
        opportunities = await self._call_provider(self._opportunity_stream_fn) or []
        executions = await self._call_provider(self._execution_history_fn) or []
        strategy_stats = await self._call_provider(self._strategy_stats_fn) or {}

        # Section 10.2 / 10.3 metrics and go-live criteria
        strategy_metrics = await self._call_provider(self._strategy_metrics_fn) or {}
        go_live = await self._call_provider(self._go_live_fn) or {}
        latency_dist = await self._call_provider(self._latency_dist_fn) or {}
        partial_triangles = await self._call_provider(self._partial_triangle_fn) or []

        base["arb"] = {
            "scanner": scanner_stats,
            "execution": execution_stats,
            "risk": risk_state,
            "wallet": wallet_stats,
            "recent_opportunities": opportunities[:20] if isinstance(opportunities, list) else [],
            "recent_executions": executions[:20] if isinstance(executions, list) else [],
            "strategy": strategy_stats,
            # Section 10.2 strategy-specific metrics
            "strategy_metrics": strategy_metrics,
            # Section 9.3 go-live criteria
            "go_live_criteria": go_live,
            # Paper trading latency distribution
            "latency_distribution": latency_dist,
            # Partial triangle events (Section 11.2)
            "partial_triangle_events": (
                partial_triangles[-20:] if isinstance(partial_triangles, list) else []
            ),
        }

        return base
