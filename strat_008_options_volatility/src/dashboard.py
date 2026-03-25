"""STRAT-008 Dashboard — IV/RV charts, Greeks display, PnL attribution, cycle tracking.

Extends the shared DashboardBase with options-specific visualization data providers.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional

from shared.dashboard_base import DashboardBase

logger = logging.getLogger(__name__)


class OptionsDashboard:
    """Dashboard for STRAT-008 with options-specific data providers.

    Wraps DashboardBase and registers callbacks that pull data from the
    strategy coordinator, volatility engine, and risk manager.

    Parameters
    ----------
    strategy_coordinator : object
        The StrategyCoordinator instance.
    vol_engine : object
        The VolatilityEngine instance.
    risk_mgr : object
        The StrategyRiskManager instance.
    host : str
        Dashboard bind address.
    port : int
        Dashboard bind port.
    template_dir : str
        Path to HTML templates directory.
    """

    def __init__(
        self,
        strategy_coordinator: Any,
        vol_engine: Any,
        risk_mgr: Any,
        host: str = "0.0.0.0",
        port: int = 8088,
        template_dir: str = "templates",
        kill_fn: Optional[Callable] = None,
    ) -> None:
        self._coordinator = strategy_coordinator
        self._vol_engine = vol_engine
        self._risk_mgr = risk_mgr

        self._base = DashboardBase(
            strategy_id="STRAT-008",
            strategy_name="Options & Volatility Strategies",
            host=host,
            port=port,
            template_dir=template_dir,
        )

        self._base.set_data_providers(
            positions_fn=self._get_positions,
            trades_fn=self._get_trades,
            metrics_fn=self._get_metrics,
            equity_fn=self._get_equity_curve,
            alerts_fn=self._get_alerts,
            config_fn=self._get_config,
            kill_fn=kill_fn,
        )

    async def start(self) -> None:
        """Start the dashboard server."""
        await self._base.start()

    def stop(self) -> None:
        """Stop the dashboard server."""
        self._base.stop()

    # ------------------------------------------------------------------
    # Data providers
    # ------------------------------------------------------------------

    def _get_positions(self) -> List[dict]:
        """Return all active positions with options-specific detail."""
        positions = self._coordinator.get_positions()

        # Enrich with current IV/RV data
        for pos in positions:
            symbol = pos.get("symbol", "")
            regime = self._vol_engine.get_regime(symbol)
            if regime:
                pos["current_iv"] = regime.iv
                pos["current_rv_7d"] = regime.rv_7d
                pos["iv_rv_ratio"] = regime.iv_rv_ratio
                pos["vol_regime"] = regime.regime

        return positions

    def _get_trades(self, limit: int = 50) -> List[dict]:
        """Return recent completed cycles as trades."""
        trades: List[dict] = []

        # Covered call completed cycles
        for cycle in self._coordinator.cc_manager.get_completed_cycles()[-limit:]:
            trades.append({
                "type": "covered_call",
                "id": cycle.cycle_id,
                "symbol": cycle.symbol,
                "entry_time": cycle.opened_at,
                "exit_time": cycle.expiration_at,
                "status": cycle.status,
                "premium": cycle.premium,
                "pnl": cycle.realized_pnl,
                "iv_at_entry": cycle.iv_at_entry,
                "strike": cycle.strike_price,
            })

        # CSP completed cycles
        for cycle in self._coordinator.csp_manager.get_completed_cycles()[-limit:]:
            trades.append({
                "type": "cash_secured_put",
                "id": cycle.cycle_id,
                "symbol": cycle.symbol,
                "entry_time": cycle.opened_at,
                "exit_time": cycle.expiration_at,
                "status": cycle.status,
                "premium": cycle.premium,
                "pnl": cycle.realized_pnl,
                "iv_at_entry": cycle.iv_at_entry,
                "strike": cycle.strike_price,
            })

        # Delta-neutral completed
        if self._coordinator.dn_manager:
            for pos in self._coordinator.dn_manager.get_completed_positions()[-limit:]:
                trades.append({
                    "type": "delta_neutral",
                    "id": pos.position_id,
                    "symbol": pos.symbol,
                    "entry_time": pos.opened_at,
                    "exit_time": pos.expiration_at,
                    "status": pos.status,
                    "premium": pos.total_premium,
                    "pnl": pos.net_pnl,
                    "iv_at_entry": pos.iv_at_entry,
                    "hedge_costs": pos.hedge_fees,
                    "rebalances": pos.rebalance_count,
                })

        # Sort by entry time descending
        trades.sort(key=lambda t: t.get("entry_time", 0), reverse=True)
        return trades[:limit]

    def _get_metrics(self) -> dict:
        """Return comprehensive metrics including options-specific ones."""
        base_metrics = self._coordinator.get_metrics()

        # Add volatility data
        vol_data = {}
        for symbol in ["BTCUSDT", "ETHUSDT"]:
            regime = self._vol_engine.get_regime(symbol)
            if regime:
                vol_data[symbol] = regime.to_dict()

        # Greeks display
        greeks = {"net_delta": 0.0, "net_gamma": 0.0, "net_theta": 0.0, "net_vega": 0.0}
        risk_state = self._risk_mgr.state
        greeks["net_delta"] = risk_state.net_delta
        greeks["net_gamma"] = risk_state.net_gamma
        greeks["net_theta"] = risk_state.net_theta
        greeks["net_vega"] = risk_state.net_vega

        # PnL attribution
        cc_pnl = base_metrics.get("covered_calls", {}).get("total_pnl", 0)
        csp_pnl = base_metrics.get("cash_secured_puts", {}).get("total_pnl", 0)
        dn_pnl = base_metrics.get("delta_neutral", {}).get("total_net_pnl", 0)

        base_metrics.update({
            "volatility": vol_data,
            "greeks": greeks,
            "pnl_attribution": {
                "covered_calls": round(cc_pnl, 4),
                "cash_secured_puts": round(csp_pnl, 4),
                "delta_neutral": round(dn_pnl, 4),
                "total": round(cc_pnl + csp_pnl + dn_pnl, 4),
            },
            "delta_compliance_pct": round(risk_state.delta_compliance_pct, 1),
            "greek_limit_breaches": risk_state.greek_limit_breaches,
        })

        return base_metrics

    def _get_equity_curve(self) -> List[dict]:
        """Return equity curve data points."""
        # This would normally come from the paper trading engine
        return [
            {"timestamp": time.time() * 1000, "equity": self._risk_mgr.equity}
        ]

    def _get_alerts(self) -> List[dict]:
        """Return risk alerts."""
        return self._risk_mgr.get_alerts()

    def _get_config(self) -> dict:
        """Return current configuration."""
        return {
            "mode": self._coordinator.mode,
            "paper_trading": True,
            "strategy_id": "STRAT-008",
        }

    # ------------------------------------------------------------------
    # Additional data for the options dashboard template
    # ------------------------------------------------------------------

    def get_iv_rv_chart_data(self) -> Dict[str, list]:
        """Return IV/RV history for charting."""
        chart_data: Dict[str, list] = {}

        for symbol in ["BTCUSDT", "ETHUSDT"]:
            regime = self._vol_engine.get_regime(symbol)
            if regime:
                chart_data[symbol] = {
                    "iv": regime.iv,
                    "rv_1d": regime.rv_1d,
                    "rv_7d": regime.rv_7d,
                    "rv_30d": regime.rv_30d,
                    "ratio": regime.iv_rv_ratio,
                    "iv_history": regime.iv_history,
                }

        return chart_data

    def get_cycle_tracking(self) -> dict:
        """Return cycle tracking data."""
        active_cc = self._coordinator.cc_manager.get_active_cycles()
        active_csp = self._coordinator.csp_manager.get_active_cycles()
        active_dn = (
            self._coordinator.dn_manager.get_active_positions()
            if self._coordinator.dn_manager
            else {}
        )

        return {
            "covered_calls": {k: v.to_dict() for k, v in active_cc.items()},
            "cash_secured_puts": {k: v.to_dict() for k, v in active_csp.items()},
            "delta_neutral": {k: v.to_dict() for k, v in active_dn.items()},
            "total_active": len(active_cc) + len(active_csp) + len(active_dn),
        }
