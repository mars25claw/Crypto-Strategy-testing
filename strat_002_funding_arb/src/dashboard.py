"""Funding Rate Arbitrage dashboard.

Extends the shared DashboardBase with strategy-specific data providers
for basis charts, predicted funding, accumulated yield, liquidation
distance gauge, and delta exposure per Section 12.4.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from shared.dashboard_base import DashboardBase

from strat_002_funding_arb.src.strategy_metrics import FundingArbMetrics

logger = logging.getLogger(__name__)


class FundingArbDashboard:
    """Dashboard for the Funding Rate Arbitrage strategy.

    Wraps DashboardBase and wires up all strategy-specific data providers.

    Parameters
    ----------
    strategy : FundingArbStrategy
        The strategy instance.
    risk_manager : FundingArbRiskManager
        The risk manager.
    wallet_manager : WalletManager
        The wallet manager.
    funding_tracker : FundingTracker
        The funding settlement tracker.
    performance_tracker : PerformanceTracker
        The shared performance tracker.
    config : dict
        Dashboard configuration.
    """

    def __init__(
        self,
        strategy: Any,
        risk_manager: Any,
        wallet_manager: Any,
        funding_tracker: Any,
        performance_tracker: Any,
        config: Optional[Dict[str, Any]] = None,
        kill_fn: Optional[Callable] = None,
        strategy_metrics: Optional[Any] = None,
    ) -> None:
        self._strategy = strategy
        self._risk = risk_manager
        self._wallet = wallet_manager
        self._funding = funding_tracker
        self._perf = performance_tracker
        self._config = config or {}
        self._kill_fn = kill_fn
        self._strategy_metrics: Optional[FundingArbMetrics] = strategy_metrics

        # Set up template directory to our strategy's templates folder
        template_dir = str(Path(__file__).parent.parent / "templates")

        self._dashboard = DashboardBase(
            strategy_id="STRAT-002",
            strategy_name="Funding Rate Arbitrage",
            host=self._config.get("host", "0.0.0.0"),
            port=self._config.get("port", 8082),
            template_dir=template_dir,
        )

        # Wire up data providers
        self._dashboard.set_data_providers(
            positions_fn=self._get_positions,
            trades_fn=self._get_trades,
            metrics_fn=self._get_metrics,
            equity_fn=self._get_equity_curve,
            alerts_fn=self._get_alerts,
            config_fn=self._get_config,
            kill_fn=self._kill_fn,
            config_update_fn=self._update_config,
        )

    async def start(self) -> None:
        """Start the dashboard server."""
        await self._dashboard.start()

    def stop(self) -> None:
        """Stop the dashboard server."""
        self._dashboard.stop()

    # ══════════════════════════════════════════════════════════════════════
    #  Data providers
    # ══════════════════════════════════════════════════════════════════════

    def _get_positions(self) -> List[Dict[str, Any]]:
        """Return all active arbitrage positions for display."""
        positions = []
        for pos_id, pos in self._strategy.positions.items():
            inst = self._strategy.instruments.get(pos.symbol)
            positions.append({
                "position_id": pos.position_id,
                "symbol": pos.symbol,
                "spot_quantity": pos.spot_quantity,
                "spot_entry_price": pos.spot_entry_price,
                "spot_notional": pos.spot_notional,
                "futures_quantity": pos.futures_quantity,
                "futures_entry_price": pos.futures_entry_price,
                "futures_notional": pos.futures_notional,
                "entry_basis_pct": round(pos.entry_basis_pct, 4),
                "current_basis_pct": round(inst.current_basis_pct(), 4) if inst else 0,
                "cumulative_funding": round(pos.cumulative_funding_income, 6),
                "funding_periods": pos.funding_periods_collected,
                "holding_days": round(pos.holding_days, 2),
                "annualized_yield": round(pos.annualized_yield, 2),
                "delta_pct": round(pos.current_delta_pct, 4),
                "predicted_funding_rate": round(
                    inst.predicted_funding_rate * 100, 4
                ) if inst else 0,
                "negative_streak": pos.negative_funding_streak,
                "low_rate_streak": pos.low_rate_streak,
            })
        return positions

    def _get_trades(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return recent trades from performance tracker."""
        if self._perf:
            return self._perf.get_recent_trades(limit)
        return []

    def _get_metrics(self) -> Dict[str, Any]:
        """Aggregate all metrics for the dashboard."""
        # Standard metrics from performance tracker
        standard = self._perf.get_metrics() if self._perf else {}

        # Strategy-specific metrics (Section 10.2)
        strategy = self._strategy.get_strategy_metrics()

        # Risk metrics
        risk = self._risk.get_risk_metrics() if self._risk else {}

        # Wallet metrics
        wallet = self._wallet.get_wallet_metrics() if self._wallet else {}

        # Funding tracker metrics
        funding = self._funding.get_funding_metrics() if self._funding else {}

        # Instrument-level data for charts
        instruments = {}
        for symbol, inst in self._strategy.instruments.items():
            instruments[symbol] = {
                "predicted_funding_rate": round(inst.predicted_funding_rate * 100, 6),
                "current_basis_pct": round(inst.current_basis_pct(), 4),
                "mark_price": inst.mark_price,
                "index_price": inst.index_price,
                "annualized_yield": round(inst.annualized_yield(), 2),
                "avg_24h_rate": round(inst.avg_funding_rate(3) * 100, 6),
                "avg_7d_rate": round(inst.avg_funding_rate(21) * 100, 6),
                "avg_1h_basis": round(inst.avg_basis_pct(60), 4),
            }

        # Basis history for chart (last 24h)
        basis_chart: Dict[str, List[Dict]] = {}
        for symbol, inst in self._strategy.instruments.items():
            points = []
            for b in list(inst.basis_history)[-100:]:  # Last 100 points
                points.append({
                    "timestamp_ms": b.timestamp_ms,
                    "basis_pct": round(b.basis_pct, 4),
                })
            basis_chart[symbol] = points

        # Section 10.2 + 10.3 strategy-specific metrics
        section_10_2 = {}
        go_live_status = {}
        if self._strategy_metrics:
            section_10_2 = self._strategy_metrics.get_all_metrics()
            go_live_status = self._strategy_metrics.get_go_live_status()

        return {
            "standard": standard,
            "strategy": strategy,
            "risk": risk,
            "wallet": wallet,
            "funding": funding,
            "instruments": instruments,
            "basis_chart": basis_chart,
            # Section 10.2 specific
            "cost_to_income_ratio": self._calc_cost_to_income_ratio(),
            "avg_holding_duration_days": self._calc_avg_holding_duration(),
            # Section 10.2 comprehensive metrics
            "section_10_2_metrics": section_10_2,
            # Section 10.3 dimensional breakdowns
            "dimensional_breakdowns": section_10_2.get("dimensional_breakdowns", {}),
            # Section 10.4 go-live criteria
            "go_live_criteria": go_live_status,
        }

    def _get_equity_curve(self) -> List[Dict[str, Any]]:
        """Return equity curve data."""
        if self._perf:
            return self._perf.get_metrics().get("equity_curve", [])
        return []

    def _get_alerts(self) -> List[Dict[str, Any]]:
        """Return recent alerts."""
        alerts = []

        # Check for active circuit breakers
        if self._strategy._circuit_breaker_active:
            alerts.append({
                "level": "critical",
                "message": f"Circuit breaker active: {self._strategy._circuit_breaker_reason}",
                "timestamp": int(time.time() * 1000),
            })

        # Check for kill switch
        if self._risk and self._risk.kill_switch_active:
            alerts.append({
                "level": "emergency",
                "message": "Kill switch is active",
                "timestamp": int(time.time() * 1000),
            })

        return alerts

    def _get_config(self) -> Dict[str, Any]:
        """Return current configuration (sanitized)."""
        return {
            "entry_mode": self._strategy._config.get("entry_mode", "standard"),
            "entry_threshold_pct": self._strategy._entry_threshold * 100,
            "max_instruments": 5,
            "max_capital_pct": 40.0,
            "paper_trading": self._config.get("paper_trading", True),
        }

    def _update_config(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle config updates from dashboard."""
        applied = []

        if "entry_mode" in params:
            self._strategy.update_entry_mode(params["entry_mode"])
            applied.append("entry_mode")

        return {"applied": applied}

    # ══════════════════════════════════════════════════════════════════════
    #  Metric calculations (Section 10.2)
    # ══════════════════════════════════════════════════════════════════════

    def _calc_cost_to_income_ratio(self) -> float:
        """Cost-to-Income ratio: total fees / total funding income.

        Target: < 30%.
        """
        metrics = self._perf.get_metrics() if self._perf else {}
        total_fees = metrics.get("total_fees", 0)
        funding_income = sum(
            p.cumulative_funding_income for p in self._strategy.positions.values()
        )
        if funding_income <= 0:
            return 0.0
        return round((total_fees / funding_income) * 100, 2)

    def _calc_avg_holding_duration(self) -> float:
        """Average holding duration in days."""
        if not self._strategy.positions:
            return 0.0
        durations = [p.holding_days for p in self._strategy.positions.values()]
        return round(sum(durations) / len(durations), 2)


# Need time import for alerts
import time
