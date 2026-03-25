"""STRAT-008 Strategy-Specific Metrics (Section 10.2) and Go-Live Criteria (Section 10.3).

Provides all strategy-specific performance metrics:
- Annualized Premium Yield
- IV/RV Spread at Entry
- Theta Decay Capture Rate
- Delta Hedge Cost (% premium)
- Net Yield After Hedging
- Greek Limit Breaches count
- Exercise Rate (calls/puts separately)
- Volatility Forecasting Accuracy (IV vs actual RV)
- Cycle-Level Attribution (per cycle PnL breakdown)

Also evaluates go-live readiness criteria:
- 90-day paper trading minimum
- Sharpe > 1.0
- Delta within limits > 90%
- Premium > hedge costs 2x
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Section 10.2: Strategy-Specific Metrics
# ---------------------------------------------------------------------------

@dataclass
class StrategySpecificMetrics:
    """Container for all Section 10.2 strategy-specific metrics."""
    # Premium Yield
    annualized_premium_yield_pct: float = 0.0
    total_premium_collected: float = 0.0
    avg_capital_deployed: float = 0.0

    # IV/RV Spread
    avg_iv_rv_spread_at_entry: float = 0.0
    iv_rv_spread_entries: List[float] = field(default_factory=list)

    # Theta Decay Capture
    theta_decay_capture_rate_pct: float = 0.0
    total_theta_expected: float = 0.0
    total_theta_captured: float = 0.0

    # Delta Hedge Cost
    delta_hedge_cost_pct: float = 0.0
    total_hedge_cost: float = 0.0

    # Net Yield
    net_yield_after_hedging: float = 0.0

    # Greek Limit Breaches
    greek_limit_breaches: int = 0

    # Exercise Rates
    call_exercise_rate_pct: float = 0.0
    put_exercise_rate_pct: float = 0.0
    call_expected_exercise_rate_pct: float = 0.0
    put_expected_exercise_rate_pct: float = 0.0

    # Volatility Forecasting Accuracy
    vol_forecast_accuracy_pct: float = 0.0
    iv_vs_rv_comparisons: List[Tuple[float, float]] = field(default_factory=list)

    # Cycle-Level Attribution
    cycle_attributions: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "annualized_premium_yield_pct": round(self.annualized_premium_yield_pct, 2),
            "total_premium_collected": round(self.total_premium_collected, 4),
            "avg_capital_deployed": round(self.avg_capital_deployed, 2),
            "avg_iv_rv_spread_at_entry": round(self.avg_iv_rv_spread_at_entry, 2),
            "theta_decay_capture_rate_pct": round(self.theta_decay_capture_rate_pct, 1),
            "delta_hedge_cost_pct": round(self.delta_hedge_cost_pct, 1),
            "net_yield_after_hedging": round(self.net_yield_after_hedging, 4),
            "greek_limit_breaches": self.greek_limit_breaches,
            "call_exercise_rate_pct": round(self.call_exercise_rate_pct, 1),
            "put_exercise_rate_pct": round(self.put_exercise_rate_pct, 1),
            "call_expected_exercise_rate_pct": round(self.call_expected_exercise_rate_pct, 1),
            "put_expected_exercise_rate_pct": round(self.put_expected_exercise_rate_pct, 1),
            "vol_forecast_accuracy_pct": round(self.vol_forecast_accuracy_pct, 1),
            "num_cycles_attributed": len(self.cycle_attributions),
        }


class StrategyMetricsCalculator:
    """Computes all strategy-specific metrics from sub-strategy managers.

    Parameters
    ----------
    cc_manager : CoveredCallManager
    csp_manager : CashSecuredPutManager
    dn_manager : DeltaNeutralManager or None
    risk_mgr : StrategyRiskManager
    vol_engine : VolatilityEngine
    """

    def __init__(
        self,
        cc_manager: Any,
        csp_manager: Any,
        dn_manager: Any,
        risk_mgr: Any,
        vol_engine: Any,
    ) -> None:
        self._cc = cc_manager
        self._csp = csp_manager
        self._dn = dn_manager
        self._risk = risk_mgr
        self._vol = vol_engine

        # Track daily equity for Sharpe calculation
        self._daily_returns: List[float] = []
        self._last_equity: float = 0.0
        self._start_timestamp: float = time.time()

        # IV vs RV tracking for forecasting accuracy
        self._iv_rv_records: List[Tuple[float, float, float]] = []  # (timestamp, entry_iv, actual_rv)

    # ------------------------------------------------------------------
    # Main computation
    # ------------------------------------------------------------------

    def compute_all(self, equity: float, avg_capital: float = 0.0) -> StrategySpecificMetrics:
        """Compute all Section 10.2 strategy-specific metrics.

        Parameters
        ----------
        equity : float
            Current equity.
        avg_capital : float
            Average capital deployed over the period.

        Returns
        -------
        StrategySpecificMetrics
        """
        m = StrategySpecificMetrics()

        # Gather completed cycles
        cc_completed = self._cc.get_completed_cycles()
        csp_completed = self._csp.get_completed_cycles()
        dn_completed = self._dn.get_completed_positions() if self._dn else []

        # --- Annualized Premium Yield ---
        total_premium = (
            sum(c.premium for c in cc_completed)
            + sum(c.premium for c in csp_completed)
            + sum(p.total_premium for p in dn_completed)
        )
        m.total_premium_collected = total_premium
        m.avg_capital_deployed = avg_capital if avg_capital > 0 else equity

        # Calculate time span
        all_opens = (
            [c.opened_at for c in cc_completed]
            + [c.opened_at for c in csp_completed]
            + [p.opened_at for p in dn_completed]
        )
        if all_opens and m.avg_capital_deployed > 0:
            span_days = max(1, (time.time() - min(all_opens)) / 86400)
            annualization = 365.0 / span_days
            m.annualized_premium_yield_pct = (
                total_premium / m.avg_capital_deployed * annualization * 100.0
            )

        # --- IV/RV Spread at Entry ---
        spreads = []
        for c in cc_completed:
            if c.iv_at_entry > 0 and c.rv_at_entry > 0:
                spreads.append(c.iv_at_entry - c.rv_at_entry)
        for c in csp_completed:
            if c.iv_at_entry > 0 and c.rv_at_entry > 0:
                spreads.append(c.iv_at_entry - c.rv_at_entry)
        for p in dn_completed:
            if p.iv_at_entry > 0 and p.rv_at_entry > 0:
                spreads.append(p.iv_at_entry - p.rv_at_entry)
        m.iv_rv_spread_entries = spreads
        m.avg_iv_rv_spread_at_entry = float(np.mean(spreads)) if spreads else 0.0

        # --- Theta Decay Capture Rate ---
        # Ratio of actual profit to initial theta expectation
        total_theta_expected = 0.0
        total_actual_profit = 0.0
        for p in dn_completed:
            if p.initial_theta != 0:
                # Expected theta income over the cycle
                expected = abs(p.initial_theta) * p.cycle_days
                total_theta_expected += expected
                total_actual_profit += max(0, p.net_pnl)
        m.total_theta_expected = total_theta_expected
        m.total_theta_captured = total_actual_profit
        if total_theta_expected > 0:
            m.theta_decay_capture_rate_pct = (
                total_actual_profit / total_theta_expected * 100.0
            )

        # --- Delta Hedge Cost (% premium) ---
        total_hedge_cost = (
            sum(c.hedge_costs for c in cc_completed)
            + sum(c.hedge_costs for c in csp_completed)
            + sum(p.hedge_fees for p in dn_completed)
        )
        m.total_hedge_cost = total_hedge_cost
        if total_premium > 0:
            m.delta_hedge_cost_pct = total_hedge_cost / total_premium * 100.0

        # --- Net Yield After Hedging ---
        total_fees = (
            sum(c.fees_paid for c in cc_completed)
            + sum(c.fees_paid for c in csp_completed)
            + sum(p.hedge_fees for p in dn_completed)
        )
        m.net_yield_after_hedging = total_premium - total_hedge_cost - total_fees

        # --- Greek Limit Breaches ---
        m.greek_limit_breaches = self._risk.state.greek_limit_breaches

        # --- Exercise Rate (calls/puts separately) ---
        if cc_completed:
            cc_exercised = sum(1 for c in cc_completed if c.status == "exercised")
            m.call_exercise_rate_pct = cc_exercised / len(cc_completed) * 100.0
            cc_deltas = [c.delta_at_entry for c in cc_completed if c.delta_at_entry > 0]
            m.call_expected_exercise_rate_pct = (
                float(np.mean(cc_deltas)) * 100.0 if cc_deltas else 0.0
            )

        if csp_completed:
            csp_exercised = sum(1 for c in csp_completed if c.status == "exercised")
            m.put_exercise_rate_pct = csp_exercised / len(csp_completed) * 100.0
            csp_deltas = [c.delta_at_entry for c in csp_completed if c.delta_at_entry > 0]
            m.put_expected_exercise_rate_pct = (
                float(np.mean(csp_deltas)) * 100.0 if csp_deltas else 0.0
            )

        # --- Volatility Forecasting Accuracy ---
        # Compare IV at entry vs actual RV over the cycle
        iv_rv_pairs = []
        for c in cc_completed:
            if c.iv_at_entry > 0 and c.rv_at_entry > 0:
                # Use rv_at_entry as proxy for actual realized over the cycle
                iv_rv_pairs.append((c.iv_at_entry, c.rv_at_entry))
        for c in csp_completed:
            if c.iv_at_entry > 0 and c.rv_at_entry > 0:
                iv_rv_pairs.append((c.iv_at_entry, c.rv_at_entry))
        for p in dn_completed:
            if p.iv_at_entry > 0 and p.rv_at_entry > 0:
                iv_rv_pairs.append((p.iv_at_entry, p.rv_at_entry))

        m.iv_vs_rv_comparisons = iv_rv_pairs
        if iv_rv_pairs:
            # Accuracy = 1 - mean absolute percentage error of IV vs RV
            errors = [
                abs(iv - rv) / rv * 100.0
                for iv, rv in iv_rv_pairs if rv > 0
            ]
            if errors:
                mape = float(np.mean(errors))
                m.vol_forecast_accuracy_pct = max(0.0, 100.0 - mape)

        # --- Cycle-Level Attribution ---
        for c in cc_completed:
            m.cycle_attributions.append({
                "cycle_id": c.cycle_id,
                "type": "covered_call",
                "premium": round(c.premium, 4),
                "hedge_cost": round(c.hedge_costs, 4),
                "underlying_pnl": round(c.underlying_pnl, 4),
                "net_pnl": round(c.realized_pnl, 4),
                "status": c.status,
            })
        for c in csp_completed:
            m.cycle_attributions.append({
                "cycle_id": c.cycle_id,
                "type": "cash_secured_put",
                "premium": round(c.premium, 4),
                "hedge_cost": round(c.hedge_costs, 4),
                "underlying_pnl": round(c.underlying_pnl, 4),
                "net_pnl": round(c.realized_pnl, 4),
                "status": c.status,
            })
        for p in dn_completed:
            attr = {
                "cycle_id": p.position_id,
                "type": "delta_neutral",
                "premium": round(p.total_premium, 4),
                "hedge_cost": round(p.hedge_fees, 4),
                "delta_pnl": round(p.delta_pnl, 4),
                "gamma_pnl": round(p.gamma_pnl, 4),
                "theta_pnl": round(p.theta_pnl, 4),
                "vega_pnl": round(p.vega_pnl, 4),
                "net_pnl": round(p.net_pnl, 4),
                "status": p.status,
            }
            m.cycle_attributions.append(attr)

        return m

    # ------------------------------------------------------------------
    # Equity tracking for Sharpe calculation
    # ------------------------------------------------------------------

    def record_daily_equity(self, equity: float) -> None:
        """Record daily equity for Sharpe ratio calculation."""
        if self._last_equity > 0:
            daily_return = (equity - self._last_equity) / self._last_equity
            self._daily_returns.append(daily_return)
        self._last_equity = equity

    def record_iv_vs_rv(self, entry_iv: float, actual_rv: float) -> None:
        """Record an IV vs actual RV observation for forecasting accuracy."""
        self._iv_rv_records.append((time.time(), entry_iv, actual_rv))

    # ------------------------------------------------------------------
    # Dimensional Breakdowns
    # ------------------------------------------------------------------

    def get_dimensional_breakdown(self) -> dict:
        """Return metrics broken down by dimensions.

        Dimensions:
        - By sub-strategy (CC, CSP, DN)
        - By symbol (BTCUSDT, ETHUSDT)
        - By IV regime (favorable, strong, neutral)
        """
        breakdown: Dict[str, Any] = {
            "by_sub_strategy": {},
            "by_symbol": {},
            "by_regime": {},
        }

        # By sub-strategy
        breakdown["by_sub_strategy"]["covered_calls"] = self._cc.get_metrics()
        breakdown["by_sub_strategy"]["cash_secured_puts"] = self._csp.get_metrics()
        if self._dn:
            breakdown["by_sub_strategy"]["delta_neutral"] = self._dn.get_metrics()

        # By symbol
        cc_by_sym: Dict[str, List] = {}
        for c in self._cc.get_completed_cycles():
            cc_by_sym.setdefault(c.symbol, []).append(c)
        for sym, cycles in cc_by_sym.items():
            total_pnl = sum(c.realized_pnl for c in cycles)
            total_premium = sum(c.premium for c in cycles)
            breakdown["by_symbol"].setdefault(sym, {})["covered_calls"] = {
                "cycles": len(cycles),
                "pnl": round(total_pnl, 4),
                "premium": round(total_premium, 4),
            }

        csp_by_sym: Dict[str, List] = {}
        for c in self._csp.get_completed_cycles():
            csp_by_sym.setdefault(c.symbol, []).append(c)
        for sym, cycles in csp_by_sym.items():
            total_pnl = sum(c.realized_pnl for c in cycles)
            total_premium = sum(c.premium for c in cycles)
            breakdown["by_symbol"].setdefault(sym, {})["cash_secured_puts"] = {
                "cycles": len(cycles),
                "pnl": round(total_pnl, 4),
                "premium": round(total_premium, 4),
            }

        return breakdown

    # ------------------------------------------------------------------
    # Section 10.3: Go-Live Criteria
    # ------------------------------------------------------------------

    def evaluate_go_live_criteria(self, equity: float) -> dict:
        """Evaluate whether the strategy is ready for live trading.

        Go-Live Criteria (90-day minimum, Section 9.3):
        1. Net positive PnL after all simulated fees and hedge costs
        2. Profit factor > 1.3
        3. Maximum drawdown < 5%
        4. IV/RV regime detection correctly avoided < 1.0 ratio periods
        5. Delta stayed within limits > 90% of the time
        6. Covered call premium income exceeded hedge costs by > 2x

        Additional:
        - Sharpe ratio > 1.0
        - 90-day paper trading minimum
        """
        now = time.time()
        days_running = (now - self._start_timestamp) / 86400.0

        # Gather all PnL
        cc_completed = self._cc.get_completed_cycles()
        csp_completed = self._csp.get_completed_cycles()
        dn_completed = self._dn.get_completed_positions() if self._dn else []

        total_pnl = (
            sum(c.realized_pnl for c in cc_completed)
            + sum(c.realized_pnl for c in csp_completed)
            + sum(p.net_pnl for p in dn_completed)
        )

        # Profit factor
        gross_profit = (
            sum(c.realized_pnl for c in cc_completed if c.realized_pnl > 0)
            + sum(c.realized_pnl for c in csp_completed if c.realized_pnl > 0)
            + sum(p.net_pnl for p in dn_completed if p.net_pnl > 0)
        )
        gross_loss = abs(
            sum(c.realized_pnl for c in cc_completed if c.realized_pnl < 0)
            + sum(c.realized_pnl for c in csp_completed if c.realized_pnl < 0)
            + sum(p.net_pnl for p in dn_completed if p.net_pnl < 0)
        )
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Sharpe ratio from daily returns
        sharpe = 0.0
        if len(self._daily_returns) >= 30:
            returns_arr = np.array(self._daily_returns)
            mean_r = float(np.mean(returns_arr))
            std_r = float(np.std(returns_arr, ddof=1))
            if std_r > 0:
                sharpe = mean_r / std_r * math.sqrt(365)  # Annualized

        # Delta compliance
        risk_state = self._risk.state
        delta_compliance_pct = risk_state.delta_compliance_pct

        # Premium vs hedge costs (CC specifically)
        cc_total_premium = sum(c.premium for c in cc_completed)
        cc_total_hedge = sum(c.hedge_costs for c in cc_completed)
        premium_hedge_ratio = (
            cc_total_premium / cc_total_hedge
            if cc_total_hedge > 0 else float("inf")
        )

        # Max drawdown
        max_dd = max(
            risk_state.daily_drawdown_pct,
            risk_state.weekly_drawdown_pct,
            risk_state.monthly_drawdown_pct,
        )

        # Evaluate each criterion
        criteria = {
            "paper_trading_90d": {
                "required": 90.0,
                "actual": round(days_running, 1),
                "passed": days_running >= 90.0,
                "description": "90-day paper trading minimum",
            },
            "net_positive_pnl": {
                "required": 0.0,
                "actual": round(total_pnl, 4),
                "passed": total_pnl > 0,
                "description": "Net positive PnL after fees and hedge costs",
            },
            "profit_factor": {
                "required": 1.3,
                "actual": round(min(profit_factor, 999.9), 2),
                "passed": profit_factor > 1.3,
                "description": "Profit factor > 1.3",
            },
            "sharpe_ratio": {
                "required": 1.0,
                "actual": round(sharpe, 2),
                "passed": sharpe > 1.0,
                "description": "Sharpe ratio > 1.0",
            },
            "max_drawdown": {
                "required": 5.0,
                "actual": round(max_dd, 2),
                "passed": max_dd < 5.0,
                "description": "Maximum drawdown < 5%",
            },
            "delta_within_limits": {
                "required": 90.0,
                "actual": round(delta_compliance_pct, 1),
                "passed": delta_compliance_pct >= 90.0,
                "description": "Delta within limits > 90% of the time",
            },
            "premium_exceeds_hedge_2x": {
                "required": 2.0,
                "actual": round(min(premium_hedge_ratio, 999.9), 2),
                "passed": premium_hedge_ratio >= 2.0,
                "description": "CC premium income > hedge costs 2x",
            },
        }

        all_passed = all(c["passed"] for c in criteria.values())

        result = {
            "ready_for_live": all_passed,
            "days_running": round(days_running, 1),
            "criteria": criteria,
            "summary": {
                "passed": sum(1 for c in criteria.values() if c["passed"]),
                "total": len(criteria),
                "blocking": [
                    name for name, c in criteria.items() if not c["passed"]
                ],
            },
        }

        logger.info(
            "Go-live evaluation: %s (%d/%d criteria met, blocking: %s)",
            "READY" if all_passed else "NOT READY",
            result["summary"]["passed"],
            result["summary"]["total"],
            result["summary"]["blocking"],
        )

        return result
