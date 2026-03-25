"""Strategy-specific risk manager for STRAT-008.

Enforces 25% total allocation, sub-strategy allocations, per-trade max loss,
Greek limits, event risk rules, drawdown protection, IV spike / flash crash
circuit breakers, and the kill switch.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.black_scholes import OptionGreeks

logger = logging.getLogger(__name__)
trade_logger = logging.getLogger("trade")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class StrategyRiskState:
    """Current risk state across all sub-strategies."""
    # Allocation tracking (USDT)
    total_capital_deployed: float = 0.0
    cc_capital: float = 0.0     # Covered calls
    csp_capital: float = 0.0    # Cash-secured puts
    dn_capital: float = 0.0     # Delta-neutral
    # Greeks aggregate
    net_delta: float = 0.0
    net_gamma: float = 0.0
    net_theta: float = 0.0
    net_vega: float = 0.0
    # Drawdown
    daily_drawdown_pct: float = 0.0
    weekly_drawdown_pct: float = 0.0
    monthly_drawdown_pct: float = 0.0
    # Risk events
    iv_spike_active: bool = False
    flash_crash_active: bool = False
    event_risk_active: bool = False
    kill_switch_active: bool = False
    # Consecutive losses
    cc_consecutive_losses: int = 0
    csp_consecutive_losses: int = 0
    # Counters
    greek_limit_breaches: int = 0
    delta_compliance_checks: int = 0
    delta_within_limits: int = 0

    @property
    def delta_compliance_pct(self) -> float:
        if self.delta_compliance_checks == 0:
            return 100.0
        return self.delta_within_limits / self.delta_compliance_checks * 100.0

    def to_dict(self) -> dict:
        return {
            "total_deployed": round(self.total_capital_deployed, 2),
            "cc_capital": round(self.cc_capital, 2),
            "csp_capital": round(self.csp_capital, 2),
            "dn_capital": round(self.dn_capital, 2),
            "net_delta": round(self.net_delta, 4),
            "net_gamma": round(self.net_gamma, 6),
            "net_theta": round(self.net_theta, 4),
            "net_vega": round(self.net_vega, 2),
            "daily_dd_pct": round(self.daily_drawdown_pct, 2),
            "weekly_dd_pct": round(self.weekly_drawdown_pct, 2),
            "monthly_dd_pct": round(self.monthly_drawdown_pct, 2),
            "iv_spike": self.iv_spike_active,
            "flash_crash": self.flash_crash_active,
            "event_risk": self.event_risk_active,
            "kill_switch": self.kill_switch_active,
            "greek_breaches": self.greek_limit_breaches,
            "delta_compliance_pct": round(self.delta_compliance_pct, 1),
        }


# ---------------------------------------------------------------------------
# StrategyRiskManager
# ---------------------------------------------------------------------------

class StrategyRiskManager:
    """STRAT-008 specific risk management.

    Enforces all risk rules from Section 5 of the instructions.

    Parameters
    ----------
    config : dict
        Strategy parameters.
    total_equity : float
        Starting equity.
    """

    def __init__(self, config: dict, total_equity: float = 10000.0) -> None:
        self._config = config
        self._equity = total_equity

        # Allocation limits (% of equity)
        self._max_total_pct = 25.0
        self._max_cc_pct = config.get("cc_max_allocation_pct", 15.0)
        self._max_csp_pct = config.get("csp_max_allocation_pct", 10.0)
        self._max_dn_pct = config.get("dn_max_allocation_pct", 15.0)

        # Per-trade loss limits
        self._cc_max_loss_pct = 10.0    # 10% below entry
        self._csp_max_loss_pct = 15.0   # 15% below current price
        self._dn_max_loss_multiple = 2.0  # 2x premium collected

        # Greek limits
        self._max_delta_per_10k = config.get("max_net_delta_per_10k", 0.15)
        self._max_gamma_per_premium = config.get("max_gamma_per_premium", -0.005)
        self._max_vega_per_iv_point = config.get("max_vega_per_iv_point", -500.0)

        # Drawdown limits
        self._daily_dd_limit = config.get("daily_drawdown_pct", 2.0)
        self._weekly_dd_limit = config.get("weekly_drawdown_pct", 4.0)
        self._monthly_dd_limit = config.get("monthly_drawdown_pct", 7.0)
        self._system_dd_limit = 15.0

        # Circuit breaker thresholds
        self._iv_spike_pp = config.get("iv_spike_threshold_pp", 20.0)
        self._flash_crash_pct = config.get("flash_crash_pct", 8.0)

        # Event risk
        self._event_tighten_hours = config.get("event_tighten_hours", 24)
        self._event_close_hours = config.get("event_close_hours", 2)

        # Consecutive loss handling
        self._max_consecutive_losses = config.get("max_consecutive_losses", 3)
        self._loss_halt_days = config.get("loss_halt_days", 14)

        # State
        self._state = StrategyRiskState()
        self._halt_until: Dict[str, float] = {}  # sub-strategy -> resume timestamp
        self._alerts: List[dict] = []

        # Equity tracking
        self._peak_equity = total_equity
        self._daily_start_equity = total_equity
        self._weekly_start_equity = total_equity
        self._monthly_start_equity = total_equity

    # ------------------------------------------------------------------
    # Equity management
    # ------------------------------------------------------------------

    def update_equity(self, equity: float) -> None:
        """Update current equity and refresh drawdown state."""
        self._equity = equity
        if equity > self._peak_equity:
            self._peak_equity = equity

        # Update drawdowns
        if self._daily_start_equity > 0:
            self._state.daily_drawdown_pct = max(
                0, (self._daily_start_equity - equity) / self._daily_start_equity * 100
            )
        if self._weekly_start_equity > 0:
            self._state.weekly_drawdown_pct = max(
                0, (self._weekly_start_equity - equity) / self._weekly_start_equity * 100
            )
        if self._monthly_start_equity > 0:
            self._state.monthly_drawdown_pct = max(
                0, (self._monthly_start_equity - equity) / self._monthly_start_equity * 100
            )

    def reset_daily(self) -> None:
        self._daily_start_equity = self._equity

    def reset_weekly(self) -> None:
        self._weekly_start_equity = self._equity

    def reset_monthly(self) -> None:
        self._monthly_start_equity = self._equity

    # ------------------------------------------------------------------
    # Allocation checks (Section 5.1)
    # ------------------------------------------------------------------

    def check_allocation(
        self, sub_strategy: str, additional_capital: float
    ) -> Tuple[bool, str]:
        """Check if additional capital can be deployed.

        Parameters
        ----------
        sub_strategy : str
            "cc" (covered calls), "csp" (cash-secured puts), "dn" (delta-neutral)
        additional_capital : float
            Additional USDT to deploy.

        Returns
        -------
        (allowed, reason)
        """
        # Total strategy limit: 25% of equity
        new_total = self._state.total_capital_deployed + additional_capital
        max_total = self._equity * (self._max_total_pct / 100.0)
        if new_total > max_total:
            return False, (
                f"Total allocation ${new_total:.2f} exceeds "
                f"{self._max_total_pct}% limit (${max_total:.2f})"
            )

        # Sub-strategy limits
        if sub_strategy == "cc":
            new_sub = self._state.cc_capital + additional_capital
            max_sub = self._equity * (self._max_cc_pct / 100.0)
            label = "Covered calls"
        elif sub_strategy == "csp":
            new_sub = self._state.csp_capital + additional_capital
            max_sub = self._equity * (self._max_csp_pct / 100.0)
            label = "Cash-secured puts"
        elif sub_strategy == "dn":
            new_sub = self._state.dn_capital + additional_capital
            max_sub = self._equity * (self._max_dn_pct / 100.0)
            label = "Delta-neutral"
        else:
            return False, f"Unknown sub-strategy: {sub_strategy}"

        if new_sub > max_sub:
            return False, (
                f"{label} allocation ${new_sub:.2f} exceeds limit ${max_sub:.2f}"
            )

        return True, ""

    def record_allocation(self, sub_strategy: str, capital: float) -> None:
        """Record capital deployment."""
        self._state.total_capital_deployed += capital
        if sub_strategy == "cc":
            self._state.cc_capital += capital
        elif sub_strategy == "csp":
            self._state.csp_capital += capital
        elif sub_strategy == "dn":
            self._state.dn_capital += capital

    def release_allocation(self, sub_strategy: str, capital: float) -> None:
        """Release capital when a position is closed."""
        self._state.total_capital_deployed = max(
            0, self._state.total_capital_deployed - capital
        )
        if sub_strategy == "cc":
            self._state.cc_capital = max(0, self._state.cc_capital - capital)
        elif sub_strategy == "csp":
            self._state.csp_capital = max(0, self._state.csp_capital - capital)
        elif sub_strategy == "dn":
            self._state.dn_capital = max(0, self._state.dn_capital - capital)

    # ------------------------------------------------------------------
    # Drawdown checks (Section 5.4)
    # ------------------------------------------------------------------

    def check_drawdown(self) -> Tuple[bool, str, float]:
        """Check if any drawdown threshold is breached.

        Returns (halted, level, pct).
        """
        s = self._state

        if s.daily_drawdown_pct >= self._daily_dd_limit:
            return True, "daily", s.daily_drawdown_pct
        if s.weekly_drawdown_pct >= self._weekly_dd_limit:
            return True, "weekly", s.weekly_drawdown_pct
        if s.monthly_drawdown_pct >= self._monthly_dd_limit:
            return True, "monthly", s.monthly_drawdown_pct

        # System-wide
        if self._peak_equity > 0:
            system_dd = (self._peak_equity - self._equity) / self._peak_equity * 100
            if system_dd >= self._system_dd_limit:
                return True, "system", system_dd

        return False, "", 0.0

    # ------------------------------------------------------------------
    # Greek limits (Section 5.3)
    # ------------------------------------------------------------------

    def update_greeks(self, greeks: OptionGreeks, notional: float = 0.0) -> None:
        """Update aggregate Greeks and check limits."""
        self._state.net_delta = greeks.delta
        self._state.net_gamma = greeks.gamma
        self._state.net_theta = greeks.theta
        self._state.net_vega = greeks.vega

        # Delta compliance tracking
        self._state.delta_compliance_checks += 1
        per_10k = notional / 10000.0 if notional > 0 else 1.0
        delta_limit = self._max_delta_per_10k * per_10k

        if abs(greeks.delta) <= delta_limit:
            self._state.delta_within_limits += 1

    def check_greek_limits(
        self, greeks: OptionGreeks, notional: float, total_premium: float
    ) -> List[str]:
        """Check if Greeks exceed limits. Returns list of breaches."""
        breaches: List[str] = []
        per_10k = notional / 10000.0 if notional > 0 else 1.0

        # Delta: +/- 0.15 per $10k
        delta_limit = self._max_delta_per_10k * per_10k
        if abs(greeks.delta) > delta_limit:
            breaches.append(
                f"Net delta |{greeks.delta:.4f}| > limit {delta_limit:.4f}"
            )

        # Gamma: -0.005 per dollar of premium
        gamma_limit = abs(self._max_gamma_per_premium * total_premium)
        if abs(greeks.gamma) > gamma_limit and gamma_limit > 0:
            breaches.append(
                f"Gamma |{greeks.gamma:.6f}| > limit {gamma_limit:.6f}"
            )

        # Vega: -$500 per 1% IV
        if greeks.vega < self._max_vega_per_iv_point:
            breaches.append(
                f"Vega {greeks.vega:.2f} < limit {self._max_vega_per_iv_point:.2f}"
            )

        if breaches:
            self._state.greek_limit_breaches += len(breaches)
            for b in breaches:
                logger.warning("Greek limit breach: %s", b)

        return breaches

    # ------------------------------------------------------------------
    # Circuit breakers (Section 5.6)
    # ------------------------------------------------------------------

    def handle_iv_spike(self) -> None:
        """Activate IV spike circuit breaker. Close all vol-selling positions."""
        self._state.iv_spike_active = True
        self._add_alert("CIRCUIT_BREAKER", "IV spike detected: closing all positions")
        logger.critical(
            "IV SPIKE circuit breaker activated — closing all vol-selling positions"
        )

    def handle_flash_crash(self) -> None:
        """Activate flash crash circuit breaker."""
        self._state.flash_crash_active = True
        self._add_alert("CIRCUIT_BREAKER", "Flash crash detected: closing all positions")
        logger.critical(
            "FLASH CRASH circuit breaker activated — closing all positions"
        )

    def activate_kill_switch(self, reason: str) -> None:
        """Activate kill switch — close everything."""
        self._state.kill_switch_active = True
        self._add_alert("KILL_SWITCH", f"Kill switch: {reason}")
        logger.critical("KILL SWITCH activated: %s", reason)

    def clear_circuit_breakers(self) -> None:
        """Clear circuit breaker states after manual review."""
        self._state.iv_spike_active = False
        self._state.flash_crash_active = False
        logger.info("Circuit breakers cleared")

    def is_trading_halted(self) -> Tuple[bool, str]:
        """Check if trading is halted for any reason."""
        if self._state.kill_switch_active:
            return True, "Kill switch active"

        if self._state.iv_spike_active:
            return True, "IV spike circuit breaker"

        if self._state.flash_crash_active:
            return True, "Flash crash circuit breaker"

        halted, level, pct = self.check_drawdown()
        if halted:
            return True, f"{level} drawdown {pct:.1f}% exceeded"

        return False, ""

    # ------------------------------------------------------------------
    # Event risk (Section 5.5)
    # ------------------------------------------------------------------

    def set_event_risk(
        self, hours_until_event: float
    ) -> str:
        """Update event risk state.

        Returns action recommendation.
        """
        if hours_until_event <= self._event_close_hours:
            self._state.event_risk_active = True
            return "close_50pct"
        elif hours_until_event <= self._event_tighten_hours:
            self._state.event_risk_active = True
            return "tighten_delta"
        else:
            self._state.event_risk_active = False
            return "normal"

    def should_block_new_vol_selling(self, hours_until_event: float) -> bool:
        """Block new vol selling within 2h of events."""
        return hours_until_event <= self._event_close_hours

    # ------------------------------------------------------------------
    # Consecutive loss handling (Section 7.6)
    # ------------------------------------------------------------------

    def record_sub_strategy_loss(self, sub_strategy: str) -> None:
        """Record a consecutive loss for a sub-strategy."""
        if sub_strategy == "cc":
            self._state.cc_consecutive_losses += 1
            if self._state.cc_consecutive_losses >= self._max_consecutive_losses:
                halt_until = time.time() + self._loss_halt_days * 86400
                self._halt_until["cc"] = halt_until
                logger.warning(
                    "Covered calls halted for %d days: %d consecutive losses",
                    self._loss_halt_days, self._state.cc_consecutive_losses,
                )
        elif sub_strategy == "csp":
            self._state.csp_consecutive_losses += 1
            if self._state.csp_consecutive_losses >= self._max_consecutive_losses:
                halt_until = time.time() + self._loss_halt_days * 86400
                self._halt_until["csp"] = halt_until
                logger.warning(
                    "CSP halted for %d days: %d consecutive losses",
                    self._loss_halt_days, self._state.csp_consecutive_losses,
                )

    def record_sub_strategy_win(self, sub_strategy: str) -> None:
        """Reset consecutive loss counter on a win."""
        if sub_strategy == "cc":
            self._state.cc_consecutive_losses = 0
        elif sub_strategy == "csp":
            self._state.csp_consecutive_losses = 0

    def is_sub_strategy_halted(self, sub_strategy: str) -> Tuple[bool, str]:
        """Check if a sub-strategy is halted due to consecutive losses."""
        halt_time = self._halt_until.get(sub_strategy, 0)
        if halt_time > 0 and time.time() < halt_time:
            remaining_h = (halt_time - time.time()) / 3600
            return True, f"Halted for {remaining_h:.1f}h more (consecutive losses)"

        # Clear expired halts
        if halt_time > 0 and time.time() >= halt_time:
            del self._halt_until[sub_strategy]
            logger.info("Halt expired for %s — resuming", sub_strategy)

        return False, ""

    # ------------------------------------------------------------------
    # Per-trade loss checks (Section 5.2)
    # ------------------------------------------------------------------

    def check_cc_stop(
        self, entry_price: float, current_price: float
    ) -> bool:
        """Covered call hard stop: 10% below spot entry."""
        if entry_price <= 0:
            return False
        loss_pct = (entry_price - current_price) / entry_price * 100
        return loss_pct >= self._cc_max_loss_pct

    def check_csp_stop(
        self, current_price: float, original_price: float
    ) -> bool:
        """CSP hard stop: cancel if underlying drops > 15% from current price."""
        if original_price <= 0:
            return False
        drop_pct = (original_price - current_price) / original_price * 100
        return drop_pct >= self._csp_max_loss_pct

    def check_dn_stop(
        self, unrealized_loss: float, premium_collected: float
    ) -> bool:
        """Delta-neutral stop: loss > 2x premium."""
        if premium_collected <= 0:
            return False
        return abs(unrealized_loss) >= self._dn_max_loss_multiple * premium_collected

    # ------------------------------------------------------------------
    # Alerts
    # ------------------------------------------------------------------

    def _add_alert(self, alert_type: str, message: str) -> None:
        self._alerts.append({
            "type": alert_type,
            "message": message,
            "timestamp": time.time(),
        })
        # Keep last 100 alerts
        if len(self._alerts) > 100:
            self._alerts = self._alerts[-100:]

    def get_alerts(self) -> List[dict]:
        return list(self._alerts)

    # ------------------------------------------------------------------
    # State access
    # ------------------------------------------------------------------

    @property
    def state(self) -> StrategyRiskState:
        return self._state

    @property
    def equity(self) -> float:
        return self._equity

    def get_state_dict(self) -> dict:
        return self._state.to_dict()

    def get_full_state(self) -> dict:
        return {
            "risk": self._state.to_dict(),
            "equity": self._equity,
            "peak_equity": self._peak_equity,
            "halts": {k: v for k, v in self._halt_until.items()},
            "alerts_count": len(self._alerts),
        }
