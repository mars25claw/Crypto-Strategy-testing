"""Monthly budget tracking for the Signal-Enhanced DCA bot.

Enforces hard monthly caps for both regular DCA purchases and crash-buy
reserve spending.  Resets at 00:00 UTC on the 1st of each month.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class MonthlyBudget:
    """Snapshot of budget usage for a single calendar month."""

    year: int
    month: int
    dca_budget: float = 0.0          # Total DCA budget for the month
    crash_reserve: float = 0.0       # Total crash-buy reserve for the month
    dca_spent: float = 0.0           # Amount spent on regular DCA
    crash_spent: float = 0.0         # Amount spent on crash-buys
    per_instrument_dca: Dict[str, float] = field(default_factory=dict)
    per_instrument_crash: Dict[str, float] = field(default_factory=dict)

    @property
    def dca_remaining(self) -> float:
        return max(0.0, self.dca_budget - self.dca_spent)

    @property
    def crash_remaining(self) -> float:
        return max(0.0, self.crash_reserve - self.crash_spent)

    @property
    def total_spent(self) -> float:
        return self.dca_spent + self.crash_spent

    @property
    def total_budget(self) -> float:
        return self.dca_budget + self.crash_reserve

    @property
    def total_remaining(self) -> float:
        return max(0.0, self.total_budget - self.total_spent)

    @property
    def utilisation_pct(self) -> float:
        if self.total_budget <= 0:
            return 0.0
        return (self.total_spent / self.total_budget) * 100.0

    def to_dict(self) -> dict:
        return {
            "year": self.year,
            "month": self.month,
            "dca_budget": self.dca_budget,
            "crash_reserve": self.crash_reserve,
            "dca_spent": round(self.dca_spent, 2),
            "crash_spent": round(self.crash_spent, 2),
            "dca_remaining": round(self.dca_remaining, 2),
            "crash_remaining": round(self.crash_remaining, 2),
            "total_spent": round(self.total_spent, 2),
            "total_budget": round(self.total_budget, 2),
            "total_remaining": round(self.total_remaining, 2),
            "utilisation_pct": round(self.utilisation_pct, 2),
            "per_instrument_dca": {k: round(v, 2) for k, v in self.per_instrument_dca.items()},
            "per_instrument_crash": {k: round(v, 2) for k, v in self.per_instrument_crash.items()},
        }


class BudgetManager:
    """Manages monthly DCA and crash-buy budgets with hard cap enforcement.

    Parameters
    ----------
    monthly_dca_budget : float
        Total USDT budget for regular DCA purchases per month.
    monthly_crash_reserve : float
        Separate USDT reserve for crash-buy opportunities per month.
    base_amounts : dict
        Per-instrument base amounts (USDT per DCA interval).
    min_purchase_usdt : float
        Minimum purchase amount (max of this and exchange minimum).
    budget_cap_multiplier : float
        Monthly cap = multiplier * base_amount * intervals_per_month.
    """

    def __init__(
        self,
        monthly_dca_budget: float = 400.0,
        monthly_crash_reserve: float = 200.0,
        base_amounts: Optional[Dict[str, float]] = None,
        min_purchase_usdt: float = 5.0,
        budget_cap_multiplier: float = 4.5,
    ) -> None:
        self._monthly_dca_budget = monthly_dca_budget
        self._monthly_crash_reserve = monthly_crash_reserve
        self._base_amounts = base_amounts or {"BTCUSDT": 50.0, "ETHUSDT": 30.0}
        self._min_purchase = min_purchase_usdt
        self._cap_multiplier = budget_cap_multiplier

        # Current month tracking
        self._current: Optional[MonthlyBudget] = None
        self._history: list[MonthlyBudget] = []

        self._ensure_current_month()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_base_amount(self, symbol: str) -> float:
        """Return the configured base DCA amount for an instrument."""
        return self._base_amounts.get(symbol, 0.0)

    def get_min_purchase(self, exchange_min: float = 0.0) -> float:
        """Return effective minimum purchase = max(configured, exchange minimum)."""
        return max(self._min_purchase, exchange_min)

    def check_dca_purchase(self, symbol: str, amount_usdt: float) -> tuple[float, str]:
        """Verify and possibly cap a regular DCA purchase.

        Returns
        -------
        (approved_amount, reason)
            approved_amount is the capped purchase amount (may be 0 if denied).
            reason describes any adjustment or empty string if OK.
        """
        self._ensure_current_month()
        budget = self._current

        if amount_usdt <= 0:
            return 0.0, "Amount must be positive"

        # Hard monthly cap
        remaining = budget.dca_remaining
        if remaining <= 0:
            return 0.0, "Monthly DCA budget exhausted"

        approved = min(amount_usdt, remaining)

        # Check minimum
        min_purchase = self.get_min_purchase()
        if approved < min_purchase:
            if remaining < min_purchase:
                return 0.0, f"Remaining budget {remaining:.2f} below minimum {min_purchase:.2f}"
            approved = min_purchase

        reason = ""
        if approved < amount_usdt:
            reason = f"Capped from {amount_usdt:.2f} to {approved:.2f} (budget remaining: {remaining:.2f})"

        return round(approved, 2), reason

    def check_crash_purchase(self, symbol: str, amount_usdt: float) -> tuple[float, str]:
        """Verify and possibly cap a crash-buy purchase.

        Returns
        -------
        (approved_amount, reason)
        """
        self._ensure_current_month()
        budget = self._current

        if amount_usdt <= 0:
            return 0.0, "Amount must be positive"

        remaining = budget.crash_remaining
        if remaining <= 0:
            return 0.0, "Monthly crash-buy reserve exhausted"

        approved = min(amount_usdt, remaining)

        min_purchase = self.get_min_purchase()
        if approved < min_purchase:
            if remaining < min_purchase:
                return 0.0, f"Crash reserve remaining {remaining:.2f} below minimum {min_purchase:.2f}"
            approved = min_purchase

        reason = ""
        if approved < amount_usdt:
            reason = f"Crash-buy capped from {amount_usdt:.2f} to {approved:.2f} (reserve remaining: {remaining:.2f})"

        return round(approved, 2), reason

    def record_dca_purchase(self, symbol: str, amount_usdt: float) -> None:
        """Record a completed DCA purchase against the budget."""
        self._ensure_current_month()
        self._current.dca_spent += amount_usdt
        self._current.per_instrument_dca[symbol] = (
            self._current.per_instrument_dca.get(symbol, 0.0) + amount_usdt
        )
        logger.info(
            "DCA purchase recorded: %s $%.2f  (month spent: $%.2f / $%.2f)",
            symbol, amount_usdt, self._current.dca_spent, self._current.dca_budget,
        )

    def record_crash_purchase(self, symbol: str, amount_usdt: float) -> None:
        """Record a completed crash-buy purchase against the reserve."""
        self._ensure_current_month()
        self._current.crash_spent += amount_usdt
        self._current.per_instrument_crash[symbol] = (
            self._current.per_instrument_crash.get(symbol, 0.0) + amount_usdt
        )
        logger.info(
            "Crash-buy recorded: %s $%.2f  (month crash spent: $%.2f / $%.2f)",
            symbol, amount_usdt, self._current.crash_spent, self._current.crash_reserve,
        )

    def get_current_budget(self) -> MonthlyBudget:
        """Return the current month's budget snapshot."""
        self._ensure_current_month()
        return self._current

    def get_budget_summary(self) -> dict:
        """Return a dict summary of current budget status."""
        self._ensure_current_month()
        return self._current.to_dict()

    def get_monthly_cap(self, symbol: str, intervals_per_month: int) -> float:
        """Calculate the per-instrument monthly cap.

        monthly_cap = budget_cap_multiplier * base_amount * intervals_per_month
        """
        base = self._base_amounts.get(symbol, 0.0)
        return self._cap_multiplier * base * intervals_per_month

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def to_state(self) -> dict:
        """Serialize budget state for persistence."""
        self._ensure_current_month()
        return {
            "current": self._current.to_dict(),
            "history": [b.to_dict() for b in self._history[-12:]],
        }

    def load_state(self, state: dict) -> None:
        """Restore budget state from persisted data."""
        current_data = state.get("current", {})
        if current_data:
            now = datetime.now(timezone.utc)
            if current_data.get("year") == now.year and current_data.get("month") == now.month:
                self._current = MonthlyBudget(
                    year=current_data["year"],
                    month=current_data["month"],
                    dca_budget=current_data.get("dca_budget", self._monthly_dca_budget),
                    crash_reserve=current_data.get("crash_reserve", self._monthly_crash_reserve),
                    dca_spent=current_data.get("dca_spent", 0.0),
                    crash_spent=current_data.get("crash_spent", 0.0),
                    per_instrument_dca=current_data.get("per_instrument_dca", {}),
                    per_instrument_crash=current_data.get("per_instrument_crash", {}),
                )
                logger.info(
                    "Budget state restored: DCA spent $%.2f / $%.2f, crash spent $%.2f / $%.2f",
                    self._current.dca_spent, self._current.dca_budget,
                    self._current.crash_spent, self._current.crash_reserve,
                )
            else:
                logger.info("Persisted budget is from a different month, starting fresh")
                self._ensure_current_month()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ensure_current_month(self) -> None:
        """Create or roll over to the current calendar month."""
        now = datetime.now(timezone.utc)
        if self._current is not None:
            if self._current.year == now.year and self._current.month == now.month:
                return
            # Month has rolled over — archive old budget
            logger.info(
                "Monthly budget reset: %04d-%02d -> %04d-%02d "
                "(previous: DCA $%.2f / $%.2f, crash $%.2f / $%.2f)",
                self._current.year, self._current.month,
                now.year, now.month,
                self._current.dca_spent, self._current.dca_budget,
                self._current.crash_spent, self._current.crash_reserve,
            )
            self._history.append(self._current)

        self._current = MonthlyBudget(
            year=now.year,
            month=now.month,
            dca_budget=self._monthly_dca_budget,
            crash_reserve=self._monthly_crash_reserve,
        )
