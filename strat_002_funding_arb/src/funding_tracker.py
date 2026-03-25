"""Funding settlement monitoring and income tracking.

Implements Section 8.3 funding settlement monitoring:
- Timer 10 minutes before each funding (00:00/08:00/16:00 UTC)
- Query /fapi/v1/income for actual funding payment
- Compare actual vs expected
- Track cumulative funding income per instrument
- Pre-funding exit for negative rates (Section 4.7)
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)
trade_logger = logging.getLogger("trade")

# Funding settlement times in UTC: 00:00, 08:00, 16:00
FUNDING_HOURS = [0, 8, 16]
PRE_FUNDING_MINUTES = 10  # Timer fires 10 minutes before
CLOCK_DRIFT_EXCLUSION_MINUTES = 5  # Never trade within 5 min of funding


class FundingSettlementRecord:
    """Record of a single funding settlement."""

    __slots__ = (
        "symbol", "timestamp_ms", "actual_rate", "expected_rate",
        "actual_payment", "expected_payment", "discrepancy_pct",
    )

    def __init__(
        self,
        symbol: str,
        timestamp_ms: int,
        actual_rate: float,
        expected_rate: float,
        actual_payment: float,
        expected_payment: float,
    ) -> None:
        self.symbol = symbol
        self.timestamp_ms = timestamp_ms
        self.actual_rate = actual_rate
        self.expected_rate = expected_rate
        self.actual_payment = actual_payment
        self.expected_payment = expected_payment
        if expected_payment != 0:
            self.discrepancy_pct = abs(actual_payment - expected_payment) / abs(expected_payment) * 100
        else:
            self.discrepancy_pct = 0.0


class FundingTracker:
    """Monitors funding settlements and tracks income.

    Parameters
    ----------
    binance_client : BinanceClient
        For querying /fapi/v1/income.
    strategy : FundingArbStrategy
        Strategy instance for updating position funding data and tactical exits.
    execution_engine : ExecutionEngine
        For executing tactical pre-funding exits.
    paper_mode : bool
        If True, simulate funding payments from live rates.
    """

    def __init__(
        self,
        binance_client: Any,
        strategy: Any,
        execution_engine: Any = None,
        paper_mode: bool = True,
    ) -> None:
        self._client = binance_client
        self._strategy = strategy
        self._execution = execution_engine
        self._paper_mode = paper_mode

        # Settlement history: symbol -> list of FundingSettlementRecord
        self._settlements: Dict[str, List[FundingSettlementRecord]] = defaultdict(list)

        # Cumulative income tracking: symbol -> total_income
        self._cumulative_income: Dict[str, float] = defaultdict(float)

        # Daily income tracking: "YYYY-MM-DD" -> symbol -> income
        self._daily_income: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

        # Background task handles
        self._settlement_task: Optional[asyncio.Task] = None
        self._running = False

        # Callbacks for alerting
        self._alert_callback: Optional[Callable] = None

    # ══════════════════════════════════════════════════════════════════════
    #  Lifecycle
    # ══════════════════════════════════════════════════════════════════════

    async def start(self) -> None:
        """Start the funding settlement monitoring loop."""
        if self._running:
            return
        self._running = True
        self._settlement_task = asyncio.create_task(
            self._settlement_loop(),
            name="funding_settlement_loop",
        )
        logger.info("FundingTracker started")

    async def stop(self) -> None:
        """Stop the monitoring loop."""
        self._running = False
        if self._settlement_task and not self._settlement_task.done():
            self._settlement_task.cancel()
            try:
                await self._settlement_task
            except asyncio.CancelledError:
                pass
        logger.info("FundingTracker stopped")

    def set_alert_callback(self, callback: Callable) -> None:
        """Register a callback for funding alerts."""
        self._alert_callback = callback

    # ══════════════════════════════════════════════════════════════════════
    #  Main settlement loop
    # ══════════════════════════════════════════════════════════════════════

    async def _settlement_loop(self) -> None:
        """Main loop: wake up 10 minutes before each funding settlement."""
        while self._running:
            try:
                next_funding = self._next_funding_time()
                pre_funding = next_funding - timedelta(minutes=PRE_FUNDING_MINUTES)

                now = datetime.now(timezone.utc)
                wait_seconds = (pre_funding - now).total_seconds()

                if wait_seconds > 0:
                    logger.info(
                        "Next funding at %s UTC, pre-check at %s (%.0fs from now)",
                        next_funding.strftime("%H:%M:%S"),
                        pre_funding.strftime("%H:%M:%S"),
                        wait_seconds,
                    )
                    # Sleep in small increments so we can be cancelled
                    while wait_seconds > 0 and self._running:
                        sleep_time = min(wait_seconds, 30.0)
                        await asyncio.sleep(sleep_time)
                        wait_seconds -= sleep_time

                if not self._running:
                    break

                # ── Pre-funding check (10 min before) ─────────────────
                await self._pre_funding_check()

                # ── Wait for actual funding settlement ────────────────
                now = datetime.now(timezone.utc)
                seconds_to_settlement = (next_funding - now).total_seconds()
                if seconds_to_settlement > 0:
                    await asyncio.sleep(seconds_to_settlement + 5)  # +5s buffer

                if not self._running:
                    break

                # ── Post-funding settlement reconciliation ────────────
                await self._post_funding_reconciliation(next_funding)

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in funding settlement loop")
                await asyncio.sleep(60)

    # ══════════════════════════════════════════════════════════════════════
    #  Pre-funding check
    # ══════════════════════════════════════════════════════════════════════

    async def _pre_funding_check(self) -> None:
        """Run 10 minutes before each funding settlement.

        1. Verify positions are still held correctly.
        2. Evaluate tactical pre-funding exits for negative rates.
        """
        logger.info("Pre-funding check: verifying positions")

        for pos_id, pos in list(self._strategy.positions.items()):
            inst = self._strategy.instruments.get(pos.symbol)
            if inst is None:
                continue

            # Verify position integrity
            logger.info(
                "Pre-funding: %s — spot_qty=%.8f futures_qty=%.8f "
                "predicted_rate=%.6f",
                pos.symbol, pos.spot_quantity, pos.futures_quantity,
                inst.predicted_funding_rate,
            )

            # Section 4.7: Tactical pre-funding exit
            exit_signal = self._strategy.evaluate_tactical_exit(pos.symbol)
            if exit_signal and self._execution:
                logger.info(
                    "Tactical pre-funding exit triggered for %s: %s",
                    pos.symbol, exit_signal.reason,
                )
                # Only close futures leg — will reopen after settlement
                # This is handled by the main loop via the exit signal

    # ══════════════════════════════════════════════════════════════════════
    #  Post-funding reconciliation
    # ══════════════════════════════════════════════════════════════════════

    async def _post_funding_reconciliation(self, funding_time: datetime) -> None:
        """Run within 5 minutes after each funding settlement.

        1. Query actual funding payments.
        2. Compare with expected.
        3. Update strategy tracking.
        """
        logger.info("Post-funding reconciliation for %s", funding_time.isoformat())

        for pos_id, pos in list(self._strategy.positions.items()):
            inst = self._strategy.instruments.get(pos.symbol)
            if inst is None:
                continue

            try:
                actual_rate, actual_payment = await self._get_funding_payment(
                    pos.symbol, funding_time
                )

                # Calculate expected payment
                expected_payment = pos.futures_notional * inst.current_funding_rate

                # Record settlement
                record = FundingSettlementRecord(
                    symbol=pos.symbol,
                    timestamp_ms=int(funding_time.timestamp() * 1000),
                    actual_rate=actual_rate,
                    expected_rate=inst.current_funding_rate,
                    actual_payment=actual_payment,
                    expected_payment=expected_payment,
                )
                self._settlements[pos.symbol].append(record)

                # Update cumulative tracking
                self._cumulative_income[pos.symbol] += actual_payment
                day_key = funding_time.strftime("%Y-%m-%d")
                self._daily_income[day_key][pos.symbol] += actual_payment

                # Update strategy position tracking
                self._strategy.update_position_funding(
                    pos.symbol, actual_rate, actual_payment
                )

                # Update the instrument's current rate from actual
                self._strategy.ingest_funding_rate(
                    pos.symbol, actual_rate,
                    int(funding_time.timestamp() * 1000),
                    predicted=False,
                )

                # Log discrepancy
                if record.discrepancy_pct > 0.1:
                    logger.warning(
                        "Funding discrepancy for %s: expected=%.6f actual=%.6f "
                        "(%.2f%% off)",
                        pos.symbol, expected_payment, actual_payment,
                        record.discrepancy_pct,
                    )

                trade_logger.info(
                    "FUNDING_SETTLEMENT\tsymbol=%s\trate=%.6f\tpayment=%.6f\t"
                    "expected=%.6f\tdiscrepancy=%.2f%%\tcumulative=%.6f",
                    pos.symbol, actual_rate, actual_payment,
                    expected_payment, record.discrepancy_pct,
                    self._cumulative_income[pos.symbol],
                )

            except Exception:
                logger.exception(
                    "Failed to reconcile funding for %s", pos.symbol
                )

    async def _get_funding_payment(
        self,
        symbol: str,
        funding_time: datetime,
    ) -> Tuple[float, float]:
        """Query actual funding payment from Binance.

        Returns (actual_rate, actual_payment).
        """
        if self._paper_mode:
            return self._simulate_funding_payment(symbol, funding_time)

        # Query /fapi/v1/income for FUNDING_FEE
        start_ms = int((funding_time - timedelta(minutes=5)).timestamp() * 1000)
        end_ms = int((funding_time + timedelta(minutes=5)).timestamp() * 1000)

        try:
            income_records = await self._client.get_income_history(
                symbol=symbol,
                income_type="FUNDING_FEE",
                start_time=start_ms,
                end_time=end_ms,
                limit=10,
            )

            if income_records:
                # Find the record closest to the funding time
                best = min(
                    income_records,
                    key=lambda r: abs(int(r.get("time", 0)) - int(funding_time.timestamp() * 1000)),
                )
                payment = float(best.get("income", 0))
                # For shorts, positive income means we received funding
                # Binance reports negative for shorts receiving funding
                # Negate it so positive = income for our strategy
                actual_payment = -payment  # Binance: negative = received by short

                # Get the actual rate from premium index
                premium = await self._client.get_premium_index(symbol=symbol)
                if isinstance(premium, dict):
                    actual_rate = float(premium.get("lastFundingRate", 0))
                elif isinstance(premium, list) and premium:
                    actual_rate = float(premium[0].get("lastFundingRate", 0))
                else:
                    actual_rate = 0.0

                return actual_rate, actual_payment

        except Exception:
            logger.exception("Failed to query funding payment for %s", symbol)

        return 0.0, 0.0

    def _simulate_funding_payment(
        self,
        symbol: str,
        funding_time: datetime,
    ) -> Tuple[float, float]:
        """Simulate funding payment for paper trading (Section 9.1).

        Uses the ACTUAL funding rate from Binance (via markPrice stream)
        to calculate the simulated payment.
        """
        inst = self._strategy.instruments.get(symbol)
        pos = None
        for p in self._strategy.positions.values():
            if p.symbol == symbol:
                pos = p
                break

        if inst is None or pos is None:
            return 0.0, 0.0

        # Use the actual current funding rate (not predicted)
        actual_rate = inst.current_funding_rate

        # Payment = Position Notional x Funding Rate
        # For short position: if rate > 0, shorts receive payment
        payment = pos.futures_notional * actual_rate

        logger.info(
            "PAPER FUNDING: %s rate=%.6f notional=%.2f payment=%.6f",
            symbol, actual_rate, pos.futures_notional, payment,
        )

        return actual_rate, payment

    # ══════════════════════════════════════════════════════════════════════
    #  Warm-up: fetch missed funding during downtime (Section 8.2)
    # ══════════════════════════════════════════════════════════════════════

    async def reconcile_downtime(self, last_check_ms: int) -> None:
        """Reconcile funding payments received during bot downtime.

        Queries /fapi/v1/income for FUNDING_FEE from last_check_ms to now.
        """
        if self._paper_mode:
            logger.info("Paper mode: skipping downtime funding reconciliation")
            return

        now_ms = int(time.time() * 1000)
        if last_check_ms <= 0:
            last_check_ms = now_ms - 86_400_000  # Default: last 24h

        for symbol in self._strategy.instruments:
            try:
                income_records = await self._client.get_income_history(
                    symbol=symbol,
                    income_type="FUNDING_FEE",
                    start_time=last_check_ms,
                    end_time=now_ms,
                    limit=100,
                )

                missed_total = 0.0
                for record in income_records:
                    payment = -float(record.get("income", 0))
                    missed_total += payment
                    ts = int(record.get("time", 0))
                    # Ingest as historical funding
                    self._strategy.ingest_funding_rate(
                        symbol, 0.0, ts, predicted=False
                    )

                if missed_total != 0:
                    self._cumulative_income[symbol] += missed_total
                    logger.info(
                        "Downtime funding reconciliation: %s received %.6f USDT "
                        "(%d periods)",
                        symbol, missed_total, len(income_records),
                    )

            except Exception:
                logger.exception(
                    "Failed to reconcile downtime funding for %s", symbol
                )

    # ══════════════════════════════════════════════════════════════════════
    #  Metrics and state
    # ══════════════════════════════════════════════════════════════════════

    def get_cumulative_income(self) -> Dict[str, float]:
        """Return cumulative funding income per instrument."""
        return dict(self._cumulative_income)

    def get_daily_income(self) -> Dict[str, Dict[str, float]]:
        """Return daily funding income breakdown."""
        return {k: dict(v) for k, v in self._daily_income.items()}

    def get_settlement_history(
        self, symbol: Optional[str] = None, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Return recent settlement records."""
        if symbol:
            records = self._settlements.get(symbol, [])
        else:
            records = []
            for sym_records in self._settlements.values():
                records.extend(sym_records)
            records.sort(key=lambda r: r.timestamp_ms)

        recent = records[-limit:]
        return [
            {
                "symbol": r.symbol,
                "timestamp_ms": r.timestamp_ms,
                "actual_rate": r.actual_rate,
                "expected_rate": r.expected_rate,
                "actual_payment": r.actual_payment,
                "expected_payment": r.expected_payment,
                "discrepancy_pct": r.discrepancy_pct,
            }
            for r in recent
        ]

    def get_funding_metrics(self) -> Dict[str, Any]:
        """Return funding tracker metrics for dashboard."""
        total_income = sum(self._cumulative_income.values())
        total_settlements = sum(len(v) for v in self._settlements.values())

        # Average capture rate
        all_records = []
        for records in self._settlements.values():
            all_records.extend(records)

        avg_rate = 0.0
        positive_count = 0
        if all_records:
            rates = [r.actual_rate for r in all_records]
            avg_rate = sum(rates) / len(rates)
            positive_count = sum(1 for r in rates if r > 0)

        return {
            "total_funding_income": total_income,
            "per_instrument_income": dict(self._cumulative_income),
            "total_settlements_tracked": total_settlements,
            "average_funding_rate_captured": avg_rate,
            "funding_win_loss_pct": (
                positive_count / len(all_records) * 100
                if all_records else 0.0
            ),
            "next_funding_time": self._next_funding_time().isoformat(),
        }

    def get_state(self) -> Dict[str, Any]:
        """Serialize for persistence."""
        return {
            "cumulative_income": dict(self._cumulative_income),
            "daily_income": {k: dict(v) for k, v in self._daily_income.items()},
        }

    def restore_state(self, state: Dict[str, Any]) -> None:
        """Restore from persistence."""
        for sym, income in state.get("cumulative_income", {}).items():
            self._cumulative_income[sym] = income
        for day, sym_income in state.get("daily_income", {}).items():
            for sym, income in sym_income.items():
                self._daily_income[day][sym] = income
        logger.info(
            "FundingTracker state restored: %d instruments tracked",
            len(self._cumulative_income),
        )

    # ══════════════════════════════════════════════════════════════════════
    #  Helpers
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _next_funding_time() -> datetime:
        """Calculate the next funding settlement time."""
        now = datetime.now(timezone.utc)

        for hour in FUNDING_HOURS:
            candidate = now.replace(hour=hour, minute=0, second=0, microsecond=0)
            if candidate > now:
                return candidate

        # Next day 00:00 UTC
        tomorrow = now + timedelta(days=1)
        return tomorrow.replace(hour=0, minute=0, second=0, microsecond=0)

    @staticmethod
    def seconds_until_next_funding() -> float:
        """Seconds until the next funding settlement."""
        now = datetime.now(timezone.utc)
        next_time = FundingTracker._next_funding_time()
        return (next_time - now).total_seconds()

    @staticmethod
    def is_within_funding_exclusion_zone() -> bool:
        """Clock drift protection: return True if within 5 minutes of any
        funding timestamp (00:00/08:00/16:00 UTC).

        Never execute trades within this window to avoid funding rate
        settlement race conditions and clock drift risks.
        """
        now = datetime.now(timezone.utc)
        exclusion = CLOCK_DRIFT_EXCLUSION_MINUTES

        for hour in FUNDING_HOURS:
            funding_time = now.replace(hour=hour, minute=0, second=0, microsecond=0)

            # Check within +/- exclusion minutes of this funding time
            delta_seconds = abs((now - funding_time).total_seconds())
            # Handle wrap-around for times near midnight
            if delta_seconds > 43200:  # More than 12 hours -> wrap
                delta_seconds = 86400 - delta_seconds

            if delta_seconds <= exclusion * 60:
                logger.warning(
                    "CLOCK DRIFT PROTECTION: Within %d minutes of funding "
                    "timestamp %02d:00 UTC (delta=%.0fs). Trade execution blocked.",
                    exclusion, hour, delta_seconds,
                )
                return True

        return False
