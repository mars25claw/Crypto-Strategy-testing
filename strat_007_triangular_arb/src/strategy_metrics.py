"""STRAT-007: Strategy-specific performance metrics (Section 10.2 & 10.3).

Tracks and computes all metrics required by the build instructions:

Section 10.2 — Strategy-Specific Metrics:
- Opportunities Detected/Hour
- Opportunities Taken vs Skipped (with reasons)
- Execution Success Rate
- Average Net Profit Per Trade (USDT and %)
- Latency Breakdown (detection-to-order, order-to-fill, total cycle)
- Slippage Analysis per leg
- Wallet Balance Drift
- Stale Opportunity Rate
- Mode A vs Mode B Attribution
- Time of Day Analysis
- Profit Per Dollar of Volume

Section 10.3 — Dimensional Breakdowns:
- By mode (A vs B)
- By symbol / triangle path
- By time of day (hour buckets)
- By fee tier / BNB discount state

Section 9.3 — Go-Live Criteria:
- 45-day minimum paper trading
- > 80% win rate
- > 0.03% avg net profit per trade
- No unhedged position > 60s
- Sub-200ms latency consistently
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trade record for metrics
# ---------------------------------------------------------------------------

@dataclass
class TradeRecord:
    """Single arb execution record for metrics analysis."""

    timestamp: float = 0.0
    mode: str = ""              # "spot_futures" or "triangular"
    symbol: str = ""            # Asset (Mode A) or path_id (Mode B)
    success: bool = False
    profit_usdt: float = 0.0
    profit_pct: float = 0.0
    fees_usdt: float = 0.0
    volume_usdt: float = 0.0    # Trade size
    legs_filled: int = 0
    legs_total: int = 0
    execution_time_ms: float = 0.0
    detection_to_order_ms: float = 0.0
    order_to_fill_ms: float = 0.0
    slippage_usdt: float = 0.0
    leg_slippages: List[float] = field(default_factory=list)  # Per-leg slippage
    bnb_discount_active: bool = False
    was_stale: bool = False
    hour_utc: int = 0


@dataclass
class SkipRecord:
    """Record of a skipped opportunity for analysis."""

    timestamp: float = 0.0
    mode: str = ""
    symbol: str = ""
    reason: str = ""
    hour_utc: int = 0


# ---------------------------------------------------------------------------
# Strategy metrics engine
# ---------------------------------------------------------------------------

class StrategyMetrics:
    """Comprehensive metrics tracker for STRAT-007.

    Collects trade records, skip records, and wallet snapshots to compute
    all Section 10.2 and 10.3 metrics on demand.

    Parameters
    ----------
    max_history : int
        Maximum trade records to retain (default: 10000).
    """

    def __init__(self, max_history: int = 10000) -> None:
        # Trade history
        self._trades: Deque[TradeRecord] = deque(maxlen=max_history)
        self._skips: Deque[SkipRecord] = deque(maxlen=max_history)

        # Opportunity detection timestamps (for per-hour rate)
        self._detection_times: Deque[float] = deque(maxlen=100000)

        # Wallet balance snapshots for drift tracking
        self._wallet_snapshots: Deque[Dict[str, float]] = deque(maxlen=8640)  # ~1 day at 10s

        # Latency breakdown samples
        self._latency_detection_to_order: Deque[float] = deque(maxlen=5000)
        self._latency_order_to_fill: Deque[float] = deque(maxlen=5000)
        self._latency_total_cycle: Deque[float] = deque(maxlen=5000)

        # Go-live tracking
        self._paper_start_time: float = time.time()
        self._max_unhedged_duration_s: float = 0.0

        # Counters
        self._total_detected = 0
        self._total_taken = 0
        self._total_skipped = 0

        logger.info("StrategyMetrics initialized: max_history=%d", max_history)

    # ------------------------------------------------------------------
    # Recording methods
    # ------------------------------------------------------------------

    def record_detection(self) -> None:
        """Record that an opportunity was detected (for hourly rate)."""
        self._detection_times.append(time.time())
        self._total_detected += 1

    def record_trade(self, record: TradeRecord) -> None:
        """Record a completed arb execution."""
        record.hour_utc = datetime.fromtimestamp(
            record.timestamp, tz=timezone.utc
        ).hour
        self._trades.append(record)
        self._total_taken += 1

        # Track latency breakdowns
        if record.detection_to_order_ms > 0:
            self._latency_detection_to_order.append(record.detection_to_order_ms)
        if record.order_to_fill_ms > 0:
            self._latency_order_to_fill.append(record.order_to_fill_ms)
        if record.execution_time_ms > 0:
            self._latency_total_cycle.append(record.execution_time_ms)

    def record_skip(self, mode: str, symbol: str, reason: str) -> None:
        """Record that an opportunity was skipped."""
        now = time.time()
        rec = SkipRecord(
            timestamp=now,
            mode=mode,
            symbol=symbol,
            reason=reason,
            hour_utc=datetime.fromtimestamp(now, tz=timezone.utc).hour,
        )
        self._skips.append(rec)
        self._total_skipped += 1

    def record_wallet_snapshot(
        self, spot_usdt: float, futures_usdt: float, total_usdt: float,
    ) -> None:
        """Record a wallet balance snapshot for drift analysis."""
        self._wallet_snapshots.append({
            "timestamp": time.time(),
            "spot_usdt": spot_usdt,
            "futures_usdt": futures_usdt,
            "total_usdt": total_usdt,
        })

    def record_unhedged_duration(self, duration_s: float) -> None:
        """Track maximum unhedged position duration (for go-live criteria)."""
        if duration_s > self._max_unhedged_duration_s:
            self._max_unhedged_duration_s = duration_s

    # ------------------------------------------------------------------
    # Section 10.2: Strategy-Specific Metrics
    # ------------------------------------------------------------------

    def opportunities_detected_per_hour(self) -> float:
        """Raw count of detected arb opportunities in the last hour."""
        now = time.time()
        cutoff = now - 3600
        return sum(1 for t in self._detection_times if t > cutoff)

    def opportunities_taken_vs_skipped(self) -> Dict[str, Any]:
        """Ratio and reasons for skipping."""
        total = self._total_taken + self._total_skipped
        ratio = self._total_taken / total if total > 0 else 0.0

        # Aggregate skip reasons
        reason_counts: Dict[str, int] = defaultdict(int)
        for skip in self._skips:
            reason_counts[skip.reason] += 1

        return {
            "taken": self._total_taken,
            "skipped": self._total_skipped,
            "total": total,
            "taken_ratio": round(ratio, 4),
            "skip_reasons": dict(reason_counts),
        }

    def execution_success_rate(self) -> float:
        """Percentage of arb attempts where all legs filled."""
        if not self._trades:
            return 0.0
        successes = sum(1 for t in self._trades if t.success)
        return round(successes / len(self._trades) * 100.0, 2)

    def avg_net_profit_per_trade(self) -> Dict[str, float]:
        """Average net profit per trade in USDT and percentage."""
        successful = [t for t in self._trades if t.success]
        if not successful:
            return {"usdt": 0.0, "pct": 0.0}
        avg_usdt = sum(t.profit_usdt for t in successful) / len(successful)
        avg_pct = sum(t.profit_pct for t in successful) / len(successful)
        return {"usdt": round(avg_usdt, 6), "pct": round(avg_pct, 6)}

    def latency_breakdown(self) -> Dict[str, Dict[str, float]]:
        """Latency breakdown: detection-to-order, order-to-fill, total cycle."""
        def _stats(samples: Deque[float]) -> Dict[str, float]:
            if not samples:
                return {"avg_ms": 0, "min_ms": 0, "max_ms": 0, "p50_ms": 0, "p95_ms": 0}
            s = sorted(samples)
            n = len(s)
            return {
                "avg_ms": round(sum(s) / n, 1),
                "min_ms": round(s[0], 1),
                "max_ms": round(s[-1], 1),
                "p50_ms": round(s[n // 2], 1),
                "p95_ms": round(s[int(n * 0.95)], 1),
            }

        return {
            "detection_to_order": _stats(self._latency_detection_to_order),
            "order_to_fill": _stats(self._latency_order_to_fill),
            "total_cycle": _stats(self._latency_total_cycle),
        }

    def slippage_analysis(self) -> Dict[str, Any]:
        """Slippage analysis per leg and aggregate."""
        if not self._trades:
            return {"avg_slippage_usdt": 0, "total_slippage_usdt": 0, "per_leg": []}

        total_slip = sum(t.slippage_usdt for t in self._trades)
        avg_slip = total_slip / len(self._trades)

        # Per-leg aggregation
        leg_slips: Dict[int, List[float]] = defaultdict(list)
        for t in self._trades:
            for i, s in enumerate(t.leg_slippages):
                leg_slips[i].append(s)

        per_leg = {}
        for leg_idx, slips in sorted(leg_slips.items()):
            per_leg[f"leg_{leg_idx + 1}"] = {
                "avg_usdt": round(sum(slips) / len(slips), 6) if slips else 0,
                "max_usdt": round(max(slips), 6) if slips else 0,
                "samples": len(slips),
            }

        return {
            "avg_slippage_usdt": round(avg_slip, 6),
            "total_slippage_usdt": round(total_slip, 4),
            "per_leg": per_leg,
        }

    def wallet_balance_drift(self) -> Dict[str, float]:
        """How much capital has migrated between wallets over time."""
        if len(self._wallet_snapshots) < 2:
            return {"spot_drift_usdt": 0, "futures_drift_usdt": 0, "drift_pct": 0}

        first = self._wallet_snapshots[0]
        last = self._wallet_snapshots[-1]

        spot_drift = last["spot_usdt"] - first["spot_usdt"]
        futures_drift = last["futures_usdt"] - first["futures_usdt"]
        total = last.get("total_usdt", 1) or 1
        drift_pct = abs(spot_drift) / total * 100.0

        return {
            "spot_drift_usdt": round(spot_drift, 4),
            "futures_drift_usdt": round(futures_drift, 4),
            "drift_pct": round(drift_pct, 2),
        }

    def stale_opportunity_rate(self) -> float:
        """Percentage of taken opportunities that were stale by execution time."""
        if not self._trades:
            return 0.0
        stale = sum(1 for t in self._trades if t.was_stale)
        return round(stale / len(self._trades) * 100.0, 2)

    def mode_attribution(self) -> Dict[str, Dict[str, Any]]:
        """P&L breakdown by arb mode (Mode A vs Mode B)."""
        result: Dict[str, Dict[str, Any]] = {}
        for mode_val in ("spot_futures", "triangular"):
            mode_trades = [t for t in self._trades if t.mode == mode_val]
            total_profit = sum(t.profit_usdt for t in mode_trades)
            total_volume = sum(t.volume_usdt for t in mode_trades)
            wins = sum(1 for t in mode_trades if t.success and t.profit_usdt > 0)
            count = len(mode_trades)

            result[mode_val] = {
                "count": count,
                "total_profit_usdt": round(total_profit, 4),
                "total_volume_usdt": round(total_volume, 2),
                "win_rate": round(wins / count * 100.0, 2) if count > 0 else 0,
                "avg_profit_pct": round(
                    sum(t.profit_pct for t in mode_trades) / count, 6
                ) if count > 0 else 0,
            }

        return result

    def time_of_day_analysis(self) -> Dict[int, Dict[str, Any]]:
        """When are opportunities most frequent? Bucketed by UTC hour."""
        hourly: Dict[int, Dict[str, Any]] = {}
        for h in range(24):
            h_trades = [t for t in self._trades if t.hour_utc == h]
            h_detections = sum(
                1 for t in self._detection_times
                if datetime.fromtimestamp(t, tz=timezone.utc).hour == h
            )
            total_profit = sum(t.profit_usdt for t in h_trades)
            count = len(h_trades)
            hourly[h] = {
                "detections": h_detections,
                "executions": count,
                "total_profit_usdt": round(total_profit, 4),
                "avg_profit_usdt": round(total_profit / count, 6) if count > 0 else 0,
            }
        return hourly

    def profit_per_dollar_of_volume(self) -> float:
        """Efficiency measure: thin margins x high volume."""
        successful = [t for t in self._trades if t.success]
        if not successful:
            return 0.0
        total_profit = sum(t.profit_usdt for t in successful)
        total_volume = sum(t.volume_usdt for t in successful) or 1
        return round(total_profit / total_volume, 8)

    # ------------------------------------------------------------------
    # Section 9.3: Go-Live Criteria
    # ------------------------------------------------------------------

    def go_live_criteria(self) -> Dict[str, Any]:
        """Evaluate go-live criteria (Section 9.3).

        - 45-day minimum paper trading
        - Win rate > 80%
        - Avg net profit per trade > 0.03%
        - No unhedged position > 60s
        - Sub-200ms latency consistently
        """
        elapsed_days = (time.time() - self._paper_start_time) / 86400.0
        successful = [t for t in self._trades if t.success]
        total = len(successful)

        win_count = sum(1 for t in successful if t.profit_usdt > 0)
        win_rate = (win_count / total * 100.0) if total > 0 else 0.0

        avg_profit = self.avg_net_profit_per_trade()
        avg_profit_pct = avg_profit["pct"]

        latency_stats = self.latency_breakdown()
        avg_latency = latency_stats["total_cycle"]["avg_ms"]
        p95_latency = latency_stats["total_cycle"]["p95_ms"]

        criteria = {
            "min_days": {
                "required": 45,
                "actual": round(elapsed_days, 1),
                "passed": elapsed_days >= 45,
            },
            "win_rate": {
                "required_pct": 80.0,
                "actual_pct": round(win_rate, 2),
                "passed": win_rate > 80.0,
            },
            "avg_net_profit": {
                "required_pct": 0.03,
                "actual_pct": round(avg_profit_pct, 6),
                "passed": avg_profit_pct > 0.03,
            },
            "max_unhedged_duration": {
                "required_s": 60,
                "actual_s": round(self._max_unhedged_duration_s, 1),
                "passed": self._max_unhedged_duration_s <= 60,
            },
            "latency": {
                "required_ms": 200,
                "avg_ms": round(avg_latency, 1),
                "p95_ms": round(p95_latency, 1),
                "passed": p95_latency < 200 if p95_latency > 0 else True,
            },
            "total_trades": total,
            "all_passed": False,  # Set below
        }

        criteria["all_passed"] = all(
            criteria[k]["passed"]
            for k in ("min_days", "win_rate", "avg_net_profit",
                      "max_unhedged_duration", "latency")
        )

        return criteria

    # ------------------------------------------------------------------
    # Section 10.3: Dimensional Breakdowns
    # ------------------------------------------------------------------

    def dimensional_breakdown_by_symbol(self) -> Dict[str, Dict[str, Any]]:
        """Breakdown by symbol / triangle path."""
        symbols: Dict[str, List[TradeRecord]] = defaultdict(list)
        for t in self._trades:
            symbols[t.symbol].append(t)

        result = {}
        for sym, trades in symbols.items():
            profit = sum(t.profit_usdt for t in trades)
            wins = sum(1 for t in trades if t.success and t.profit_usdt > 0)
            count = len(trades)
            result[sym] = {
                "count": count,
                "total_profit_usdt": round(profit, 4),
                "win_rate": round(wins / count * 100.0, 2) if count > 0 else 0,
                "avg_execution_ms": round(
                    sum(t.execution_time_ms for t in trades) / count, 1
                ) if count > 0 else 0,
                "avg_slippage_usdt": round(
                    sum(t.slippage_usdt for t in trades) / count, 6
                ) if count > 0 else 0,
            }
        return result

    def dimensional_breakdown_by_hour(self) -> Dict[int, Dict[str, Any]]:
        """Alias for time_of_day_analysis."""
        return self.time_of_day_analysis()

    def dimensional_breakdown_by_fee_state(self) -> Dict[str, Dict[str, Any]]:
        """Breakdown by BNB discount state."""
        with_bnb = [t for t in self._trades if t.bnb_discount_active]
        without_bnb = [t for t in self._trades if not t.bnb_discount_active]

        def _summary(trades: List[TradeRecord]) -> Dict[str, Any]:
            if not trades:
                return {"count": 0, "total_profit_usdt": 0, "avg_profit_pct": 0}
            profit = sum(t.profit_usdt for t in trades)
            avg_pct = sum(t.profit_pct for t in trades) / len(trades)
            return {
                "count": len(trades),
                "total_profit_usdt": round(profit, 4),
                "avg_profit_pct": round(avg_pct, 6),
                "total_fees_usdt": round(sum(t.fees_usdt for t in trades), 4),
            }

        return {
            "with_bnb_discount": _summary(with_bnb),
            "without_bnb_discount": _summary(without_bnb),
        }

    # ------------------------------------------------------------------
    # Aggregate report
    # ------------------------------------------------------------------

    def get_full_report(self) -> Dict[str, Any]:
        """Return all Section 10.2, 10.3 metrics, and go-live criteria."""
        return {
            # Section 10.2
            "opportunities_per_hour": self.opportunities_detected_per_hour(),
            "taken_vs_skipped": self.opportunities_taken_vs_skipped(),
            "execution_success_rate": self.execution_success_rate(),
            "avg_net_profit": self.avg_net_profit_per_trade(),
            "latency_breakdown": self.latency_breakdown(),
            "slippage_analysis": self.slippage_analysis(),
            "wallet_balance_drift": self.wallet_balance_drift(),
            "stale_opportunity_rate": self.stale_opportunity_rate(),
            "mode_attribution": self.mode_attribution(),
            "time_of_day": self.time_of_day_analysis(),
            "profit_per_dollar_of_volume": self.profit_per_dollar_of_volume(),
            # Section 10.3 — Dimensional Breakdowns
            "by_symbol": self.dimensional_breakdown_by_symbol(),
            "by_hour": self.dimensional_breakdown_by_hour(),
            "by_fee_state": self.dimensional_breakdown_by_fee_state(),
            # Section 9.3 — Go-Live Criteria
            "go_live_criteria": self.go_live_criteria(),
        }

    def get_dashboard_summary(self) -> Dict[str, Any]:
        """Compact summary suitable for real-time dashboard push."""
        avg_profit = self.avg_net_profit_per_trade()
        return {
            "opportunities_per_hour": self.opportunities_detected_per_hour(),
            "taken": self._total_taken,
            "skipped": self._total_skipped,
            "success_rate": self.execution_success_rate(),
            "avg_profit_usdt": avg_profit["usdt"],
            "avg_profit_pct": avg_profit["pct"],
            "stale_rate": self.stale_opportunity_rate(),
            "profit_per_volume": self.profit_per_dollar_of_volume(),
            "mode_a_profit": self.mode_attribution().get("spot_futures", {}).get("total_profit_usdt", 0),
            "mode_b_profit": self.mode_attribution().get("triangular", {}).get("total_profit_usdt", 0),
            "go_live_all_passed": self.go_live_criteria()["all_passed"],
        }
