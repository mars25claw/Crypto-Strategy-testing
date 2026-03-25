"""Shared risk manager enforcing cross-strategy risk rules.

Integrates drawdown tracking, correlation-based position limits,
consecutive-loss size reduction, and cross-strategy exposure checks.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from shared.config_loader import RiskConfig

logger = logging.getLogger(__name__)
trade_logger = logging.getLogger("trade")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PositionRecord:
    """In-memory record of an open position tracked by the risk manager."""

    strategy_id: str
    symbol: str
    direction: str          # "LONG" or "SHORT"
    size_usdt: float
    leverage: int = 1
    opened_at: float = field(default_factory=time.time)


@dataclass
class ExposureSummary:
    """Snapshot of current exposure metrics."""

    total_exposure_usdt: float = 0.0
    total_exposure_pct: float = 0.0
    long_exposure_usdt: float = 0.0
    long_exposure_pct: float = 0.0
    short_exposure_usdt: float = 0.0
    short_exposure_pct: float = 0.0
    net_directional_usdt: float = 0.0
    net_directional_pct: float = 0.0
    per_asset: Dict[str, float] = field(default_factory=dict)
    per_strategy: Dict[str, float] = field(default_factory=dict)
    position_count: int = 0


# ---------------------------------------------------------------------------
# Correlation matrix
# ---------------------------------------------------------------------------

class CorrelationMatrix:
    """Maintains a 30-day rolling correlation matrix between assets.

    Updated weekly via :meth:`update` with close-price arrays keyed by symbol.
    """

    WINDOW_DAYS: int = 30
    HIGH_CORRELATION_THRESHOLD: float = 0.75

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # symbol -> list of daily close prices (most recent WINDOW_DAYS entries)
        self._prices: Dict[str, List[float]] = {}
        # (sym_a, sym_b) -> correlation  (sorted key, so a < b)
        self._correlations: Dict[Tuple[str, str], float] = {}
        self._last_update: float = 0.0

    # -- public API ----------------------------------------------------------

    def update(self, daily_closes: Dict[str, List[float]]) -> None:
        """Recompute correlation matrix from the latest daily close arrays.

        Parameters
        ----------
        daily_closes:
            Mapping of symbol -> list[float] of the most recent N daily closes.
        """
        with self._lock:
            self._prices = {
                sym: closes[-self.WINDOW_DAYS:]
                for sym, closes in daily_closes.items()
                if len(closes) >= 2
            }
            self._recompute()
            self._last_update = time.time()

    def get_correlation(self, sym_a: str, sym_b: str) -> float:
        """Return the pairwise correlation between two symbols (0.0 if unknown)."""
        if sym_a == sym_b:
            return 1.0
        key = (min(sym_a, sym_b), max(sym_a, sym_b))
        with self._lock:
            return self._correlations.get(key, 0.0)

    def get_highly_correlated(self, symbol: str, threshold: float | None = None) -> List[str]:
        """Return symbols with correlation > *threshold* to *symbol*."""
        threshold = threshold or self.HIGH_CORRELATION_THRESHOLD
        result: List[str] = []
        with self._lock:
            for (a, b), corr in self._correlations.items():
                if corr > threshold:
                    if a == symbol:
                        result.append(b)
                    elif b == symbol:
                        result.append(a)
        return result

    def needs_update(self, interval_seconds: float = 7 * 86400) -> bool:
        """Return True when more than *interval_seconds* have elapsed since last update."""
        return (time.time() - self._last_update) > interval_seconds

    # -- internals -----------------------------------------------------------

    def _recompute(self) -> None:
        symbols = sorted(self._prices.keys())
        self._correlations.clear()
        if len(symbols) < 2:
            return

        # Align to equal length
        min_len = min(len(self._prices[s]) for s in symbols)
        if min_len < 5:
            return

        arrays = {s: np.array(self._prices[s][-min_len:]) for s in symbols}

        # Compute log returns
        returns: Dict[str, np.ndarray] = {}
        for s, arr in arrays.items():
            with np.errstate(divide="ignore", invalid="ignore"):
                r = np.diff(np.log(arr))
            if np.any(np.isnan(r)) or np.any(np.isinf(r)):
                continue
            returns[s] = r

        ret_symbols = sorted(returns.keys())
        for i, sa in enumerate(ret_symbols):
            for sb in ret_symbols[i + 1:]:
                corr = float(np.corrcoef(returns[sa], returns[sb])[0, 1])
                if np.isnan(corr):
                    corr = 0.0
                key = (min(sa, sb), max(sa, sb))
                self._correlations[key] = corr


# ---------------------------------------------------------------------------
# DrawdownState  (in-memory companion to the DB DrawdownTracker row)
# ---------------------------------------------------------------------------

class DrawdownState:
    """In-memory drawdown tracking with daily/weekly/monthly windows."""

    def __init__(
        self,
        peak_equity: float,
        current_equity: float,
        daily_start: float,
        weekly_start: float,
        monthly_start: float,
    ) -> None:
        self.peak_equity = peak_equity
        self.current_equity = current_equity
        self.daily_start_equity = daily_start
        self.weekly_start_equity = weekly_start
        self.monthly_start_equity = monthly_start

    # -- drawdown percentages (positive means loss) --------------------------

    @property
    def overall_drawdown_pct(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return max(0.0, (self.peak_equity - self.current_equity) / self.peak_equity * 100.0)

    @property
    def daily_drawdown_pct(self) -> float:
        if self.daily_start_equity <= 0:
            return 0.0
        return max(0.0, (self.daily_start_equity - self.current_equity) / self.daily_start_equity * 100.0)

    @property
    def weekly_drawdown_pct(self) -> float:
        if self.weekly_start_equity <= 0:
            return 0.0
        return max(0.0, (self.weekly_start_equity - self.current_equity) / self.weekly_start_equity * 100.0)

    @property
    def monthly_drawdown_pct(self) -> float:
        if self.monthly_start_equity <= 0:
            return 0.0
        return max(0.0, (self.monthly_start_equity - self.current_equity) / self.monthly_start_equity * 100.0)

    def update_equity(self, equity: float) -> None:
        self.current_equity = equity
        if equity > self.peak_equity:
            self.peak_equity = equity

    def reset_daily(self) -> None:
        self.daily_start_equity = self.current_equity

    def reset_weekly(self) -> None:
        self.weekly_start_equity = self.current_equity

    def reset_monthly(self) -> None:
        self.monthly_start_equity = self.current_equity

    def to_dict(self) -> dict:
        return {
            "peak_equity": self.peak_equity,
            "current_equity": self.current_equity,
            "drawdown_pct": self.overall_drawdown_pct,
            "daily_drawdown_pct": self.daily_drawdown_pct,
            "weekly_drawdown_pct": self.weekly_drawdown_pct,
            "monthly_drawdown_pct": self.monthly_drawdown_pct,
            "daily_start_equity": self.daily_start_equity,
            "weekly_start_equity": self.weekly_start_equity,
            "monthly_start_equity": self.monthly_start_equity,
        }


# ---------------------------------------------------------------------------
# Cross-strategy position reader
# ---------------------------------------------------------------------------

class CrossStrategyReader:
    """Reads other strategies' position files from a shared state directory.

    Each strategy writes ``data/state/<strategy_id>/positions.json`` which is
    a JSON list of objects with ``symbol``, ``direction``, ``size_usdt``,
    ``leverage``.
    """

    def __init__(self, state_dir: str = "data/state") -> None:
        self._state_dir = Path(state_dir)

    def get_all_positions(self, exclude_strategy: str | None = None) -> List[Dict[str, Any]]:
        """Return position dicts from every strategy except *exclude_strategy*."""
        positions: List[Dict[str, Any]] = []
        if not self._state_dir.exists():
            return positions

        for strategy_dir in self._state_dir.iterdir():
            if not strategy_dir.is_dir():
                continue
            if exclude_strategy and strategy_dir.name == exclude_strategy:
                continue
            pos_file = strategy_dir / "positions.json"
            if not pos_file.exists():
                continue
            try:
                data = json.loads(pos_file.read_text())
                if isinstance(data, list):
                    for p in data:
                        p["strategy_id"] = strategy_dir.name
                    positions.extend(data)
            except Exception as exc:
                logger.warning("Failed to read positions for %s: %s", strategy_dir.name, exc)

        return positions

    def get_total_exposure(self, exclude_strategy: str | None = None) -> Dict[str, float]:
        """Return per-symbol total exposure (USDT) from all other strategies."""
        exposure: Dict[str, float] = defaultdict(float)
        for pos in self.get_all_positions(exclude_strategy=exclude_strategy):
            exposure[pos.get("symbol", "UNKNOWN")] += abs(pos.get("size_usdt", 0.0))
        return dict(exposure)


# ---------------------------------------------------------------------------
# RiskManager
# ---------------------------------------------------------------------------

class RiskManager:
    """Shared risk manager enforcing cross-strategy risk rules.

    Parameters
    ----------
    config:
        :class:`~shared.config_loader.RiskConfig` with all thresholds.
    database_manager:
        A :class:`~shared.database.DatabaseManager` instance (or None for paper mode).
    cross_strategy_reader:
        A :class:`CrossStrategyReader` (or compatible) for reading other strategies' positions.
    """

    # Consecutive-loss size reduction schedule: {loss_count: multiplier}
    SIZE_REDUCTION_SCHEDULE: Dict[int, float] = {
        3: 0.75,    # after 3 consecutive losses reduce size to 75%
        5: 0.50,    # after 5 consecutive losses reduce to 50%
        7: 0.25,    # after 7 reduce to 25%
        10: 0.0,    # after 10 halt trading (size = 0)
    }

    def __init__(
        self,
        config: RiskConfig,
        database_manager: Any = None,
        cross_strategy_reader: CrossStrategyReader | None = None,
    ) -> None:
        self._config = config
        self._db = database_manager
        self._cross_reader = cross_strategy_reader or CrossStrategyReader()

        self._lock = threading.Lock()

        # In-memory position tracking  strategy_id -> {symbol -> PositionRecord}
        self._positions: Dict[str, Dict[str, PositionRecord]] = defaultdict(dict)

        # Consecutive losses per strategy
        self._consecutive_losses: Dict[str, int] = defaultdict(int)

        # Trade results history (for streak tracking)
        self._trade_results: Dict[str, List[bool]] = defaultdict(list)

        # Drawdown state (initialised on first update_equity call)
        self._drawdown: Optional[DrawdownState] = None
        self._current_equity: float = 0.0

        # Correlation
        self.correlation_matrix = CorrelationMatrix()

        logger.info(
            "RiskManager initialised: max_capital=%.1f%%, max_per_trade=%.1f%%, "
            "max_concurrent=%d, daily_dd=%.1f%%, system_dd=%.1f%%",
            config.max_capital_pct,
            config.max_per_trade_pct,
            config.max_concurrent_positions,
            config.daily_drawdown_pct,
            config.system_wide_drawdown_pct,
        )

    # ======================================================================
    #  Core entry gate
    # ======================================================================

    def check_entry_allowed(
        self,
        strategy_id: str,
        symbol: str,
        direction: str,
        size_usdt: float,
        leverage: int,
    ) -> Tuple[bool, str]:
        """Evaluate whether a new position entry is allowed.

        Returns
        -------
        (allowed, reason)
            *allowed* is True if the trade may proceed.
            *reason* is an empty string on success or an explanation when rejected.
        """
        with self._lock:
            equity = self._current_equity
            if equity <= 0:
                return False, "Equity not initialised (call update_equity first)"

            # 1. Per-trade size limit
            max_trade = equity * (self._config.max_per_trade_pct / 100.0)
            if size_usdt > max_trade:
                return False, (
                    f"Trade size {size_usdt:.2f} exceeds max_per_trade "
                    f"({self._config.max_per_trade_pct}% of equity = {max_trade:.2f})"
                )

            # 2. Leverage limit
            if leverage > self._config.max_leverage:
                return False, (
                    f"Leverage {leverage}x exceeds max_leverage {self._config.max_leverage}x"
                )

            # 3. Total strategy exposure
            exposure = self._compute_exposure_unlocked()
            new_total = exposure.total_exposure_usdt + size_usdt
            max_total = equity * (self._config.max_capital_pct / 100.0)
            if new_total > max_total:
                return False, (
                    f"Total exposure {new_total:.2f} would exceed max_capital "
                    f"({self._config.max_capital_pct}% of equity = {max_total:.2f})"
                )

            # 4. Per-asset exposure
            asset_exp = exposure.per_asset.get(symbol, 0.0) + size_usdt
            max_asset = equity * (self._config.max_per_asset_pct / 100.0)
            if asset_exp > max_asset:
                return False, (
                    f"Asset {symbol} exposure {asset_exp:.2f} would exceed max_per_asset "
                    f"({self._config.max_per_asset_pct}% = {max_asset:.2f})"
                )

            # 5. Directional exposure
            direction_upper = direction.upper()
            if direction_upper == "LONG":
                new_long = exposure.long_exposure_usdt + size_usdt
                max_long = equity * (self._config.max_long_exposure_pct / 100.0)
                if new_long > max_long:
                    return False, (
                        f"Long exposure {new_long:.2f} would exceed limit "
                        f"({self._config.max_long_exposure_pct}% = {max_long:.2f})"
                    )
            elif direction_upper == "SHORT":
                new_short = exposure.short_exposure_usdt + size_usdt
                max_short = equity * (self._config.max_short_exposure_pct / 100.0)
                if new_short > max_short:
                    return False, (
                        f"Short exposure {new_short:.2f} would exceed limit "
                        f"({self._config.max_short_exposure_pct}% = {max_short:.2f})"
                    )

            # Net directional
            if direction_upper == "LONG":
                new_net = (exposure.long_exposure_usdt + size_usdt) - exposure.short_exposure_usdt
            else:
                new_net = exposure.long_exposure_usdt - (exposure.short_exposure_usdt + size_usdt)
            max_net = equity * (self._config.max_net_directional_pct / 100.0)
            if abs(new_net) > max_net:
                return False, (
                    f"Net directional {new_net:.2f} would exceed limit "
                    f"({self._config.max_net_directional_pct}% = {max_net:.2f})"
                )

            # 6. Max concurrent positions
            total_positions = sum(len(syms) for syms in self._positions.values())
            if total_positions >= self._config.max_concurrent_positions:
                return False, (
                    f"Already at max concurrent positions ({self._config.max_concurrent_positions})"
                )

            # 7. Correlation check
            corr_reject = self._check_correlation_risk(symbol, direction_upper)
            if corr_reject:
                return False, corr_reject

            # 8. Cross-strategy exposure check
            cross_exposure = self._cross_reader.get_total_exposure(exclude_strategy=strategy_id)
            cross_asset = cross_exposure.get(symbol, 0.0) + asset_exp
            # Use 2x the per-asset limit as the system-wide per-asset cap
            system_asset_max = equity * (self._config.max_per_asset_pct / 100.0) * 2.0
            if cross_asset > system_asset_max:
                return False, (
                    f"Cross-strategy exposure for {symbol} ({cross_asset:.2f}) "
                    f"exceeds system limit ({system_asset_max:.2f})"
                )

            # 9. Drawdown checks
            halted, level, dd_pct = self._check_drawdown_unlocked()
            if halted:
                return False, f"Trading halted: {level} drawdown {dd_pct:.2f}% exceeds limit"

            # 10. Consecutive loss / size reduction
            consec = self._consecutive_losses.get(strategy_id, 0)
            multiplier = self._get_size_multiplier(consec)
            if multiplier <= 0.0:
                return False, (
                    f"Trading halted for {strategy_id}: {consec} consecutive losses"
                )
            if multiplier < 1.0:
                effective_size = size_usdt * multiplier
                logger.info(
                    "Size reduced %.0f%% for %s due to %d consecutive losses: "
                    "%.2f -> %.2f USDT",
                    (1 - multiplier) * 100, strategy_id, consec,
                    size_usdt, effective_size,
                )

        return True, ""

    # ======================================================================
    #  Position tracking
    # ======================================================================

    def record_position_change(
        self,
        strategy_id: str,
        symbol: str,
        direction: str,
        size_usdt: float,
        is_open: bool,
    ) -> None:
        """Record a position open or close.

        Parameters
        ----------
        is_open:
            True when opening / adding to a position, False when closing.
        """
        with self._lock:
            if is_open:
                self._positions[strategy_id][symbol] = PositionRecord(
                    strategy_id=strategy_id,
                    symbol=symbol,
                    direction=direction.upper(),
                    size_usdt=size_usdt,
                )
                logger.info(
                    "Position opened: strategy=%s symbol=%s dir=%s size=%.2f",
                    strategy_id, symbol, direction, size_usdt,
                )
            else:
                self._positions[strategy_id].pop(symbol, None)
                logger.info(
                    "Position closed: strategy=%s symbol=%s", strategy_id, symbol,
                )

    # ======================================================================
    #  Equity & drawdown
    # ======================================================================

    def update_equity(self, equity: float) -> None:
        """Update the current equity value and refresh drawdown state."""
        with self._lock:
            self._current_equity = equity
            if self._drawdown is None:
                self._drawdown = DrawdownState(
                    peak_equity=equity,
                    current_equity=equity,
                    daily_start=equity,
                    weekly_start=equity,
                    monthly_start=equity,
                )
            else:
                self._drawdown.update_equity(equity)

            # Persist to DB
            if self._db is not None:
                try:
                    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                    self._db.save_drawdown_tracker({
                        "strategy_id": "__SYSTEM__",
                        "peak_equity": self._drawdown.peak_equity,
                        "current_equity": equity,
                        "drawdown_pct": self._drawdown.overall_drawdown_pct,
                        "daily_drawdown_pct": self._drawdown.daily_drawdown_pct,
                        "weekly_drawdown_pct": self._drawdown.weekly_drawdown_pct,
                        "monthly_drawdown_pct": self._drawdown.monthly_drawdown_pct,
                        "daily_start_equity": self._drawdown.daily_start_equity,
                        "weekly_start_equity": self._drawdown.weekly_start_equity,
                        "monthly_start_equity": self._drawdown.monthly_start_equity,
                        "last_reset_daily": now_str,
                        "last_reset_weekly": now_str,
                        "last_reset_monthly": now_str,
                    })
                except Exception as exc:
                    logger.warning("Failed to persist drawdown tracker: %s", exc)

    def get_current_equity(self) -> float:
        """Return the last known equity value."""
        return self._current_equity

    def check_drawdown(self) -> Tuple[bool, str, float]:
        """Check whether any drawdown threshold has been breached.

        Returns
        -------
        (halted, level, pct)
            *halted* is True if trading should stop.
            *level* is one of ``"daily"``, ``"weekly"``, ``"monthly"``, ``"system"``, or ``""``
            *pct* is the drawdown percentage at the breached level.
        """
        with self._lock:
            return self._check_drawdown_unlocked()

    def _check_drawdown_unlocked(self) -> Tuple[bool, str, float]:
        if self._drawdown is None:
            return False, "", 0.0

        dd = self._drawdown

        if dd.daily_drawdown_pct >= self._config.daily_drawdown_pct:
            return True, "daily", dd.daily_drawdown_pct

        if dd.weekly_drawdown_pct >= self._config.weekly_drawdown_pct:
            return True, "weekly", dd.weekly_drawdown_pct

        if dd.monthly_drawdown_pct >= self._config.monthly_drawdown_pct:
            return True, "monthly", dd.monthly_drawdown_pct

        if dd.overall_drawdown_pct >= self._config.system_wide_drawdown_pct:
            return True, "system", dd.overall_drawdown_pct

        return False, "", 0.0

    # ======================================================================
    #  Trade results / consecutive losses
    # ======================================================================

    def record_trade_result(self, strategy_id: str, pnl: float, is_win: bool) -> None:
        """Record the outcome of a completed trade."""
        with self._lock:
            self._trade_results[strategy_id].append(is_win)
            if is_win:
                self._consecutive_losses[strategy_id] = 0
            else:
                self._consecutive_losses[strategy_id] = (
                    self._consecutive_losses.get(strategy_id, 0) + 1
                )
            trade_logger.info(
                "RESULT\tstrategy=%s\tpnl=%.4f\twin=%s\tconsec_losses=%d",
                strategy_id, pnl, is_win, self._consecutive_losses[strategy_id],
            )

    def get_consecutive_losses(self, strategy_id: str) -> int:
        """Return the current consecutive-loss count for a strategy."""
        return self._consecutive_losses.get(strategy_id, 0)

    def _get_size_multiplier(self, consecutive_losses: int) -> float:
        """Return the position size multiplier based on consecutive losses."""
        multiplier = 1.0
        for threshold, mult in sorted(self.SIZE_REDUCTION_SCHEDULE.items()):
            if consecutive_losses >= threshold:
                multiplier = mult
        return multiplier

    # ======================================================================
    #  Exposure summary
    # ======================================================================

    def get_exposure_summary(self) -> dict:
        """Return a dict summarising current exposure across all strategies."""
        with self._lock:
            exp = self._compute_exposure_unlocked()
        return {
            "total_exposure_usdt": exp.total_exposure_usdt,
            "total_exposure_pct": exp.total_exposure_pct,
            "long_exposure_usdt": exp.long_exposure_usdt,
            "long_exposure_pct": exp.long_exposure_pct,
            "short_exposure_usdt": exp.short_exposure_usdt,
            "short_exposure_pct": exp.short_exposure_pct,
            "net_directional_usdt": exp.net_directional_usdt,
            "net_directional_pct": exp.net_directional_pct,
            "per_asset": dict(exp.per_asset),
            "per_strategy": dict(exp.per_strategy),
            "position_count": exp.position_count,
            "equity": self._current_equity,
        }

    def _compute_exposure_unlocked(self) -> ExposureSummary:
        summary = ExposureSummary()
        equity = self._current_equity if self._current_equity > 0 else 1.0

        for strat_id, symbols in self._positions.items():
            strat_total = 0.0
            for sym, pos in symbols.items():
                size = abs(pos.size_usdt)
                summary.total_exposure_usdt += size
                summary.per_asset[sym] = summary.per_asset.get(sym, 0.0) + size
                summary.position_count += 1
                strat_total += size

                if pos.direction == "LONG":
                    summary.long_exposure_usdt += size
                else:
                    summary.short_exposure_usdt += size

            summary.per_strategy[strat_id] = strat_total

        summary.net_directional_usdt = summary.long_exposure_usdt - summary.short_exposure_usdt
        summary.total_exposure_pct = (summary.total_exposure_usdt / equity) * 100.0
        summary.long_exposure_pct = (summary.long_exposure_usdt / equity) * 100.0
        summary.short_exposure_pct = (summary.short_exposure_usdt / equity) * 100.0
        summary.net_directional_pct = (summary.net_directional_usdt / equity) * 100.0

        return summary

    # ======================================================================
    #  Correlation risk
    # ======================================================================

    def _check_correlation_risk(self, symbol: str, direction: str) -> Optional[str]:
        """Return a rejection reason if adding *symbol* creates too much
        correlated exposure in the same direction, or None if OK."""
        correlated_same_direction = 0

        for _strat, sym_map in self._positions.items():
            for sym, pos in sym_map.items():
                if sym == symbol:
                    continue
                if pos.direction != direction:
                    continue
                corr = self.correlation_matrix.get_correlation(symbol, sym)
                if corr > CorrelationMatrix.HIGH_CORRELATION_THRESHOLD:
                    correlated_same_direction += 1

        if correlated_same_direction >= 2:
            return (
                f"Correlation limit: already {correlated_same_direction} {direction} "
                f"positions in assets highly correlated (>0.75) with {symbol}"
            )
        return None

    # ======================================================================
    #  Drawdown resets
    # ======================================================================

    def reset_daily_drawdown(self) -> None:
        """Reset daily drawdown counter. Called at 00:00 UTC."""
        with self._lock:
            if self._drawdown is not None:
                self._drawdown.reset_daily()
                logger.info("Daily drawdown reset. Start equity: %.2f", self._drawdown.daily_start_equity)

    def reset_weekly_drawdown(self) -> None:
        """Reset weekly drawdown counter. Called Monday 00:00 UTC."""
        with self._lock:
            if self._drawdown is not None:
                self._drawdown.reset_weekly()
                logger.info("Weekly drawdown reset. Start equity: %.2f", self._drawdown.weekly_start_equity)

    def reset_monthly_drawdown(self) -> None:
        """Reset monthly drawdown counter. Called 1st of month 00:00 UTC."""
        with self._lock:
            if self._drawdown is not None:
                self._drawdown.reset_monthly()
                logger.info("Monthly drawdown reset. Start equity: %.2f", self._drawdown.monthly_start_equity)
