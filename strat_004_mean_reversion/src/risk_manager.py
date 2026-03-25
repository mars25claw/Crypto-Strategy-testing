"""STRAT-004 Mean Reversion risk manager — extends shared.risk_manager.

Adds strategy-specific rules:
- 20% max equity, 4% per trade, 1.0% risk
- Max 3 concurrent, 6% per asset, 12% net directional
- Anti-trend safeguard: Z +/-4.0 -> emergency exit, 7-day blacklist
- Cross-strategy check: reject if STRAT-001 has same-direction position
- Drawdown: daily 2%, weekly 4%, monthly 7%
- Whipsaw: 6h min between trades, 12h after stop loss
- Consecutive loss: 3 -> reduce 25%, 5 -> halt
- Circuit breaker: 2 stop-outs in same day -> halt 24h
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from shared.config_loader import RiskConfig
from shared.risk_manager import (
    RiskManager as SharedRiskManager,
    CrossStrategyReader,
    DrawdownState,
)
from shared.cross_strategy import CrossStrategyManager

logger = logging.getLogger(__name__)
trade_logger = logging.getLogger("trade")

STRATEGY_ID = "STRAT-004"


@dataclass
class TradeRecord:
    """Record of a completed trade for metrics tracking."""

    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    pnl: float
    is_win: bool
    entry_time: float
    exit_time: float
    close_reason: str = ""


class MeanReversionRiskManager:
    """Strategy-specific risk manager for STRAT-004.

    Wraps the shared RiskManager and adds mean-reversion-specific rules:
    whipsaw protection, anti-trend blacklisting, cross-strategy conflict
    checks, and circuit breakers.

    Parameters
    ----------
    config : dict
        Strategy parameters from config.yaml.
    risk_config : RiskConfig
        Shared risk configuration dataclass.
    shared_risk : SharedRiskManager
        The shared risk manager instance.
    cross_strategy : CrossStrategyManager
        For reading STRAT-001 positions.
    """

    def __init__(
        self,
        config: dict,
        risk_config: RiskConfig,
        shared_risk: SharedRiskManager,
        cross_strategy: Optional[CrossStrategyManager] = None,
    ) -> None:
        self._config = config
        self._risk_config = risk_config
        self._shared = shared_risk
        self._cross = cross_strategy

        # Whipsaw tracking: symbol -> last trade exit time
        self._last_trade_time: Dict[str, float] = {}
        # symbol -> last stop-loss time
        self._last_stop_time: Dict[str, float] = {}

        # Anti-trend blacklist: symbol -> blacklist expiry epoch
        self._blacklist: Dict[str, float] = {}

        # Circuit breaker: count stop-outs per day
        self._daily_stop_count: int = 0
        self._daily_stop_reset_time: float = self._next_utc_midnight()
        self._halted_until: float = 0.0

        # Trade history
        self._trade_results: List[bool] = []
        self._trade_history: List[TradeRecord] = []
        self._consecutive_losses: int = 0

        # Config values
        self._min_trade_interval_h: float = config.get("min_trade_interval_hours", 6)
        self._post_stop_cooldown_h: float = config.get("post_stop_cooldown_hours", 12)
        self._blacklist_days: int = config.get("blacklist_days", 7)
        self._consec_loss_reduce_at: int = config.get("consecutive_loss_reduce_at", 3)
        self._consec_loss_reduce_pct: float = config.get("consecutive_loss_reduce_pct", 25)
        self._consec_loss_halt_at: int = config.get("consecutive_loss_halt_at", 5)

    # ==================================================================
    # Entry gate
    # ==================================================================

    def check_entry_allowed(
        self,
        strategy_id: str,
        symbol: str,
        direction: str,
        size_usdt: float,
        leverage: int,
    ) -> Tuple[bool, str]:
        """Full entry validation including STRAT-004-specific rules.

        Returns (allowed, reason).
        """
        # 0. Strategy halt check
        if self._is_halted():
            return False, "Strategy halted (circuit breaker or consecutive losses)"

        # 1. Blacklist check
        if self._is_blacklisted(symbol):
            expiry = datetime.fromtimestamp(self._blacklist[symbol], tz=timezone.utc)
            return False, f"Instrument {symbol} blacklisted until {expiry.isoformat()}"

        # 2. Whipsaw protection: 6h between trades per instrument
        if not self._check_whipsaw(symbol):
            elapsed = (time.time() - self._last_trade_time.get(symbol, 0)) / 3600
            return False, (
                f"Whipsaw protection: {elapsed:.1f}h since last trade on {symbol}, "
                f"need {self._min_trade_interval_h}h"
            )

        # 3. Post-stop cooldown: 12h after stop loss
        if not self._check_post_stop_cooldown(symbol):
            elapsed = (time.time() - self._last_stop_time.get(symbol, 0)) / 3600
            return False, (
                f"Post-stop cooldown: {elapsed:.1f}h since stop on {symbol}, "
                f"need {self._post_stop_cooldown_h}h"
            )

        # 4. Cross-strategy conflict: reject if STRAT-001 has same-direction
        conflict = self._check_cross_strategy_conflict(symbol, direction)
        if conflict:
            return False, conflict

        # 5. Consecutive loss size reduction
        if self._consecutive_losses >= self._consec_loss_halt_at:
            return False, (
                f"Strategy halted: {self._consecutive_losses} consecutive losses "
                f"(halt at {self._consec_loss_halt_at})"
            )

        # Apply size reduction if needed
        adjusted_size = size_usdt
        if self._consecutive_losses >= self._consec_loss_reduce_at:
            reduction = self._consec_loss_reduce_pct / 100.0
            adjusted_size = size_usdt * (1 - reduction)
            logger.info(
                "Size reduced by %d%% due to %d consecutive losses: %.2f -> %.2f",
                int(self._consec_loss_reduce_pct), self._consecutive_losses,
                size_usdt, adjusted_size,
            )

        # 6. Delegate to shared risk manager
        allowed, reason = self._shared.check_entry_allowed(
            strategy_id=strategy_id,
            symbol=symbol,
            direction=direction,
            size_usdt=adjusted_size,
            leverage=leverage,
        )

        return allowed, reason

    def check_instrument_allowed(self, symbol: str) -> bool:
        """Quick check if an instrument is eligible (not blacklisted, not in cooldown)."""
        if self._is_halted():
            return False
        if self._is_blacklisted(symbol):
            return False
        if not self._check_whipsaw(symbol):
            return False
        if not self._check_post_stop_cooldown(symbol):
            return False
        return True

    # ==================================================================
    # Position lifecycle
    # ==================================================================

    def record_position_open(
        self,
        symbol: str,
        direction: str,
        size_usdt: float,
        entry_price: float,
    ) -> None:
        """Record a new position opening."""
        self._shared.record_position_change(
            strategy_id=STRATEGY_ID,
            symbol=symbol,
            direction=direction,
            size_usdt=size_usdt,
            is_open=True,
        )

        # Write cross-strategy position file
        if self._cross:
            self._write_positions_file()

    def record_trade_closed(
        self,
        symbol: str,
        pnl: float,
        is_win: bool,
        close_reason: str = "",
    ) -> None:
        """Record a trade close and update all tracking."""
        now = time.time()

        # Update shared risk manager
        self._shared.record_position_change(
            strategy_id=STRATEGY_ID,
            symbol=symbol,
            direction="",
            size_usdt=0,
            is_open=False,
        )
        self._shared.record_trade_result(
            strategy_id=STRATEGY_ID, pnl=pnl, is_win=is_win,
        )

        # Update local tracking
        self._trade_results.append(is_win)
        self._last_trade_time[symbol] = now

        # Consecutive losses
        if is_win:
            self._consecutive_losses = 0
        else:
            self._consecutive_losses += 1

        # Stop-loss tracking
        if close_reason in ("stop_loss", "anti_trend_emergency_z4"):
            self._last_stop_time[symbol] = now
            self._record_daily_stop()

        # Update cross-strategy file
        if self._cross:
            self._write_positions_file()

        trade_logger.info(
            "RISK_RECORD\tsymbol=%s\tpnl=%.4f\twin=%s\tconsec_losses=%d\treason=%s",
            symbol, pnl, is_win, self._consecutive_losses, close_reason,
        )

    # ==================================================================
    # Anti-trend blacklist
    # ==================================================================

    def add_blacklist(self, symbol: str, days: Optional[int] = None) -> None:
        """Add an instrument to the anti-trend blacklist."""
        days = days or self._blacklist_days
        expiry = time.time() + days * 86400
        self._blacklist[symbol] = expiry
        logger.warning(
            "BLACKLISTED %s for %d days (until %s)",
            symbol, days,
            datetime.fromtimestamp(expiry, tz=timezone.utc).isoformat(),
        )
        trade_logger.info(
            "BLACKLIST\tsymbol=%s\tdays=%d\treason=anti_trend_z4",
            symbol, days,
        )

    def _is_blacklisted(self, symbol: str) -> bool:
        """Check if instrument is currently blacklisted."""
        expiry = self._blacklist.get(symbol, 0)
        if expiry <= 0:
            return False
        if time.time() >= expiry:
            # Expired — remove
            del self._blacklist[symbol]
            logger.info("Blacklist expired for %s", symbol)
            return False
        return True

    def get_blacklist(self) -> Dict[str, str]:
        """Return current blacklist as {symbol: expiry_iso}."""
        now = time.time()
        active = {}
        for sym, expiry in list(self._blacklist.items()):
            if expiry > now:
                active[sym] = datetime.fromtimestamp(expiry, tz=timezone.utc).isoformat()
            else:
                del self._blacklist[sym]
        return active

    # ==================================================================
    # Whipsaw protection
    # ==================================================================

    def _check_whipsaw(self, symbol: str) -> bool:
        """Return True if enough time has passed since last trade on symbol."""
        last = self._last_trade_time.get(symbol, 0)
        if last <= 0:
            return True
        elapsed_hours = (time.time() - last) / 3600
        return elapsed_hours >= self._min_trade_interval_h

    def _check_post_stop_cooldown(self, symbol: str) -> bool:
        """Return True if enough time has passed since last stop loss on symbol."""
        last = self._last_stop_time.get(symbol, 0)
        if last <= 0:
            return True
        elapsed_hours = (time.time() - last) / 3600
        return elapsed_hours >= self._post_stop_cooldown_h

    # ==================================================================
    # Cross-strategy conflict
    # ==================================================================

    def _check_cross_strategy_conflict(self, symbol: str, direction: str) -> Optional[str]:
        """Check if STRAT-001 has a same-direction position.

        Per Section 7.2: if STRAT-001 is LONG and we want to SHORT
        (counter-trend), that's rejected. Mean reversion should not
        fight an active trend-following position.
        """
        if self._cross is None:
            return None

        all_positions = self._cross.read_all_positions()
        strat001_positions = all_positions.get("STRAT-001", [])

        for pos in strat001_positions:
            if pos.get("symbol", "").upper() != symbol.upper():
                continue

            s001_direction = pos.get("direction", "").upper()

            # If STRAT-001 is LONG on this instrument and we want to SHORT
            # (mean reversion would be counter-trend — REJECTED)
            # If STRAT-001 is SHORT and we want to LONG — also counter-trend
            if s001_direction and s001_direction != direction.upper():
                return (
                    f"Cross-strategy conflict: STRAT-001 is {s001_direction} on {symbol}, "
                    f"rejecting {direction} mean reversion trade"
                )

        return None

    # ==================================================================
    # Circuit breaker
    # ==================================================================

    def _record_daily_stop(self) -> None:
        """Track daily stop-out count. 2 in same day -> 24h halt."""
        now = time.time()

        # Reset counter if new day
        if now >= self._daily_stop_reset_time:
            self._daily_stop_count = 0
            self._daily_stop_reset_time = self._next_utc_midnight()

        self._daily_stop_count += 1

        if self._daily_stop_count >= 2:
            self._halted_until = now + 24 * 3600
            logger.warning(
                "CIRCUIT BREAKER: %d stop-outs today, halting for 24h until %s",
                self._daily_stop_count,
                datetime.fromtimestamp(self._halted_until, tz=timezone.utc).isoformat(),
            )
            trade_logger.info(
                "CIRCUIT_BREAKER\tstops_today=%d\thalted_until=%s",
                self._daily_stop_count,
                datetime.fromtimestamp(self._halted_until, tz=timezone.utc).isoformat(),
            )

    def _is_halted(self) -> bool:
        """Check if strategy is halted by circuit breaker or consecutive losses."""
        if self._halted_until > 0 and time.time() < self._halted_until:
            return True
        if self._consecutive_losses >= self._consec_loss_halt_at:
            return True
        return False

    # ==================================================================
    # Drawdown delegation
    # ==================================================================

    def update_equity(self, equity: float) -> None:
        """Update equity in the shared risk manager."""
        self._shared.update_equity(equity)

    def get_current_equity(self) -> float:
        """Get current equity from shared risk manager."""
        return self._shared.get_current_equity()

    def check_drawdown(self) -> Tuple[bool, str, float]:
        """Check drawdown limits."""
        return self._shared.check_drawdown()

    # ==================================================================
    # Trade results access
    # ==================================================================

    def get_trade_results(self) -> List[bool]:
        """Return list of win/loss booleans."""
        return list(self._trade_results)

    def get_consecutive_losses(self) -> int:
        """Return current consecutive loss count."""
        return self._consecutive_losses

    def get_risk_status(self) -> dict:
        """Return comprehensive risk status for the dashboard."""
        halted, level, dd_pct = self._shared.check_drawdown()
        exposure = self._shared.get_exposure_summary()

        return {
            "strategy_id": STRATEGY_ID,
            "is_halted": self._is_halted(),
            "halted_until": (
                datetime.fromtimestamp(self._halted_until, tz=timezone.utc).isoformat()
                if self._halted_until > time.time() else None
            ),
            "consecutive_losses": self._consecutive_losses,
            "daily_stop_count": self._daily_stop_count,
            "blacklist": self.get_blacklist(),
            "drawdown_halted": halted,
            "drawdown_level": level,
            "drawdown_pct": round(dd_pct, 2),
            "exposure": exposure,
            "trade_count": len(self._trade_results),
            "win_count": sum(1 for r in self._trade_results if r),
            "loss_count": sum(1 for r in self._trade_results if not r),
        }

    # ==================================================================
    # Helpers
    # ==================================================================

    def _write_positions_file(self) -> None:
        """Write current positions to cross-strategy shared file."""
        if self._cross is None:
            return
        # Gather positions from shared risk manager
        positions = []
        strat_positions = self._shared._positions.get(STRATEGY_ID, {})
        for sym, rec in strat_positions.items():
            positions.append({
                "symbol": sym,
                "direction": rec.direction,
                "size_usdt": rec.size_usdt,
                "entry_price": 0,
                "strategy_id": STRATEGY_ID,
            })
        try:
            self._cross.write_positions(positions)
        except Exception as e:
            logger.warning("Failed to write cross-strategy positions: %s", e)

    @staticmethod
    def _next_utc_midnight() -> float:
        """Return epoch timestamp of the next UTC midnight."""
        now = datetime.now(timezone.utc)
        tomorrow = now.replace(hour=0, minute=0, second=0, microsecond=0)
        if tomorrow <= now:
            from datetime import timedelta
            tomorrow += timedelta(days=1)
        return tomorrow.timestamp()
