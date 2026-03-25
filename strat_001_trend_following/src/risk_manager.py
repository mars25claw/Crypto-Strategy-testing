"""STRAT-001 Risk Manager — Sections 5 & 6.

Extends the shared RiskManager with strategy-specific rules:
  - 30% max equity, 5% per trade, 1.5% risk base
  - Risk adjustments: ADX 20-25 halve, volume 1.0-1.5x reduce 25%,
    same-asset 50%, correlated direction 33%
  - Kelly after 100 trades: K/2, rolling 100, recalc every 20, 0.25-3.0%
  - Leverage: max 5x, prefer 3x, ATR/Price >3% -> 2x, >5% -> 1x
  - Drawdown: daily 3%, weekly 6%, monthly 10%
  - Exposure: 5 concurrent, 10% per asset, 25% long/short, 20% net
  - BTC+ETH same direction <= 15%
  - Compound growth: 75% size for first 5 trades after DD recovery
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from shared.config_loader import RiskConfig
from shared.risk_manager import RiskManager, CorrelationMatrix, CrossStrategyReader
from shared.alerting import AlertLevel

from . import STRATEGY_ID
from .strategy import SignalDirection

logger = logging.getLogger(__name__)
trade_logger = logging.getLogger("trade")


@dataclass
class PositionSizeResult:
    """Result of position sizing calculation."""
    size_usdt: float
    size_qty: float
    risk_pct: float
    leverage: int
    adjustments: List[str]
    rejected: bool = False
    reject_reason: str = ""


class TrendRiskManager:
    """Strategy-specific risk manager wrapping the shared RiskManager.

    Parameters
    ----------
    config : dict
        ``strategy_params`` from config.yaml.
    shared_risk : RiskManager
        The shared cross-strategy risk manager.
    """

    # BTC-ETH correlation pair special limit
    BTC_ETH_PAIR = frozenset({"BTCUSDT", "ETHUSDT"})
    BTC_ETH_MAX_PCT = 15.0

    def __init__(self, config: Dict[str, Any], shared_risk: RiskManager,
                 alerter: Optional[Any] = None) -> None:
        self.cfg = config
        self.shared = shared_risk
        self._alerter = alerter

        # Trade history for Kelly criterion
        self._trade_history: deque = deque(maxlen=200)
        self._total_trade_count: int = 0
        self._kelly_pct: Optional[float] = None
        self._last_kelly_calc_count: int = 0

        # Kelly halt flag (Section 5.3): negative Kelly -> halt + performance review
        self._kelly_halt: bool = False

        # Drawdown recovery tracking (Section 6.4)
        self._in_dd_recovery: bool = False
        self._dd_recovery_trades_remaining: int = 0
        self._peak_equity_for_recovery: float = 0.0

        # Monthly halt tracking
        self._monthly_halted: bool = False
        self._monthly_halt_resume_trades: int = 0

        # Consecutive loss tracking (local to this strategy)
        self._consecutive_losses: int = 0
        self._halted_by_losses: bool = False

    # ======================================================================
    # Position sizing (Sections 5.2, 6.1, 6.2)
    # ======================================================================

    def calculate_position_size(
        self,
        symbol: str,
        direction: SignalDirection,
        entry_price: float,
        atr_value: float,
        adx_value: float = 30.0,
        volume_ratio: float = 2.0,
        size_multiplier: float = 1.0,
    ) -> PositionSizeResult:
        """Calculate the position size for a new entry.

        Returns a :class:`PositionSizeResult` with size and adjustments.
        """
        adjustments: List[str] = []
        equity = self.shared.get_current_equity()

        if equity <= 0:
            return PositionSizeResult(
                0, 0, 0, 1, [], rejected=True, reject_reason="Equity not initialised"
            )

        # Base risk percentage
        base_risk_pct = self.cfg.get("risk_per_trade_pct", 1.5)

        # Kelly criterion override (Section 5.3)
        if self._kelly_pct is not None and self._total_trade_count >= self.cfg.get("kelly_min_trades", 100):
            base_risk_pct = self._kelly_pct
            adjustments.append(f"Kelly={self._kelly_pct:.2f}%")

        risk_pct = base_risk_pct

        # ADX 20-25 -> halve (Section 5.2)
        adx_weak_low = self.cfg.get("adx_weak_low", 20)
        adx_strong = self.cfg.get("adx_strong", 25)
        if adx_weak_low <= adx_value < adx_strong:
            risk_pct *= 0.5
            adjustments.append(f"ADX_weak({adx_value:.1f})->50%")

        # Volume 1.0-1.5x -> reduce 25% (Section 5.2)
        vol_strong = self.cfg.get("volume_strong_multiplier", 1.5)
        vol_weak_low = self.cfg.get("volume_weak_low", 1.0)
        if vol_weak_low <= volume_ratio < vol_strong:
            risk_pct *= 0.75
            adjustments.append(f"Vol_weak({volume_ratio:.2f})->75%")

        # Same-asset cross-strategy check (Section 5.2)
        cross_positions = self.shared._cross_reader.get_all_positions(
            exclude_strategy=STRATEGY_ID
        )
        same_asset = [p for p in cross_positions if p.get("symbol") == symbol]
        if same_asset:
            risk_pct *= 0.5
            adjustments.append("Same_asset_other_strat->50%")

        # Correlated direction check (Section 5.2)
        correlated_same_dir = self._count_correlated_same_direction(symbol, direction.value)
        if correlated_same_dir >= 2:
            risk_pct *= 0.67
            adjustments.append(f"Corr_same_dir({correlated_same_dir})->67%")

        # Apply signal-based size multiplier (from strategy ADX/volume checks)
        risk_pct *= size_multiplier
        if size_multiplier < 1.0:
            adjustments.append(f"Signal_mult={size_multiplier:.2f}")

        # Consecutive loss reduction (Section 7.3)
        consec = self._consecutive_losses
        if consec >= 7:
            return PositionSizeResult(
                0, 0, risk_pct, 1, adjustments,
                rejected=True, reject_reason=f"Halted: {consec} consecutive losses",
            )
        elif consec >= 5:
            risk_pct *= 0.5
            adjustments.append(f"Consec_loss({consec})->50%")
        elif consec >= 3:
            risk_pct *= 0.75
            adjustments.append(f"Consec_loss({consec})->75%")

        # Drawdown recovery (Section 6.4)
        if self._in_dd_recovery and self._dd_recovery_trades_remaining > 0:
            recovery_pct = self.cfg.get("dd_recovery_size_pct", 75) / 100.0
            risk_pct *= recovery_pct
            adjustments.append(f"DD_recovery({self._dd_recovery_trades_remaining})->{recovery_pct*100:.0f}%")

        # Monthly halt recovery
        if self._monthly_halt_resume_trades > 0:
            risk_pct *= 0.5
            adjustments.append(f"Monthly_recovery({self._monthly_halt_resume_trades})->50%")

        # Apply floor and ceiling
        kelly_floor = self.cfg.get("kelly_floor_pct", 0.25)
        kelly_ceiling = self.cfg.get("kelly_ceiling_pct", 3.0)
        risk_pct = max(kelly_floor, min(kelly_ceiling, risk_pct))

        # Calculate position size
        risk_amount = equity * (risk_pct / 100.0)
        stop_distance = self.cfg.get("hard_stop_atr_mult", 2.0) * atr_value

        if stop_distance <= 0 or entry_price <= 0:
            return PositionSizeResult(
                0, 0, risk_pct, 1, adjustments,
                rejected=True, reject_reason="Invalid stop distance or price",
            )

        # Size in USDT
        stop_distance_pct = stop_distance / entry_price
        size_usdt = risk_amount / stop_distance_pct

        # Per-trade cap: 5% of equity
        max_trade = equity * (self.cfg.get("max_per_trade_pct", 5.0) / 100.0)
        if size_usdt > max_trade:
            size_usdt = max_trade
            adjustments.append(f"Capped_at_{self.cfg.get('max_per_trade_pct', 5.0)}%")

        # Determine leverage (Section 6.2)
        leverage = self._calculate_leverage(atr_value, entry_price)

        # Convert to contracts
        size_qty = size_usdt / entry_price

        logger.info(
            "%s position size: risk=%.2f%% size_usdt=%.2f qty=%.6f lev=%dx adj=%s",
            symbol, risk_pct, size_usdt, size_qty, leverage, adjustments,
        )

        return PositionSizeResult(
            size_usdt=size_usdt,
            size_qty=size_qty,
            risk_pct=risk_pct,
            leverage=leverage,
            adjustments=adjustments,
        )

    # ======================================================================
    # Leverage calculation (Section 6.2)
    # ======================================================================

    def _calculate_leverage(self, atr_value: float, entry_price: float) -> int:
        """Determine leverage based on volatility."""
        max_lev = self.cfg.get("max_leverage", 5)
        preferred = self.cfg.get("preferred_leverage", 3)
        high_vol = self.cfg.get("high_vol_atr_pct", 3.0)
        extreme_vol = self.cfg.get("extreme_vol_atr_pct", 5.0)

        if entry_price <= 0:
            return 1

        atr_pct = (atr_value / entry_price) * 100.0

        if atr_pct > extreme_vol:
            return 1
        elif atr_pct > high_vol:
            return 2
        else:
            return min(preferred, max_lev)

    # ======================================================================
    # Kelly criterion (Section 5.3)
    # ======================================================================

    def _recalculate_kelly(self) -> None:
        """Recalculate Kelly criterion from recent trade history."""
        min_trades = self.cfg.get("kelly_min_trades", 100)
        if self._total_trade_count < min_trades:
            return

        recalc_interval = self.cfg.get("kelly_recalc_interval", 20)
        if self._total_trade_count - self._last_kelly_calc_count < recalc_interval:
            return

        # Use last 100 trades
        trades = list(self._trade_history)[-100:]
        if len(trades) < 50:
            return

        wins = [t for t in trades if t["pnl"] > 0]
        losses = [t for t in trades if t["pnl"] <= 0]

        if not losses:
            self._kelly_pct = self.cfg.get("kelly_ceiling_pct", 3.0)
            self._last_kelly_calc_count = self._total_trade_count
            return

        win_rate = len(wins) / len(trades)
        avg_win = sum(t["pnl"] for t in wins) / len(wins) if wins else 0
        avg_loss = abs(sum(t["pnl"] for t in losses) / len(losses)) if losses else 1

        if avg_loss == 0:
            return

        r_ratio = avg_win / avg_loss
        kelly = win_rate - ((1 - win_rate) / r_ratio)

        if kelly < 0:
            logger.critical(
                "Negative Kelly (%.4f) — HALTING strategy for performance review", kelly
            )
            self._kelly_halt = True
            self._kelly_pct = None
            # Trigger CRITICAL alert for performance review
            if self._alerter:
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.ensure_future(self._alerter.send(
                            f"KELLY HALT: Negative Kelly criterion ({kelly:.4f}). "
                            f"Strategy halted — performance review required. "
                            f"Win rate: {win_rate:.2%}, R-ratio: {r_ratio:.2f}",
                            level=AlertLevel.CRITICAL,
                            strategy_id=STRATEGY_ID,
                        ))
                    else:
                        loop.run_until_complete(self._alerter.send(
                            f"KELLY HALT: Negative Kelly criterion ({kelly:.4f}). "
                            f"Strategy halted — performance review required.",
                            level=AlertLevel.CRITICAL,
                            strategy_id=STRATEGY_ID,
                        ))
                except Exception as e:
                    logger.error("Failed to send Kelly halt alert: %s", e)
            return

        # Half-Kelly
        fraction = self.cfg.get("kelly_fraction", 0.5)
        kelly_pct = kelly * fraction * 100.0

        # Apply floor and ceiling
        floor = self.cfg.get("kelly_floor_pct", 0.25)
        ceiling = self.cfg.get("kelly_ceiling_pct", 3.0)
        kelly_pct = max(floor, min(ceiling, kelly_pct))

        self._kelly_pct = kelly_pct
        self._last_kelly_calc_count = self._total_trade_count

        logger.info(
            "Kelly recalculated: W=%.2f R=%.2f K=%.4f K/2=%.2f%%",
            win_rate, r_ratio, kelly, kelly_pct,
        )

    # ======================================================================
    # Trade result recording
    # ======================================================================

    def record_trade_result(self, trade: Dict[str, Any]) -> None:
        """Record a completed trade for risk calculations."""
        pnl = trade.get("realized_pnl", 0)
        is_win = pnl > 0

        self._trade_history.append(trade)
        self._total_trade_count += 1

        # Update consecutive losses
        if is_win:
            self._consecutive_losses = 0
            self._halted_by_losses = False
        else:
            self._consecutive_losses += 1
            if self._consecutive_losses >= 7:
                self._halted_by_losses = True
                logger.critical(
                    "STRAT-001 halted: %d consecutive losses", self._consecutive_losses
                )

        # Update shared risk manager
        self.shared.record_trade_result(STRATEGY_ID, pnl, is_win)

        # Recalculate Kelly
        self._recalculate_kelly()

        # DD recovery tracking
        if self._in_dd_recovery and self._dd_recovery_trades_remaining > 0:
            self._dd_recovery_trades_remaining -= 1
            if self._dd_recovery_trades_remaining <= 0:
                self._in_dd_recovery = False
                logger.info("Drawdown recovery period complete — full sizing resumed")

        # Monthly halt recovery
        if self._monthly_halt_resume_trades > 0:
            self._monthly_halt_resume_trades -= 1

    # ======================================================================
    # Entry gate check
    # ======================================================================

    def check_entry_allowed(
        self,
        strategy_id: str,
        symbol: str,
        direction: str,
        size_usdt: float,
        leverage: int,
    ) -> Tuple[bool, str]:
        """Full pre-trade check combining local and shared rules."""

        # Local halts
        if self._kelly_halt:
            return False, "Halted: Negative Kelly criterion — performance review required"

        if self._halted_by_losses:
            return False, f"Halted: {self._consecutive_losses} consecutive losses"

        if self._monthly_halted:
            return False, "Monthly drawdown halt active"

        # BTC+ETH same direction special limit (Section 5.6)
        if symbol in self.BTC_ETH_PAIR:
            btc_eth_exposure = self._get_btc_eth_same_direction_exposure(direction)
            equity = self.shared.get_current_equity()
            max_combined = equity * (self.BTC_ETH_MAX_PCT / 100.0)
            if btc_eth_exposure + size_usdt > max_combined:
                return False, (
                    f"BTC+ETH same-direction exposure "
                    f"{btc_eth_exposure + size_usdt:.2f} > limit {max_combined:.2f}"
                )

        # Shared risk manager gate
        return self.shared.check_entry_allowed(
            strategy_id=strategy_id,
            symbol=symbol,
            direction=direction,
            size_usdt=size_usdt,
            leverage=leverage,
        )

    def _get_btc_eth_same_direction_exposure(self, direction: str) -> float:
        """Get combined BTC+ETH exposure in the same direction."""
        exposure = 0.0
        for strat_id, sym_map in self.shared._positions.items():
            for sym, pos in sym_map.items():
                if sym in self.BTC_ETH_PAIR and pos.direction == direction.upper():
                    exposure += abs(pos.size_usdt)
        return exposure

    def _count_correlated_same_direction(self, symbol: str, direction: str) -> int:
        """Count positions in highly correlated assets with same direction."""
        count = 0
        threshold = self.cfg.get("correlation_threshold", 0.75)

        for strat_id, sym_map in self.shared._positions.items():
            for sym, pos in sym_map.items():
                if sym == symbol:
                    continue
                if pos.direction != direction.upper():
                    continue
                corr = self.shared.correlation_matrix.get_correlation(symbol, sym)
                if corr > threshold:
                    count += 1
        return count

    # ======================================================================
    # Drawdown event handlers
    # ======================================================================

    def on_daily_drawdown_hit(self) -> None:
        """Handle daily drawdown limit breach (Section 5.5)."""
        logger.warning("STRAT-001 daily drawdown limit hit — halting new entries")

    def on_weekly_drawdown_hit(self) -> None:
        """Handle weekly drawdown limit breach."""
        logger.warning("STRAT-001 weekly drawdown limit hit — reducing positions 50%")

    def on_monthly_drawdown_hit(self) -> None:
        """Handle monthly drawdown limit breach."""
        logger.critical("STRAT-001 monthly drawdown limit hit — closing ALL positions")
        self._monthly_halted = True
        self._monthly_halt_resume_trades = 10

    def on_drawdown_recovery(self) -> None:
        """Handle new equity high after a drawdown (Section 6.4)."""
        dd_recovery_trades = self.cfg.get("dd_recovery_trades", 5)
        self._in_dd_recovery = True
        self._dd_recovery_trades_remaining = dd_recovery_trades
        logger.info(
            "New equity high — drawdown recovery: %d trades at 75%% size",
            dd_recovery_trades,
        )

    # ======================================================================
    # State
    # ======================================================================

    def get_state(self) -> dict:
        """Return risk manager state for persistence/dashboard."""
        return {
            "total_trades": self._total_trade_count,
            "consecutive_losses": self._consecutive_losses,
            "halted_by_losses": self._halted_by_losses,
            "kelly_pct": self._kelly_pct,
            "kelly_halt": self._kelly_halt,
            "in_dd_recovery": self._in_dd_recovery,
            "dd_recovery_trades_remaining": self._dd_recovery_trades_remaining,
            "monthly_halted": self._monthly_halted,
            "monthly_halt_resume_trades": self._monthly_halt_resume_trades,
        }
