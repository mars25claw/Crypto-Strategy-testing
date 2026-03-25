"""Fallback rule-based strategy for catastrophic ML model failure.

When the ML model is corrupted, produces NaN, or is otherwise completely
unavailable, this simple RSI + MACD rule set keeps the strategy operational
at reduced sizing until the model is restored.

NOT a permanent solution -- triggers CRITICAL alert for manual attention.
"""

from __future__ import annotations

import logging
import time
from typing import Optional, Tuple

import numpy as np

from shared.indicators import IndicatorBuffer, rsi, macd

logger = logging.getLogger(__name__)


class FallbackSignal:
    """Result of the fallback strategy evaluation."""

    def __init__(
        self,
        signal: str = "NONE",
        reason: str = "",
        rsi_value: float = 50.0,
        macd_hist: float = 0.0,
    ):
        self.signal = signal            # "LONG", "SHORT", "NONE"
        self.reason = reason
        self.rsi_value = rsi_value
        self.macd_hist = macd_hist
        self.confidence = "MODERATE"    # Always moderate for fallback
        self.size_multiplier = 0.50     # 50% position sizes
        self.timestamp_ms = int(time.time() * 1000)

    def to_dict(self) -> dict:
        return {
            "signal": self.signal,
            "reason": self.reason,
            "rsi_value": round(self.rsi_value, 2),
            "macd_hist": round(self.macd_hist, 6),
            "confidence": self.confidence,
            "size_multiplier": self.size_multiplier,
            "timestamp_ms": self.timestamp_ms,
        }


class FallbackStrategy:
    """Simple RSI + MACD fallback for when the ML model is down.

    Rules:
    - RSI(14) on 4h < 30 with MACD histogram confirming (crossing up) -> LONG
    - RSI(14) on 4h > 70 with MACD histogram confirming (crossing down) -> SHORT
    - Position sizes capped at 50% of normal
    - Only trades when model is completely unavailable

    Parameters
    ----------
    rsi_oversold : float
        RSI level for LONG signal (default 30).
    rsi_overbought : float
        RSI level for SHORT signal (default 70).
    rsi_period : int
        RSI lookback period (default 14).
    """

    def __init__(
        self,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
        rsi_period: int = 14,
    ) -> None:
        self._rsi_oversold = rsi_oversold
        self._rsi_overbought = rsi_overbought
        self._rsi_period = rsi_period
        self._active = False
        self._activation_time: float = 0.0

        logger.info(
            "FallbackStrategy initialised: RSI<%d=LONG, RSI>%d=SHORT, 50%% sizing",
            int(rsi_oversold), int(rsi_overbought),
        )

    @property
    def is_active(self) -> bool:
        return self._active

    def activate(self, reason: str = "ML model failure") -> None:
        """Activate fallback mode."""
        if not self._active:
            self._active = True
            self._activation_time = time.time()
            logger.critical(
                "FALLBACK STRATEGY ACTIVATED: %s -- trading at 50%% sizing "
                "until ML model is restored",
                reason,
            )

    def deactivate(self) -> None:
        """Deactivate fallback mode (ML model restored)."""
        if self._active:
            duration = time.time() - self._activation_time
            self._active = False
            logger.info(
                "Fallback strategy deactivated (was active for %.1f minutes)",
                duration / 60,
            )

    def evaluate(self, buf_4h: IndicatorBuffer) -> FallbackSignal:
        """Evaluate the fallback strategy.

        Parameters
        ----------
        buf_4h : IndicatorBuffer
            4-hour kline buffer with at least 50 candles.

        Returns
        -------
        FallbackSignal with signal direction and metadata.
        """
        if not self._active:
            return FallbackSignal(signal="NONE", reason="Fallback not active")

        if len(buf_4h) < 30:
            return FallbackSignal(
                signal="NONE",
                reason=f"Insufficient 4h data ({len(buf_4h)} candles, need 30)",
            )

        closes = buf_4h.get_closes()

        # RSI(14) on 4h
        rsi_vals = rsi(closes, self._rsi_period)
        current_rsi = float(rsi_vals[-1]) if not np.isnan(rsi_vals[-1]) else 50.0

        # MACD histogram on 4h
        _, _, hist = macd(closes, 12, 26, 9)
        current_hist = float(hist[-1]) if not np.isnan(hist[-1]) else 0.0
        prev_hist = float(hist[-2]) if len(hist) > 1 and not np.isnan(hist[-2]) else 0.0

        # Signal logic
        if current_rsi < self._rsi_oversold:
            # MACD confirmation: histogram crossing from negative to less negative (or positive)
            if current_hist > prev_hist:
                return FallbackSignal(
                    signal="LONG",
                    reason=f"RSI={current_rsi:.1f} < {self._rsi_oversold}, MACD confirming",
                    rsi_value=current_rsi,
                    macd_hist=current_hist,
                )
            else:
                return FallbackSignal(
                    signal="NONE",
                    reason=f"RSI={current_rsi:.1f} oversold but MACD not confirming",
                    rsi_value=current_rsi,
                    macd_hist=current_hist,
                )

        elif current_rsi > self._rsi_overbought:
            # MACD confirmation: histogram crossing from positive to less positive (or negative)
            if current_hist < prev_hist:
                return FallbackSignal(
                    signal="SHORT",
                    reason=f"RSI={current_rsi:.1f} > {self._rsi_overbought}, MACD confirming",
                    rsi_value=current_rsi,
                    macd_hist=current_hist,
                )
            else:
                return FallbackSignal(
                    signal="NONE",
                    reason=f"RSI={current_rsi:.1f} overbought but MACD not confirming",
                    rsi_value=current_rsi,
                    macd_hist=current_hist,
                )

        return FallbackSignal(
            signal="NONE",
            reason=f"RSI={current_rsi:.1f} in neutral zone",
            rsi_value=current_rsi,
            macd_hist=current_hist,
        )

    def get_status(self) -> dict:
        """Return fallback strategy status for dashboard."""
        return {
            "active": self._active,
            "activation_time": self._activation_time,
            "uptime_minutes": round((time.time() - self._activation_time) / 60, 1) if self._active else 0,
            "rsi_oversold": self._rsi_oversold,
            "rsi_overbought": self._rsi_overbought,
        }
