"""Daily regime classification for STRAT-004 Mean Reversion.

Runs at 00:00 UTC per instrument. Uses Hurst exponent (R/S analysis) on
100-day rolling closes and ADX(14) on daily candles to determine whether
the market is mean-reverting, random-walk, or trending.

Regime outcomes per instrument:
    AGGRESSIVE  — Hurst < 0.40 and ADX < 20
    STANDARD    — Hurst 0.40–0.50 and ADX < 25
    PAUSE       — Hurst 0.50–0.55 (random walk)
    DISABLE     — Hurst > 0.55 or ADX >= 25
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

from shared.indicators import hurst_exponent, adx as compute_adx, atr as compute_atr

logger = logging.getLogger(__name__)
system_logger = logging.getLogger("system")


# ---------------------------------------------------------------------------
# Regime enum
# ---------------------------------------------------------------------------

class RegimeState(str, Enum):
    AGGRESSIVE = "AGGRESSIVE"   # H < 0.40, ADX < 20 — strong mean reversion
    STANDARD = "STANDARD"       # H 0.40–0.50, ADX < 25 — moderate
    PAUSE = "PAUSE"             # H 0.50–0.55 — random walk
    DISABLE = "DISABLE"         # H > 0.55 or ADX >= 25 — trending


# ---------------------------------------------------------------------------
# Per-instrument regime record
# ---------------------------------------------------------------------------

@dataclass
class InstrumentRegime:
    """Current regime classification for a single instrument."""

    symbol: str
    state: RegimeState = RegimeState.DISABLE
    hurst: float = 0.5
    adx_value: float = 25.0
    daily_atr: float = 0.0
    daily_atr_avg: float = 0.0
    classified_at: float = 0.0          # epoch
    classification_duration_ms: int = 0

    @property
    def is_tradeable(self) -> bool:
        """True when mean reversion signals may be evaluated."""
        return self.state in (RegimeState.AGGRESSIVE, RegimeState.STANDARD)

    @property
    def is_aggressive(self) -> bool:
        return self.state == RegimeState.AGGRESSIVE

    @property
    def size_multiplier(self) -> float:
        """Position-size multiplier based on regime strength.

        AGGRESSIVE → 1.25 (Hurst < 0.40 bonus), STANDARD → 1.0.
        ADX 20–25 (transitional) reduces by 25%.
        """
        if not self.is_tradeable:
            return 0.0
        mult = 1.25 if self.is_aggressive else 1.0
        if 20.0 <= self.adx_value < 25.0:
            mult *= 0.75  # transitional — reduced size
        return mult

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "state": self.state.value,
            "hurst": round(self.hurst, 4),
            "adx": round(self.adx_value, 2),
            "daily_atr": round(self.daily_atr, 4),
            "daily_atr_avg": round(self.daily_atr_avg, 4),
            "is_tradeable": self.is_tradeable,
            "size_multiplier": round(self.size_multiplier, 2),
            "classified_at": self.classified_at,
            "classification_duration_ms": self.classification_duration_ms,
        }


# ---------------------------------------------------------------------------
# RegimeClassifier
# ---------------------------------------------------------------------------

class RegimeClassifier:
    """Classifies market regime per instrument using daily candle data.

    Parameters
    ----------
    config : dict
        Strategy parameters (from ``strategy_params`` in config.yaml).
    binance_client :
        Async Binance REST client for fetching historical daily candles.
    """

    def __init__(self, config: dict, binance_client: Any = None) -> None:
        self._config = config
        self._client = binance_client

        # Thresholds from config
        self._hurst_window: int = config.get("hurst_window", 100)
        self._hurst_aggressive: float = config.get("hurst_aggressive_threshold", 0.40)
        self._hurst_standard: float = config.get("hurst_standard_threshold", 0.50)
        self._hurst_pause: float = config.get("hurst_pause_threshold", 0.55)
        self._adx_period: int = config.get("adx_period", 14)
        self._adx_ideal: float = config.get("adx_ideal_threshold", 20.0)
        self._adx_disable: float = config.get("adx_disable_threshold", 25.0)
        self._timeout_s: int = config.get("regime_recalc_timeout_s", 60)

        # Per-instrument regime state
        self.regimes: Dict[str, InstrumentRegime] = {}

        # History for dashboard
        self._classification_history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def classify_all(self, instruments: List[str]) -> Dict[str, InstrumentRegime]:
        """Run regime classification for all instruments.

        Fetches daily candles, computes Hurst + ADX, and stores results.
        Each instrument must complete within the configured timeout.

        Returns
        -------
        dict
            symbol -> InstrumentRegime
        """
        start = time.time()
        logger.info("Starting regime classification for %d instruments", len(instruments))

        results: Dict[str, InstrumentRegime] = {}

        for symbol in instruments:
            try:
                regime = await asyncio.wait_for(
                    self._classify_instrument(symbol),
                    timeout=self._timeout_s,
                )
                results[symbol] = regime
                self.regimes[symbol] = regime
                logger.info(
                    "Regime %s: %s (H=%.4f, ADX=%.2f, tradeable=%s)",
                    symbol, regime.state.value, regime.hurst,
                    regime.adx_value, regime.is_tradeable,
                )
            except asyncio.TimeoutError:
                logger.error(
                    "Regime classification timed out for %s (limit=%ds)",
                    symbol, self._timeout_s,
                )
                # Keep previous regime or default to DISABLE
                if symbol not in self.regimes:
                    self.regimes[symbol] = InstrumentRegime(
                        symbol=symbol, state=RegimeState.DISABLE,
                    )
                results[symbol] = self.regimes[symbol]
            except Exception:
                logger.exception("Regime classification failed for %s", symbol)
                if symbol not in self.regimes:
                    self.regimes[symbol] = InstrumentRegime(
                        symbol=symbol, state=RegimeState.DISABLE,
                    )
                results[symbol] = self.regimes[symbol]

        elapsed = time.time() - start
        tradeable_count = sum(1 for r in results.values() if r.is_tradeable)

        # Store history entry
        self._classification_history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "duration_s": round(elapsed, 2),
            "total": len(instruments),
            "tradeable": tradeable_count,
            "regimes": {s: r.state.value for s, r in results.items()},
        })
        # Keep last 30 entries
        if len(self._classification_history) > 30:
            self._classification_history = self._classification_history[-30:]

        system_logger.info(
            "regime_classification\ttotal=%d\ttradeable=%d\tduration=%.1fs",
            len(instruments), tradeable_count, elapsed,
        )

        return results

    def classify_from_buffers(
        self,
        symbol: str,
        daily_closes: np.ndarray,
        daily_highs: np.ndarray,
        daily_lows: np.ndarray,
    ) -> InstrumentRegime:
        """Classify regime directly from in-memory candle buffers.

        Used during warm-up or when daily candles are already available
        without needing a REST fetch.
        """
        start = time.time()
        regime = self._compute_regime(symbol, daily_closes, daily_highs, daily_lows)
        regime.classification_duration_ms = int((time.time() - start) * 1000)
        self.regimes[symbol] = regime
        return regime

    def get_regime(self, symbol: str) -> InstrumentRegime:
        """Return current regime for a symbol (DISABLE if unknown)."""
        return self.regimes.get(
            symbol,
            InstrumentRegime(symbol=symbol, state=RegimeState.DISABLE),
        )

    def is_tradeable(self, symbol: str) -> bool:
        """Convenience check: is the instrument eligible for mean reversion?"""
        return self.get_regime(symbol).is_tradeable

    def check_regime_change_exit(self, symbol: str) -> bool:
        """Return True if the current regime requires exiting open positions.

        Exit trigger: Hurst > 0.55 or ADX > 30.
        """
        regime = self.get_regime(symbol)
        adx_exit = self._config.get("adx_exit_threshold", 30.0)
        return regime.hurst > self._hurst_pause or regime.adx_value > adx_exit

    def get_tradeable_instruments(self) -> List[str]:
        """Return list of symbols currently classified as tradeable."""
        return [s for s, r in self.regimes.items() if r.is_tradeable]

    def get_classification_history(self) -> List[Dict[str, Any]]:
        """Return recent classification history for the dashboard."""
        return list(self._classification_history)

    def get_all_regimes(self) -> Dict[str, dict]:
        """Return all regime data as plain dicts (for API / dashboard)."""
        return {s: r.to_dict() for s, r in self.regimes.items()}

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _classify_instrument(self, symbol: str) -> InstrumentRegime:
        """Fetch daily candles and compute regime for a single instrument."""
        start = time.time()

        # Fetch enough daily candles: need hurst_window + some ADX warm-up
        needed = self._hurst_window + self._adx_period + 30
        klines = await self._client.get_futures_klines(
            symbol=symbol,
            interval="1d",
            limit=min(needed, 500),
        )

        if not klines or len(klines) < self._hurst_window:
            logger.warning(
                "Insufficient daily data for %s: got %d, need %d",
                symbol, len(klines) if klines else 0, self._hurst_window,
            )
            return InstrumentRegime(symbol=symbol, state=RegimeState.DISABLE)

        # Parse OHLC
        closes = np.array([float(k[4]) for k in klines], dtype=np.float64)
        highs = np.array([float(k[2]) for k in klines], dtype=np.float64)
        lows = np.array([float(k[3]) for k in klines], dtype=np.float64)

        regime = self._compute_regime(symbol, closes, highs, lows)
        regime.classification_duration_ms = int((time.time() - start) * 1000)
        return regime

    def _compute_regime(
        self,
        symbol: str,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
    ) -> InstrumentRegime:
        """Pure computation: Hurst + ADX → regime state."""

        # --- Hurst exponent on last hurst_window closes ---
        hurst_data = closes[-self._hurst_window:]
        h = hurst_exponent(hurst_data, max_lag=self._hurst_window // 2)
        if np.isnan(h):
            h = 0.5  # default to random walk

        # --- ADX(14) on daily data ---
        adx_arr, _, _ = compute_adx(highs, lows, closes, period=self._adx_period)
        # Latest valid ADX
        adx_val = 25.0  # default
        for i in range(len(adx_arr) - 1, -1, -1):
            if not np.isnan(adx_arr[i]):
                adx_val = float(adx_arr[i])
                break

        # --- ATR for volatility context ---
        atr_arr = compute_atr(highs, lows, closes, period=14)
        daily_atr = 0.0
        daily_atr_avg = 0.0
        valid_atr = atr_arr[~np.isnan(atr_arr)]
        if len(valid_atr) > 0:
            daily_atr = float(valid_atr[-1])
            daily_atr_avg = float(np.mean(valid_atr[-50:])) if len(valid_atr) >= 50 else float(np.mean(valid_atr))

        # --- Combined regime decision ---
        # Rule: active ONLY if Hurst < 0.50 AND ADX < 25
        if h >= self._hurst_pause:
            # Hurst > 0.55 → trending / DISABLE
            state = RegimeState.DISABLE
        elif h >= self._hurst_standard:
            # Hurst 0.50–0.55 → random walk / PAUSE
            state = RegimeState.PAUSE
        elif adx_val >= self._adx_disable:
            # ADX >= 25 → trending, override Hurst
            state = RegimeState.DISABLE
        elif h < self._hurst_aggressive and adx_val < self._adx_ideal:
            # Hurst < 0.40 and ADX < 20 → AGGRESSIVE
            state = RegimeState.AGGRESSIVE
        else:
            # Hurst 0.40–0.50 and ADX < 25 → STANDARD
            state = RegimeState.STANDARD

        return InstrumentRegime(
            symbol=symbol,
            state=state,
            hurst=h,
            adx_value=adx_val,
            daily_atr=daily_atr,
            daily_atr_avg=daily_atr_avg,
            classified_at=time.time(),
        )
