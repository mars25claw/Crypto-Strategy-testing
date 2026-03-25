"""Master strategy coordinator for STRAT-008.

Orchestrates hourly volatility regime assessment, IV/RV ratio evaluation,
sub-strategy selection based on conditions and mode (Synthetic vs Full),
cycle management, and coordination between all sub-strategies.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from shared.indicators import ema, rsi, atr, adx
from shared.indicators import IndicatorBuffer

from src.volatility_engine import VolatilityEngine, VolRegime
from src.covered_calls import CoveredCallManager
from src.cash_secured_puts import CashSecuredPutManager
from src.delta_neutral import DeltaNeutralManager
from src.risk_manager import StrategyRiskManager
from src.black_scholes import OptionGreeks
from src.strategy_metrics import StrategyMetricsCalculator, StrategySpecificMetrics

# MVRV thresholds (Section 6: MVRV Ratio Integration)
MVRV_HIGH_RISK = 3.5      # High risk of sell-off
MVRV_UNDERVALUED = 1.0    # Extreme undervaluation

logger = logging.getLogger(__name__)
trade_logger = logging.getLogger("trade")
perf_logger = logging.getLogger("performance")

# ---------------------------------------------------------------------------
# Operating modes
# ---------------------------------------------------------------------------

MODE_SYNTHETIC = "synthetic"    # Binance only: Sub-A + Sub-B
MODE_FULL = "full"              # Binance + Deribit: Sub-A + Sub-B + Sub-C + Sub-D


# ---------------------------------------------------------------------------
# StrategyCoordinator
# ---------------------------------------------------------------------------

class StrategyCoordinator:
    """Master coordinator for all STRAT-008 sub-strategies.

    Responsibilities:
    - Hourly vol regime assessment
    - IV/RV ratio classification and sub-strategy selection
    - 7-day cycle management for each sub-strategy
    - Greek monitoring (Deribit mode)
    - Coordination of entries, exits, and rolling

    Parameters
    ----------
    config : dict
        Strategy parameters from config.yaml.
    vol_engine : VolatilityEngine
        Shared volatility engine instance.
    risk_mgr : StrategyRiskManager
        Strategy-specific risk manager.
    mode : str
        "synthetic" or "full".
    """

    def __init__(
        self,
        config: dict,
        vol_engine: VolatilityEngine,
        risk_mgr: StrategyRiskManager,
        mode: str = MODE_SYNTHETIC,
    ) -> None:
        self._config = config
        self._vol_engine = vol_engine
        self._risk_mgr = risk_mgr
        self._mode = mode

        # Instruments
        self._symbols = ["BTCUSDT", "ETHUSDT"]

        # Sub-strategy managers
        self._cc_mgr = CoveredCallManager(config)
        self._csp_mgr = CashSecuredPutManager(config)
        self._dn_mgr = DeltaNeutralManager(config) if mode == MODE_FULL else None

        # Indicator buffers: symbol -> timeframe -> IndicatorBuffer
        self._buffers: Dict[str, Dict[str, IndicatorBuffer]] = {}
        for sym in self._symbols:
            self._buffers[sym] = {
                "1m": IndicatorBuffer(max_size=50000),  # ~35 days
                "1h": IndicatorBuffer(max_size=1000),
                "1d": IndicatorBuffer(max_size=365),
            }

        # Current prices
        self._prices: Dict[str, float] = {}

        # Current regime per symbol
        self._regimes: Dict[str, VolRegime] = {}

        # Last assessment timestamps
        self._last_regime_assessment: float = 0.0
        self._last_cycle_check: float = 0.0

        # Pending actions queue
        self._pending_actions: List[dict] = []

        # Performance tracking
        self._total_premium_collected = 0.0
        self._total_hedge_costs = 0.0
        self._total_cycles_completed = 0

        # MVRV Ratio Integration (Glassnode)
        self._mvrv_ratio: Optional[float] = None
        self._mvrv_last_update: float = 0.0
        self._mvrv_adjustment: Dict[str, float] = {}  # sub-strategy -> size multiplier

        # Strategy-specific metrics calculator
        self._metrics_calc = StrategyMetricsCalculator(
            cc_manager=self._cc_mgr,
            csp_manager=self._csp_mgr,
            dn_manager=self._dn_mgr,
            risk_mgr=self._risk_mgr,
            vol_engine=self._vol_engine,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def cc_manager(self) -> CoveredCallManager:
        return self._cc_mgr

    @property
    def csp_manager(self) -> CashSecuredPutManager:
        return self._csp_mgr

    @property
    def dn_manager(self) -> Optional[DeltaNeutralManager]:
        return self._dn_mgr

    # ------------------------------------------------------------------
    # Data ingestion callbacks
    # ------------------------------------------------------------------

    def on_kline_1m(self, symbol: str, candle: dict) -> None:
        """Process 1-minute kline data."""
        buf = self._buffers.get(symbol, {}).get("1m")
        if buf:
            buf.add_candle(candle)

        close = float(candle.get("close", 0))
        ts = int(candle.get("timestamp", 0))
        if close > 0:
            self._vol_engine.add_1m_candle(symbol, ts, close)
            self._prices[symbol] = close

    def on_kline_1h(self, symbol: str, candle: dict) -> None:
        """Process 1-hour kline data."""
        buf = self._buffers.get(symbol, {}).get("1h")
        if buf:
            buf.add_candle(candle)

        close = float(candle.get("close", 0))
        if close > 0:
            self._prices[symbol] = close
            self._vol_engine.update_hourly_price(symbol, close)

    def on_kline_1d(self, symbol: str, candle: dict) -> None:
        """Process daily kline data."""
        buf = self._buffers.get(symbol, {}).get("1d")
        if buf:
            buf.add_candle(candle)

    def on_mark_price(self, symbol: str, price: float) -> None:
        """Process mark price update."""
        self._prices[symbol] = price

    def on_premium_update(
        self, symbol: str, spot_price: float, futures_price: float
    ) -> None:
        """Process spot/futures premium for synthetic IV."""
        self._vol_engine.update_futures_premium(symbol, spot_price, futures_price)

    # ------------------------------------------------------------------
    # MVRV Ratio Integration
    # ------------------------------------------------------------------

    def update_mvrv_ratio(self, mvrv: float) -> None:
        """Update the MVRV ratio from Glassnode data.

        Adjusts sub-strategy sizing based on MVRV:
        - MVRV > 3.5: high risk of sell-off -> reduce put selling, increase call selling
        - MVRV < 1.0: extreme undervaluation -> reduce call selling, increase put selling
        - Between 1.0 and 3.5: normal operation

        Parameters
        ----------
        mvrv : float
            Market Value to Realized Value ratio.
        """
        self._mvrv_ratio = mvrv
        self._mvrv_last_update = time.time()

        if mvrv > MVRV_HIGH_RISK:
            # High risk of sell-off
            self._mvrv_adjustment = {
                "covered_calls": 1.3,       # Increase call selling (profit from pullback)
                "cash_secured_puts": 0.5,   # Reduce put selling (danger of assignment)
                "delta_neutral": 0.8,       # Slightly reduce
            }
            logger.info(
                "MVRV=%.2f > %.1f: HIGH RISK — "
                "increasing call selling, reducing put selling",
                mvrv, MVRV_HIGH_RISK,
            )
        elif mvrv < MVRV_UNDERVALUED:
            # Extreme undervaluation
            self._mvrv_adjustment = {
                "covered_calls": 0.5,       # Reduce call selling (don't cap upside)
                "cash_secured_puts": 1.3,   # Increase put selling (good entry level)
                "delta_neutral": 0.8,
            }
            logger.info(
                "MVRV=%.2f < %.1f: UNDERVALUED — "
                "reducing call selling, increasing put selling",
                mvrv, MVRV_UNDERVALUED,
            )
        else:
            # Normal
            self._mvrv_adjustment = {
                "covered_calls": 1.0,
                "cash_secured_puts": 1.0,
                "delta_neutral": 1.0,
            }
            logger.debug("MVRV=%.2f: normal range, no adjustment", mvrv)

    def get_mvrv_adjustment(self, sub_strategy: str) -> float:
        """Return the MVRV-based sizing multiplier for a sub-strategy."""
        return self._mvrv_adjustment.get(sub_strategy, 1.0)

    @property
    def mvrv_ratio(self) -> Optional[float]:
        return self._mvrv_ratio

    # ------------------------------------------------------------------
    # Hourly regime assessment (Section 3.1)
    # ------------------------------------------------------------------

    async def assess_volatility_regimes(self) -> Dict[str, VolRegime]:
        """Run hourly volatility regime assessment for all symbols.

        This is the core hourly evaluation:
        1. Calculate RV (1d/7d/30d)
        2. Get IV (Deribit or synthetic)
        3. Compute IV/RV ratio
        4. Classify regime
        """
        now = time.time()
        self._last_regime_assessment = now

        deribit_available = self._mode == MODE_FULL

        regimes = {}
        for symbol in self._symbols:
            regime = self._vol_engine.assess_regime(symbol, deribit_available)
            self._regimes[symbol] = regime
            regimes[symbol] = regime

            perf_logger.info(
                "VOL_REGIME\t%s\trv_1d=%.1f\trv_7d=%.1f\trv_30d=%.1f\t"
                "iv=%.1f\tiv_src=%s\tratio=%.2f\tregime=%s",
                symbol, regime.rv_1d, regime.rv_7d, regime.rv_30d,
                regime.iv, regime.iv_source, regime.iv_rv_ratio, regime.regime,
            )

        return regimes

    # ------------------------------------------------------------------
    # Sub-strategy selection
    # ------------------------------------------------------------------

    def select_sub_strategies(
        self, symbol: str, regime: VolRegime
    ) -> List[str]:
        """Determine which sub-strategies should be active.

        IV/RV ratio classification:
        - > 1.5: strong — all eligible, increase sizes
        - > 1.3: favorable — all eligible
        - 1.1-1.3: neutral — reduce activity
        - < 1.0: no vol selling

        Returns list of eligible sub-strategy names.
        """
        ratio = regime.iv_rv_ratio
        eligible: List[str] = []

        if ratio < 1.0:
            logger.info(
                "%s: IV/RV ratio %.2f < 1.0 — NO vol selling", symbol, ratio,
            )
            return eligible

        if ratio < 1.1:
            logger.info(
                "%s: IV/RV ratio %.2f < 1.1 — reducing activity", symbol, ratio,
            )
            return eligible

        # Ratio >= 1.1: at minimum, covered calls and CSP are eligible
        eligible.append("covered_calls")
        eligible.append("cash_secured_puts")

        # Delta-neutral only in Full mode with ratio > 1.4
        if self._mode == MODE_FULL and ratio >= 1.4:
            eligible.append("delta_neutral")

        label = "STRONG" if ratio >= 1.5 else "FAVORABLE" if ratio >= 1.3 else "NEUTRAL"
        logger.info(
            "%s: IV/RV=%.2f (%s) — eligible: %s",
            symbol, ratio, label, eligible,
        )

        return eligible

    # ------------------------------------------------------------------
    # Covered call evaluation
    # ------------------------------------------------------------------

    async def evaluate_covered_call(
        self,
        symbol: str,
        available_spot_qty: float,
        equity: float,
    ) -> Optional[dict]:
        """Evaluate and potentially create a covered call position.

        Returns action dict or None.
        """
        regime = self._regimes.get(symbol)
        if not regime:
            return None

        # Trading halt check
        halted, reason = self._risk_mgr.is_trading_halted()
        if halted:
            return None

        sub_halted, sub_reason = self._risk_mgr.is_sub_strategy_halted("cc")
        if sub_halted:
            return None

        # Get indicators
        daily_buf = self._buffers.get(symbol, {}).get("1d")
        if not daily_buf or len(daily_buf) < 50:
            return None

        closes = daily_buf.get_closes()
        highs = daily_buf.get_highs()
        lows = daily_buf.get_lows()

        ema_20 = ema(closes, 20)
        ema_50 = ema(closes, 50)
        daily_atr_arr = atr(highs, lows, closes, 14)
        adx_arr, plus_di, minus_di = adx(highs, lows, closes, 14)

        current_price = self._prices.get(symbol, 0)
        if current_price <= 0:
            return None

        ema20_val = float(ema_20[-1]) if not np.isnan(ema_20[-1]) else 0
        ema50_val = float(ema_50[-1]) if not np.isnan(ema_50[-1]) else 0
        atr_val = float(daily_atr_arr[-1]) if not np.isnan(daily_atr_arr[-1]) else 0
        adx_val = float(adx_arr[-1]) if not np.isnan(adx_arr[-1]) else 0
        plus_di_val = float(plus_di[-1]) if not np.isnan(plus_di[-1]) else 0

        should, reason = self._cc_mgr.should_enter(
            symbol=symbol,
            iv=regime.iv,
            rv_7d=regime.rv_7d,
            iv_rv_ratio=regime.iv_rv_ratio,
            current_price=current_price,
            ema_20=ema20_val,
            ema_50=ema50_val,
            daily_atr=atr_val,
            adx_value=adx_val,
            adx_plus_di=plus_di_val,
        )

        if not should:
            logger.debug("CC skip %s: %s", symbol, reason)
            return None

        # Apply MVRV adjustment to sizing
        mvrv_mult = self.get_mvrv_adjustment("covered_calls")
        adjusted_spot_qty = available_spot_qty * mvrv_mult

        # Allocation check
        spot_value = current_price * adjusted_spot_qty
        allowed, alloc_reason = self._risk_mgr.check_allocation("cc", spot_value)
        if not allowed:
            logger.debug("CC allocation blocked %s: %s", symbol, alloc_reason)
            return None

        # Calculate entry
        cycle = self._cc_mgr.calculate_entry(
            symbol=symbol,
            current_price=current_price,
            spot_quantity=adjusted_spot_qty,
            iv=regime.iv,
            rv_7d=regime.rv_7d,
        )

        return {
            "sub_strategy": "covered_calls",
            "action": "open",
            "symbol": symbol,
            "cycle": cycle,
            "spot_value": spot_value,
            "daily_atr": atr_val,
        }

    # ------------------------------------------------------------------
    # Cash-secured put evaluation
    # ------------------------------------------------------------------

    async def evaluate_cash_secured_put(
        self,
        symbol: str,
        available_usdt: float,
        equity: float,
    ) -> Optional[dict]:
        """Evaluate and potentially create a CSP position."""
        regime = self._regimes.get(symbol)
        if not regime:
            return None

        halted, _ = self._risk_mgr.is_trading_halted()
        if halted:
            return None

        sub_halted, _ = self._risk_mgr.is_sub_strategy_halted("csp")
        if sub_halted:
            return None

        daily_buf = self._buffers.get(symbol, {}).get("1d")
        if not daily_buf or len(daily_buf) < 20:
            return None

        closes = daily_buf.get_closes()
        rsi_arr = rsi(closes, 14)
        current_price = self._prices.get(symbol, 0)

        if current_price <= 0:
            return None

        rsi_val = float(rsi_arr[-1]) if not np.isnan(rsi_arr[-1]) else 50.0

        # 7-day price change
        if len(closes) >= 7:
            price_7d_ago = closes[-7]
            price_7d_change = (closes[-1] - price_7d_ago) / price_7d_ago * 100
        else:
            price_7d_change = 0.0

        should, reason = self._csp_mgr.should_enter(
            symbol=symbol,
            iv_rv_ratio=regime.iv_rv_ratio,
            rsi_14_daily=rsi_val,
            price_7d_change_pct=price_7d_change,
            available_usdt=available_usdt,
            equity=equity,
        )

        if not should:
            logger.debug("CSP skip %s: %s", symbol, reason)
            return None

        # Apply MVRV adjustment to USDT sizing
        mvrv_mult = self.get_mvrv_adjustment("cash_secured_puts")
        adjusted_usdt = available_usdt * mvrv_mult

        # Calculate entry
        cycle = self._csp_mgr.calculate_entry(
            symbol=symbol,
            current_price=current_price,
            available_usdt=adjusted_usdt,
            equity=equity,
            iv=regime.iv,
            rv_7d=regime.rv_7d,
        )

        allowed, alloc_reason = self._risk_mgr.check_allocation(
            "csp", cycle.reserved_usdt
        )
        if not allowed:
            logger.debug("CSP allocation blocked %s: %s", symbol, alloc_reason)
            return None

        highs = daily_buf.get_highs()
        lows = daily_buf.get_lows()
        daily_atr_arr = atr(highs, lows, closes, 14)
        atr_val = float(daily_atr_arr[-1]) if not np.isnan(daily_atr_arr[-1]) else 0

        return {
            "sub_strategy": "cash_secured_puts",
            "action": "open",
            "symbol": symbol,
            "cycle": cycle,
            "daily_atr": atr_val,
        }

    # ------------------------------------------------------------------
    # Delta-neutral evaluation
    # ------------------------------------------------------------------

    async def evaluate_delta_neutral(
        self,
        symbol: str,
        deribit_connected: bool,
        deribit_balance_ok: bool,
    ) -> Optional[dict]:
        """Evaluate delta-neutral entry (Deribit required)."""
        if self._mode != MODE_FULL or not self._dn_mgr:
            return None

        regime = self._regimes.get(symbol)
        if not regime:
            return None

        halted, _ = self._risk_mgr.is_trading_halted()
        if halted:
            return None

        base_asset = symbol.replace("USDT", "")

        should, reason = self._dn_mgr.should_enter(
            symbol=base_asset,
            iv_rv_ratio=regime.iv_rv_ratio,
            deribit_connected=deribit_connected,
            deribit_balance_ok=deribit_balance_ok,
        )

        if not should:
            logger.debug("DN skip %s: %s", symbol, reason)
            return None

        return {
            "sub_strategy": "delta_neutral",
            "action": "evaluate",
            "symbol": symbol,
            "base_asset": base_asset,
            "regime": regime,
        }

    # ------------------------------------------------------------------
    # Exit checks
    # ------------------------------------------------------------------

    async def check_all_exits(self) -> List[dict]:
        """Check exit conditions for all active positions across sub-strategies.

        Returns list of exit actions to execute.
        """
        all_actions: List[dict] = []

        for symbol in self._symbols:
            current_price = self._prices.get(symbol, 0)
            if current_price <= 0:
                continue

            regime = self._regimes.get(symbol)
            current_iv = regime.iv if regime else 0
            rv_7d = regime.rv_7d if regime else 0

            # Get daily ATR
            daily_buf = self._buffers.get(symbol, {}).get("1d")
            atr_val = 0.0
            if daily_buf and len(daily_buf) > 14:
                closes = daily_buf.get_closes()
                highs = daily_buf.get_highs()
                lows = daily_buf.get_lows()
                atr_arr = atr(highs, lows, closes, 14)
                if not np.isnan(atr_arr[-1]):
                    atr_val = float(atr_arr[-1])

            # Covered call exits
            cc_exits = self._cc_mgr.check_exits(
                symbol, current_price, atr_val, current_iv, rv_7d,
            )
            for exit_action in cc_exits:
                exit_action["sub_strategy"] = "covered_calls"
                exit_action["symbol"] = symbol
                exit_action["current_price"] = current_price
                all_actions.append(exit_action)

            # CSP exits
            csp_exits = self._csp_mgr.check_exits(
                symbol, current_price, atr_val,
            )
            for exit_action in csp_exits:
                exit_action["sub_strategy"] = "cash_secured_puts"
                exit_action["symbol"] = symbol
                exit_action["current_price"] = current_price
                all_actions.append(exit_action)

            # RV exceeds entry IV check (Section 4.4)
            for cycle in self._cc_mgr.get_active_cycles().values():
                if cycle.symbol == symbol:
                    if self._vol_engine.check_rv_exceeds_entry_iv(
                        symbol, cycle.iv_at_entry
                    ):
                        all_actions.append({
                            "sub_strategy": "covered_calls",
                            "action": "rv_exit",
                            "cycle_id": cycle.cycle_id,
                            "symbol": symbol,
                            "current_price": current_price,
                            "reason": "RV exceeds entry IV — edge lost",
                        })

        # Delta-neutral exits
        if self._dn_mgr:
            for pos_id, pos in self._dn_mgr.get_active_positions().items():
                exit_action = self._dn_mgr.check_exits(pos_id, 0.0)
                if exit_action:
                    exit_action["sub_strategy"] = "delta_neutral"
                    all_actions.append(exit_action)

        return all_actions

    # ------------------------------------------------------------------
    # Circuit breaker checks
    # ------------------------------------------------------------------

    async def check_circuit_breakers(self) -> List[dict]:
        """Check all circuit breakers (IV spike, flash crash).

        Returns list of emergency actions.
        """
        actions: List[dict] = []

        for symbol in self._symbols:
            # IV spike check
            spike, magnitude = self._vol_engine.check_iv_spike(symbol)
            if spike:
                self._risk_mgr.handle_iv_spike()
                actions.append({
                    "type": "iv_spike",
                    "symbol": symbol,
                    "magnitude_pp": magnitude,
                    "action": "close_all_vol_selling",
                })

            # Flash crash check
            crash, drop_pct = self._vol_engine.check_flash_crash(symbol)
            if crash:
                self._risk_mgr.handle_flash_crash()
                actions.append({
                    "type": "flash_crash",
                    "symbol": symbol,
                    "drop_pct": drop_pct,
                    "action": "close_all_positions",
                })

        return actions

    # ------------------------------------------------------------------
    # Cycle management
    # ------------------------------------------------------------------

    async def check_rolling(self) -> List[dict]:
        """Check if any expired cycles should be rolled into new ones.

        Section 8.3: If conditions still favorable, enter new cycle within 1h.
        """
        actions: List[dict] = []

        for symbol in self._symbols:
            regime = self._regimes.get(symbol)
            if not regime:
                continue

            # Check CC rolling
            if self._cc_mgr.should_roll(symbol, regime.iv_rv_ratio):
                actions.append({
                    "sub_strategy": "covered_calls",
                    "action": "roll",
                    "symbol": symbol,
                    "iv_rv_ratio": regime.iv_rv_ratio,
                })

            # Check CSP rolling
            if self._csp_mgr.should_roll(symbol, regime.iv_rv_ratio):
                actions.append({
                    "sub_strategy": "cash_secured_puts",
                    "action": "roll",
                    "symbol": symbol,
                    "iv_rv_ratio": regime.iv_rv_ratio,
                })

        return actions

    # ------------------------------------------------------------------
    # Aggregate metrics
    # ------------------------------------------------------------------

    def get_metrics(self) -> dict:
        """Return comprehensive strategy metrics."""
        cc_metrics = self._cc_mgr.get_metrics()
        csp_metrics = self._csp_mgr.get_metrics()
        dn_metrics = self._dn_mgr.get_metrics() if self._dn_mgr else {}

        total_premium = (
            cc_metrics.get("total_premium", 0)
            + csp_metrics.get("total_premium", 0)
            + dn_metrics.get("total_premium_collected", 0)
        )
        total_pnl = (
            cc_metrics.get("total_pnl", 0)
            + csp_metrics.get("total_pnl", 0)
            + dn_metrics.get("total_net_pnl", 0)
        )

        # Strategy-specific metrics (Section 10.2)
        equity = self._risk_mgr.equity
        specific_metrics = self._metrics_calc.compute_all(equity)

        # Go-live criteria (Section 10.3)
        go_live = self._metrics_calc.evaluate_go_live_criteria(equity)

        return {
            "mode": self._mode,
            "total_premium_collected": round(total_premium, 4),
            "total_pnl": round(total_pnl, 4),
            "covered_calls": cc_metrics,
            "cash_secured_puts": csp_metrics,
            "delta_neutral": dn_metrics,
            "risk": self._risk_mgr.get_state_dict(),
            "regimes": self._vol_engine.get_all_regimes(),
            "mvrv_ratio": self._mvrv_ratio,
            "mvrv_adjustments": dict(self._mvrv_adjustment),
            "strategy_specific_metrics": specific_metrics.to_dict(),
            "dimensional_breakdown": self._metrics_calc.get_dimensional_breakdown(),
            "go_live": go_live,
        }

    def get_positions(self) -> List[dict]:
        """Return all active positions across sub-strategies."""
        positions: List[dict] = []

        for cycle in self._cc_mgr.get_active_cycles().values():
            positions.append({
                "type": "covered_call",
                **cycle.to_dict(),
            })

        for cycle in self._csp_mgr.get_active_cycles().values():
            positions.append({
                "type": "cash_secured_put",
                **cycle.to_dict(),
            })

        if self._dn_mgr:
            for pos in self._dn_mgr.get_active_positions().values():
                positions.append({
                    "type": "delta_neutral",
                    **pos.to_dict(),
                })

        return positions

    def get_state(self) -> dict:
        """Get full state for persistence."""
        return {
            "mode": self._mode,
            "regimes": self._vol_engine.get_all_regimes(),
            "covered_calls": self._cc_mgr.get_state(),
            "cash_secured_puts": self._csp_mgr.get_state(),
            "delta_neutral": self._dn_mgr.get_state() if self._dn_mgr else {},
            "risk": self._risk_mgr.get_full_state(),
            "prices": dict(self._prices),
            "last_regime_assessment": self._last_regime_assessment,
        }
