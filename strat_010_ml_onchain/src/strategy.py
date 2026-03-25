"""Core ML strategy for STRAT-010.

Hourly cycle: calculate features -> run inference -> check signal ->
check persistence -> apply filters -> execute or manage exits.

Entry: P_final > 0.65 LONG, P_final < 0.35 SHORT, else no trade.
Exit: model reversal, 3h uncertainty, tiered TP, hard stop, 48h time exit,
      feature anomaly (exchange net flow reversal).
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from shared.binance_client import BinanceClient
from shared.indicators import IndicatorBuffer, atr, ema
from shared.paper_trading import PaperTradingEngine

from src.feature_engine import FeatureEngine, FeatureSnapshot
from src.ml_models import InferenceResult, ModelManager
from src.fallback_strategy import FallbackStrategy
from src.risk_manager import StrategyRiskManager, TradeOutcome

logger = logging.getLogger(__name__)
trade_logger = logging.getLogger("trade")


# ── Position tracking ──────────────────────────────────────────────────────

@dataclass
class MLPosition:
    """Tracks an open position managed by the ML strategy."""

    trade_id: str
    symbol: str
    direction: str       # "LONG" or "SHORT"
    entry_price: float
    quantity: float
    size_usdt: float
    leverage: int
    entry_time_ms: int
    entry_p_final: float
    entry_confidence: str
    atr_at_entry: float

    # TP/SL levels
    tp1_price: float = 0.0
    tp2_price: float = 0.0
    hard_stop_price: float = 0.0
    trailing_active: bool = False
    trailing_stop_price: float = 0.0
    highest_since_entry: float = 0.0
    lowest_since_entry: float = float("inf")

    # Tranches
    tp1_filled: bool = False
    tp2_filled: bool = False
    remaining_pct: float = 1.0

    # Model exit tracking
    uncertainty_hours: int = 0
    stop_tightened: bool = False

    def age_hours(self) -> float:
        return (time.time() * 1000 - self.entry_time_ms) / 3_600_000

    def to_dict(self) -> dict:
        return {
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "direction": self.direction,
            "entry_price": self.entry_price,
            "quantity": self.quantity,
            "size_usdt": self.size_usdt,
            "leverage": self.leverage,
            "entry_time_ms": self.entry_time_ms,
            "entry_p_final": round(self.entry_p_final, 4),
            "entry_confidence": self.entry_confidence,
            "atr_at_entry": self.atr_at_entry,
            "tp1_price": self.tp1_price,
            "tp2_price": self.tp2_price,
            "hard_stop_price": self.hard_stop_price,
            "trailing_active": self.trailing_active,
            "trailing_stop_price": self.trailing_stop_price,
            "tp1_filled": self.tp1_filled,
            "tp2_filled": self.tp2_filled,
            "remaining_pct": self.remaining_pct,
            "age_hours": round(self.age_hours(), 2),
            "uncertainty_hours": self.uncertainty_hours,
        }


# ── Strategy ───────────────────────────────────────────────────────────────

class MLStrategy:
    """Core ML & On-Chain Quantitative Models strategy.

    Parameters
    ----------
    binance_client : BinanceClient
        Shared Binance REST client.
    model_manager : ModelManager
        ML model inference manager.
    feature_engines : dict[str, FeatureEngine]
        Per-instrument feature engines.
    risk_manager : StrategyRiskManager
        Strategy-specific risk manager.
    paper_engine : PaperTradingEngine | None
        Paper trading engine (None for live).
    params : dict
        Strategy parameters from config.yaml.
    """

    def __init__(
        self,
        binance_client: BinanceClient,
        model_manager: ModelManager,
        feature_engines: Dict[str, FeatureEngine],
        risk_manager: StrategyRiskManager,
        paper_engine: Optional[PaperTradingEngine] = None,
        params: Optional[dict] = None,
    ) -> None:
        self._client = binance_client
        self._models = model_manager
        self._engines = feature_engines
        self._risk = risk_manager
        self._paper = paper_engine
        self._params = params or {}

        # Fallback
        self._fallback = FallbackStrategy()

        # Positions: symbol -> MLPosition
        self._positions: Dict[str, MLPosition] = {}

        # Signal persistence: symbol -> list of recent signals
        self._signal_history: Dict[str, Deque[str]] = {
            sym: deque(maxlen=10) for sym in feature_engines
        }

        # Prediction history for metrics
        self._prediction_history: List[Dict[str, Any]] = []

        # Book ticker cache: symbol -> {bestBid, bestAsk}
        self._book_tickers: Dict[str, Dict[str, float]] = {}

        # Order book cache: symbol -> snapshot
        self._order_books: Dict[str, dict] = {}

        # Params
        self._persistence_hours = self._params.get("signal_persistence_hours", 2)
        self._min_atr_pct = self._params.get("min_atr_pct", 0.5) / 100.0
        self._max_holding_hours = self._params.get("max_holding_hours", 48)
        self._tighten_hours = self._params.get("tighten_stop_hours", 24)
        self._tighten_atr_mult = self._params.get("tighten_stop_atr_mult", 1.0)
        self._limit_timeout = self._params.get("limit_order_timeout_seconds", 30)
        self._min_features = self._params.get("min_features_available", 25)

        # TP/SL multipliers
        self._tp1_pct = self._params.get("tp_tranche1_pct", 0.30)
        self._tp1_atr = self._params.get("tp_tranche1_atr_mult", 1.5)
        self._tp2_pct = self._params.get("tp_tranche2_pct", 0.30)
        self._tp2_atr = self._params.get("tp_tranche2_atr_mult", 3.0)
        self._trail_act_atr = self._params.get("trailing_activation_atr_mult", 2.0)
        self._trail_dist_atr = self._params.get("trailing_distance_atr_mult", 1.5)
        self._hard_stop_atr = self._params.get("hard_stop_atr_mult", 2.0)

        logger.info("MLStrategy initialised with %d instruments", len(feature_engines))

    # ══════════════════════════════════════════════════════════════════
    #  WS callbacks
    # ══════════════════════════════════════════════════════════════════

    async def on_book_ticker(self, data: dict) -> None:
        """Handle bookTicker updates for execution pricing."""
        symbol = data.get("s", "")
        if symbol:
            self._book_tickers[symbol] = {
                "bestBid": float(data.get("b", 0)),
                "bestAsk": float(data.get("a", 0)),
            }

    async def on_depth(self, symbol: str, data: dict) -> None:
        """Handle depth updates -- delegate to feature engine."""
        engine = self._engines.get(symbol)
        if engine:
            engine.on_depth(data)
        self._order_books[symbol] = data

    async def on_mark_price(self, data: dict) -> None:
        """Handle markPrice for funding rate tracking."""
        symbol = data.get("s", "")
        engine = self._engines.get(symbol)
        if engine:
            funding = float(data.get("r", 0))
            predicted = float(data.get("P", 0))
            premium = float(data.get("p", 0)) if "p" in data else np.nan
            engine.update_derivatives(
                funding_rate=funding,
                predicted_funding=predicted,
                premium_index=premium,
            )

    async def on_liquidation(self, symbol: str, data: dict) -> None:
        """Handle forceOrder events."""
        engine = self._engines.get(symbol)
        if engine:
            engine.on_liquidation(data)

    async def on_kline(self, symbol: str, timeframe: str, data: dict) -> None:
        """Handle kline updates. Trigger hourly evaluation on 1h close."""
        engine = self._engines.get(symbol)
        if engine is None:
            return

        kline = data.get("k", data)
        is_closed = kline.get("x", False)

        if is_closed:
            candle = {
                "timestamp": kline.get("t", 0),
                "open": kline.get("o", 0),
                "high": kline.get("h", 0),
                "low": kline.get("l", 0),
                "close": kline.get("c", 0),
                "volume": kline.get("v", 0),
            }
            engine.on_kline(timeframe, candle)

            # Trigger hourly evaluation on 1h close
            if timeframe == "1h":
                await self._hourly_evaluation(symbol)

        # Update price tracking for open positions
        if symbol in self._positions:
            current_price = float(kline.get("c", 0))
            if current_price > 0:
                pos = self._positions[symbol]
                pos.highest_since_entry = max(pos.highest_since_entry, current_price)
                pos.lowest_since_entry = min(pos.lowest_since_entry, current_price)

    # ══════════════════════════════════════════════════════════════════
    #  Hourly evaluation cycle
    # ══════════════════════════════════════════════════════════════════

    async def _hourly_evaluation(self, symbol: str) -> None:
        """Main hourly cycle for one instrument."""
        engine = self._engines.get(symbol)
        if engine is None:
            return

        logger.info("=== Hourly evaluation: %s ===", symbol)

        # Check if strategy is halted
        if self._risk.is_halted:
            logger.info("Strategy halted -- skipping evaluation for %s", symbol)
            return

        # Check model freshness
        if self._models.should_halt_for_staleness():
            self._risk.halt("Model staleness > 60 days")
            return
        self._risk.set_model_freshness(self._models.get_freshness_multiplier())

        # 1. Calculate features
        snapshot = engine.calculate_features()
        logger.info(
            "[%s] Features: %d/%d available, freshness=%s, anomalies=%d",
            symbol, snapshot.available_count, 35,
            snapshot.freshness_ok, len(snapshot.anomaly_flags),
        )

        # Feature availability check
        if snapshot.available_count < self._min_features:
            logger.warning(
                "[%s] Only %d features available (need %d) -- skipping",
                symbol, snapshot.available_count, self._min_features,
            )
            return

        # 2. Check if we have an open position -> manage exits first
        if symbol in self._positions:
            await self._manage_position(symbol, engine, snapshot)
            return  # Don't evaluate new entry while in a position

        # 3. Run model inference
        inference = await self._run_inference(symbol, engine)
        if inference is None:
            return

        # Record prediction
        self._prediction_history.append({
            "timestamp_ms": inference.timestamp_ms,
            "symbol": symbol,
            "p_final": inference.p_final,
            "signal": inference.signal,
            "confidence": inference.confidence,
        })

        # 4. Check signal
        if inference.signal == "NONE":
            self._signal_history[symbol].append("NONE")
            logger.info("[%s] No signal (P=%.4f in uncertainty zone)", symbol, inference.p_final)
            return

        # 5. Signal persistence
        self._signal_history[symbol].append(inference.signal)
        if not self._check_persistence(symbol, inference.signal):
            logger.info(
                "[%s] Signal %s not persistent yet (need %d consecutive)",
                symbol, inference.signal, self._persistence_hours,
            )
            return

        # 6. Apply filters
        if not await self._apply_filters(symbol, inference, engine):
            return

        # 7. Execute entry
        await self._execute_entry(symbol, inference, engine)

    async def _run_inference(
        self, symbol: str, engine: FeatureEngine,
    ) -> Optional[InferenceResult]:
        """Run ML model inference, falling back to rules if needed."""
        # Check for catastrophic model failure
        if self._models.consecutive_failures >= 3:
            self._fallback.activate("3+ consecutive inference failures")
            self._risk.record_inference_failure()

        if self._fallback.is_active:
            fb_signal = self._fallback.evaluate(engine.buffers["4h"])
            logger.info("[%s] Fallback signal: %s", symbol, fb_signal.signal)
            # Convert to InferenceResult format
            result = InferenceResult(
                p_final=0.8 if fb_signal.signal == "LONG" else (0.2 if fb_signal.signal == "SHORT" else 0.5),
                signal=fb_signal.signal,
                confidence=fb_signal.confidence,
                timestamp_ms=int(time.time() * 1000),
            )
            return result if fb_signal.signal != "NONE" else None

        try:
            xgb_features = engine.get_lagged_feature_vector()
            lstm_sequence = engine.get_sequence_for_lstm(
                self._params.get("lstm_sequence_length", 24)
            )

            inference = await self._models.predict(xgb_features, lstm_sequence)
            self._risk.record_inference_success()

            # If model was in fallback, check if we can deactivate
            if self._fallback.is_active and inference.xgb_available:
                self._fallback.deactivate()

            return inference

        except Exception:
            logger.exception("[%s] Model inference exception", symbol)
            self._risk.record_inference_failure()
            return None

    # ── Signal persistence ───────────────────────────────────────────

    def _check_persistence(self, symbol: str, signal: str) -> bool:
        """Check that the same signal has persisted for N consecutive hours."""
        history = list(self._signal_history[symbol])
        needed = self._persistence_hours
        if len(history) < needed:
            return False
        return all(h == signal for h in history[-needed:])

    # ── Filters ──────────────────────────────────────────────────────

    async def _apply_filters(
        self, symbol: str, inference: InferenceResult, engine: FeatureEngine,
    ) -> bool:
        """Apply all entry filters. Returns True if trade is allowed."""

        # Trend alignment
        trend = engine.get_trend_alignment()
        if inference.signal == "LONG" and trend != "LONG":
            if inference.confidence != "HIGH":
                logger.info("[%s] LONG blocked: trend=%s, confidence not HIGH", symbol, trend)
                return False
        elif inference.signal == "SHORT" and trend != "SHORT":
            if inference.confidence != "HIGH":
                logger.info("[%s] SHORT blocked: trend=%s, confidence not HIGH", symbol, trend)
                return False

        # Volatility check
        atr_val = engine.get_atr_4h()
        price = engine.get_current_price()
        if np.isnan(atr_val) or np.isnan(price) or price <= 0:
            logger.info("[%s] ATR or price unavailable", symbol)
            return False
        atr_pct = atr_val / price
        if atr_pct < self._min_atr_pct:
            logger.info(
                "[%s] ATR %.4f%% < min %.4f%% -- insufficient volatility",
                symbol, atr_pct * 100, self._min_atr_pct * 100,
            )
            return False

        # Whipsaw protection
        if self._risk.check_whipsaw(symbol):
            return False

        # Fee threshold: expected profit must exceed 3x round-trip fees
        # Approximate: 2 * 0.04% taker = 0.08% round-trip, 3x = 0.24%
        min_expected = 0.0024 * price
        expected_profit = atr_val * self._tp1_atr * self._tp1_pct  # conservative estimate
        if expected_profit < min_expected:
            logger.info("[%s] Expected profit below 3x fee threshold", symbol)
            return False

        return True

    # ── Entry execution ──────────────────────────────────────────────

    async def _execute_entry(
        self, symbol: str, inference: InferenceResult, engine: FeatureEngine,
    ) -> None:
        """Execute a trade entry with LIMIT->MARKET fallback."""
        atr_val = engine.get_atr_4h()
        price = engine.get_current_price()
        if np.isnan(atr_val) or np.isnan(price):
            return

        atr_pct = atr_val / price
        stop_distance_pct = self._hard_stop_atr * atr_val / price

        # Calculate position size
        equity = self._paper.get_equity() if self._paper else 10000.0
        size_usdt, leverage, reason = self._risk.calculate_position_size(
            equity=equity,
            stop_distance_pct=stop_distance_pct,
            confidence=inference.confidence,
            atr_pct=atr_pct,
        )

        if size_usdt <= 0:
            logger.info("[%s] Position size zero: %s", symbol, reason)
            return

        quantity = size_usdt / price
        side = "BUY" if inference.signal == "LONG" else "SELL"

        # Calculate TP/SL levels
        if inference.signal == "LONG":
            tp1 = price + atr_val * self._tp1_atr
            tp2 = price + atr_val * self._tp2_atr
            hard_stop = price - atr_val * self._hard_stop_atr
        else:
            tp1 = price - atr_val * self._tp1_atr
            tp2 = price - atr_val * self._tp2_atr
            hard_stop = price + atr_val * self._hard_stop_atr

        # Execute order: LIMIT at best bid/ask, 30s timeout, then MARKET
        fill_price = await self._place_entry_order(symbol, side, quantity, price)
        if fill_price is None:
            return

        # Create position record
        trade_id = f"S010-{symbol}-{uuid.uuid4().hex[:8]}"
        position = MLPosition(
            trade_id=trade_id,
            symbol=symbol,
            direction=inference.signal,
            entry_price=fill_price,
            quantity=quantity,
            size_usdt=size_usdt,
            leverage=leverage,
            entry_time_ms=int(time.time() * 1000),
            entry_p_final=inference.p_final,
            entry_confidence=inference.confidence,
            atr_at_entry=atr_val,
            tp1_price=tp1,
            tp2_price=tp2,
            hard_stop_price=hard_stop,
            highest_since_entry=fill_price,
            lowest_since_entry=fill_price,
        )
        self._positions[symbol] = position

        # Place stop loss on exchange
        await self._place_stop_loss(symbol, position)

        trade_logger.info(
            "ENTRY\t%s\t%s\t%s\tprice=%.6f\tqty=%.6f\tsize=%.2f\tlev=%d\t"
            "P=%.4f\tconf=%s\tatr=%.6f\tstop=%.6f\ttp1=%.6f\ttp2=%.6f",
            trade_id, symbol, inference.signal, fill_price, quantity,
            size_usdt, leverage, inference.p_final, inference.confidence,
            atr_val, hard_stop, tp1, tp2,
        )

    async def _place_entry_order(
        self, symbol: str, side: str, quantity: float, target_price: float,
    ) -> Optional[float]:
        """Place LIMIT order, fallback to MARKET after timeout."""
        if self._paper:
            ob = self._order_books.get(symbol, {"bids": [[target_price, 100]], "asks": [[target_price, 100]]})
            result = self._paper.simulate_market_order(symbol, side, quantity, ob)
            return result.fill_price

        # Live: try LIMIT first
        try:
            book = self._book_tickers.get(symbol, {})
            limit_price = book.get("bestBid" if side == "BUY" else "bestAsk", target_price)

            order = await self._client.place_futures_order(
                symbol=symbol,
                side=side,
                type="LIMIT",
                quantity=quantity,
                price=limit_price,
                time_in_force="GTC",
            )
            order_id = order.get("orderId")

            # Wait for fill
            await asyncio.sleep(self._limit_timeout)

            # Check if filled
            open_orders = await self._client.get_futures_open_orders(symbol)
            still_open = any(o.get("orderId") == order_id for o in open_orders)

            if still_open:
                # Cancel and fallback to MARKET
                await self._client.cancel_futures_order(symbol, order_id)
                order = await self._client.place_futures_order(
                    symbol=symbol,
                    side=side,
                    type="MARKET",
                    quantity=quantity,
                )
                return float(order.get("avgPrice", target_price))
            else:
                return limit_price

        except Exception:
            logger.exception("Entry order failed for %s", symbol)
            return None

    async def _place_stop_loss(self, symbol: str, pos: MLPosition) -> None:
        """Place server-side STOP_MARKET order."""
        if self._paper:
            return  # Paper trading handles stops differently

        try:
            close_side = "SELL" if pos.direction == "LONG" else "BUY"
            await self._client.place_futures_order(
                symbol=symbol,
                side=close_side,
                type="STOP_MARKET",
                quantity=pos.quantity,
                stop_price=pos.hard_stop_price,
                reduce_only=True,
            )
        except Exception:
            logger.exception("Failed to place stop loss for %s", symbol)

    # ══════════════════════════════════════════════════════════════════
    #  Position management (exits)
    # ══════════════════════════════════════════════════════════════════

    async def _manage_position(
        self, symbol: str, engine: FeatureEngine, snapshot: FeatureSnapshot,
    ) -> None:
        """Evaluate all exit conditions for an open position."""
        pos = self._positions.get(symbol)
        if pos is None:
            return

        price = engine.get_current_price()
        if np.isnan(price):
            return

        # 1. Model-based exit: re-evaluate model
        inference = await self._run_inference(symbol, engine)
        if inference is not None:
            exit_reason = self._check_model_exit(pos, inference)
            if exit_reason:
                await self._close_position(symbol, price, exit_reason)
                return

        # 2. Time-based exit: 48h max
        if pos.age_hours() >= self._max_holding_hours:
            await self._close_position(symbol, price, f"Time exit: {pos.age_hours():.1f}h > {self._max_holding_hours}h")
            return

        # 3. Tighten stop after 24h if break-even or better
        if pos.age_hours() >= self._tighten_hours and not pos.stop_tightened:
            unrealised = self._calc_unrealised(pos, price)
            if unrealised >= 0:
                atr_val = pos.atr_at_entry
                if pos.direction == "LONG":
                    new_stop = price - atr_val * self._tighten_atr_mult
                    if new_stop > pos.hard_stop_price:
                        pos.hard_stop_price = new_stop
                        pos.stop_tightened = True
                        logger.info("[%s] Stop tightened to %.6f (24h+, break-even)", symbol, new_stop)
                else:
                    new_stop = price + atr_val * self._tighten_atr_mult
                    if new_stop < pos.hard_stop_price:
                        pos.hard_stop_price = new_stop
                        pos.stop_tightened = True
                        logger.info("[%s] Stop tightened to %.6f (24h+, break-even)", symbol, new_stop)

        # 4. Feature anomaly exit: exchange net flow reversal
        if self._check_feature_anomaly_exit(pos, engine):
            await self._close_position(symbol, price, "Feature anomaly: exchange net flow reversal")
            return

        # 5. Check tiered TP
        await self._check_take_profit(symbol, pos, price)

        # 6. Check hard stop (for paper mode)
        if self._paper:
            if pos.direction == "LONG" and price <= pos.hard_stop_price:
                await self._close_position(symbol, price, "Hard stop hit (LONG)")
                self._risk.record_stop_loss(symbol)
                return
            elif pos.direction == "SHORT" and price >= pos.hard_stop_price:
                await self._close_position(symbol, price, "Hard stop hit (SHORT)")
                self._risk.record_stop_loss(symbol)
                return

        # 7. Trailing stop
        if pos.trailing_active:
            trail_hit = self._check_trailing_stop(pos, price)
            if trail_hit:
                await self._close_position(symbol, price, "Trailing stop hit")
                return

    def _check_model_exit(self, pos: MLPosition, inference: InferenceResult) -> Optional[str]:
        """Check model-based exit conditions."""
        # Reversal: position is LONG but model now predicts SHORT (P < 0.40)
        if pos.direction == "LONG" and inference.p_final < 0.40:
            return f"Model reversal: P={inference.p_final:.4f} < 0.40 (LONG position)"

        if pos.direction == "SHORT" and inference.p_final > 0.60:
            return f"Model reversal: P={inference.p_final:.4f} > 0.60 (SHORT position)"

        # Uncertainty for 3 consecutive hours
        if 0.40 <= inference.p_final <= 0.60:
            pos.uncertainty_hours += 1
            if pos.uncertainty_hours >= 3:
                return f"Uncertainty exit: P={inference.p_final:.4f} uncertain for {pos.uncertainty_hours}h"
        else:
            pos.uncertainty_hours = 0

        return None

    def _check_feature_anomaly_exit(self, pos: MLPosition, engine: FeatureEngine) -> bool:
        """Check if on-chain features signal a regime change against the position."""
        net_flow = engine.onchain.exchange_net_flow

        if np.isnan(net_flow):
            return False

        # LONG position but massive exchange inflow (selling pressure)
        if pos.direction == "LONG" and net_flow > 2.0:  # >2 sigma inflow
            logger.warning(
                "[%s] Exchange net flow reversal: %.2f (LONG position)",
                pos.symbol, net_flow,
            )
            return True

        # SHORT position but massive exchange outflow (accumulation)
        if pos.direction == "SHORT" and net_flow < -2.0:
            logger.warning(
                "[%s] Exchange net flow reversal: %.2f (SHORT position)",
                pos.symbol, net_flow,
            )
            return True

        return False

    async def _check_take_profit(self, symbol: str, pos: MLPosition, price: float) -> None:
        """Check and execute tiered take-profit."""
        atr_val = pos.atr_at_entry

        if pos.direction == "LONG":
            profit_atr = (price - pos.entry_price) / atr_val if atr_val > 0 else 0

            # TP1: 30% at 1.5x ATR
            if not pos.tp1_filled and price >= pos.tp1_price:
                close_qty = pos.quantity * self._tp1_pct
                await self._partial_close(symbol, close_qty, price, "TP1 hit (1.5x ATR)")
                pos.tp1_filled = True
                pos.remaining_pct -= self._tp1_pct

            # TP2: 30% at 3x ATR
            if not pos.tp2_filled and price >= pos.tp2_price:
                close_qty = pos.quantity * self._tp2_pct
                await self._partial_close(symbol, close_qty, price, "TP2 hit (3x ATR)")
                pos.tp2_filled = True
                pos.remaining_pct -= self._tp2_pct

            # Activate trailing stop at 2x ATR
            if not pos.trailing_active and profit_atr >= self._trail_act_atr:
                pos.trailing_active = True
                pos.trailing_stop_price = price - atr_val * self._trail_dist_atr
                logger.info("[%s] Trailing stop activated at %.6f", symbol, pos.trailing_stop_price)

        else:  # SHORT
            profit_atr = (pos.entry_price - price) / atr_val if atr_val > 0 else 0

            if not pos.tp1_filled and price <= pos.tp1_price:
                close_qty = pos.quantity * self._tp1_pct
                await self._partial_close(symbol, close_qty, price, "TP1 hit (1.5x ATR)")
                pos.tp1_filled = True
                pos.remaining_pct -= self._tp1_pct

            if not pos.tp2_filled and price <= pos.tp2_price:
                close_qty = pos.quantity * self._tp2_pct
                await self._partial_close(symbol, close_qty, price, "TP2 hit (3x ATR)")
                pos.tp2_filled = True
                pos.remaining_pct -= self._tp2_pct

            if not pos.trailing_active and profit_atr >= self._trail_act_atr:
                pos.trailing_active = True
                pos.trailing_stop_price = price + atr_val * self._trail_dist_atr
                logger.info("[%s] Trailing stop activated at %.6f", symbol, pos.trailing_stop_price)

    def _check_trailing_stop(self, pos: MLPosition, price: float) -> bool:
        """Update trailing stop and check if hit."""
        atr_val = pos.atr_at_entry

        if pos.direction == "LONG":
            # Update trailing stop higher
            new_trail = price - atr_val * self._trail_dist_atr
            if new_trail > pos.trailing_stop_price:
                pos.trailing_stop_price = new_trail
            return price <= pos.trailing_stop_price
        else:
            new_trail = price + atr_val * self._trail_dist_atr
            if new_trail < pos.trailing_stop_price:
                pos.trailing_stop_price = new_trail
            return price >= pos.trailing_stop_price

    # ── Close / Partial close ────────────────────────────────────────

    async def _close_position(self, symbol: str, price: float, reason: str) -> None:
        """Fully close a position."""
        pos = self._positions.get(symbol)
        if pos is None:
            return

        pnl = self._calc_unrealised(pos, price)

        if self._paper:
            self._paper.close_position(symbol, price)
        else:
            close_side = "SELL" if pos.direction == "LONG" else "BUY"
            remaining_qty = pos.quantity * pos.remaining_pct
            try:
                await self._client.cancel_all_futures_orders(symbol)
                await self._client.place_futures_order(
                    symbol=symbol,
                    side=close_side,
                    type="MARKET",
                    quantity=remaining_qty,
                    reduce_only=True,
                )
            except Exception:
                logger.exception("Failed to close position %s", symbol)

        # Record trade outcome
        pnl_pct = pnl / pos.size_usdt if pos.size_usdt > 0 else 0
        is_correct = (pos.direction == "LONG" and price > pos.entry_price) or \
                     (pos.direction == "SHORT" and price < pos.entry_price)

        outcome = TradeOutcome(
            symbol=symbol,
            direction=pos.direction,
            pnl=pnl,
            pnl_pct=pnl_pct,
            predicted_p=pos.entry_p_final,
            actual_direction_correct=is_correct,
            timestamp_ms=int(time.time() * 1000),
        )
        self._risk.record_trade(outcome)

        trade_logger.info(
            "EXIT\t%s\t%s\t%s\tentry=%.6f\texit=%.6f\tpnl=%.4f\tpnl_pct=%.4f%%\t"
            "reason=%s\tholding=%.1fh",
            pos.trade_id, symbol, pos.direction, pos.entry_price, price,
            pnl, pnl_pct * 100, reason, pos.age_hours(),
        )

        del self._positions[symbol]

    async def _partial_close(
        self, symbol: str, qty: float, price: float, reason: str,
    ) -> None:
        """Partially close a position (tranche TP)."""
        pos = self._positions.get(symbol)
        if pos is None:
            return

        if self._paper:
            # Paper mode: simulate the partial close
            pass  # PnL tracked at full close
        else:
            close_side = "SELL" if pos.direction == "LONG" else "BUY"
            try:
                await self._client.place_futures_order(
                    symbol=symbol,
                    side=close_side,
                    type="MARKET",
                    quantity=qty,
                    reduce_only=True,
                )
            except Exception:
                logger.exception("Partial close failed for %s", symbol)

        trade_logger.info(
            "PARTIAL_CLOSE\t%s\t%s\tqty=%.6f\tprice=%.6f\treason=%s",
            pos.trade_id, symbol, qty, price, reason,
        )

    def _calc_unrealised(self, pos: MLPosition, price: float) -> float:
        """Calculate unrealised PnL for a position."""
        if pos.direction == "LONG":
            return (price - pos.entry_price) * pos.quantity * pos.remaining_pct
        else:
            return (pos.entry_price - price) * pos.quantity * pos.remaining_pct

    # ══════════════════════════════════════════════════════════════════
    #  Public API (for main.py and dashboard)
    # ══════════════════════════════════════════════════════════════════

    def get_positions(self) -> List[dict]:
        """Return open positions as dicts."""
        return [pos.to_dict() for pos in self._positions.values()]

    def get_position_count(self) -> int:
        return len(self._positions)

    def get_prediction_history(self, limit: int = 100) -> List[dict]:
        return self._prediction_history[-limit:]

    def get_state(self) -> Dict[str, Any]:
        """Export full state for persistence."""
        return {
            "positions": {sym: pos.to_dict() for sym, pos in self._positions.items()},
            "signal_history": {sym: list(hist) for sym, hist in self._signal_history.items()},
            "prediction_count": len(self._prediction_history),
            "fallback_active": self._fallback.is_active,
        }
