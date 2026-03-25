"""STRAT-001 Main Entry Point.

Lifecycle:
  1. Load config (YAML + env overrides)
  2. Initialise all components (client, WS manager, risk, filters, strategy, exits, scaling)
  3. Warm up indicators (200 candles x 4 TF x 10 instruments)
  4. Reconcile with Binance (Section 8.2)
  5. Register with WS manager (STRAT-001, 80 streams)
  6. Start event loop, heartbeat, state persistence, dashboard
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Shared library imports
from shared.config_loader import ConfigLoader, BotConfig
from shared.binance_client import BinanceClient, BinanceClientError
from shared.binance_ws_manager import WebSocketManager
from shared.rate_limiter import RateLimiter, Priority, ApiType
from shared.risk_manager import RiskManager, CrossStrategyReader
from shared.database import DatabaseManager
from shared.state_persistence import StatePersistence
from shared.heartbeat import HeartbeatMonitor
from shared.memory_manager import MemoryManager
from shared.log_manager import setup_logging
from shared.alerting import AlertLevel
from shared.reconciliation import StartupReconciler
from shared.performance_tracker import (
    PerformanceTracker, DimensionalBreakdown, GoLiveCriteriaChecker,
)
from shared.paper_trading import PaperTradingEngine
from shared.utils import TimeSync, ExchangeInfo

# Strategy imports
from . import STRATEGY_ID, STRATEGY_NAME
from .strategy import TrendFollowingStrategy, SignalDirection, SignalStatus
from .exit_manager import ExitManager, ExitReason
from .scaling import ScalingManager
from .risk_manager import TrendRiskManager
from .filters import FilterEngine
from .dashboard import TrendDashboard
from .strategy_metrics import TrendStrategyMetrics

logger = logging.getLogger(__name__)
trade_logger = logging.getLogger("trade")
system_logger = logging.getLogger("system")

# ---------------------------------------------------------------------------
# WebSocket stream definitions (Section 2.1)
# ---------------------------------------------------------------------------

KLINE_TIMEFRAMES = ("1m", "15m", "4h", "1d")
STREAM_TYPES = (
    "kline_1m", "kline_15m", "kline_4h", "kline_1d",
    "depth20@100ms", "aggTrade", "markPrice@1s", "bookTicker",
)
# Note: forceOrder streams are subscribed but handled at the WS manager level
STREAMS_PER_INSTRUMENT = 8
MAX_INSTRUMENTS = 10
MAX_STREAMS = STREAMS_PER_INSTRUMENT * MAX_INSTRUMENTS  # 80


# ---------------------------------------------------------------------------
# TrendBot
# ---------------------------------------------------------------------------

class TrendBot:
    """Main orchestrator for STRAT-001 Trend Following & Momentum.

    Parameters
    ----------
    config_path : str
        Path to config.yaml.
    """

    def __init__(self, config_path: str = "config.yaml") -> None:
        self.base_path = Path(__file__).resolve().parent.parent
        self.config_path = self.base_path / config_path

        # Load configuration
        self.config_loader = ConfigLoader(str(self.config_path))
        self.config: BotConfig = self.config_loader.config
        self.cfg = self.config.strategy_params

        # Setup logging
        setup_logging(
            strategy_id=STRATEGY_ID,
            log_dir=str(self.base_path / self.config.logging.log_dir),
            level=self.config.logging.level,
        )

        logger.info("=" * 60)
        logger.info("STRAT-001 Trend Following & Momentum — Starting")
        logger.info("Mode: %s", self.config.mode)
        logger.info("Instruments: %s", self.config.instruments)
        logger.info("=" * 60)

        # Core components (initialised in start())
        self.time_sync = TimeSync()
        self.exchange_info = ExchangeInfo()
        self.rate_limiter: Optional[RateLimiter] = None
        self.client: Optional[BinanceClient] = None
        self.ws_manager: Optional[WebSocketManager] = None
        self.db: Optional[DatabaseManager] = None
        self.state_persistence: Optional[StatePersistence] = None
        self.heartbeat: Optional[HeartbeatMonitor] = None
        self.memory_mgr: Optional[MemoryManager] = None
        self.paper: Optional[PaperTradingEngine] = None
        self.alerter: Optional[Any] = None
        self.perf_tracker: Optional[PerformanceTracker] = None
        self.reconciler: Optional[StartupReconciler] = None

        # Strategy components
        self.shared_risk: Optional[RiskManager] = None
        self.risk_mgr: Optional[TrendRiskManager] = None
        self.filters: Optional[FilterEngine] = None
        self.strategy: Optional[TrendFollowingStrategy] = None
        self.exit_mgr: Optional[ExitManager] = None
        self.scaling_mgr: Optional[ScalingManager] = None
        self.dashboard: Optional[TrendDashboard] = None
        self.strat_metrics: Optional[TrendStrategyMetrics] = None
        self.dimensional: Optional[DimensionalBreakdown] = None
        self.go_live_checker: Optional[GoLiveCriteriaChecker] = None

        # Runtime state
        self._running = False
        self._shutdown_event = asyncio.Event()
        self._last_equity_update = 0.0
        self._last_health_check = 0.0
        self._ws_reconnect_cooldown_until = 0.0

        # Best bid/ask cache for order placement
        self._book_ticker: Dict[str, Dict[str, float]] = {}

        # Mark price cache
        self._mark_prices: Dict[str, float] = {}

    # ======================================================================
    # Initialisation
    # ======================================================================

    async def _init_components(self) -> None:
        """Initialise all sub-components."""

        # Database
        db_url = self.config.database.url
        if not db_url.startswith("/") and ":///" in db_url:
            # Make relative SQLite paths absolute to base_path
            parts = db_url.split(":///", 1)
            if len(parts) == 2 and not parts[1].startswith("/"):
                db_url = f"{parts[0]}:///{self.base_path / parts[1]}"
        self.db = DatabaseManager(db_url)

        # Rate limiter
        self.rate_limiter = RateLimiter(
            budget=self.config.rate_limit_weight_per_min,
            burst=self.config.rate_limit_burst_weight,
        )

        # Binance REST client
        self.client = BinanceClient(
            api_key=self.config.binance.api_key,
            api_secret=self.config.binance.api_secret,
            time_sync=self.time_sync,
            exchange_info=self.exchange_info,
            rate_limiter=self.rate_limiter,
            futures_base_url=self.config.binance.futures_base_url,
            recv_window=self.config.binance.recv_window,
        )
        await self.client.start()

        # Time sync
        await self.client.sync_time()

        # Exchange info
        await self.client.load_exchange_info(self.config.instruments)

        # Alerting
        from shared.alerting import AlertManager
        self.alerter = AlertManager(self.config.alerting, "STRAT-001")

        # Paper trading engine
        if self.config.mode == "paper":
            self.paper = PaperTradingEngine(
                starting_equity=self.config.paper_trading.starting_equity,
                maker_fee_pct=self.config.paper_trading.maker_fee_pct,
                taker_fee_pct=self.config.paper_trading.taker_fee_pct,
            )

        # Shared risk manager
        cross_reader = CrossStrategyReader(
            state_dir=str(self.base_path / self.config.state.state_dir)
        )
        self.shared_risk = RiskManager(
            config=self.config.risk,
            database_manager=self.db,
            cross_strategy_reader=cross_reader,
        )

        # Set initial equity
        if self.config.mode == "paper":
            equity = self.config.paper_trading.starting_equity
        else:
            account = await self.client.get_futures_account()
            equity = float(account.get("totalWalletBalance", 0))
        self.shared_risk.update_equity(equity)

        # Strategy-specific risk manager (pass alerter for Kelly halt alerts)
        self.risk_mgr = TrendRiskManager(self.cfg, self.shared_risk, alerter=self.alerter)

        # Filters
        self.filters = FilterEngine(self.cfg, self.shared_risk)

        # Exit manager (with trade-close callback for metrics/dimensional/go-live)
        self.exit_mgr = ExitManager(
            self.cfg, self.client, self.alerter, self.paper,
            on_trade_close_callback=self._on_trade_close,
        )

        # Strategy engine
        self.strategy = TrendFollowingStrategy(
            config=self.cfg,
            client=self.client,
            risk_manager=self.risk_mgr,
            filters=self.filters,
            exit_manager=self.exit_mgr,
            alerter=self.alerter,
            paper_engine=self.paper,
        )

        # Scaling manager
        self.scaling_mgr = ScalingManager(
            config=self.cfg,
            client=self.client,
            exit_manager=self.exit_mgr,
            risk_manager=self.risk_mgr,
            alerter=self.alerter,
            paper_engine=self.paper,
        )

        # Performance tracker
        self.perf_tracker = PerformanceTracker(
            strategy_id=STRATEGY_ID,
            risk_free_rate=0.05,
        )

        # Strategy-specific metrics (Section 10.2)
        self.strat_metrics = TrendStrategyMetrics()

        # Dimensional breakdowns (Section 10.3)
        self.dimensional = DimensionalBreakdown()

        # Go-live criteria checker (Section 9.4)
        if self.config.mode == "paper":
            self.go_live_checker = GoLiveCriteriaChecker(
                tracker=self.perf_tracker,
                min_days=self.cfg.get("paper_min_days", 60),
                min_trades=self.cfg.get("paper_min_trades", 50),
                min_win_rate=self.cfg.get("paper_min_win_rate", 35.0) / 100.0,
                min_profit_factor=self.cfg.get("paper_min_profit_factor", 1.3),
                max_drawdown=self.cfg.get("paper_max_drawdown", 12.0),
                min_sharpe=self.cfg.get("paper_min_sharpe", 0.8),
                max_single_loss_pct=self.cfg.get("paper_max_single_loss_pct", 2.5),
            )
            self.go_live_checker.set_paper_start()
            logger.info("Go-live criteria checker initialised for paper mode")

        # State persistence
        self.state_persistence = StatePersistence(
            state_dir=str(self.base_path / self.config.state.state_dir),
            strategy_id=STRATEGY_ID,
            save_interval=self.config.state.persistence_interval,
            max_snapshots=self.config.state.snapshot_count,
        )

        # Heartbeat monitor
        self.heartbeat = HeartbeatMonitor(
            strategy_id=STRATEGY_ID,
            interval=self.config.heartbeat.interval,
            timeout=self.config.heartbeat.timeout,
            max_restarts_per_hour=self.config.heartbeat.max_restarts_per_hour,
        )

        # Memory manager
        self.memory_mgr = MemoryManager(
            check_interval=self.config.memory.check_interval,
            warn_mb=self.config.memory.warn_mb,
            restart_mb=self.config.memory.restart_mb,
        )

        # WebSocket manager
        self.ws_manager = WebSocketManager(
            futures_ws_url=self.config.binance.futures_ws_url,
            binance_client=self.client,
        )

        logger.info("All components initialised")

    # ======================================================================
    # Warm-up (200 candles x 4 TF x 10 instruments)
    # ======================================================================

    async def _warm_up_indicators(self) -> None:
        """Fetch historical klines and populate indicator buffers."""
        logger.info("Warming up indicators: 200 candles x %d TF x %d instruments",
                     len(KLINE_TIMEFRAMES), len(self.config.instruments))

        for symbol in self.config.instruments:
            for tf in KLINE_TIMEFRAMES:
                interval_map = {
                    "1m": "1m", "15m": "15m", "4h": "4h", "1d": "1d"
                }
                interval = interval_map.get(tf, tf)

                try:
                    klines = await self.client.get_futures_klines(
                        symbol=symbol,
                        interval=interval,
                        limit=200,
                    )

                    buf = self.strategy.buffers[symbol][tf]
                    for k in klines:
                        buf.add_candle({
                            "timestamp": k[0],
                            "open": float(k[1]),
                            "high": float(k[2]),
                            "low": float(k[3]),
                            "close": float(k[4]),
                            "volume": float(k[5]),
                        })

                    logger.debug(
                        "Warmed up %s %s: %d candles", symbol, tf, len(buf)
                    )

                except Exception as e:
                    logger.error("Warm-up failed for %s %s: %s", symbol, tf, e)

                # Small delay to respect rate limits
                await asyncio.sleep(0.05)

        # Calculate initial Hurst exponents
        for symbol in self.config.instruments:
            buf_1d = self.strategy.buffers[symbol].get("1d")
            if buf_1d and len(buf_1d) >= 100:
                self.filters.update_hurst(symbol, buf_1d.get_closes())

        logger.info("Indicator warm-up complete")

    # ======================================================================
    # Reconciliation (Section 8.2)
    # ======================================================================

    async def _reconcile(self) -> None:
        """Run startup reconciliation against Binance state."""
        logger.info("Starting reconciliation...")

        self.reconciler = StartupReconciler(
            client=self.client,
            state_persistence=self.state_persistence,
            exit_manager=self.exit_mgr,
            strategy_id=STRATEGY_ID,
        )

        try:
            result = await asyncio.wait_for(
                self.reconciler.reconcile(),
                timeout=60.0,
            )

            if result.has_issues:
                logger.warning(
                    "Reconciliation found issues: orphans=%d closed=%d "
                    "cancelled=%d fills=%d discrepancies=%d (took %dms)",
                    len(result.orphan_positions), len(result.closed_positions),
                    len(result.cancelled_orders), len(result.detected_fills),
                    len(result.discrepancies), result.duration_ms,
                )
                if self.alerter:
                    await self.alerter.send(
                        level=AlertLevel.WARNING,
                        title="Reconciliation discrepancies",
                        message=f"Reconciliation completed with {len(result.discrepancies)} discrepancies",
                    )
            else:
                logger.info("Reconciliation complete — no issues (%dms)", result.duration_ms)

        except asyncio.TimeoutError:
            logger.critical("Reconciliation timed out at 60s — halting")
            if self.alerter:
                await self.alerter.send(
                    level=AlertLevel.EMERGENCY,
                    title="Reconciliation timeout",
                    message="Reconciliation timeout — manual intervention required",
                )
            raise

    # ======================================================================
    # WebSocket registration (Section 12.1)
    # ======================================================================

    async def _register_ws_streams(self) -> None:
        """Register all required streams with the WS manager."""
        from shared.binance_ws_manager import ConnectionType

        subscriptions = []
        for symbol in self.config.instruments:
            sym_lower = symbol.lower()
            subscriptions.extend([
                (f"{sym_lower}@kline_1m", self._on_kline),
                (f"{sym_lower}@kline_15m", self._on_kline),
                (f"{sym_lower}@kline_4h", self._on_kline),
                (f"{sym_lower}@kline_1d", self._on_kline),
                (f"{sym_lower}@depth20@100ms", self._on_depth),
                (f"{sym_lower}@aggTrade", self._on_agg_trade),
                (f"{sym_lower}@markPrice@1s", self._on_mark_price),
                (f"{sym_lower}@bookTicker", self._on_book_ticker),
            ])

        logger.info(
            "Registering %d streams for %d instruments with WS manager",
            len(subscriptions), len(self.config.instruments),
        )

        self.ws_manager.register_strategy(
            strategy_id=STRATEGY_ID,
            subscriptions=subscriptions,
            conn_type=ConnectionType.FUTURES,
        )

        # Also register user data stream
        self.ws_manager.register_strategy(
            strategy_id=f"{STRATEGY_ID}_user",
            subscriptions=[
                ("ORDER_TRADE_UPDATE", self._on_user_data),
                ("ACCOUNT_UPDATE", self._on_user_data),
            ],
            conn_type=ConnectionType.FUTURES_USER,
        )

    # ======================================================================
    # WS message handlers
    # ======================================================================

    async def _on_kline(self, data: dict) -> None:
        """Handle kline messages from all timeframes."""
        kline = data.get("k", {})
        symbol = kline.get("s", "")
        interval = kline.get("i", "")

        # Map Binance intervals to our TF names
        tf_map = {"1m": "1m", "15m": "15m", "4h": "4h", "1d": "1d"}
        tf = tf_map.get(interval)
        if not tf or symbol not in self.config.instruments:
            return

        await self.strategy.on_kline(symbol, tf, kline)

        # On 4h close, process new signals
        if kline.get("x") and tf == "4h":
            await self._process_4h_close(symbol)

        # On 15m close, check pullback entries
        if kline.get("x") and tf == "15m":
            await self._process_15m_close(symbol)

    async def _on_agg_trade(self, data: dict) -> None:
        """Handle aggTrade messages."""
        symbol = data.get("s", "")
        if symbol in self.config.instruments:
            await self.strategy.on_agg_trade(symbol, data)

    async def _on_mark_price(self, data: dict) -> None:
        """Handle mark price updates for PnL tracking."""
        symbol = data.get("s", "")
        mark = float(data.get("p", 0))
        if symbol in self.config.instruments and mark > 0:
            self._mark_prices[symbol] = mark

            # Update exit manager with current price
            if symbol in self.exit_mgr.positions:
                indicators = self._get_current_indicators(symbol)
                await self.exit_mgr.process_tick(symbol, mark, indicators)

    async def _on_book_ticker(self, data: dict) -> None:
        """Handle bookTicker for spread monitoring and order placement."""
        symbol = data.get("s", "")
        if symbol not in self.config.instruments:
            return

        bid = float(data.get("b", 0))
        ask = float(data.get("a", 0))
        self._book_ticker[symbol] = {"bid": bid, "ask": ask}
        self.filters.update_spread(symbol, bid, ask)

    async def _on_depth(self, data: dict) -> None:
        """Handle depth updates (for liquidity assessment)."""
        pass  # Used for paper trading fill simulation

    async def _on_force_order(self, data: dict) -> None:
        """Handle liquidation events for cascade detection."""
        order = data.get("o", {})
        symbol = order.get("s", "")
        qty = float(order.get("q", 0))
        price = float(order.get("p", 0))
        side = order.get("S", "")

        system_logger.info(
            "LIQUIDATION %s: %s qty=%.4f price=%.4f", symbol, side, qty, price
        )

    async def _on_user_data(self, data: dict) -> None:
        """Handle user data stream events."""
        event_type = data.get("e", "")

        if event_type == "ORDER_TRADE_UPDATE":
            order_data = data.get("o", {})
            await self.strategy.on_order_update(order_data)

            # Check if this is an exit fill
            status = order_data.get("X", "")
            if status == "FILLED":
                symbol = order_data.get("s", "")
                # Reconcile position after fill
                await self._reconcile_position(symbol)

        elif event_type == "ACCOUNT_UPDATE":
            balances = data.get("a", {}).get("B", [])
            for bal in balances:
                if bal.get("a") == "USDT":
                    equity = float(bal.get("wb", 0))
                    self.shared_risk.update_equity(equity)

        elif event_type == "MARGIN_CALL":
            logger.critical("MARGIN CALL received: %s", data)
            if self.alerter:
                await self.alerter.send(
                    level=AlertLevel.EMERGENCY,
                    title="MARGIN CALL",
                    message=f"MARGIN CALL: {data}",
                )

    # ======================================================================
    # Signal processing
    # ======================================================================

    async def _process_4h_close(self, symbol: str) -> None:
        """Process a 4h candle close — check for new signals and scale-ins."""
        # Update whipsaw 2-candle confirmation (Section 7.2)
        snap_4h = self.strategy.compute_indicators(symbol, "4h")
        if snap_4h and math.isfinite(snap_4h.ema_20):
            buf_4h = self.strategy.buffers[symbol].get("4h")
            if buf_4h and len(buf_4h) > 0:
                close_price = float(buf_4h.get_closes()[-1])
                self.filters.update_4h_candle(symbol, close_price, snap_4h.ema_20)

        # Section 11.8: Indicator anomaly recalculation
        # When 3 consecutive anomalies occur, halt and recalculate from raw klines
        if self.strategy.is_halted_by_anomaly(symbol):
            logger.warning(
                "%s halted by indicator anomaly — initiating recalculation from raw klines",
                symbol,
            )
            await self._recalculate_indicators_from_raw(symbol)

        # Check for scale-in opportunities on existing positions
        if symbol in self.exit_mgr.positions:
            snap = snap_4h if snap_4h else self.strategy.compute_indicators(symbol, "4h")
            opp = self.scaling_mgr.evaluate_scale_in(
                symbol=symbol,
                current_price=self._mark_prices.get(symbol, 0),
                ema_20_4h=snap.ema_20,
            )
            if opp:
                await self.scaling_mgr.execute_scale_in(opp)

        # New signal is already detected in strategy.on_kline
        signal = self.strategy.pending_signals.get(symbol)
        if signal and signal.status == SignalStatus.PENDING_PULLBACK:
            logger.info(
                "New pending signal: %s %s — waiting for 15m pullback",
                symbol, signal.direction.value,
            )

    async def _process_15m_close(self, symbol: str) -> None:
        """Process a 15m candle close — check pullback entry conditions."""
        signal = self.strategy.pending_signals.get(symbol)
        if signal is None or signal.status != SignalStatus.PENDING_PULLBACK:
            return

        current_price = self._mark_prices.get(symbol, 0)
        if current_price <= 0:
            return

        ready = await self.strategy.check_pullback_entry(symbol, signal, current_price)
        if not ready:
            # Check if signal was cancelled/expired
            if signal.status in (SignalStatus.CANCELLED, SignalStatus.EXPIRED):
                self.strategy.pending_signals.pop(symbol, None)
            return

        # Run full filter pipeline
        book = self._book_ticker.get(symbol, {})
        filter_result = self.filters.evaluate(
            symbol=symbol,
            direction=signal.direction,
            entry_price=current_price,
            atr_value=signal.atr_value,
            best_bid=book.get("bid", 0),
            best_ask=book.get("ask", 0),
        )

        if not filter_result.passed:
            logger.info("%s signal rejected by filters: %s", symbol, filter_result.reject_reason)
            self.strategy.pending_signals.pop(symbol, None)
            return

        # Apply filter size adjustments
        total_multiplier = signal.size_multiplier * filter_result.size_multiplier

        # Calculate position size
        size_result = self.risk_mgr.calculate_position_size(
            symbol=symbol,
            direction=signal.direction,
            entry_price=current_price,
            atr_value=signal.atr_value,
            adx_value=signal.adx_value,
            volume_ratio=signal.volume_ratio,
            size_multiplier=total_multiplier,
        )

        if size_result.rejected:
            logger.info("%s sizing rejected: %s", symbol, size_result.reject_reason)
            self.strategy.pending_signals.pop(symbol, None)
            return

        # Check with shared risk manager
        allowed, reason = self.risk_mgr.check_entry_allowed(
            strategy_id=STRATEGY_ID,
            symbol=symbol,
            direction=signal.direction.value,
            size_usdt=size_result.size_usdt,
            leverage=size_result.leverage,
        )

        if not allowed:
            logger.info("%s entry denied by risk manager: %s", symbol, reason)
            self.strategy.pending_signals.pop(symbol, None)
            return

        # Re-check spread right before order (Section 7.1)
        if not self.filters.recheck_spread_before_order(symbol):
            logger.info("%s entry cancelled: spread check failed at order time", symbol)
            return  # Don't remove signal — try again next candle

        # Execute entry
        best_price = book.get("bid", current_price) if signal.direction == SignalDirection.LONG \
            else book.get("ask", current_price)

        fill_result = await self.strategy.execute_entry(
            signal=signal,
            position_size_qty=size_result.size_qty,
            best_price=best_price,
        )

        if fill_result is None:
            logger.warning("%s entry execution failed", symbol)
            self.strategy.pending_signals.pop(symbol, None)
            return

        # Entry successful — place bracket orders
        fill_price = float(fill_result.get("avgPrice", current_price))
        fill_qty = float(fill_result.get("executedQty", size_result.size_qty))

        # Handle partial fill (Section 11.2)
        if fill_qty < size_result.size_qty * 0.5:
            logger.warning(
                "%s partial fill: %.6f / %.6f (< 50%%) — closing partial",
                symbol, fill_qty, size_result.size_qty,
            )
            # Close partial if profitable, hold with tight stop if within 0.3% of entry
            pnl_pct = abs(fill_price - current_price) / fill_price * 100
            if pnl_pct > 0.3:
                side = "SELL" if signal.direction == SignalDirection.LONG else "BUY"
                try:
                    if not self.paper:
                        await self.client.place_futures_order(
                            symbol=symbol, side=side, type="MARKET",
                            quantity=fill_qty, reduce_only=True,
                        )
                except Exception as e:
                    logger.error("Partial fill close failed for %s: %s", symbol, e)
                self.strategy.pending_signals.pop(symbol, None)
                return
            # else: proceed with tightened stop

        # Place bracket orders (stop + TP) within 500ms
        bracket = await self.strategy.place_bracket_orders(
            symbol=symbol,
            direction=signal.direction,
            entry_price=fill_price,
            quantity=fill_qty,
            atr_value=signal.atr_value,
        )

        # Create managed position
        stop_order_id = None
        if bracket.get("stop_order") and isinstance(bracket["stop_order"], dict):
            stop_order_id = bracket["stop_order"].get("orderId")

        pos = self.exit_mgr.create_position(
            symbol=symbol,
            direction=signal.direction,
            entry_price=fill_price,
            quantity=fill_qty,
            atr_value=signal.atr_value,
            leverage=size_result.leverage,
            fees=float(fill_result.get("fees", 0)),
            stop_order_id=stop_order_id,
        )

        # Register with scaling manager
        self.scaling_mgr.register_entry(symbol, fill_qty)

        # Report to shared risk manager (within 100ms — Section 12.3)
        self.shared_risk.record_position_change(
            strategy_id=STRATEGY_ID,
            symbol=symbol,
            direction=signal.direction.value,
            size_usdt=fill_qty * fill_price,
            is_open=True,
        )

        # Alert
        if self.alerter:
            await self.alerter.send(
                level=AlertLevel.INFO,
                title=f"ENTRY {symbol} {signal.direction.value}",
                message=(
                    f"ENTRY {symbol} {signal.direction.value}: qty={fill_qty:.6f} "
                    f"@ {fill_price:.4f}, stop={pos.hard_stop_price:.4f}, "
                    f"TP1={pos.tp.tp1_price:.4f}, lev={size_result.leverage}x, "
                    f"risk={size_result.risk_pct:.2f}%"
                ),
            )

        trade_logger.info(
            "ENTRY\tsymbol=%s\tdir=%s\tprice=%.4f\tqty=%.6f\tatr=%.4f\t"
            "adx=%.1f\trsi=%.1f\tvol_r=%.2f\tlev=%dx\trisk=%.2f%%\t"
            "size_usdt=%.2f\tadjustments=%s",
            symbol, signal.direction.value, fill_price, fill_qty,
            signal.atr_value, signal.adx_value, signal.rsi_value,
            signal.volume_ratio, size_result.leverage, size_result.risk_pct,
            size_result.size_usdt, size_result.adjustments,
        )

        # Remove processed signal
        self.strategy.pending_signals.pop(symbol, None)

    # ======================================================================
    # Helper methods
    # ======================================================================

    def _get_current_indicators(self, symbol: str) -> Dict[str, Any]:
        """Get current indicator values for exit checking."""
        snap = self.strategy.compute_indicators(symbol, "4h")
        if not snap:
            return {}

        # Check for reverse crossover
        prev_fast_above = snap.prev_ema_20 > snap.prev_ema_50 if (
            snap.prev_ema_20 and snap.prev_ema_50
        ) else None
        curr_fast_above = snap.ema_20 > snap.ema_50

        cross_reversed = False
        pos = self.exit_mgr.positions.get(symbol)
        if pos and prev_fast_above is not None:
            if pos.direction == SignalDirection.LONG and prev_fast_above and not curr_fast_above:
                cross_reversed = True
            elif pos.direction == SignalDirection.SHORT and not prev_fast_above and curr_fast_above:
                cross_reversed = True

        return {
            "ema_20": snap.ema_20,
            "ema_50": snap.ema_50,
            "ema_200": snap.ema_200,
            "rsi": snap.rsi_value,
            "adx": snap.adx_value,
            "atr": snap.atr_value,
            "cross_reversed": cross_reversed,
        }

    async def _reconcile_position(self, symbol: str) -> None:
        """Reconcile actual vs expected position size after a fill."""
        if self.paper:
            return

        try:
            account = await self.client.get_futures_account()
            for p in account.get("positions", []):
                if p.get("symbol") == symbol:
                    actual_qty = abs(float(p.get("positionAmt", 0)))
                    pos = self.exit_mgr.positions.get(symbol)
                    if pos and abs(pos.remaining_quantity - actual_qty) > 0.001:
                        logger.warning(
                            "%s position mismatch: local=%.6f actual=%.6f",
                            symbol, pos.remaining_quantity, actual_qty,
                        )
                        pos.remaining_quantity = actual_qty
                    break
        except Exception as e:
            logger.warning("Post-fill reconciliation failed for %s: %s", symbol, e)

    # ======================================================================
    # Section 11.8: Indicator anomaly recalculation from raw kline data
    # ======================================================================

    async def _recalculate_indicators_from_raw(self, symbol: str) -> None:
        """When 3 consecutive indicator anomalies occur, fetch 200 fresh candles
        per timeframe and rebuild all indicator buffers from scratch.
        Only resume after successful recalculation."""
        logger.info("Recalculating indicators for %s from raw kline data (200 candles)", symbol)

        if self.alerter:
            await self.alerter.send(
                level=AlertLevel.WARNING,
                title="Indicator anomaly",
                message=f"Indicator anomaly halt for {symbol} — recalculating from raw data",
            )

        success = True
        for tf in KLINE_TIMEFRAMES:
            try:
                klines = await self.client.get_futures_klines(
                    symbol=symbol,
                    interval=tf,
                    limit=200,
                )

                # Clear and rebuild the buffer
                buf = self.strategy.buffers[symbol][tf]
                buf.clear()
                for k in klines:
                    buf.add_candle({
                        "timestamp": k[0],
                        "open": float(k[1]),
                        "high": float(k[2]),
                        "low": float(k[3]),
                        "close": float(k[4]),
                        "volume": float(k[5]),
                    })

                logger.info(
                    "Rebuilt %s %s buffer: %d candles from raw data",
                    symbol, tf, len(buf),
                )

            except Exception as e:
                logger.error(
                    "Failed to recalculate %s %s from raw data: %s", symbol, tf, e
                )
                success = False

            await asyncio.sleep(0.05)  # Rate limit respect

        if success:
            # Reset anomaly counter — strategy resumes for this symbol
            self.strategy.reset_anomaly_count(symbol)
            logger.info(
                "%s indicator recalculation complete — anomaly halt cleared", symbol
            )
            if self.alerter:
                await self.alerter.send(
                    level=AlertLevel.INFO,
                    title="Indicator recalculation OK",
                    message=f"{symbol} indicator recalculation successful — resuming",
                )
        else:
            logger.critical(
                "%s indicator recalculation FAILED — symbol remains halted", symbol
            )
            if self.alerter:
                await self.alerter.send(
                    level=AlertLevel.CRITICAL,
                    title="Indicator recalculation failed",
                    message=f"CRITICAL: {symbol} indicator recalculation failed — manual intervention required",
                )

    # ======================================================================
    # Trade close handler (Section 10.2, 10.3, 9.4)
    # ======================================================================

    async def _on_trade_close(self, trade_record: Dict[str, Any]) -> None:
        """Handle post-trade-close processing: metrics, dimensional, go-live check."""
        # Enrich trade record with additional context for metrics
        symbol = trade_record.get("symbol", "")
        direction = trade_record.get("direction", "")

        # Add standard fields expected by PerformanceTracker
        if "trade_id" not in trade_record:
            trade_record["trade_id"] = f"{symbol}_{trade_record.get('entry_time_ms', 0)}"
        if "side" not in trade_record:
            trade_record["side"] = direction
        if "pnl" not in trade_record:
            trade_record["pnl"] = trade_record.get("realized_pnl", 0)
        if "pnl_pct" not in trade_record:
            entry_p = trade_record.get("entry_price", 0)
            exit_p = trade_record.get("exit_price", 0)
            if entry_p > 0:
                mult = 1 if direction == "LONG" else -1
                trade_record["pnl_pct"] = mult * ((exit_p - entry_p) / entry_p) * 100
            else:
                trade_record["pnl_pct"] = 0

        # Record stop-loss events for whipsaw 2-candle confirmation
        exit_reason = trade_record.get("exit_reason", "")
        if exit_reason in ("HARD_STOP", "TRAILING_STOP"):
            self.filters.record_stop_loss(symbol, direction)

        # Record in strategy-specific metrics (Section 10.2)
        if self.strat_metrics:
            self.strat_metrics.record_trade(trade_record)

        # Record in dimensional breakdowns (Section 10.3)
        if self.dimensional:
            self.dimensional.record_trade_dimensional(trade_record)

        # Record in performance tracker
        if self.perf_tracker:
            self.perf_tracker.record_trade(trade_record)

        # Record in risk manager
        self.risk_mgr.record_trade_result(trade_record)

        # Go-live criteria check (Section 9.4)
        if self.go_live_checker:
            ready, results = self.go_live_checker.check()
            if ready:
                logger.info(
                    "Go-live criteria met! Results: %s", results
                )
                if self.alerter:
                    await self.alerter.send(
                        level=AlertLevel.INFO,
                        title="Go-live criteria met",
                        message=f"Go-live criteria met — all checks passed. Results: {results}",
                    )

    def switch_mode(self, new_mode: str) -> bool:
        """Attempt to switch trading mode. Prevents switching to 'live'
        if go-live criteria are not met (Section 9.4).

        Returns True if mode switch succeeded, False if blocked.
        """
        if new_mode == "live" and self.go_live_checker:
            ready, results = self.go_live_checker.check()
            if not ready:
                logger.warning(
                    "Cannot switch to live mode — go-live criteria not met: %s",
                    {k: v for k, v in results.items() if not v["passed"]},
                )
                return False

        self.config.mode = new_mode
        logger.info("Mode switched to: %s", new_mode)
        return True

    # ======================================================================
    # Periodic tasks
    # ======================================================================

    async def _periodic_equity_update(self) -> None:
        """Update equity every 60 seconds (Section 5.1)."""
        while self._running:
            try:
                if self.config.mode == "paper" and self.paper:
                    equity = self.paper._equity
                else:
                    account = await self.client.get_futures_account()
                    equity = float(account.get("totalWalletBalance", 0))

                self.shared_risk.update_equity(equity)

                # Check drawdown
                halted, level, dd_pct = self.shared_risk.check_drawdown()
                if halted:
                    logger.critical(
                        "%s drawdown limit hit: %s %.2f%%", STRATEGY_ID, level, dd_pct
                    )
                    if level == "daily":
                        self.risk_mgr.on_daily_drawdown_hit()
                        await self.exit_mgr.close_losing_positions(0.5)
                    elif level == "weekly":
                        self.risk_mgr.on_weekly_drawdown_hit()
                    elif level == "monthly":
                        self.risk_mgr.on_monthly_drawdown_hit()
                        await self.exit_mgr.close_all_positions("Monthly DD limit")
                    elif level == "system":
                        await self.emergency_shutdown("System-wide drawdown limit")

            except Exception as e:
                logger.warning("Equity update error: %s", e)

            await asyncio.sleep(60)

    async def _periodic_health_check(self) -> None:
        """Periodic health checks: orders, config reload, Hurst recalc."""
        while self._running:
            try:
                # Config hot-reload
                if self.config_loader.check_reload():
                    self.cfg = self.config_loader.config.strategy_params
                    logger.info("Config reloaded — effective on next trade")

                # Health check open orders every 30s (Section 2.2)
                if not self.paper:
                    open_orders = await self.client.get_futures_open_orders()
                    # Verify all expected orders still exist
                    for symbol, pos in self.exit_mgr.positions.items():
                        if pos.hard_stop_order_id:
                            found = any(
                                o.get("orderId") == pos.hard_stop_order_id
                                for o in open_orders
                            )
                            if not found:
                                logger.warning(
                                    "%s hard stop order %s missing — replacing",
                                    symbol, pos.hard_stop_order_id,
                                )
                                # Re-place the stop
                                side = "SELL" if pos.direction == SignalDirection.LONG else "BUY"
                                try:
                                    order = await self.client.place_futures_order(
                                        symbol=symbol,
                                        side=side,
                                        type="STOP_MARKET",
                                        quantity=pos.remaining_quantity,
                                        stop_price=pos.hard_stop_price,
                                        reduce_only=True,
                                    )
                                    pos.hard_stop_order_id = order.get("orderId")
                                except BinanceClientError as e:
                                    logger.error(
                                        "%s CRITICAL: cannot replace missing stop: %s",
                                        symbol, e,
                                    )

            except Exception as e:
                logger.warning("Health check error: %s", e)

            await asyncio.sleep(30)

    async def _periodic_hurst_update(self) -> None:
        """Recalculate Hurst exponents daily at 00:00 UTC (Section 7.7)."""
        while self._running:
            for symbol in self.config.instruments:
                buf_1d = self.strategy.buffers[symbol].get("1d")
                if buf_1d and len(buf_1d) >= 100:
                    self.filters.update_hurst(symbol, buf_1d.get_closes())
            await asyncio.sleep(86400)  # Once per day

    async def _periodic_correlation_update(self) -> None:
        """Update correlation matrix weekly (Section 7.5)."""
        while self._running:
            try:
                daily_closes = {}
                for symbol in self.config.instruments:
                    buf_1d = self.strategy.buffers[symbol].get("1d")
                    if buf_1d and len(buf_1d) >= 30:
                        daily_closes[symbol] = buf_1d.get_closes().tolist()

                if len(daily_closes) >= 2:
                    self.shared_risk.correlation_matrix.update(daily_closes)
                    logger.info("Correlation matrix updated for %d symbols", len(daily_closes))
            except Exception as e:
                logger.warning("Correlation update error: %s", e)

            await asyncio.sleep(7 * 86400)  # Weekly

    async def _state_persistence_loop(self) -> None:
        """Persist state every 5 seconds (Section 8.1)."""
        while self._running:
            try:
                state = {
                    "positions": self.exit_mgr.get_positions_state(),
                    "pending_signals": self.strategy.get_pending_signals_state(),
                    "indicators": {},  # Indicator values per symbol/tf
                    "risk_state": self.risk_mgr.get_state(),
                    "filter_state": self.filters.get_state(),
                    "drawdown": self.shared_risk._drawdown.to_dict() if self.shared_risk._drawdown else {},
                    "timestamp": int(time.time() * 1000),
                }
                await self.state_persistence.save(state)
            except Exception as e:
                logger.warning("State persistence error: %s", e)

            await asyncio.sleep(self.config.state.persistence_interval)

    # ======================================================================
    # Emergency shutdown
    # ======================================================================

    async def emergency_shutdown(self, reason: str) -> None:
        """Kill switch: cancel all orders, close all positions, halt."""
        logger.critical("EMERGENCY SHUTDOWN: %s", reason)

        if self.alerter:
            await self.alerter.send(
                level=AlertLevel.EMERGENCY,
                title="EMERGENCY SHUTDOWN",
                message=f"EMERGENCY SHUTDOWN: {reason}",
            )

        # Close all positions
        closed = await self.exit_mgr.close_all_positions(reason)
        logger.info("Emergency: closed %d positions", closed)

        # Cancel all remaining orders
        if not self.paper:
            for symbol in self.config.instruments:
                try:
                    await self.client.cancel_all_futures_orders(symbol)
                except Exception as e:
                    logger.error("Cancel orders failed for %s: %s", symbol, e)

        self._running = False
        self._shutdown_event.set()

    # ======================================================================
    # Main lifecycle
    # ======================================================================

    async def start(self) -> None:
        """Start the bot: init, warm up, reconcile, run."""
        try:
            # 1. Init all components
            await self._init_components()

            # 2. Warm up indicators
            await self._warm_up_indicators()

            # 3. Reconcile with Binance
            if self.config.mode == "live":
                await self._reconcile()

            # 4. Register WS streams
            await self._register_ws_streams()

            # 5. Start WS connections
            await self.ws_manager.start()

            # 6. Start dashboard
            self.dashboard = TrendDashboard(
                bot=self,
                port=self.config.dashboard.port,
                host=self.config.dashboard.host,
            )
            await self.dashboard.start()

            # 7. Start periodic tasks
            self._running = True
            tasks = [
                asyncio.create_task(self._periodic_equity_update()),
                asyncio.create_task(self._periodic_health_check()),
                asyncio.create_task(self._periodic_hurst_update()),
                asyncio.create_task(self._periodic_correlation_update()),
                asyncio.create_task(self._state_persistence_loop()),
            ]

            # Start heartbeat
            if self.heartbeat:
                tasks.append(asyncio.create_task(self.heartbeat.start()))

            # Start memory manager
            if self.memory_mgr:
                tasks.append(asyncio.create_task(self.memory_mgr.start()))

            logger.info("=" * 60)
            logger.info("STRAT-001 is LIVE — monitoring %d instruments", len(self.config.instruments))
            logger.info("=" * 60)

            if self.alerter:
                await self.alerter.send(
                    level=AlertLevel.INFO,
                    title="STRAT-001 started",
                    message=(
                        f"STRAT-001 started in {self.config.mode} mode. "
                        f"Monitoring {len(self.config.instruments)} instruments."
                    ),
                )

            # Wait for shutdown signal
            await self._shutdown_event.wait()

            # Cancel tasks
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

        except Exception as e:
            logger.critical("Fatal error: %s", e, exc_info=True)
            if self.alerter:
                await self.alerter.send(
                    level=AlertLevel.EMERGENCY,
                    title="FATAL ERROR",
                    message=str(e),
                )
            raise
        finally:
            await self._cleanup()

    async def _cleanup(self) -> None:
        """Graceful cleanup."""
        logger.info("Cleaning up...")
        self._running = False

        if self.dashboard:
            await self.dashboard.stop()
        if self.ws_manager:
            await self.ws_manager.stop()
        if self.client:
            await self.client.close()
        if self.state_persistence:
            # Final state save
            try:
                state = {
                    "positions": self.exit_mgr.get_positions_state() if self.exit_mgr else [],
                    "shutdown": True,
                    "timestamp": int(time.time() * 1000),
                }
                await self.state_persistence.save(state)
            except Exception:
                pass

        logger.info("STRAT-001 shutdown complete")

    def request_shutdown(self) -> None:
        """Request a graceful shutdown (called from signal handler)."""
        logger.info("Shutdown requested")
        self._shutdown_event.set()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point for STRAT-001."""
    config_path = "config.yaml"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    bot = TrendBot(config_path=config_path)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Signal handlers for graceful shutdown
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, bot.request_shutdown)

    try:
        loop.run_until_complete(bot.start())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        loop.close()


if __name__ == "__main__":
    main()
