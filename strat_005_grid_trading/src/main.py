"""STRAT-005 Grid Trading Bot — Main Entry Point.

Handles:
- Startup reconciliation (critical for 20-50 open orders)
- Warm up: 200 candles per TF (1m, 15m, 4h, 1d)
- WebSocket stream registration including USER DATA STREAM
  (ORDER_TRADE_UPDATE is primary trigger)
- Daily grid refresh task at 00:00 UTC
- Order monitoring every 60 seconds
- State persistence every 5 seconds
- Kill switch, heartbeat, memory management
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent dirs to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.binance_client import BinanceClient, BinanceClientError
from shared.binance_ws_manager import WebSocketManager, ConnectionType
from shared.config_loader import ConfigLoader, BotConfig
from shared.state_persistence import StatePersistence
from shared.rate_limiter import RateLimiter, Priority
from shared.paper_trading import PaperTradingEngine
from shared.performance_tracker import PerformanceTracker
from shared.risk_manager import RiskManager, RiskConfig, CrossStrategyReader
from shared.circuit_breaker import CircuitBreaker
from shared.kill_switch import KillSwitch
from shared.heartbeat import HeartbeatMonitor
from shared.memory_manager import MemoryManager
from shared.log_manager import setup_logging
from shared.alerting import AlertManager
from shared.database import DatabaseManager
from shared.cross_strategy import CrossStrategyManager
from shared.utils import TimeSync, ExchangeInfo
from shared.reconciliation import StartupReconciler

from src.strategy import GridStrategy, GridSide, BreakoutDirection, InstrumentState
from src.grid_manager import GridManager
from src.risk_manager import GridRiskManager
from src.dashboard import GridDashboard
from src.strategy_metrics import GridMetrics

logger = logging.getLogger(__name__)
trade_logger = logging.getLogger("trade")
system_logger = logging.getLogger("system")

STRATEGY_ID = "STRAT-005"
STRATEGY_NAME = "Grid Trading"


class GridTradingBot:
    """Main orchestrator for the STRAT-005 Grid Trading Bot."""

    def __init__(self, config_path: str = "config.yaml") -> None:
        self._config_loader = ConfigLoader(config_path)
        self._cfg: BotConfig = self._config_loader.config
        self._params: dict = self._cfg.strategy_params
        self._running = False
        self._shutdown_event = asyncio.Event()

        # ── Core infrastructure ──
        self._time_sync = TimeSync()
        self._exchange_info = ExchangeInfo()
        self._rate_limiter = RateLimiter(
            budget=self._cfg.rate_limit_weight_per_min,
            burst=self._cfg.rate_limit_burst_weight,
        )

        # ── Binance client ──
        self._client = BinanceClient(
            api_key=self._cfg.binance.api_key,
            api_secret=self._cfg.binance.api_secret,
            time_sync=self._time_sync,
            exchange_info=self._exchange_info,
            rate_limiter=self._rate_limiter,
            spot_base_url=self._cfg.binance.spot_base_url,
            futures_base_url=self._cfg.binance.futures_base_url,
            recv_window=self._cfg.binance.recv_window,
        )

        # ── WebSocket manager ──
        self._ws_manager = WebSocketManager(
            spot_ws_url=self._cfg.binance.spot_ws_url + "/stream",
            futures_ws_url=self._cfg.binance.futures_ws_url + "/stream",
            binance_client=self._client,
        )

        # ── Database ──
        self._db = DatabaseManager(self._cfg.database.url)

        # ── State persistence ──
        self._state = StatePersistence(
            state_dir=self._cfg.state.state_dir,
            strategy_id=STRATEGY_ID,
            save_interval=self._cfg.state.persistence_interval,
            max_snapshots=self._cfg.state.snapshot_count,
        )

        # ── Paper trading ──
        self._paper: Optional[PaperTradingEngine] = None
        if self._cfg.paper_trading.enabled or self._cfg.mode == "paper":
            self._paper = PaperTradingEngine(
                starting_equity=self._cfg.paper_trading.starting_equity,
                maker_fee_pct=self._cfg.paper_trading.maker_fee_pct,
                taker_fee_pct=self._cfg.paper_trading.taker_fee_pct,
            )

        # ── Strategy ──
        self._strategy = GridStrategy(self._params, self._exchange_info)

        # ── Grid manager ──
        self._grid_manager = GridManager(
            strategy=self._strategy,
            binance_client=self._client,
            exchange_info=self._exchange_info,
            paper_engine=self._paper,
            config=self._params,
        )

        # ── Risk management ──
        self._cross_strategy = CrossStrategyManager(
            strategy_id=STRATEGY_ID,
            shared_dir=str(Path(self._cfg.state.state_dir).parent / "shared"),
        )
        shared_risk_config = RiskConfig(
            max_capital_pct=self._cfg.risk.max_capital_pct,
            max_per_trade_pct=self._cfg.risk.max_per_trade_pct,
            max_leverage=self._cfg.risk.max_leverage,
            max_concurrent_positions=self._cfg.risk.max_concurrent_positions,
            max_per_asset_pct=self._cfg.risk.max_per_asset_pct,
            daily_drawdown_pct=self._cfg.risk.daily_drawdown_pct,
            weekly_drawdown_pct=self._cfg.risk.weekly_drawdown_pct,
            monthly_drawdown_pct=self._cfg.risk.monthly_drawdown_pct,
            system_wide_drawdown_pct=self._cfg.risk.system_wide_drawdown_pct,
        )
        self._shared_risk = RiskManager(
            config=shared_risk_config,
            database_manager=self._db,
        )
        self._risk = GridRiskManager(
            shared_risk=self._shared_risk,
            strategy=self._strategy,
            cross_strategy=self._cross_strategy,
            config={**self._cfg.risk.__dict__, **self._params},
        )

        # ── Performance ──
        self._perf = PerformanceTracker(strategy_id=STRATEGY_ID)

        # ── Circuit breaker ──
        self._circuit_breaker = CircuitBreaker(config={
            "flash_crash_pct": self._params.get("flash_crash_pct", 5.0),
            "flash_crash_minutes": self._params.get("flash_crash_minutes", 5),
        })

        # ── Kill switch ──
        self._kill_switch = KillSwitch(
            binance_client=self._client,
            state_persistence=self._state,
            database_manager=self._db,
            max_execution_time=15,
        )

        # ── Heartbeat ──
        self._heartbeat = HeartbeatMonitor(
            strategy_id=STRATEGY_ID,
            interval=self._cfg.heartbeat.interval,
            timeout=self._cfg.heartbeat.timeout,
            max_restarts_per_hour=self._cfg.heartbeat.max_restarts_per_hour,
        )

        # ── Memory ──
        self._memory = MemoryManager(
            check_interval=self._cfg.memory.check_interval,
            warn_mb=self._cfg.memory.warn_mb,
            restart_mb=self._cfg.memory.restart_mb,
        )

        # ── Alerting ──
        self._alerting = AlertManager(self._cfg.alerting, STRATEGY_ID)

        # ── Strategy Metrics (Section 10.2 + 10.3 + 10.4) ──
        self._strategy_metrics = GridMetrics(
            strategy=self._strategy,
            grid_manager=self._grid_manager,
            risk_manager=self._risk,
        )

        # ── Dashboard ──
        template_dir = str(Path(__file__).parent.parent / "templates")
        self._dashboard = GridDashboard(
            host=self._cfg.dashboard.host,
            port=self._cfg.dashboard.port,
            template_dir=template_dir,
            strategy_metrics=self._strategy_metrics,
        )

        # ── Background tasks ──
        self._tasks: List[asyncio.Task] = []

    # ======================================================================
    #  Lifecycle
    # ======================================================================

    async def start(self) -> None:
        """Start the grid trading bot."""
        # Setup logging
        setup_logging(
            STRATEGY_ID,
            log_dir=self._cfg.logging.log_dir,
            level=self._cfg.logging.level,
        )

        logger.info("=" * 70)
        logger.info("  STRAT-005 Grid Trading Bot Starting")
        logger.info("  Mode: %s", self._cfg.mode)
        logger.info("  Instruments: %s", self._cfg.instruments)
        logger.info("=" * 70)

        # Start client
        await self._client.start()
        await self._client.sync_time()

        # Load exchange info
        if self._cfg.mode == "live" or not self._paper:
            await self._client.load_exchange_info(self._cfg.instruments)

        # Initialize equity
        if self._paper:
            equity = self._paper.get_equity()
        else:
            account = await self._client.get_futures_account()
            equity = float(account.get("totalWalletBalance", 0))
        self._risk.update_equity(equity)

        # Load persisted state
        saved = self._state.load()
        grid_state = saved.get("custom", {}).get("grid_manager", {})
        if grid_state:
            self._grid_manager.load_state_dict(grid_state)
            logger.info("Loaded persisted grid state")

        # Startup reconciliation (Section 8.2)
        if not self._paper:
            await self._reconcile()
        else:
            logger.info("Paper mode — skipping exchange reconciliation")

        # Warm up indicators (200 candles per TF)
        await self._warmup_indicators()

        # Initialize instruments
        num_instruments = len(self._cfg.instruments)
        for symbol in self._cfg.instruments:
            if symbol not in self._grid_manager.instruments:
                capital = self._risk.calculate_capital_per_instrument(num_instruments)
                self._grid_manager.init_instrument(symbol, capital)
                logger.info("[%s] Initialized with capital=%.2f", symbol, capital)

        # Register WebSocket streams
        self._register_ws_streams()

        # Start WebSocket
        await self._ws_manager.start()

        # Start state persistence
        await self._state.start()

        # Start heartbeat
        await self._heartbeat.start()

        # Setup dashboard
        self._dashboard.set_providers(
            positions_fn=lambda: GridDashboard.build_positions_data(self._grid_manager.instruments),
            trades_fn=lambda limit: self._perf.get_recent_trades(limit),
            metrics_fn=lambda: self._get_all_metrics(),
            equity_fn=lambda: self._perf.get_metrics().get("equity_curve", []),
            alerts_fn=lambda: self._alerting.get_recent_alerts(),
            config_fn=lambda: {"strategy_params": self._params, "mode": self._cfg.mode},
            kill_fn=lambda reason: self._kill_switch.execute(reason),
            grid_data_fn=lambda: GridDashboard.build_grid_data(self._grid_manager.instruments),
            risk_fn=lambda: self._risk.get_risk_summary(self._grid_manager.instruments),
        )
        await self._dashboard.start()

        # Deploy grids for instruments that don't have active grids
        for symbol in self._cfg.instruments:
            state = self._grid_manager.instruments.get(symbol)
            if state and not state.active:
                allowed, reason = self._risk.check_deployment_allowed(
                    symbol, state.allocated_capital,
                )
                if allowed:
                    await self._deploy_instrument(symbol)
                else:
                    logger.warning("[%s] Deployment blocked: %s", symbol, reason)

        # Start background tasks
        self._running = True
        self._tasks.append(asyncio.create_task(self._order_monitor_loop(), name="order_monitor"))
        self._tasks.append(asyncio.create_task(self._daily_refresh_loop(), name="daily_refresh"))
        self._tasks.append(asyncio.create_task(self._state_persist_loop(), name="state_persist"))
        self._tasks.append(asyncio.create_task(self._risk_check_loop(), name="risk_check"))
        self._tasks.append(asyncio.create_task(self._paper_fill_loop(), name="paper_fill"))
        self._tasks.append(asyncio.create_task(self._memory.start(), name="memory"))

        logger.info("STRAT-005 Grid Trading Bot fully started")
        await self._alerting.send_info("Bot Started", f"Grid Trading bot started in {self._cfg.mode} mode")

    async def stop(self) -> None:
        """Gracefully shut down the bot."""
        logger.info("Shutting down STRAT-005 Grid Trading Bot...")
        self._running = False

        # Cancel background tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

        # Persist final state
        self._persist_state()
        await self._state.stop()

        # Stop components
        self._heartbeat.stop()
        self._dashboard.stop()
        self._memory.stop()
        await self._ws_manager.stop()
        await self._client.close()
        await self._alerting.close()

        logger.info("STRAT-005 Grid Trading Bot stopped")

    # ======================================================================
    #  Startup: Reconciliation (Section 8.2)
    # ======================================================================

    async def _reconcile(self) -> None:
        """Run startup reconciliation for grid orders.

        Section 8.2: Critical for 20-50 open orders.
        Step 1: Fetch all open orders from Binance
        Step 2: Compare with persisted state
        Step 3-7: Resolve discrepancies
        """
        logger.info("=== Grid Startup Reconciliation ===")

        reconciler = StartupReconciler(
            binance_client=self._client,
            state_persistence=self._state,
            database_manager=self._db,
            logger=logger,
        )

        try:
            result = await reconciler.reconcile()
            if result.has_issues:
                logger.warning(
                    "Reconciliation found issues: orphans=%d fills=%d cancelled=%d",
                    len(result.orphan_positions),
                    len(result.detected_fills),
                    len(result.cancelled_orders),
                )
                # Process detected fills for grid state
                for fill in result.detected_fills:
                    if fill.get("action") == "mark_filled" and fill.get("filled_qty", 0) > 0:
                        symbol = fill.get("symbol", "")
                        order_id = str(fill.get("order_id", ""))
                        await self._grid_manager.handle_fill(
                            symbol=symbol,
                            order_id=order_id,
                            side="BUY",  # Direction determined by grid level
                            fill_price=fill.get("avg_price", 0),
                            fill_qty=fill.get("filled_qty", 0),
                            fill_time_ms=int(time.time() * 1000),
                        )
            else:
                logger.info("Reconciliation complete — no issues")

        except Exception as e:
            logger.error("Reconciliation failed: %s", e, exc_info=True)
            await self._alerting.send_critical(
                "Reconciliation Failed",
                f"Startup reconciliation error: {e}",
            )

    # ======================================================================
    #  Startup: Warm up indicators
    # ======================================================================

    async def _warmup_indicators(self) -> None:
        """Fetch 200 historical candles per timeframe per instrument."""
        timeframes = ["1m", "15m", "4h", "1d"]
        logger.info("Warming up indicators: 200 candles per TF for %d instruments", len(self._cfg.instruments))

        for symbol in self._cfg.instruments:
            for tf in timeframes:
                try:
                    klines = await self._client.get_futures_klines(symbol, tf, limit=200)
                    for k in klines:
                        self._strategy.process_kline(symbol, tf, {
                            "t": k[0], "o": k[1], "h": k[2], "l": k[3],
                            "c": k[4], "v": k[5], "T": k[6],
                        })
                    logger.info("[%s] Warmed %s: %d candles", symbol, tf, len(klines))
                except Exception as e:
                    logger.error("[%s] Failed to warm %s: %s", symbol, tf, e)

                # Rate limit courtesy
                await asyncio.sleep(0.2)

    # ======================================================================
    #  Grid deployment
    # ======================================================================

    async def _deploy_instrument(self, symbol: str) -> bool:
        """Calculate range, parameters, and deploy grid for one instrument."""
        state = self._grid_manager.instruments.get(symbol)
        if not state:
            return False

        # Get current price
        price = await self._get_current_price(symbol)
        if price <= 0:
            logger.error("[%s] Cannot deploy: no price data", symbol)
            return False

        state.current_price = price

        # Run all filters
        allowed, reason = self._strategy.check_all_filters(
            symbol, state.best_bid or price, state.best_ask or price,
        )
        if not allowed:
            logger.warning("[%s] Filters blocked deployment: %s", symbol, reason)
            return False

        # Calculate range
        range_result = self._strategy.calculate_range(symbol, price)
        if range_result is None:
            logger.warning("[%s] Range calculation failed", symbol)
            return False

        upper, lower, width = range_result

        # Calculate grid parameters
        params = self._strategy.calculate_grid_parameters(
            symbol, price, upper, lower, width, state.allocated_capital,
        )
        if params is None:
            logger.warning("[%s] Grid parameter calculation failed", symbol)
            return False

        state.grid_params = params

        # Deploy
        success = await self._grid_manager.deploy_grid(symbol)
        if success:
            await self._alerting.send_info(
                f"Grid Deployed: {symbol}",
                f"Range: {lower:.2f}-{upper:.2f} ({width:.1f}%), "
                f"{params.num_levels} levels, spacing {params.grid_spacing_pct:.3f}%",
            )
        return success

    # ======================================================================
    #  WebSocket Registration
    # ======================================================================

    def _register_ws_streams(self) -> None:
        """Register all required WebSocket streams per Section 2."""
        for symbol in self._cfg.instruments:
            sym_lower = symbol.lower()

            # Market data streams (Section 2.1)
            market_streams = [
                (f"{sym_lower}@kline_1m", self._on_kline),
                (f"{sym_lower}@kline_15m", self._on_kline),
                (f"{sym_lower}@kline_4h", self._on_kline),
                (f"{sym_lower}@kline_1d", self._on_kline),
                (f"{sym_lower}@bookTicker", self._on_book_ticker),
                (f"{sym_lower}@markPrice@1s", self._on_mark_price),
                (f"{sym_lower}@aggTrade", self._on_agg_trade),
            ]

            self._ws_manager.register_strategy(
                strategy_id=STRATEGY_ID,
                subscriptions=market_streams,
                conn_type=ConnectionType.FUTURES,
                on_reconnect=self._on_reconnect,
            )

        # User data stream (Section 2.3) — ORDER_TRADE_UPDATE is PRIMARY trigger
        self._ws_manager.register_strategy(
            strategy_id=f"{STRATEGY_ID}_user",
            subscriptions=[
                ("ORDER_TRADE_UPDATE", self._on_order_update),
                ("ACCOUNT_UPDATE", self._on_account_update),
            ],
            conn_type=ConnectionType.FUTURES_USER,
        )

        logger.info("Registered WS streams for %d instruments + user data", len(self._cfg.instruments))

    # ======================================================================
    #  WebSocket Callbacks
    # ======================================================================

    async def _on_kline(self, data: dict) -> None:
        """Process kline updates for indicator buffers."""
        k = data.get("k", data)
        symbol = k.get("s", "")
        interval = k.get("i", "")
        is_closed = k.get("x", False)

        if symbol and interval:
            self._strategy.process_kline(symbol, interval, k)

            # Update current price from 1m close
            if interval == "1m":
                state = self._grid_manager.instruments.get(symbol)
                if state:
                    state.current_price = float(k.get("c", 0))

                    # Paper trading: check fills on every price update
                    if self._paper and state.active:
                        await self._grid_manager.simulate_paper_fills(
                            symbol, state.current_price,
                        )

    async def _on_book_ticker(self, data: dict) -> None:
        """Update best bid/ask for spread monitoring."""
        symbol = data.get("s", "")
        state = self._grid_manager.instruments.get(symbol)
        if state:
            state.best_bid = float(data.get("b", 0))
            state.best_ask = float(data.get("a", 0))

    async def _on_mark_price(self, data: dict) -> None:
        """Update mark price for PnL tracking."""
        symbol = data.get("s", "")
        state = self._grid_manager.instruments.get(symbol)
        if state:
            mark = float(data.get("p", 0))
            state.mark_price = mark
            self._strategy.update_unrealized_pnl(state, mark)

    async def _on_agg_trade(self, data: dict) -> None:
        """Process aggregate trades for volume analysis."""
        symbol = data.get("s", "")
        state = self._grid_manager.instruments.get(symbol)
        if state:
            state.current_price = float(data.get("p", 0))

    async def _on_order_update(self, data: dict) -> None:
        """Handle ORDER_TRADE_UPDATE — the PRIMARY trigger for grid operation.

        Section 3.4: Fill detection drives the next order placement.
        Must place counter-order within 500ms.
        """
        order = data.get("o", data)
        symbol = order.get("s", "")
        order_id = str(order.get("i", order.get("orderId", "")))
        status = order.get("X", order.get("status", ""))
        side = order.get("S", order.get("side", ""))
        fill_price = float(order.get("ap", order.get("avgPrice", 0)))  # Average fill price
        fill_qty = float(order.get("z", order.get("executedQty", 0)))  # Cumulative filled qty
        event_time_ms = int(order.get("T", order.get("updateTime", time.time() * 1000)))

        if status == "FILLED":
            cycle = await self._grid_manager.handle_fill(
                symbol=symbol,
                order_id=order_id,
                side=side,
                fill_price=fill_price,
                fill_qty=fill_qty,
                fill_time_ms=event_time_ms,
            )

            if cycle:
                # Record the cycle as a trade for performance tracking
                self._perf.record_trade({
                    "trade_id": cycle.cycle_id,
                    "symbol": symbol,
                    "side": "LONG",
                    "entry_price": cycle.buy_price,
                    "exit_price": cycle.sell_price,
                    "quantity": cycle.quantity,
                    "pnl": cycle.net_profit,
                    "pnl_pct": cycle.net_profit / (cycle.buy_price * cycle.quantity) * 100.0
                    if cycle.buy_price * cycle.quantity > 0 else 0.0,
                    "fees": cycle.fees,
                    "entry_time_ms": cycle.buy_time_ms,
                    "exit_time_ms": cycle.sell_time_ms,
                })

                if self._paper:
                    self._paper.update_equity(cycle.net_profit)

        elif status == "CANCELED":
            # Section 11.2: Order cancelled by Binance
            logger.info("[%s] Order %s cancelled: %s", symbol, order_id, status)

    async def _on_account_update(self, data: dict) -> None:
        """Handle account balance updates."""
        balances = data.get("a", {}).get("B", [])
        for bal in balances:
            if bal.get("a") == "USDT":
                equity = float(bal.get("wb", 0))
                self._risk.update_equity(equity)

    async def _on_reconnect(self, conn_type: ConnectionType) -> None:
        """Handle WebSocket reconnection — verify grid state."""
        logger.warning("WebSocket reconnected: %s — verifying grid state", conn_type.value)
        for symbol in self._cfg.instruments:
            await self._grid_manager.monitor_orders(symbol)

    # ======================================================================
    #  Background Tasks
    # ======================================================================

    async def _order_monitor_loop(self) -> None:
        """Section 8.3: Verify all grid orders every 60 seconds."""
        interval = self._params.get("order_monitor_interval_s", 60)
        while self._running:
            try:
                await asyncio.sleep(interval)
                for symbol in self._cfg.instruments:
                    state = self._grid_manager.instruments.get(symbol)
                    if state and state.active:
                        missing = await self._grid_manager.monitor_orders(symbol)
                        if missing and missing > 0:
                            logger.info("[%s] Order monitor: replaced %d missing orders", symbol, missing)
                self._heartbeat.emit()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Order monitor error: %s", e, exc_info=True)

    async def _daily_refresh_loop(self) -> None:
        """Section 4.5: Recalculate range at 00:00 UTC daily."""
        while self._running:
            try:
                now = datetime.now(timezone.utc)
                # Calculate seconds until next 00:00 UTC
                tomorrow = now.replace(hour=0, minute=0, second=0, microsecond=0)
                if now.hour > 0 or now.minute > 0:
                    tomorrow = tomorrow.replace(day=now.day + 1)
                wait_seconds = (tomorrow - now).total_seconds()
                logger.info("Daily refresh scheduled in %.0f seconds", wait_seconds)

                await asyncio.sleep(wait_seconds)

                if not self._running:
                    break

                logger.info("=== Daily Grid Refresh at 00:00 UTC ===")

                # Reset daily drawdown
                self._shared_risk.reset_daily_drawdown()
                if now.weekday() == 0:
                    self._shared_risk.reset_weekly_drawdown()
                if now.day == 1:
                    self._shared_risk.reset_monthly_drawdown()

                for symbol in self._cfg.instruments:
                    state = self._grid_manager.instruments.get(symbol)
                    if not state or not state.active:
                        continue

                    price = state.current_price or state.mark_price
                    if price <= 0:
                        continue

                    should_refresh, new_range = self._strategy.should_refresh_grid(state, price)
                    if should_refresh:
                        logger.info("[%s] Grid boundaries changed > 5%% — resetting", symbol)
                        await self._grid_manager.reset_grid(symbol, "daily_refresh_boundary_change")
                    else:
                        logger.info("[%s] Grid boundaries within 5%% — no change", symbol)

                    # Recalculate per-level quantities with updated equity
                    equity = self._risk.get_equity()
                    capital = self._risk.calculate_capital_per_instrument(len(self._cfg.instruments))
                    state.allocated_capital = capital

                    # Check profit extraction
                    if self._strategy.should_extract_profit(state):
                        extracted = await self._grid_manager.extract_profit(symbol)
                        await self._alerting.send_info(
                            f"Profit Extracted: {symbol}",
                            f"Extracted {extracted:.4f} USDT",
                        )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Daily refresh error: %s", e, exc_info=True)
                await asyncio.sleep(60)

    async def _risk_check_loop(self) -> None:
        """Periodic risk checks on active grids."""
        while self._running:
            try:
                await asyncio.sleep(5)

                for symbol in self._cfg.instruments:
                    state = self._grid_manager.instruments.get(symbol)
                    if not state or not state.active:
                        continue

                    # Runtime risk check
                    action, reason = self._risk.check_runtime_risks(state)

                    if action == "liquidate":
                        logger.critical("[%s] %s — LIQUIDATING", symbol, reason)
                        await self._grid_manager._liquidate_inventory(symbol)
                        await self._grid_manager.cancel_all_orders(symbol)
                        state.active = False
                        state.halted = True
                        state.halt_reason = reason
                        await self._alerting.send_emergency(f"Liquidation: {symbol}", reason)

                    elif action == "cancel_buys":
                        logger.warning("[%s] %s — cancelling buy orders", symbol, reason)
                        await self._grid_manager.cancel_buy_orders(symbol)

                    elif action == "halt_grid":
                        logger.warning("[%s] %s — halting grid", symbol, reason)
                        await self._grid_manager.cancel_all_orders(symbol)
                        state.active = False
                        state.halted = True
                        state.halt_reason = reason
                        self._risk.halt_after_shutdown(symbol)
                        await self._alerting.send_warning(f"Grid Halted: {symbol}", reason)

                    elif action == "halt_grid_trend":
                        logger.warning("[%s] %s — trend detected, shutting down", symbol, reason)
                        await self._grid_manager.cancel_buy_orders(symbol)
                        state.halted = True
                        state.halt_reason = reason
                        await self._alerting.send_warning(f"Trend Detected: {symbol}", reason)

                    # Breakout detection
                    if state.active:
                        breakout = self._strategy.check_breakout(state)
                        if breakout == BreakoutDirection.UPSIDE:
                            await self._grid_manager.handle_upside_breakout(symbol)
                            await self._alerting.send_warning(
                                f"Upside Breakout: {symbol}",
                                f"Price above upper boundary at {state.current_price:.2f}",
                            )
                        elif breakout == BreakoutDirection.DOWNSIDE:
                            await self._grid_manager.handle_downside_breakout(symbol)
                            await self._alerting.send_warning(
                                f"Downside Breakout: {symbol}",
                                f"Price below lower boundary at {state.current_price:.2f}",
                            )

                # Report positions for cross-strategy
                self._risk.report_positions(self._grid_manager.instruments)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Risk check error: %s", e, exc_info=True)

    async def _state_persist_loop(self) -> None:
        """Persist grid state every 5 seconds (Section 8.1)."""
        while self._running:
            try:
                await asyncio.sleep(5)
                self._persist_state()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("State persist error: %s", e, exc_info=True)

    async def _paper_fill_loop(self) -> None:
        """Paper trading: check for fills more frequently."""
        if not self._paper:
            return
        while self._running:
            try:
                await asyncio.sleep(1)
                for symbol in self._cfg.instruments:
                    state = self._grid_manager.instruments.get(symbol)
                    if state and state.active and state.current_price > 0:
                        await self._grid_manager.simulate_paper_fills(
                            symbol, state.current_price,
                        )
            except asyncio.CancelledError:
                break
            except Exception:
                pass

    def _persist_state(self) -> None:
        """Save grid manager state to the state persistence layer."""
        self._state.update_state("custom", {
            "grid_manager": self._grid_manager.get_state_dict(),
        })

        # Section 8.1: All required state fields
        self._state.update_state("orders", [])  # Tracked in grid_manager
        self._state.update_state("positions",
                                 GridDashboard.build_positions_data(self._grid_manager.instruments))
        self._state.update_state("performance_counters", {
            "total_cycles": sum(s.total_cycles for s in self._grid_manager.instruments.values()),
            "total_profit": sum(s.total_cycle_profit for s in self._grid_manager.instruments.values()),
        })

    # ======================================================================
    #  Helpers
    # ======================================================================

    async def _get_current_price(self, symbol: str) -> float:
        """Get current price from the indicator buffer or REST API."""
        state = self._grid_manager.instruments.get(symbol)
        if state and state.current_price > 0:
            return state.current_price

        try:
            ticker = await self._client.get_ticker_24hr(symbol)
            return float(ticker.get("lastPrice", 0))
        except Exception:
            return 0.0

    def _get_all_metrics(self) -> dict:
        """Combine performance and grid metrics."""
        perf = self._perf.get_metrics()
        grid = self._grid_manager.get_all_metrics()
        return {**perf, "grid_metrics": grid}


# ======================================================================
#  Entry point
# ======================================================================

async def main() -> None:
    """Application entry point."""
    config_path = os.environ.get("CONFIG_PATH", "config.yaml")
    bot = GridTradingBot(config_path=config_path)

    # Signal handlers
    loop = asyncio.get_running_loop()

    def _signal_handler():
        logger.info("Received shutdown signal")
        asyncio.ensure_future(bot.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)

    try:
        await bot.start()
        # Run forever until shutdown
        while bot._running:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        await bot.stop()


if __name__ == "__main__":
    asyncio.run(main())
