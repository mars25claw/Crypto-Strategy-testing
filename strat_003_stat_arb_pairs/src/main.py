"""STRAT-003 Entry Point: Statistical Arbitrage Pairs Trading Bot.

Startup sequence:
1. Load config, initialise shared infrastructure
2. Warm up 180 days of daily closes for 20 assets
3. Run initial cointegration tests
4. Register 140 WebSocket streams (7 per asset x 20 assets)
5. Schedule daily recalculation at 00:00 UTC
6. Run main event loop
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure shared library is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from shared.binance_client import BinanceClient
from shared.binance_ws_manager import WebSocketManager, ConnectionType
from shared.circuit_breaker import CircuitBreaker
from shared.config_loader import ConfigLoader, RiskConfig
from shared.dashboard_base import DashboardBase
from shared.database import DatabaseManager
from shared.heartbeat import HeartbeatMonitor
from shared.kill_switch import KillSwitch
from shared.log_manager import setup_logging
from shared.paper_trading import PaperTradingEngine
from shared.performance_tracker import PerformanceTracker
from shared.rate_limiter import RateLimiter
from shared.risk_manager import RiskManager, CrossStrategyReader
from shared.state_persistence import StatePersistence
from shared.utils import TimeSync, ExchangeInfo

from src import STRATEGY_ID, STRATEGY_NAME
from src.cointegration import CointegrationEngine
from src.dashboard import PairsDashboard
from src.risk_manager import PairsRiskManager
from src.strategy import PairsStrategy

logger = logging.getLogger(__name__)
system_logger = logging.getLogger("system")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WARMUP_KLINE_LIMIT = 500  # max per API call
DAILY_RECALC_HOUR_UTC = 0
STREAMS_PER_ASSET = 7
MAX_ASSETS = 20


class StatArbBot:
    """Main bot orchestrator for STRAT-003."""

    def __init__(self, config_path: str = "config.yaml") -> None:
        # ── Config ─────────────────────────────────────────────────────
        self._config_loader = ConfigLoader(config_path)
        self._cfg = self._config_loader.config
        self._params = self._cfg.strategy_params

        # ── Logging ────────────────────────────────────────────────────
        setup_logging(
            strategy_id=STRATEGY_ID,
            log_dir=self._cfg.logging.log_dir,
            level=self._cfg.logging.level,
            rotate_days=self._cfg.logging.rotate_days,
        )
        logger.info("Initialising %s (%s)", STRATEGY_ID, STRATEGY_NAME)

        # ── Core infrastructure ────────────────────────────────────────
        self._time_sync = TimeSync()
        self._exchange_info = ExchangeInfo()
        self._rate_limiter = RateLimiter(
            weight_per_minute=self._cfg.rate_limit_weight_per_min,
            burst_weight=self._cfg.rate_limit_burst_weight,
        )

        self._client = BinanceClient(
            api_key=self._cfg.binance.api_key,
            api_secret=self._cfg.binance.api_secret,
            time_sync=self._time_sync,
            exchange_info=self._exchange_info,
            rate_limiter=self._rate_limiter,
            spot_base_url=self._cfg.binance.spot_base_url,
            futures_base_url=self._cfg.binance.futures_base_url,
        )

        self._ws_manager = WebSocketManager(
            futures_ws_url=self._cfg.binance.futures_ws_url + "/stream",
            binance_client=self._client,
        )

        # ── Database ───────────────────────────────────────────────────
        self._db = DatabaseManager(self._cfg.database.url)

        # ── State persistence ──────────────────────────────────────────
        self._state = StatePersistence(
            state_dir=self._cfg.state.state_dir,
            strategy_id=STRATEGY_ID,
            save_interval=self._cfg.state.persistence_interval,
            max_snapshots=self._cfg.state.snapshot_count,
        )

        # ── Risk management ────────────────────────────────────────────
        risk_config = RiskConfig(
            max_capital_pct=self._cfg.risk.max_capital_pct,
            max_per_trade_pct=self._cfg.risk.max_per_trade_pct,
            risk_per_trade_pct=self._cfg.risk.risk_per_trade_pct,
            max_leverage=self._cfg.risk.max_leverage,
            preferred_leverage=self._cfg.risk.preferred_leverage,
            max_concurrent_positions=self._cfg.risk.max_concurrent_positions,
            max_per_asset_pct=self._cfg.risk.max_per_asset_pct,
            max_long_exposure_pct=self._cfg.risk.max_long_exposure_pct,
            max_short_exposure_pct=self._cfg.risk.max_short_exposure_pct,
            max_net_directional_pct=self._cfg.risk.max_net_directional_pct,
            daily_drawdown_pct=self._cfg.risk.daily_drawdown_pct,
            weekly_drawdown_pct=self._cfg.risk.weekly_drawdown_pct,
            monthly_drawdown_pct=self._cfg.risk.monthly_drawdown_pct,
            system_wide_drawdown_pct=self._cfg.risk.system_wide_drawdown_pct,
        )
        self._shared_risk = RiskManager(
            config=risk_config,
            database_manager=self._db,
            cross_strategy_reader=CrossStrategyReader(self._cfg.state.state_dir),
        )

        # ── Paper trading ──────────────────────────────────────────────
        self._paper_engine: Optional[PaperTradingEngine] = None
        if self._cfg.paper_trading.enabled or self._cfg.mode == "paper":
            self._paper_engine = PaperTradingEngine(
                starting_equity=self._cfg.paper_trading.starting_equity,
                maker_fee_pct=self._cfg.paper_trading.maker_fee_pct,
                taker_fee_pct=self._cfg.paper_trading.taker_fee_pct,
            )
            self._shared_risk.update_equity(self._cfg.paper_trading.starting_equity)

        # ── Cointegration engine ───────────────────────────────────────
        self._coint_engine = CointegrationEngine(self._params)

        # ── Pairs risk manager ─────────────────────────────────────────
        self._pairs_risk = PairsRiskManager(self._params, self._shared_risk)

        # ── Strategy engine ────────────────────────────────────────────
        self._strategy = PairsStrategy(self._params, self._coint_engine)
        self._strategy.set_execution_callbacks(
            execute_order=self._execute_order,
            get_equity=self._get_equity,
            risk_check=self._risk_check_entry,
        )

        # ── Performance tracker ────────────────────────────────────────
        self._perf_tracker = PerformanceTracker(STRATEGY_ID)

        # ── Circuit breaker ────────────────────────────────────────────
        self._circuit_breaker = CircuitBreaker({
            "flash_crash_pct": 10.0,
            "flash_crash_minutes": 5,
            "consecutive_losses": self._params.get("consecutive_loss_halt", 5),
        })

        # ── Kill switch ────────────────────────────────────────────────
        self._kill_switch = KillSwitch(
            binance_client=self._client,
            state_persistence=self._state,
            database_manager=self._db,
        )

        # ── Heartbeat ──────────────────────────────────────────────────
        self._heartbeat = HeartbeatMonitor(
            strategy_id=STRATEGY_ID,
            interval=self._cfg.heartbeat.interval,
            timeout=self._cfg.heartbeat.timeout,
            max_restarts_per_hour=self._cfg.heartbeat.max_restarts_per_hour,
            on_restart=self._handle_restart,
        )

        # ── Dashboard ──────────────────────────────────────────────────
        self._dashboard = PairsDashboard(
            strategy_id=STRATEGY_ID,
            strategy_name=STRATEGY_NAME,
            host=self._cfg.dashboard.host,
            port=self._cfg.dashboard.port,
            strategy=self._strategy,
            coint_engine=self._coint_engine,
            pairs_risk=self._pairs_risk,
            perf_tracker=self._perf_tracker,
        )

        # ── Instruments ────────────────────────────────────────────────
        self._instruments = self._cfg.instruments[:MAX_ASSETS]

        # ── Control ────────────────────────────────────────────────────
        self._running = False
        self._daily_prices: Dict[str, List[float]] = {}

    # ======================================================================
    #  Startup
    # ======================================================================

    async def start(self) -> None:
        """Full startup sequence."""
        self._running = True
        system_logger.info("startup strategy=%s mode=%s", STRATEGY_ID, self._cfg.mode)

        # 1. Start HTTP client
        await self._client.start()
        await self._client.sync_time()
        await self._client.load_exchange_info(self._instruments)

        # 2. Load persisted state
        saved = self._state.load()
        if saved.get("custom"):
            coint_state = saved["custom"].get("cointegration")
            if coint_state:
                self._coint_engine.load_state_dict(coint_state)
                logger.info("Restored cointegration state")
            strat_state = saved["custom"].get("strategy")
            if strat_state:
                self._strategy.load_state_dict(strat_state)
                logger.info("Restored strategy state with %d positions",
                            len(self._strategy._positions))

        # 3. Warm up 180 days of daily closes
        logger.info("Warming up %d instruments with %d days of daily data...",
                     len(self._instruments), self._params.get("lookback_days", 180))
        await self._warmup_daily_data()

        # 4. Run initial cointegration screening
        logger.info("Running initial cointegration screening...")
        result = self._coint_engine.run_full_screening(
            self._daily_prices,
            self._instruments,
        )
        logger.info(
            "Initial screening: %d qualified pairs in %.1fs",
            len(result.qualified_pairs), result.run_duration_s,
        )

        # 5. Register WebSocket streams (7 per asset x 20 = 140)
        self._register_ws_streams()

        # 6. Start subsystems
        await self._ws_manager.start()
        await self._state.start()
        await self._heartbeat.start()
        await self._dashboard.start()

        # 7. Start background tasks
        asyncio.create_task(self._daily_recalc_loop(), name="daily-recalc")
        asyncio.create_task(self._btc_flash_check_loop(), name="btc-flash-check")
        asyncio.create_task(self._equity_update_loop(), name="equity-update")
        asyncio.create_task(self._state_persist_loop(), name="state-persist")
        asyncio.create_task(self._config_reload_loop(), name="config-reload")

        # Register signal handlers
        for sig in (signal.SIGINT, signal.SIGTERM):
            asyncio.get_event_loop().add_signal_handler(
                sig, lambda s=sig: asyncio.create_task(self.shutdown(str(s))),
            )

        logger.info("STRAT-003 fully started. %d instruments, %d qualified pairs.",
                     len(self._instruments), len(self._coint_engine.qualified_pairs))
        system_logger.info(
            "startup_complete instruments=%d qualified_pairs=%d",
            len(self._instruments), len(self._coint_engine.qualified_pairs),
        )

    # ======================================================================
    #  Warmup
    # ======================================================================

    async def _warmup_daily_data(self) -> None:
        """Fetch 180 days of daily closes for all instruments.

        Spreads requests over time to respect rate limits (400 weight budget).
        """
        lookback = self._params.get("lookback_days", 180)

        for symbol in self._instruments:
            try:
                klines = await self._client.get_futures_klines(
                    symbol=symbol,
                    interval="1d",
                    limit=lookback,
                )
                closes = [float(k[4]) for k in klines if float(k[4]) > 0]
                self._daily_prices[symbol] = closes
                logger.debug("Warmed up %s: %d daily candles", symbol, len(closes))
                await asyncio.sleep(0.3)  # Pace requests
            except Exception:
                logger.exception("Failed to warm up %s", symbol)
                self._daily_prices[symbol] = []

        # Also warm up 1h candles for spread calculation
        for symbol in self._instruments:
            try:
                klines_1h = await self._client.get_futures_klines(
                    symbol=symbol,
                    interval="1h",
                    limit=200,
                )
                for k in klines_1h:
                    close = float(k[4])
                    volume = float(k[5])
                    if close > 0:
                        self._strategy._prices_1h[symbol].append(close)
                        self._strategy._volume_1h[symbol].append(volume)
                await asyncio.sleep(0.3)
            except Exception:
                logger.exception("Failed to warm up 1h data for %s", symbol)

    # ======================================================================
    #  WebSocket registration
    # ======================================================================

    def _register_ws_streams(self) -> None:
        """Register 7 streams per asset on the futures WS connection."""
        subscriptions = []

        for symbol in self._instruments:
            sym_lower = symbol.lower()

            subscriptions.extend([
                (f"{sym_lower}@kline_1m", self._strategy.on_kline_1m),
                (f"{sym_lower}@kline_5m", self._strategy.on_kline_5m),
                (f"{sym_lower}@kline_1h", self._strategy.on_kline_1h),
                (f"{sym_lower}@kline_1d", self._strategy.on_kline_1d),
                (f"{sym_lower}@bookTicker", self._strategy.on_book_ticker),
                (f"{sym_lower}@depth20@100ms", self._strategy.on_depth),
                (f"{sym_lower}@markPrice@1s", self._strategy.on_mark_price),
            ])

        total_streams = len(subscriptions)
        logger.info(
            "Registering %d WebSocket streams (%d assets x %d streams)",
            total_streams, len(self._instruments), STREAMS_PER_ASSET,
        )

        self._ws_manager.register_strategy(
            strategy_id=STRATEGY_ID,
            subscriptions=subscriptions,
            conn_type=ConnectionType.FUTURES,
            on_reconnect=self._on_ws_reconnect,
        )

    async def _on_ws_reconnect(self, conn_type: ConnectionType) -> None:
        """Handle WebSocket reconnection."""
        logger.warning("WebSocket reconnected (%s), recalculating spreads", conn_type.value)
        system_logger.info("ws_reconnect strategy=%s conn=%s", STRATEGY_ID, conn_type.value)

    # ======================================================================
    #  Background loops
    # ======================================================================

    async def _daily_recalc_loop(self) -> None:
        """Run cointegration recalculation daily at 00:00 UTC.

        Section 3.1: Explicit scheduling of cointegration retesting at
        00:00 UTC daily. Must complete within 5 minutes. Uses
        asyncio.create_task with timeout.
        """
        while self._running:
            try:
                now = datetime.now(timezone.utc)
                # Calculate seconds until next 00:00 UTC
                next_midnight = now.replace(
                    hour=DAILY_RECALC_HOUR_UTC, minute=0, second=0, microsecond=0,
                )
                if now >= next_midnight:
                    from datetime import timedelta
                    next_midnight = next_midnight + timedelta(days=1)
                wait_seconds = (next_midnight - now).total_seconds()
                logger.info("Next daily recalc in %.0f seconds", wait_seconds)
                await asyncio.sleep(wait_seconds)

                if not self._running:
                    break

                logger.info("Starting daily cointegration recalculation (5-min timeout)...")
                system_logger.info("daily_recalc_start strategy=%s", STRATEGY_ID)

                # Run the recalculation as a task with a 5-minute timeout
                recalc_task = asyncio.create_task(
                    self._execute_daily_recalc(),
                    name="daily-recalc-execution",
                )
                try:
                    await asyncio.wait_for(recalc_task, timeout=300)  # 5 minutes
                except asyncio.TimeoutError:
                    logger.error(
                        "Daily cointegration recalculation exceeded 5-minute deadline. "
                        "Cancelling and using stale parameters."
                    )
                    recalc_task.cancel()
                    try:
                        await recalc_task
                    except asyncio.CancelledError:
                        pass
                    continue

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Daily recalc loop error")
                await asyncio.sleep(60)

    async def _execute_daily_recalc(self) -> None:
        """Execute the daily cointegration recalculation.

        Separated to allow wrapping in asyncio.wait_for with 5-min timeout.
        """
        # Re-fetch daily data
        await self._warmup_daily_data()

        # Track which pairs were previously qualified
        old_pairs = {
            (p.asset_a, p.asset_b) for p in self._coint_engine.qualified_pairs
        }

        # Run screening
        result = self._coint_engine.run_full_screening(
            self._daily_prices,
            self._instruments,
        )

        new_pairs = {
            (p.asset_a, p.asset_b) for p in result.qualified_pairs
        }

        # Check for disqualified pairs that have open positions (Section 4.5)
        # Mark removed pairs as broken for 30 days
        removed = old_pairs - new_pairs
        for a, b in removed:
            self._coint_engine.mark_pair_broken(a, b)
            await self._strategy.handle_cointegration_breakdown(a, b)

        # Reset daily drawdown
        self._shared_risk.reset_daily_drawdown()
        now_utc = datetime.now(timezone.utc)
        if now_utc.weekday() == 0:
            self._shared_risk.reset_weekly_drawdown()
        if now_utc.day == 1:
            self._shared_risk.reset_monthly_drawdown()

        logger.info(
            "Daily recalc complete: %d qualified, %d removed (broken 30d), %.1fs",
            len(result.qualified_pairs), len(removed), result.run_duration_s,
        )
        system_logger.info(
            "daily_recalc_complete qualified=%d removed=%d duration=%.1f",
            len(result.qualified_pairs), len(removed), result.run_duration_s,
        )

    async def _btc_flash_check_loop(self) -> None:
        """Check BTC flash crash every 5 minutes."""
        while self._running:
            try:
                await asyncio.sleep(300)  # 5 minutes
                if not self._running:
                    break
                await self._strategy.check_btc_flash()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("BTC flash check error")

    async def _equity_update_loop(self) -> None:
        """Update equity tracking every 30 seconds."""
        while self._running:
            try:
                await asyncio.sleep(30)
                if not self._running:
                    break
                equity = self._get_equity()
                self._shared_risk.update_equity(equity)

                # Update heartbeat
                self._heartbeat.set_positions_count(len(self._strategy._positions) * 2)
                total_pnl = sum(p.current_pnl for p in self._strategy._positions.values())
                self._heartbeat.set_unrealized_pnl(total_pnl)

                # Update performance tracker
                self._perf_tracker.update_unrealized_pnl(total_pnl)

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Equity update loop error")

    async def _state_persist_loop(self) -> None:
        """Persist strategy state every 5 seconds."""
        while self._running:
            try:
                await asyncio.sleep(5)
                if not self._running:
                    break
                self._state.update_state("custom", {
                    "cointegration": self._coint_engine.to_state_dict(),
                    "strategy": self._strategy.to_state_dict(),
                })
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("State persist loop error")

    async def _config_reload_loop(self) -> None:
        """Check for config changes every 60 seconds."""
        while self._running:
            try:
                await asyncio.sleep(60)
                if not self._running:
                    break
                if self._config_loader.check_reload():
                    logger.info("Configuration reloaded")
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Config reload error")

    # ======================================================================
    #  Execution bridge
    # ======================================================================

    async def _execute_order(self, order: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Execute an order (paper or live)."""
        symbol = order["symbol"]
        side = order["side"]
        qty = order["quantity"]
        order_type = order.get("type", "MARKET")
        price = order.get("price")

        if self._paper_engine is not None:
            # Paper trading simulation
            book = {"bids": [], "asks": []}
            bt = self._strategy._book_tickers.get(symbol, {})
            bid = bt.get("bid", self._strategy._mark_prices.get(symbol, 0))
            ask = bt.get("ask", self._strategy._mark_prices.get(symbol, 0))
            if bid > 0 and ask > 0:
                book = {
                    "bids": [[str(bid), "100"]],
                    "asks": [[str(ask), "100"]],
                }

            if order_type == "MARKET":
                result = self._paper_engine.simulate_market_order(
                    symbol, side, qty, book,
                )
            else:
                current_price = self._strategy._mark_prices.get(symbol, 0)
                result = self._paper_engine.simulate_limit_order(
                    symbol, side, qty, price or current_price,
                    current_price, book,
                )
                if result is None:
                    # Limit not filled, convert to market (after timeout simulation)
                    result = self._paper_engine.simulate_market_order(
                        symbol, side, qty, book,
                    )

            return {
                "symbol": symbol,
                "side": side,
                "type": order_type,
                "quantity": qty,
                "avgPrice": result.fill_price,
                "status": "FILLED",
                "fees": result.fees,
            }
        else:
            # Live trading
            try:
                qty_rounded = self._exchange_info.round_quantity(symbol, qty)
                if qty_rounded <= 0:
                    logger.warning("Quantity rounds to 0 for %s", symbol)
                    return None

                if order_type == "MARKET":
                    return await self._client.place_futures_order(
                        symbol=symbol,
                        side=side,
                        type="MARKET",
                        quantity=qty_rounded,
                        reduce_only=order.get("reduce_only", False),
                    )
                else:
                    price_rounded = self._exchange_info.round_price(symbol, price) if price else 0
                    return await self._client.place_futures_order(
                        symbol=symbol,
                        side=side,
                        type="LIMIT",
                        quantity=qty_rounded,
                        price=price_rounded,
                        reduce_only=order.get("reduce_only", False),
                    )
            except Exception:
                logger.exception("Order execution failed: %s", order)
                return None

    def _get_equity(self) -> float:
        """Get current equity."""
        if self._paper_engine:
            return self._paper_engine.get_equity()
        return self._shared_risk.get_current_equity()

    def _risk_check_entry(
        self,
        params: Any,
        direction: str,
        equity: float,
    ) -> tuple:
        """Risk check bridge for strategy."""
        return self._pairs_risk.check_pair_entry(params, direction, equity)

    # ======================================================================
    #  Shutdown
    # ======================================================================

    async def shutdown(self, reason: str = "shutdown") -> None:
        """Gracefully shut down the bot."""
        if not self._running:
            return
        self._running = False
        logger.info("Shutting down: %s", reason)
        system_logger.info("shutdown strategy=%s reason=%s", STRATEGY_ID, reason)

        # Stop subsystems
        self._heartbeat.stop()
        self._dashboard.stop()
        await self._ws_manager.stop()
        await self._state.stop()
        await self._client.close()

        logger.info("Shutdown complete")

    async def _handle_restart(self, strategy_id: str) -> None:
        """Handle heartbeat-triggered restart."""
        logger.warning("Heartbeat restart triggered for %s", strategy_id)

    # ======================================================================
    #  Dashboard data providers
    # ======================================================================

    def get_config_dict(self) -> dict:
        """Return config as dict for dashboard."""
        return {
            "strategy_id": STRATEGY_ID,
            "mode": self._cfg.mode,
            "paper_trading": self._cfg.paper_trading.enabled,
            "instruments": self._instruments,
            "risk": {
                "max_capital_pct": self._cfg.risk.max_capital_pct,
                "max_per_trade_pct": self._cfg.risk.max_per_trade_pct,
                "max_leverage": self._cfg.risk.max_leverage,
            },
            "strategy_params": self._params,
        }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main():
    """Main entry point."""
    config_path = os.environ.get("CONFIG_PATH", "config.yaml")
    bot = StatArbBot(config_path)

    try:
        await bot.start()
        # Keep running
        while bot._running:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        await bot.shutdown("process_exit")


if __name__ == "__main__":
    asyncio.run(main())
