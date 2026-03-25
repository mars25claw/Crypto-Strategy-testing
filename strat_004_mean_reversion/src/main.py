"""STRAT-004 Mean Reversion — Main entry point.

Startup sequence:
1. Load configuration (config.yaml + env overrides)
2. Initialize components (client, WS manager, risk, regime, strategy, dashboard)
3. Warm up: 200 candles per TF per instrument, 100 daily candles for Hurst
4. Run initial regime classification
5. Register WebSocket streams and start the event loop
6. Schedule daily regime recalculation at 00:00 UTC
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

# Ensure shared library is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from shared.config_loader import ConfigLoader, RiskConfig
from shared.binance_client import BinanceClient
from shared.binance_ws_manager import WebSocketManager, ConnectionType
from shared.risk_manager import RiskManager as SharedRiskManager, CrossStrategyReader
from shared.cross_strategy import CrossStrategyManager
from shared.paper_trading import PaperTradingEngine
from shared.kill_switch import KillSwitch
from shared.log_manager import setup_logging
from shared.rate_limiter import RateLimiter
from shared.utils import TimeSync, ExchangeInfo
from shared.indicators import IndicatorBuffer

from src.regime_classifier import RegimeClassifier
from src.strategy import MeanReversionStrategy
from src.risk_manager import MeanReversionRiskManager
from src.dashboard import MeanReversionDashboard
from src.strategy_metrics import MeanReversionMetrics

logger = logging.getLogger(__name__)
system_logger = logging.getLogger("system")

STRATEGY_ID = "STRAT-004"
STRATEGY_NAME = "Mean Reversion"


class MeanReversionBot:
    """Main bot orchestrator for STRAT-004.

    Coordinates all components: config, exchange client, WebSocket streams,
    risk management, regime classification, strategy logic, and dashboard.
    """

    # Timeframes and their Binance interval strings
    TIMEFRAMES = {
        "1m": "1m",
        "15m": "15m",
        "4h": "4h",
        "1d": "1d",
    }

    # Candles to warm up per timeframe
    WARMUP_CANDLES = {
        "1m": 200,
        "15m": 200,
        "4h": 200,
        "1d": 100,
    }

    def __init__(self, config_path: str = "config.yaml") -> None:
        self._config_path = config_path
        self._running = False
        self._shutdown_event = asyncio.Event()

        # Components (initialized in start())
        self._config_loader: Optional[ConfigLoader] = None
        self._config: Optional[Any] = None
        self._strategy_params: dict = {}

        self._time_sync: Optional[TimeSync] = None
        self._exchange_info: Optional[ExchangeInfo] = None
        self._rate_limiter: Optional[RateLimiter] = None
        self._client: Optional[BinanceClient] = None
        self._ws_manager: Optional[WebSocketManager] = None

        self._shared_risk: Optional[SharedRiskManager] = None
        self._risk_manager: Optional[MeanReversionRiskManager] = None
        self._cross_strategy: Optional[CrossStrategyManager] = None

        self._regime: Optional[RegimeClassifier] = None
        self._strategy: Optional[MeanReversionStrategy] = None
        self._paper: Optional[PaperTradingEngine] = None
        self._kill_switch: Optional[KillSwitch] = None
        self._dashboard: Optional[MeanReversionDashboard] = None
        self._strategy_metrics: Optional[MeanReversionMetrics] = None

        self._instruments: List[str] = []

    # ==================================================================
    # Lifecycle
    # ==================================================================

    async def start(self) -> None:
        """Initialize all components and start the bot."""
        start_time = time.time()
        system_logger.info("STRAT-004 Mean Reversion starting...")

        # 1. Load configuration
        self._load_config()

        # 2. Initialize components
        await self._init_components()

        # 3. Warm up candle data
        await self._warmup()

        # 4. Run initial regime classification
        await self._initial_regime_classification()

        # 5. Register WebSocket streams
        self._register_streams()

        # 6. Start WebSocket manager
        await self._ws_manager.start()

        # 7. Start dashboard
        if self._dashboard:
            await self._dashboard.start()

        elapsed = time.time() - start_time
        self._running = True

        system_logger.info(
            "STRAT-004 started in %.1fs — instruments=%d, mode=%s, port=%d",
            elapsed, len(self._instruments),
            "paper" if self._paper else "live",
            self._config.dashboard.port if self._config else 8084,
        )
        logger.info(
            "Bot ready: %d instruments, %d tradeable",
            len(self._instruments),
            len(self._regime.get_tradeable_instruments()) if self._regime else 0,
        )

        # 8. Start event loop
        await self._run_loop()

    async def stop(self) -> None:
        """Gracefully shut down all components."""
        self._running = False
        system_logger.info("STRAT-004 shutting down...")

        if self._ws_manager:
            await self._ws_manager.stop()

        if self._dashboard:
            self._dashboard.stop()

        if self._client:
            await self._client.close()

        system_logger.info("STRAT-004 shutdown complete")

    # ==================================================================
    # Initialization
    # ==================================================================

    def _load_config(self) -> None:
        """Load and validate configuration."""
        self._config_loader = ConfigLoader(self._config_path)
        self._config = self._config_loader.config
        self._strategy_params = self._config.strategy_params or {}
        self._instruments = self._config.instruments[:8]  # max 8

        logger.info(
            "Config loaded: strategy=%s, mode=%s, instruments=%s",
            self._config.strategy_id, self._config.mode, self._instruments,
        )

    async def _init_components(self) -> None:
        """Initialize all bot components."""
        cfg = self._config
        params = self._strategy_params

        # Logging
        try:
            setup_logging(
                log_dir=cfg.logging.log_dir,
                level=cfg.logging.level,
                rotate_days=cfg.logging.rotate_days,
            )
        except Exception:
            logging.basicConfig(level=logging.INFO)

        # Time sync & exchange info
        self._time_sync = TimeSync()
        self._exchange_info = ExchangeInfo()

        # Rate limiter
        self._rate_limiter = RateLimiter(
            budget=cfg.rate_limit_weight_per_min,
            burst=cfg.rate_limit_burst_weight,
        )

        # Binance REST client
        api_key = os.environ.get("BINANCE_API_KEY", cfg.binance.api_key)
        api_secret = os.environ.get("BINANCE_API_SECRET", cfg.binance.api_secret)

        self._client = BinanceClient(
            api_key=api_key,
            api_secret=api_secret,
            time_sync=self._time_sync,
            exchange_info=self._exchange_info,
            rate_limiter=self._rate_limiter,
            futures_base_url=cfg.binance.futures_base_url,
            recv_window=cfg.binance.recv_window,
        )
        await self._client.start()
        await self._client.sync_time()
        await self._client.load_exchange_info(self._instruments)

        # Paper trading engine
        if cfg.paper_trading.enabled:
            self._paper = PaperTradingEngine(
                starting_equity=cfg.paper_trading.starting_equity,
                maker_fee_pct=cfg.paper_trading.maker_fee_pct,
                taker_fee_pct=cfg.paper_trading.taker_fee_pct,
            )
            logger.info("Paper trading enabled: equity=%.2f", cfg.paper_trading.starting_equity)

        # Cross-strategy manager
        shared_dir = os.environ.get("SHARED_STATE_DIR", "data/shared")
        self._cross_strategy = CrossStrategyManager(
            strategy_id=STRATEGY_ID,
            shared_dir=shared_dir,
        )

        # Shared risk manager with STRAT-004-specific limits
        risk_config = RiskConfig(
            max_capital_pct=cfg.risk.max_capital_pct,
            max_per_trade_pct=cfg.risk.max_per_trade_pct,
            risk_per_trade_pct=cfg.risk.risk_per_trade_pct,
            max_leverage=cfg.risk.max_leverage,
            max_concurrent_positions=cfg.risk.max_concurrent_positions,
            max_per_asset_pct=cfg.risk.max_per_asset_pct,
            max_long_exposure_pct=cfg.risk.max_long_exposure_pct,
            max_short_exposure_pct=cfg.risk.max_short_exposure_pct,
            max_net_directional_pct=cfg.risk.max_net_directional_pct,
            daily_drawdown_pct=cfg.risk.daily_drawdown_pct,
            weekly_drawdown_pct=cfg.risk.weekly_drawdown_pct,
            monthly_drawdown_pct=cfg.risk.monthly_drawdown_pct,
            system_wide_drawdown_pct=cfg.risk.system_wide_drawdown_pct,
        )

        self._shared_risk = SharedRiskManager(
            config=risk_config,
            cross_strategy_reader=CrossStrategyReader(state_dir=shared_dir),
        )

        # Set initial equity
        initial_equity = cfg.paper_trading.starting_equity if self._paper else 10000.0
        self._shared_risk.update_equity(initial_equity)

        # STRAT-004 risk manager
        self._risk_manager = MeanReversionRiskManager(
            config=params,
            risk_config=risk_config,
            shared_risk=self._shared_risk,
            cross_strategy=self._cross_strategy,
        )

        # Regime classifier
        self._regime = RegimeClassifier(
            config=params,
            binance_client=self._client,
        )

        # Strategy
        self._strategy = MeanReversionStrategy(
            config=params,
            risk_manager=self._risk_manager,
            regime_classifier=self._regime,
            binance_client=self._client,
            paper_engine=self._paper,
        )
        self._strategy.init_buffers(self._instruments)

        # Kill switch
        self._kill_switch = KillSwitch(
            binance_client=self._client,
        )

        # WebSocket manager
        ws_url = cfg.binance.futures_ws_url
        if not ws_url.endswith("/stream"):
            ws_url = ws_url.rstrip("/") + "/stream"

        self._ws_manager = WebSocketManager(
            futures_ws_url=ws_url,
            binance_client=self._client,
        )

        # Strategy Metrics (Section 10.2 + 10.3 + 10.4)
        self._strategy_metrics = MeanReversionMetrics(
            strategy=self._strategy,
            risk_manager=self._risk_manager,
            regime_classifier=self._regime,
        )

        # Dashboard
        self._dashboard = MeanReversionDashboard(
            strategy=self._strategy,
            regime_classifier=self._regime,
            risk_manager=self._risk_manager,
            host=cfg.dashboard.host,
            port=cfg.dashboard.port,
            strategy_metrics=self._strategy_metrics,
        )

    # ==================================================================
    # Warm-up
    # ==================================================================

    async def _warmup(self) -> None:
        """Fetch historical candles to fill indicator buffers.

        200 candles per TF per instrument, 100 daily candles for Hurst.
        """
        logger.info("Starting warm-up: %d instruments x %d timeframes", len(self._instruments), len(self.TIMEFRAMES))
        start = time.time()

        for symbol in self._instruments:
            for tf_key, tf_interval in self.TIMEFRAMES.items():
                candles_needed = self.WARMUP_CANDLES.get(tf_key, 200)
                try:
                    klines = await self._client.get_futures_klines(
                        symbol=symbol,
                        interval=tf_interval,
                        limit=candles_needed,
                    )

                    buf = self._strategy.get_buffer(symbol, tf_key)
                    if buf and klines:
                        for k in klines:
                            buf.add_candle({
                                "timestamp": k[0],
                                "open": k[1],
                                "high": k[2],
                                "low": k[3],
                                "close": k[4],
                                "volume": k[5],
                            })
                        logger.debug("Warmed %s %s: %d candles", symbol, tf_key, len(klines))

                except Exception as e:
                    logger.error("Warm-up failed for %s %s: %s", symbol, tf_key, e)

                # Small delay to respect rate limits
                await asyncio.sleep(0.1)

        elapsed = time.time() - start
        logger.info("Warm-up complete in %.1fs", elapsed)

    # ==================================================================
    # Initial regime classification
    # ==================================================================

    async def _initial_regime_classification(self) -> None:
        """Run regime classification using warmed-up daily buffers."""
        logger.info("Running initial regime classification...")

        import numpy as np

        for symbol in self._instruments:
            buf = self._strategy.get_buffer(symbol, "1d")
            if buf and len(buf) >= 100:
                self._regime.classify_from_buffers(
                    symbol=symbol,
                    daily_closes=buf.get_closes(),
                    daily_highs=buf.get_highs(),
                    daily_lows=buf.get_lows(),
                )
            else:
                logger.warning(
                    "Insufficient daily data for %s (%d candles), fetching via REST",
                    symbol, len(buf) if buf else 0,
                )

        # Fallback: fetch via REST for any missing
        missing = [s for s in self._instruments if not self._regime.is_tradeable(s) and s not in self._regime.regimes]
        if missing:
            await self._regime.classify_all(missing)

        tradeable = self._regime.get_tradeable_instruments()
        logger.info(
            "Regime classification done: %d/%d instruments tradeable",
            len(tradeable), len(self._instruments),
        )
        for symbol in self._instruments:
            r = self._regime.get_regime(symbol)
            logger.info(
                "  %s: %s (H=%.3f, ADX=%.1f)",
                symbol, r.state.value, r.hurst, r.adx_value,
            )

    # ==================================================================
    # WebSocket stream registration
    # ==================================================================

    def _register_streams(self) -> None:
        """Register all WebSocket streams for all instruments."""
        subscriptions = []

        for symbol in self._instruments:
            sym_lower = symbol.lower()

            # Kline streams
            for tf in ["1m", "15m", "4h", "1d"]:
                stream = f"{sym_lower}@kline_{tf}"
                subscriptions.append((
                    stream,
                    self._make_kline_handler(symbol, tf),
                ))

            # Book ticker (best bid/ask)
            subscriptions.append((
                f"{sym_lower}@bookTicker",
                self._make_book_ticker_handler(symbol),
            ))

            # Depth (order book)
            subscriptions.append((
                f"{sym_lower}@depth20@100ms",
                self._make_depth_handler(symbol),
            ))

            # Mark price
            subscriptions.append((
                f"{sym_lower}@markPrice@1s",
                self._make_mark_price_handler(symbol),
            ))

            # Aggregate trades
            subscriptions.append((
                f"{sym_lower}@aggTrade",
                self._make_agg_trade_handler(symbol),
            ))

        self._ws_manager.register_strategy(
            strategy_id=STRATEGY_ID,
            subscriptions=subscriptions,
            conn_type=ConnectionType.FUTURES,
            on_reconnect=self._on_ws_reconnect,
        )

        logger.info(
            "Registered %d WebSocket streams for %d instruments",
            len(subscriptions), len(self._instruments),
        )

    # Stream handler factories (closures to capture symbol)

    def _make_kline_handler(self, symbol: str, timeframe: str):
        async def handler(data: dict):
            kline = data.get("k", data)
            await self._strategy.on_kline(symbol, timeframe, kline)
        return handler

    def _make_book_ticker_handler(self, symbol: str):
        async def handler(data: dict):
            await self._strategy.on_book_ticker(symbol, data)
        return handler

    def _make_depth_handler(self, symbol: str):
        async def handler(data: dict):
            await self._strategy.on_depth(symbol, data)
        return handler

    def _make_mark_price_handler(self, symbol: str):
        async def handler(data: dict):
            await self._strategy.on_mark_price(symbol, data)
        return handler

    def _make_agg_trade_handler(self, symbol: str):
        async def handler(data: dict):
            await self._strategy.on_agg_trade(symbol, data)
        return handler

    async def _on_ws_reconnect(self, conn_type: ConnectionType) -> None:
        """Handle WebSocket reconnection."""
        logger.warning("WebSocket reconnected: %s", conn_type.value)
        # Could trigger state reconciliation here

    # ==================================================================
    # Main event loop
    # ==================================================================

    async def _run_loop(self) -> None:
        """Main loop: schedule daily regime recalc, monitor health, handle config reload."""
        logger.info("Event loop started")

        # Schedule tasks
        regime_task = asyncio.create_task(
            self._regime_recalc_scheduler(), name="regime-scheduler"
        )
        health_task = asyncio.create_task(
            self._health_monitor(), name="health-monitor"
        )
        equity_task = asyncio.create_task(
            self._equity_updater(), name="equity-updater"
        )
        config_task = asyncio.create_task(
            self._config_reload_checker(), name="config-reload"
        )

        try:
            await self._shutdown_event.wait()
        except asyncio.CancelledError:
            pass
        finally:
            for task in [regime_task, health_task, equity_task, config_task]:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            await self.stop()

    async def _regime_recalc_scheduler(self) -> None:
        """Recalculate regime at 00:00 UTC daily."""
        while self._running:
            try:
                # Sleep until next 00:00 UTC
                now = datetime.now(timezone.utc)
                next_midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
                if next_midnight <= now:
                    from datetime import timedelta
                    next_midnight += timedelta(days=1)

                sleep_seconds = (next_midnight - now).total_seconds()
                logger.info("Next regime recalculation in %.0f seconds", sleep_seconds)
                await asyncio.sleep(sleep_seconds)

                if not self._running:
                    break

                # Run classification
                logger.info("Daily regime recalculation starting...")
                await self._regime.classify_all(self._instruments)

                tradeable = self._regime.get_tradeable_instruments()
                logger.info(
                    "Daily regime done: %d/%d tradeable",
                    len(tradeable), len(self._instruments),
                )

                # Reset daily drawdown
                self._shared_risk.reset_daily_drawdown()

                # Check for Monday (weekly reset)
                if datetime.now(timezone.utc).weekday() == 0:
                    self._shared_risk.reset_weekly_drawdown()

                # Check for 1st of month
                if datetime.now(timezone.utc).day == 1:
                    self._shared_risk.reset_monthly_drawdown()

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Regime recalculation error")
                await asyncio.sleep(60)  # retry after 1 min

    async def _health_monitor(self) -> None:
        """Periodic health checks."""
        while self._running:
            try:
                await asyncio.sleep(30)
                if not self._running:
                    break

                # Check WebSocket health
                if self._ws_manager:
                    health = self._ws_manager.get_health()
                    for conn_name, conn_info in health.get("connections", {}).items():
                        if not conn_info.get("healthy", True):
                            logger.warning("Unhealthy connection: %s", conn_name)

                # Check drawdown
                if self._shared_risk:
                    halted, level, pct = self._shared_risk.check_drawdown()
                    if halted:
                        logger.warning("Drawdown halt active: %s %.2f%%", level, pct)

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Health monitor error")

    async def _equity_updater(self) -> None:
        """Update equity in risk manager periodically."""
        while self._running:
            try:
                await asyncio.sleep(10)
                if not self._running:
                    break

                if self._paper:
                    equity = self._paper.get_equity()
                    self._shared_risk.update_equity(equity)
                elif self._client:
                    try:
                        account = await self._client.get_futures_account()
                        equity = float(account.get("totalWalletBalance", 0))
                        if equity > 0:
                            self._shared_risk.update_equity(equity)
                    except Exception:
                        pass  # Non-critical

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Equity updater error")

    async def _config_reload_checker(self) -> None:
        """Check for config file changes and reload."""
        while self._running:
            try:
                await asyncio.sleep(30)
                if not self._running:
                    break
                if self._config_loader and self._config_loader.check_reload():
                    logger.info("Configuration reloaded")
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Config reload error")


# ==================================================================
# Entry point
# ==================================================================

def main() -> None:
    """Main entry point."""
    # Setup basic logging before config loads
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config_path = os.environ.get("CONFIG_PATH", "config.yaml")
    bot = MeanReversionBot(config_path=config_path)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Signal handlers for graceful shutdown
    def _signal_handler(sig, frame):
        logger.info("Received signal %s, initiating shutdown...", sig)
        bot._shutdown_event.set()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    try:
        loop.run_until_complete(bot.start())
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received")
    finally:
        loop.run_until_complete(bot.stop())
        loop.close()


if __name__ == "__main__":
    main()
