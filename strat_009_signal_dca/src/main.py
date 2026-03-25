"""Entry point for the STRAT-009 Signal-Enhanced DCA bot.

Orchestrates:
1. Load config and initialise logging
2. Init all components (client, risk, budget, strategy, dashboard)
3. Warm up historical data (200 daily candles, Fear & Greed)
4. Restore persisted state
5. Start scheduler (DCA schedule, hourly crash monitor, daily signal refresh)
6. Start dashboard
7. Run event loop
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

# Ensure the project root and shared lib are importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT.parent))

from shared.config_loader import ConfigLoader, BotConfig, RiskConfig
from shared.binance_client import BinanceClient
from shared.rate_limiter import RateLimiter, ApiType
from shared.utils import TimeSync, ExchangeInfo
from shared.external_data import FearGreedClient, GlassnodeClient
from shared.state_persistence import StatePersistence
from shared.heartbeat import HeartbeatMonitor
from shared.kill_switch import KillSwitch
from shared.log_manager import setup_logging
from shared.paper_trading import PaperTradingEngine
from shared.risk_manager import RiskManager, CrossStrategyReader

from src.strategy import SignalDCAStrategy
from src.budget_manager import BudgetManager
from src.risk_manager import DCARiskManager
from src.dashboard import DCADashboard

logger = logging.getLogger(__name__)

STRATEGY_ID = "STRAT-009"
STRATEGY_NAME = "Signal-Enhanced DCA"


class DCABot:
    """Main orchestrator for the Signal-Enhanced DCA bot."""

    def __init__(self) -> None:
        self._running = False
        self._shutdown_event = asyncio.Event()

        # Components (initialised in start())
        self._config_loader: ConfigLoader = None
        self._config: BotConfig = None
        self._binance: BinanceClient = None
        self._rate_limiter: RateLimiter = None
        self._time_sync: TimeSync = None
        self._exchange_info: ExchangeInfo = None
        self._state: StatePersistence = None
        self._heartbeat: HeartbeatMonitor = None
        self._kill_switch: KillSwitch = None
        self._shared_risk: RiskManager = None
        self._dca_risk: DCARiskManager = None
        self._budget: BudgetManager = None
        self._strategy: SignalDCAStrategy = None
        self._dashboard: DCADashboard = None
        self._paper_engine: PaperTradingEngine = None
        self._fg_client: FearGreedClient = None
        self._glassnode: GlassnodeClient = None

        # Scheduler tasks
        self._tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        """Initialize all components and start the bot."""
        logger.info("=" * 60)
        logger.info("Starting %s (%s)", STRATEGY_NAME, STRATEGY_ID)
        logger.info("=" * 60)

        # 1. Load config
        config_path = os.environ.get("CONFIG_PATH", str(PROJECT_ROOT / "config.yaml"))
        self._config_loader = ConfigLoader(config_path)
        self._config = self._config_loader.config
        params = self._config.strategy_params

        # 2. Setup logging
        setup_logging(
            strategy_id=STRATEGY_ID,
            log_dir=self._config.logging.log_dir,
            level=self._config.logging.level,
        )
        logger.info("Config loaded: mode=%s, instruments=%s", self._config.mode, self._config.instruments)

        # 3. Init Binance client
        self._time_sync = TimeSync()
        self._exchange_info = ExchangeInfo()
        self._rate_limiter = RateLimiter(
            budget=self._config.rate_limit_weight_per_min,
            burst=self._config.rate_limit_burst_weight,
        )

        self._binance = BinanceClient(
            api_key=self._config.binance.api_key,
            api_secret=self._config.binance.api_secret,
            time_sync=self._time_sync,
            exchange_info=self._exchange_info,
            rate_limiter=self._rate_limiter,
            spot_base_url=self._config.binance.spot_base_url,
            recv_window=self._config.binance.recv_window,
        )
        await self._binance.start()
        logger.info("Binance client started")

        # 4. External data clients
        self._fg_client = FearGreedClient()

        glassnode_key = params.get("glassnode_api_key", "")
        if params.get("sopr_enabled", False) and glassnode_key:
            self._glassnode = GlassnodeClient(api_key=glassnode_key)
            logger.info("Glassnode client enabled for SOPR data")

        # 5. Paper trading engine
        if self._config.mode == "paper":
            self._paper_engine = PaperTradingEngine(
                starting_equity=self._config.paper_trading.starting_equity,
                maker_fee_pct=self._config.paper_trading.maker_fee_pct,
                taker_fee_pct=self._config.paper_trading.taker_fee_pct,
            )
            logger.info(
                "Paper trading mode: starting equity $%.2f",
                self._config.paper_trading.starting_equity,
            )

        # 6. Risk managers
        risk_config = self._config.risk
        cross_reader = CrossStrategyReader(state_dir=self._config.state.state_dir)
        self._shared_risk = RiskManager(
            config=risk_config,
            cross_strategy_reader=cross_reader,
        )

        equity = self._config.paper_trading.starting_equity if self._config.mode == "paper" else 10000.0
        self._shared_risk.update_equity(equity)

        self._dca_risk = DCARiskManager(
            shared_risk_manager=self._shared_risk,
            strategy_params=params,
        )

        # 7. Budget manager
        self._budget = BudgetManager(
            monthly_dca_budget=params.get("monthly_dca_budget", 400.0),
            monthly_crash_reserve=params.get("monthly_crash_reserve", 200.0),
            base_amounts=params.get("base_amounts", {"BTCUSDT": 50.0, "ETHUSDT": 30.0}),
            min_purchase_usdt=params.get("min_purchase_usdt", 5.0),
            budget_cap_multiplier=params.get("budget_cap_multiplier", 4.5),
        )

        # 8. Strategy
        self._strategy = SignalDCAStrategy(
            config=self._config,
            binance_client=self._binance,
            budget_manager=self._budget,
            risk_manager=self._dca_risk,
            fear_greed_client=self._fg_client,
            paper_engine=self._paper_engine,
            glassnode_client=self._glassnode,
        )

        # 9. State persistence
        self._state = StatePersistence(
            state_dir=self._config.state.state_dir,
            strategy_id=STRATEGY_ID,
            save_interval=self._config.state.persistence_interval,
            max_snapshots=self._config.state.snapshot_count,
        )

        # Load persisted state
        persisted = self._state.load()
        custom = persisted.get("custom", {})
        if custom:
            strategy_state = custom.get("strategy", {})
            if strategy_state:
                self._strategy.load_state(strategy_state)
            risk_state = custom.get("risk", {})
            if risk_state:
                self._dca_risk.load_state(risk_state)
            budget_state = custom.get("budget", {})
            if budget_state:
                self._budget.load_state(budget_state)
            logger.info("Persisted state restored successfully")

        # 10. Heartbeat
        self._heartbeat = HeartbeatMonitor(
            strategy_id=STRATEGY_ID,
            interval=self._config.heartbeat.interval,
            timeout=self._config.heartbeat.timeout,
            max_restarts_per_hour=self._config.heartbeat.max_restarts_per_hour,
        )

        # 11. Kill switch
        self._kill_switch = KillSwitch(
            binance_client=self._binance,
            state_persistence=None,
            max_execution_time=15,
        )

        # 12. Warm up data
        logger.info("Warming up historical data...")
        await self._strategy.warm_up_data()

        # Compute next DCA time
        if self._strategy._next_dca_time <= 0:
            self._strategy._next_dca_time = self._strategy.compute_next_dca_time()
        logger.info(
            "Next DCA scheduled: %s",
            datetime.fromtimestamp(self._strategy._next_dca_time, tz=timezone.utc).isoformat(),
        )

        # Check for missed DCA during downtime
        await self._check_missed_dca()

        # 13. Dashboard
        self._dashboard = DCADashboard(
            strategy_id=STRATEGY_ID,
            strategy_name=STRATEGY_NAME,
            host=self._config.dashboard.host,
            port=self._config.dashboard.port,
        )
        self._dashboard.set_data_providers(
            positions_fn=lambda: [h.to_dict() for h in self._dca_risk.get_holdings().values()],
            trades_fn=lambda limit: self._strategy.get_recent_purchases(limit),
            metrics_fn=self._strategy.get_metrics,
            equity_fn=lambda: [],
            alerts_fn=lambda: [],
            config_fn=lambda: {"mode": self._config.mode, "paper_trading": self._config.mode == "paper",
                               "instruments": self._config.instruments,
                               "schedule": self._config.strategy_params.get("dca_schedule", "weekly")},
            kill_fn=lambda reason: self._handle_kill_switch(reason),
        )
        self._dashboard.set_dca_providers(
            budget_fn=self._budget.get_budget_summary,
            countdown_fn=self._strategy.get_next_dca_countdown,
            signals_fn=lambda: {s: sig.to_dict() for s, sig in self._strategy._signals.items()},
            purchases_fn=self._strategy.get_recent_purchases,
            crash_cooldowns_fn=self._strategy.get_crash_cooldown_status,
            risk_status_fn=self._dca_risk.get_status,
        )

        # 14. Start everything
        self._running = True
        await self._heartbeat.start()
        await self._state.start()
        await self._dashboard.start()

        # Start scheduler tasks
        self._tasks.append(asyncio.create_task(self._dca_scheduler_loop(), name="dca_scheduler"))
        self._tasks.append(asyncio.create_task(self._crash_monitor_loop(), name="crash_monitor"))
        self._tasks.append(asyncio.create_task(self._signal_refresh_loop(), name="signal_refresh"))
        self._tasks.append(asyncio.create_task(self._risk_monitor_loop(), name="risk_monitor"))
        self._tasks.append(asyncio.create_task(self._state_save_loop(), name="state_saver"))
        self._tasks.append(asyncio.create_task(self._config_reload_loop(), name="config_reloader"))

        logger.info("=" * 60)
        logger.info("%s bot fully started and running", STRATEGY_NAME)
        logger.info("Dashboard: http://%s:%d", self._config.dashboard.host, self._config.dashboard.port)
        logger.info("Mode: %s", self._config.mode)
        logger.info("=" * 60)

    async def _check_missed_dca(self) -> None:
        """Check if a DCA was missed during downtime and execute if within 1 hour."""
        next_time = self._strategy._next_dca_time
        now = time.time()

        if next_time > 0 and now > next_time:
            elapsed_minutes = (now - next_time) / 60
            max_delay = self._config.strategy_params.get("max_schedule_delay_minutes", 60)
            if elapsed_minutes <= max_delay:
                logger.info(
                    "Missed DCA detected (%.1f min ago), executing now...",
                    elapsed_minutes,
                )
                purchases = await self._strategy.execute_scheduled_dca()
                if purchases:
                    logger.info("Missed DCA executed: %d purchases", len(purchases))
            else:
                logger.info(
                    "Missed DCA was %.1f min ago (> %d min limit), skipping to next",
                    elapsed_minutes, max_delay,
                )
                self._strategy._next_dca_time = self._strategy.compute_next_dca_time()

    # ------------------------------------------------------------------
    # Scheduler loops
    # ------------------------------------------------------------------

    async def _dca_scheduler_loop(self) -> None:
        """Main DCA schedule loop. Checks every 30 seconds if DCA is due."""
        while self._running:
            try:
                await asyncio.sleep(30)

                if self._kill_switch.is_triggered():
                    continue

                now = time.time()
                next_time = self._strategy._next_dca_time

                if next_time > 0 and now >= next_time:
                    logger.info("DCA schedule triggered!")
                    purchases = await self._strategy.execute_scheduled_dca()
                    if purchases:
                        logger.info(
                            "DCA executed: %d purchases, total $%.2f",
                            len(purchases),
                            sum(p.amount_usdt for p in purchases),
                        )

                    # Check take-profit after DCA
                    tp_result = await self._strategy.execute_take_profit()
                    if tp_result:
                        logger.info("Take-profit rebalancing executed: %s", tp_result)

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("DCA scheduler error")
                await asyncio.sleep(60)

    async def _crash_monitor_loop(self) -> None:
        """Hourly crash-buy detection loop."""
        while self._running:
            try:
                await asyncio.sleep(3600)  # Check every hour

                if self._kill_switch.is_triggered():
                    continue

                triggers = self._strategy.detect_crash_buys()
                if triggers:
                    logger.info("Crash-buy triggers detected: %s", triggers)
                    purchases = await self._strategy.execute_crash_buys(triggers)
                    if purchases:
                        logger.info(
                            "Crash-buys executed: %d purchases, total $%.2f",
                            len(purchases),
                            sum(p.amount_usdt for p in purchases),
                        )

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Crash monitor error")
                await asyncio.sleep(300)

    async def _signal_refresh_loop(self) -> None:
        """Refresh signals daily at 00:00 UTC + periodically update prices."""
        while self._running:
            try:
                # Refresh Fear & Greed
                await self._strategy.update_fear_greed()

                # Refresh SOPR if enabled
                if self._config.strategy_params.get("sopr_enabled", False):
                    await self._strategy.update_sopr()

                # Recalculate signals for all instruments
                for symbol in self._config.instruments:
                    self._strategy.calculate_signals(symbol)

                # Refresh price data
                for symbol in self._config.instruments:
                    try:
                        klines = await self._binance.get_spot_klines(
                            symbol=symbol, interval="1d", limit=2,
                        )
                        if klines:
                            latest = klines[-1]
                            self._strategy.update_daily_candle(symbol, {
                                "timestamp": latest[0],
                                "open": float(latest[1]),
                                "high": float(latest[2]),
                                "low": float(latest[3]),
                                "close": float(latest[4]),
                                "volume": float(latest[5]),
                            })

                        # Hourly candles
                        hourly = await self._binance.get_spot_klines(
                            symbol=symbol, interval="1h", limit=2,
                        )
                        if hourly:
                            latest_h = hourly[-1]
                            self._strategy.update_hourly_candle(symbol, {
                                "timestamp": latest_h[0],
                                "open": float(latest_h[1]),
                                "high": float(latest_h[2]),
                                "low": float(latest_h[3]),
                                "close": float(latest_h[4]),
                                "volume": float(latest_h[5]),
                            })

                        # Book ticker for spread check
                        depth = await self._binance.get_spot_depth(symbol=symbol, limit=5)
                        if depth:
                            bids = depth.get("bids", [])
                            asks = depth.get("asks", [])
                            if bids and asks:
                                self._strategy.update_book_ticker(
                                    symbol,
                                    bid=float(bids[0][0]),
                                    ask=float(asks[0][0]),
                                )

                    except Exception as exc:
                        logger.warning("Failed to refresh data for %s: %s", symbol, exc)

                # Update equity on shared risk manager
                if self._config.mode == "paper" and self._paper_engine:
                    self._shared_risk.update_equity(self._paper_engine.get_equity())
                else:
                    try:
                        account = await self._binance.get_spot_account()
                        balances = account.get("balances", [])
                        total_usdt = 0.0
                        for bal in balances:
                            asset = bal.get("asset", "")
                            free = float(bal.get("free", 0))
                            locked = float(bal.get("locked", 0))
                            total = free + locked
                            if asset == "USDT":
                                total_usdt += total
                        portfolio = self._dca_risk.get_portfolio_value()
                        self._shared_risk.update_equity(total_usdt + portfolio)
                    except Exception as exc:
                        logger.warning("Failed to update equity: %s", exc)

                # Update heartbeat
                self._heartbeat.set_positions_count(len(self._dca_risk.get_holdings()))
                pnl = self._dca_risk.get_portfolio_value() - self._dca_risk.get_total_invested()
                self._heartbeat.set_unrealized_pnl(pnl)

                logger.debug("Signal refresh complete")

                # Sleep until next refresh (1 hour)
                await asyncio.sleep(3600)

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Signal refresh error")
                await asyncio.sleep(300)

    async def _risk_monitor_loop(self) -> None:
        """Periodic risk checks (emergency stop, diversification)."""
        while self._running:
            try:
                await asyncio.sleep(3600)  # Hourly

                # Emergency stop-loss check
                halted, reason = self._dca_risk.check_emergency_stop()
                if halted:
                    logger.critical("Emergency stop active: %s", reason)

                # Diversification check
                over_concentrated = self._dca_risk.check_diversification()
                if over_concentrated:
                    logger.warning(
                        "Diversification alert: %s",
                        {k: f"{v}%" for k, v in over_concentrated.items()},
                    )

                # Config reload check
                self._config_loader.check_reload()

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Risk monitor error")
                await asyncio.sleep(300)

    async def _state_save_loop(self) -> None:
        """Periodic state persistence (every 30 seconds)."""
        while self._running:
            try:
                await asyncio.sleep(30)

                custom_state = {
                    "strategy": self._strategy.to_state(),
                    "risk": self._dca_risk.to_state(),
                    "budget": self._budget.to_state(),
                }
                self._state.update_state("custom", custom_state)

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("State save error")
                await asyncio.sleep(10)

    async def _config_reload_loop(self) -> None:
        """Check for config file changes every 60 seconds."""
        while self._running:
            try:
                await asyncio.sleep(60)
                if self._config_loader.check_reload():
                    self._config = self._config_loader.config
                    logger.info("Configuration reloaded")
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Config reload error")

    # ------------------------------------------------------------------
    # Kill switch handler
    # ------------------------------------------------------------------

    async def _handle_kill_switch(self, reason: str) -> dict:
        """Handle kill switch activation for DCA.

        DCA kill switch halts purchases but does NOT liquidate holdings.
        """
        self._dca_risk.activate_kill_switch(reason)
        logger.critical("DCA Kill Switch: %s — purchases halted, holdings retained", reason)
        return {
            "status": "halted",
            "reason": reason,
            "holdings_retained": True,
            "message": "DCA purchases halted. Holdings are NOT liquidated.",
        }

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    async def stop(self) -> None:
        """Graceful shutdown."""
        logger.info("Shutting down %s...", STRATEGY_NAME)
        self._running = False

        # Cancel scheduler tasks
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

        # Save final state
        try:
            custom_state = {
                "strategy": self._strategy.to_state(),
                "risk": self._dca_risk.to_state(),
                "budget": self._budget.to_state(),
            }
            self._state.update_state("custom", custom_state)
            await self._state.stop()
        except Exception as exc:
            logger.error("Failed to save final state: %s", exc)

        # Stop components
        self._heartbeat.stop()
        self._dashboard.stop()

        if self._fg_client:
            await self._fg_client.close()
        if self._glassnode:
            await self._glassnode.close()
        await self._binance.close()

        logger.info("%s shutdown complete", STRATEGY_NAME)

    async def run(self) -> None:
        """Main entry point: start and run until shutdown signal."""
        await self.start()

        # Wait for shutdown
        try:
            await self._shutdown_event.wait()
        except asyncio.CancelledError:
            pass
        finally:
            await self.stop()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point."""
    bot = DCABot()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Signal handlers for graceful shutdown
    def _signal_handler():
        logger.info("Received shutdown signal")
        bot._shutdown_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)

    try:
        loop.run_until_complete(bot.run())
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
        loop.run_until_complete(bot.stop())
    finally:
        loop.close()


if __name__ == "__main__":
    main()
