"""Entry point for STRAT-002: Funding Rate Arbitrage Bot.

Orchestrates:
- Dual-wallet setup (spot + futures)
- Warm-up: 500 funding rates per instrument, 24h basis history
- Funding settlement timers at 00:00/08:00/16:00 UTC
- Weekly rebalancing Monday 00:00 UTC
- Main strategy loop: entry/exit evaluation, delta monitoring
- State persistence every 5 seconds
- Orphan detection on startup
- Startup reconciliation (Section 8.2)
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure the project root is on sys.path for shared imports
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from shared.binance_client import BinanceClient
from shared.binance_ws_manager import WebSocketManager, ConnectionType
from shared.config_loader import ConfigLoader, RiskConfig
from shared.rate_limiter import RateLimiter, Priority, ApiType
from shared.risk_manager import RiskManager as SharedRiskManager
from shared.state_persistence import StatePersistence
from shared.performance_tracker import PerformanceTracker
from shared.paper_trading import PaperTradingEngine
from shared.utils import TimeSync, ExchangeInfo
from shared.alerting import AlertLevel

from strat_002_funding_arb.src.strategy import FundingArbStrategy, EntrySignal, ExitSignal
from strat_002_funding_arb.src.execution import ExecutionEngine
from strat_002_funding_arb.src.funding_tracker import FundingTracker
from strat_002_funding_arb.src.wallet_manager import WalletManager
from strat_002_funding_arb.src.risk_manager import FundingArbRiskManager
from strat_002_funding_arb.src.dashboard import FundingArbDashboard
from strat_002_funding_arb.src.strategy_metrics import FundingArbMetrics

logger = logging.getLogger(__name__)
system_logger = logging.getLogger("system")
trade_logger = logging.getLogger("trade")

STRATEGY_ID = "STRAT-002"
STRATEGY_NAME = "Funding Rate Arbitrage"


class FundingArbBot:
    """Main orchestrator for the Funding Rate Arbitrage bot.

    Parameters
    ----------
    config_path : str
        Path to the config.yaml file.
    """

    def __init__(self, config_path: str = "config.yaml") -> None:
        # Load configuration
        self._config_loader = ConfigLoader(config_path)
        self._cfg = self._config_loader.config
        self._params = self._cfg.strategy_params

        # Determine mode
        self._paper_mode = self._cfg.mode == "paper" or self._cfg.paper_trading.enabled
        self._running = False
        self._startup_time = time.time()

        # Core components (initialized in start())
        self._time_sync = TimeSync()
        self._exchange_info = ExchangeInfo()
        self._rate_limiter: Optional[RateLimiter] = None
        self._binance_client: Optional[BinanceClient] = None
        self._ws_manager: Optional[WebSocketManager] = None

        # Strategy components
        self._strategy: Optional[FundingArbStrategy] = None
        self._execution: Optional[ExecutionEngine] = None
        self._funding_tracker: Optional[FundingTracker] = None
        self._wallet_manager: Optional[WalletManager] = None
        self._risk_manager: Optional[FundingArbRiskManager] = None
        self._shared_risk: Optional[SharedRiskManager] = None

        # Infrastructure
        self._state: Optional[StatePersistence] = None
        self._perf_tracker: Optional[PerformanceTracker] = None
        self._paper_engine: Optional[PaperTradingEngine] = None
        self._dashboard: Optional[FundingArbDashboard] = None
        self._strategy_metrics: Optional[FundingArbMetrics] = None

        # Background tasks
        self._main_loop_task: Optional[asyncio.Task] = None
        self._daily_reset_task: Optional[asyncio.Task] = None
        self._weekly_rebalance_task: Optional[asyncio.Task] = None
        self._delta_monitor_task: Optional[asyncio.Task] = None
        self._daily_review_task: Optional[asyncio.Task] = None
        self._time_sync_task: Optional[asyncio.Task] = None

        logger.info(
            "FundingArbBot created: mode=%s instruments=%s",
            "paper" if self._paper_mode else "live",
            self._cfg.instruments,
        )

    # ══════════════════════════════════════════════════════════════════════
    #  Lifecycle
    # ══════════════════════════════════════════════════════════════════════

    async def start(self) -> None:
        """Initialize all components and start the bot."""
        system_logger.info("FundingArbBot starting — mode=%s",
                           "paper" if self._paper_mode else "live")
        self._running = True

        # ── 1. Initialize infrastructure ──────────────────────────────────
        self._rate_limiter = RateLimiter(
            budget=self._cfg.rate_limit_weight_per_min,
            burst=self._cfg.rate_limit_burst_weight,
        )

        self._binance_client = BinanceClient(
            api_key=self._cfg.binance.api_key,
            api_secret=self._cfg.binance.api_secret,
            time_sync=self._time_sync,
            exchange_info=self._exchange_info,
            rate_limiter=self._rate_limiter,
            spot_base_url=self._cfg.binance.spot_base_url,
            futures_base_url=self._cfg.binance.futures_base_url,
        )
        await self._binance_client.start()

        # Time sync
        try:
            await self._binance_client.sync_time()
        except Exception as exc:
            logger.warning("Initial time sync failed: %s", exc)

        # Exchange info
        try:
            await self._binance_client.load_exchange_info(self._cfg.instruments)
        except Exception as exc:
            logger.warning("Exchange info load failed: %s", exc)

        # WebSocket manager
        self._ws_manager = WebSocketManager(
            spot_ws_url=self._cfg.binance.spot_ws_url + "/stream",
            futures_ws_url=self._cfg.binance.futures_ws_url + "/stream",
            binance_client=self._binance_client,
        )

        # State persistence
        self._state = StatePersistence(
            state_dir=self._cfg.state.state_dir,
            strategy_id=STRATEGY_ID,
            save_interval=self._cfg.state.persistence_interval,
            max_snapshots=self._cfg.state.snapshot_count,
        )

        # Performance tracker
        self._perf_tracker = PerformanceTracker(strategy_id=STRATEGY_ID)

        # Paper trading engine
        if self._paper_mode:
            self._paper_engine = PaperTradingEngine(
                starting_equity=self._cfg.paper_trading.starting_equity,
                maker_fee_pct=self._cfg.paper_trading.maker_fee_pct,
                taker_fee_pct=self._cfg.paper_trading.taker_fee_pct,
            )

        # ── 2. Initialize strategy components ─────────────────────────────
        self._strategy = FundingArbStrategy(
            config=self._params,
            instruments=self._cfg.instruments,
        )

        self._execution = ExecutionEngine(
            binance_client=self._binance_client,
            paper_mode=self._paper_mode,
            paper_engine=self._paper_engine,
            config=self._params,
        )

        self._funding_tracker = FundingTracker(
            binance_client=self._binance_client,
            strategy=self._strategy,
            execution_engine=self._execution,
            paper_mode=self._paper_mode,
        )

        self._wallet_manager = WalletManager(
            binance_client=self._binance_client,
            strategy=self._strategy,
            config=self._params,
            paper_mode=self._paper_mode,
        )

        # Risk managers
        risk_config = RiskConfig(
            max_capital_pct=40.0,
            max_per_trade_pct=20.0,
            max_leverage=3,
            preferred_leverage=2,
            max_concurrent_positions=5,
            max_per_asset_pct=20.0,
            max_long_exposure_pct=40.0,
            max_short_exposure_pct=40.0,
            max_net_directional_pct=1.0,
            daily_drawdown_pct=1.5,
            weekly_drawdown_pct=3.0,
            monthly_drawdown_pct=5.0,
            system_wide_drawdown_pct=15.0,
        )
        self._shared_risk = SharedRiskManager(config=risk_config)
        self._risk_manager = FundingArbRiskManager(
            shared_risk=self._shared_risk,
            strategy=self._strategy,
            config=self._params,
        )

        # Wire wallet manager callbacks
        self._wallet_manager.set_callbacks(
            emergency_exit_callback=self._emergency_exit_instrument,
        )

        # ── 3. Load persisted state ───────────────────────────────────────
        loaded_state = self._state.load()
        if loaded_state.get("custom"):
            custom = loaded_state["custom"]
            if "strategy" in custom:
                self._strategy.restore_state(custom["strategy"])
            if "funding_tracker" in custom:
                self._funding_tracker.restore_state(custom["funding_tracker"])
            if "wallet_manager" in custom:
                self._wallet_manager.restore_state(custom["wallet_manager"])
            if "risk_manager" in custom:
                self._risk_manager.restore_state(custom["risk_manager"])
            logger.info("State restored from persistence")

        # ── 4. Initialize wallets ─────────────────────────────────────────
        if self._paper_mode:
            self._wallet_manager.initialize_paper_wallets(
                self._cfg.paper_trading.starting_equity
            )
            equity = self._cfg.paper_trading.starting_equity
        else:
            snapshot = await self._wallet_manager.get_wallet_snapshot()
            equity = snapshot.total_equity

        self._risk_manager.update_equity(equity)

        # ── 5. Warm-up data ───────────────────────────────────────────────
        await self._warm_up()

        # ── 6. Startup reconciliation ─────────────────────────────────────
        await self._startup_reconciliation()

        # ── 7. Register WebSocket streams ─────────────────────────────────
        self._register_ws_streams()

        # ── 8. Start all components ───────────────────────────────────────
        await self._ws_manager.start()
        await self._state.start()
        await self._funding_tracker.start()
        await self._wallet_manager.start()

        # Strategy Metrics (Section 10.2 + 10.3 + 10.4)
        self._strategy_metrics = FundingArbMetrics(
            strategy=self._strategy,
            risk_manager=self._risk_manager,
            funding_tracker=self._funding_tracker,
            wallet_manager=self._wallet_manager,
        )
        # Restore metrics state if available
        if loaded_state.get("custom", {}).get("strategy_metrics"):
            self._strategy_metrics.restore_state(loaded_state["custom"]["strategy_metrics"])

        # Dashboard
        self._dashboard = FundingArbDashboard(
            strategy=self._strategy,
            risk_manager=self._risk_manager,
            wallet_manager=self._wallet_manager,
            funding_tracker=self._funding_tracker,
            performance_tracker=self._perf_tracker,
            config={
                "host": self._cfg.dashboard.host,
                "port": self._cfg.dashboard.port,
                "paper_trading": self._paper_mode,
            },
            kill_fn=self._kill_switch,
        )
        await self._dashboard.start()

        # ── 9. Start background tasks ─────────────────────────────────────
        self._main_loop_task = asyncio.create_task(
            self._main_loop(), name="main_loop"
        )
        self._daily_reset_task = asyncio.create_task(
            self._daily_reset_loop(), name="daily_reset"
        )
        self._weekly_rebalance_task = asyncio.create_task(
            self._weekly_rebalance_loop(), name="weekly_rebalance"
        )
        self._delta_monitor_task = asyncio.create_task(
            self._delta_monitor_loop(), name="delta_monitor"
        )
        self._daily_review_task = asyncio.create_task(
            self._daily_review_loop(), name="daily_review"
        )
        self._time_sync_task = asyncio.create_task(
            self._time_sync_loop(), name="time_sync"
        )

        system_logger.info(
            "FundingArbBot started — mode=%s instruments=%d equity=%.2f",
            "paper" if self._paper_mode else "live",
            len(self._cfg.instruments),
            equity,
        )

    async def stop(self) -> None:
        """Gracefully shut down all components."""
        system_logger.info("FundingArbBot stopping")
        self._running = False

        # Cancel background tasks
        for task in [
            self._main_loop_task, self._daily_reset_task,
            self._weekly_rebalance_task, self._delta_monitor_task,
            self._daily_review_task, self._time_sync_task,
        ]:
            if task and not task.done():
                task.cancel()

        # Wait for cancellations
        tasks = [t for t in [
            self._main_loop_task, self._daily_reset_task,
            self._weekly_rebalance_task, self._delta_monitor_task,
            self._daily_review_task, self._time_sync_task,
        ] if t and not t.done()]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        # Save final state
        self._persist_state()

        # Stop components
        if self._dashboard:
            self._dashboard.stop()
        if self._wallet_manager:
            await self._wallet_manager.stop()
        if self._funding_tracker:
            await self._funding_tracker.stop()
        if self._ws_manager:
            await self._ws_manager.stop()
        if self._state:
            await self._state.stop()
        if self._binance_client:
            await self._binance_client.close()

        system_logger.info("FundingArbBot stopped")

    # ══════════════════════════════════════════════════════════════════════
    #  Warm-up (Section 2.2)
    # ══════════════════════════════════════════════════════════════════════

    async def _warm_up(self) -> None:
        """Warm up with historical data: 500 funding rates + 24h basis."""
        logger.info("Starting warm-up: fetching historical data")

        for symbol in self._cfg.instruments:
            try:
                # Fetch 500 historical funding rates
                rates = await self._binance_client.get_funding_rate_history(
                    symbol=symbol, limit=500,
                )
                for r in rates:
                    self._strategy.ingest_funding_rate(
                        symbol=symbol,
                        rate=float(r.get("fundingRate", 0)),
                        timestamp_ms=int(r.get("fundingTime", 0)),
                        predicted=False,
                    )
                logger.info(
                    "%s: Loaded %d historical funding rates", symbol, len(rates)
                )

                # Fetch current premium index for predicted rate
                premium = await self._binance_client.get_premium_index(symbol=symbol)
                if isinstance(premium, dict):
                    inst = self._strategy.instruments.get(symbol)
                    if inst:
                        inst.predicted_funding_rate = float(premium.get("lastFundingRate", 0))
                        inst.mark_price = float(premium.get("markPrice", 0))
                        inst.index_price = float(premium.get("indexPrice", 0))
                        inst.next_funding_time_ms = int(premium.get("nextFundingTime", 0))

                # Fetch 24h ticker for volume
                try:
                    ticker = await self._binance_client.get_ticker_24hr(symbol=symbol)
                    if isinstance(ticker, dict):
                        inst = self._strategy.instruments.get(symbol)
                        if inst:
                            inst.futures_volume_24h = float(ticker.get("quoteVolume", 0))
                except Exception:
                    pass

                await asyncio.sleep(0.2)  # Rate limit friendliness

            except Exception:
                logger.exception("Warm-up failed for %s", symbol)

        logger.info("Warm-up complete")

    # ══════════════════════════════════════════════════════════════════════
    #  Startup reconciliation (Section 8.2)
    # ══════════════════════════════════════════════════════════════════════

    async def _startup_reconciliation(self) -> None:
        """Reconcile account state on startup."""
        logger.info("Running startup reconciliation")

        if self._paper_mode:
            logger.info("Paper mode: skipping live reconciliation")
            return

        try:
            # Query both accounts
            spot_data = await self._binance_client.get_spot_account()
            futures_data = await self._binance_client.get_futures_account()

            # Build spot balances (base assets, not USDT)
            spot_balances: Dict[str, float] = {}
            for bal in spot_data.get("balances", []):
                asset = bal["asset"]
                qty = float(bal.get("free", 0)) + float(bal.get("locked", 0))
                if qty > 0 and asset != "USDT":
                    spot_balances[asset + "USDT"] = qty

            # Build futures positions
            futures_positions: Dict[str, float] = {}
            for pos in futures_data.get("positions", []):
                qty = float(pos.get("positionAmt", 0))
                if qty != 0:
                    futures_positions[pos["symbol"]] = qty

            # Orphan detection
            orphans = self._strategy.detect_orphans(spot_balances, futures_positions)
            if orphans:
                logger.critical(
                    "ORPHANS DETECTED: %d — correcting within 60 seconds",
                    len(orphans),
                )
                for orphan in orphans:
                    await self._correct_orphan(orphan)

            # Reconcile funding payments during downtime
            last_save = self._state.get_state("last_save_timestamp_ms", 0)
            if last_save > 0:
                await self._funding_tracker.reconcile_downtime(last_save)

        except Exception:
            logger.exception("Startup reconciliation failed")

    async def _correct_orphan(self, orphan: Dict[str, Any]) -> None:
        """Correct an orphaned position within 60 seconds.

        Section 8.2 & 11.1: On startup, if a spot position exists without
        a matching futures position (or vice versa), this is an ORPHAN.
        Log as CRITICAL, close the orphaned leg at market within 60 seconds,
        and send an alert.
        """
        symbol = orphan["symbol"]
        orphan_type = orphan["type"]

        logger.critical(
            "ORPHAN DETECTED on startup: %s — type=%s. "
            "Closing orphaned leg at market within 60 seconds.",
            symbol, orphan_type,
        )

        try:
            deadline = asyncio.get_event_loop().time() + 60  # 60 second deadline

            if orphan_type == "spot_without_futures":
                qty = orphan["spot_qty"]
                logger.critical(
                    "ORPHAN: %s has spot (%.8f) but no futures short — "
                    "selling spot at market",
                    symbol, qty,
                )
                await asyncio.wait_for(
                    self._binance_client.place_spot_order(
                        symbol=symbol, side="SELL", type="MARKET", quantity=qty,
                    ),
                    timeout=60.0,
                )

            elif orphan_type == "futures_without_spot":
                qty = abs(orphan["futures_qty"])
                logger.critical(
                    "ORPHAN: %s has futures (%.8f) but no spot — "
                    "closing futures short at market",
                    symbol, qty,
                )
                await asyncio.wait_for(
                    self._binance_client.place_futures_order(
                        symbol=symbol, side="BUY", type="MARKET",
                        quantity=qty, reduce_only=True,
                    ),
                    timeout=60.0,
                )

            logger.info("Orphan closed for %s within deadline", symbol)

            # Send alert via alerting system if available
            system_logger.critical(
                "ORPHAN_CLOSED symbol=%s type=%s — "
                "position was directionally exposed on startup and has been closed",
                symbol, orphan_type,
            )

        except asyncio.TimeoutError:
            logger.critical(
                "ORPHAN CORRECTION TIMEOUT: Failed to close %s orphan for %s "
                "within 60 seconds — MANUAL INTERVENTION REQUIRED",
                orphan_type, symbol,
            )
        except Exception:
            logger.exception(
                "ORPHAN CORRECTION FAILED for %s — MANUAL INTERVENTION REQUIRED",
                symbol,
            )

    # ══════════════════════════════════════════════════════════════════════
    #  WebSocket stream registration (Section 2.1)
    # ══════════════════════════════════════════════════════════════════════

    def _register_ws_streams(self) -> None:
        """Register all required WebSocket streams including user data streams.

        Section 2.1: Market data streams per instrument.
        Section 2.3: User data streams — BOTH spot and futures must be
        active simultaneously for position reconciliation.
        """
        for symbol in self._cfg.instruments:
            sym_lower = symbol.lower()

            # Futures market data streams
            futures_subs = [
                (f"{sym_lower}@markPrice@1s", self._strategy.on_mark_price),
                (f"{sym_lower}@bookTicker", self._strategy.on_futures_book_ticker),
                (f"{sym_lower}@forceOrder", self._strategy.on_force_order),
                (f"{sym_lower}@depth20@100ms", self._on_futures_depth),
            ]
            self._ws_manager.register_strategy(
                strategy_id=STRATEGY_ID,
                subscriptions=futures_subs,
                conn_type=ConnectionType.FUTURES,
            )

            # Spot market data streams
            spot_subs = [
                (f"{sym_lower}@bookTicker", self._strategy.on_spot_book_ticker),
                (f"{sym_lower}@depth20@100ms", self._on_spot_depth),
            ]
            self._ws_manager.register_strategy(
                strategy_id=STRATEGY_ID,
                subscriptions=spot_subs,
                conn_type=ConnectionType.SPOT,
            )

        # ── Section 2.3: User Data Streams ─────────────────────────────
        # Register FUTURES user data stream (ORDER_TRADE_UPDATE, ACCOUNT_UPDATE)
        self._ws_manager.register_user_data_stream(
            strategy_id=STRATEGY_ID,
            conn_type=ConnectionType.FUTURES,
            callbacks={
                "ORDER_TRADE_UPDATE": self._on_futures_order_update,
                "ACCOUNT_UPDATE": self._on_futures_account_update,
            },
        )

        # Register SPOT user data stream (executionReport, outboundAccountPosition)
        self._ws_manager.register_user_data_stream(
            strategy_id=STRATEGY_ID,
            conn_type=ConnectionType.SPOT,
            callbacks={
                "executionReport": self._on_spot_execution_report,
                "outboundAccountPosition": self._on_spot_account_position,
            },
        )

        logger.info(
            "Registered %d market streams + 2 user data streams (spot + futures) "
            "across %d instruments",
            len(self._cfg.instruments) * 6,
            len(self._cfg.instruments),
        )

    # ── User data stream handlers (Section 2.3) ───────────────────────

    async def _on_futures_order_update(self, data: Dict[str, Any]) -> None:
        """Handle futures ORDER_TRADE_UPDATE events.

        Processes fill confirmations, partial fills, and order status changes
        for futures leg reconciliation.
        """
        order = data.get("o", data)
        symbol = order.get("s", "")
        status = order.get("X", "")  # Order status
        side = order.get("S", "")
        avg_price = float(order.get("ap", 0))
        filled_qty = float(order.get("z", 0))

        logger.info(
            "FUTURES ORDER_TRADE_UPDATE: %s %s %s qty=%.8f avgPx=%.4f",
            symbol, side, status, filled_qty, avg_price,
        )

        if status == "FILLED":
            # Update strategy position tracking
            self._strategy.on_futures_fill(symbol, side, filled_qty, avg_price)

    async def _on_futures_account_update(self, data: Dict[str, Any]) -> None:
        """Handle futures ACCOUNT_UPDATE events.

        Reconciles margin balances and position changes from the exchange.
        """
        event_reason = data.get("a", {}).get("m", "")
        positions = data.get("a", {}).get("P", [])
        balances = data.get("a", {}).get("B", [])

        logger.info(
            "FUTURES ACCOUNT_UPDATE: reason=%s positions=%d balances=%d",
            event_reason, len(positions), len(balances),
        )

    async def _on_spot_execution_report(self, data: Dict[str, Any]) -> None:
        """Handle spot executionReport events.

        Processes spot order fills for spot leg reconciliation.
        """
        symbol = data.get("s", "")
        status = data.get("X", "")  # Order status
        side = data.get("S", "")
        filled_qty = float(data.get("z", 0))
        avg_price = float(data.get("Z", 0)) / filled_qty if filled_qty > 0 else 0

        logger.info(
            "SPOT executionReport: %s %s %s qty=%.8f avgPx=%.4f",
            symbol, side, status, filled_qty, avg_price,
        )

        if status == "FILLED":
            self._strategy.on_spot_fill(symbol, side, filled_qty, avg_price)

    async def _on_spot_account_position(self, data: Dict[str, Any]) -> None:
        """Handle spot outboundAccountPosition events.

        Reconciles spot account balances after fills or transfers.
        """
        balances = data.get("B", [])
        logger.info(
            "SPOT outboundAccountPosition: %d balance updates",
            len(balances),
        )

    async def _on_spot_depth(self, data: Dict[str, Any]) -> None:
        """Handle spot depth update."""
        symbol = data.get("s", "")
        if symbol:
            await self._strategy.on_depth_update(symbol, "spot", data)

    async def _on_futures_depth(self, data: Dict[str, Any]) -> None:
        """Handle futures depth update."""
        symbol = data.get("s", "")
        if symbol:
            await self._strategy.on_depth_update(symbol, "futures", data)

    # ══════════════════════════════════════════════════════════════════════
    #  Main strategy loop
    # ══════════════════════════════════════════════════════════════════════

    async def _main_loop(self) -> None:
        """Main evaluation loop: runs every 30 seconds."""
        logger.info("Main strategy loop started")

        while self._running:
            try:
                await asyncio.sleep(30)
                if not self._running:
                    break

                # Update equity
                snapshot = await self._wallet_manager.get_wallet_snapshot()
                equity = snapshot.total_equity
                self._risk_manager.update_equity(equity)

                # Check circuit breakers
                cb_signal = self._strategy.check_circuit_breakers()
                if cb_signal and cb_signal.urgency == "emergency":
                    logger.critical("Circuit breaker triggered: %s", cb_signal.reason)
                    await self._execute_emergency_exit(cb_signal)
                    continue

                # Evaluate exits first
                exit_signals = self._strategy.evaluate_exits()
                for signal in exit_signals:
                    await self._process_exit(signal)

                # Evaluate entries
                entry_signals = self._strategy.evaluate_entries(equity)
                for signal in entry_signals:
                    await self._process_entry(signal, equity)

                # Persist state
                self._persist_state()

                # Hot-reload config check
                if self._config_loader.check_reload():
                    self._params = self._config_loader.config.strategy_params
                    logger.info("Config reloaded")

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in main loop")
                await asyncio.sleep(10)

    # ══════════════════════════════════════════════════════════════════════
    #  Entry/exit processing
    # ══════════════════════════════════════════════════════════════════════

    async def _process_entry(self, signal: EntrySignal, equity: float) -> None:
        """Process a single entry signal."""
        logger.info(
            "Processing entry: %s yield=%.2f%% allocation=%.1f%%",
            signal.symbol, signal.annualized_yield, signal.allocation_pct,
        )

        # Risk check
        allocation_usdt = equity * (signal.allocation_pct / 100)
        size_mult = self._risk_manager.get_size_multiplier()
        allocation_usdt *= size_mult

        allowed, reason = self._risk_manager.check_entry_allowed(
            signal.symbol, allocation_usdt, equity,
        )
        if not allowed:
            logger.info("Entry rejected by risk: %s — %s", signal.symbol, reason)
            return

        inst = self._strategy.instruments.get(signal.symbol)
        if inst is None:
            return

        # Calculate position sizes
        spot_qty, futures_qty, margin_req = self._wallet_manager.calculate_position_sizes(
            symbol=signal.symbol,
            allocation_usdt=allocation_usdt,
            spot_price=inst.spot_best_ask if inst.spot_best_ask > 0 else inst.index_price,
            futures_price=inst.futures_best_bid if inst.futures_best_bid > 0 else inst.mark_price,
        )

        # Pre-entry validation
        snap = await self._wallet_manager.get_wallet_snapshot()
        valid, reason = await self._execution.validate_pre_entry(
            symbol=signal.symbol,
            spot_quantity=spot_qty,
            futures_quantity=futures_qty,
            spot_usdt_balance=snap.spot_usdt,
            futures_usdt_balance=snap.futures_available,
            spot_depth=inst.spot_depth,
            futures_depth=inst.futures_depth,
            spot_best_ask=inst.spot_best_ask,
            futures_best_bid=inst.futures_best_bid,
        )
        if not valid:
            logger.info("Pre-entry validation failed: %s — %s", signal.symbol, reason)
            return

        # Execute entry
        result = await self._execution.execute_entry(
            symbol=signal.symbol,
            spot_quantity=spot_qty,
            futures_quantity=futures_qty,
        )

        if result.success:
            # In paper mode, fill prices from instrument state
            if self._paper_mode and inst:
                if result.spot_fill:
                    result.spot_fill["avgPrice"] = str(inst.spot_best_ask or inst.index_price)
                    result.spot_fill_price = inst.spot_best_ask or inst.index_price
                if result.futures_fill:
                    result.futures_fill["avgPrice"] = str(inst.futures_best_bid or inst.mark_price)
                    result.futures_fill_price = inst.futures_best_bid or inst.mark_price

            pos = self._strategy.create_position(
                signal, result.spot_fill, result.futures_fill,
            )

            # Update wallet in paper mode
            if self._paper_mode:
                self._wallet_manager.paper_deduct_spot(result.spot_fill_price * spot_qty)
                self._wallet_manager.paper_add_spot_asset(signal.symbol, spot_qty, result.spot_fill_price)
                self._wallet_manager.paper_deduct_futures_margin(margin_req)

            # Record with shared risk
            self._shared_risk.record_position_change(
                strategy_id=STRATEGY_ID,
                symbol=signal.symbol,
                direction="LONG",
                size_usdt=allocation_usdt,
                is_open=True,
            )

            logger.info(
                "Position opened: %s spot=%.8f@%.2f futures=%.8f@%.2f basis=%.4f%%",
                signal.symbol, spot_qty, result.spot_fill_price,
                futures_qty, result.futures_fill_price, result.actual_basis_pct,
            )
        else:
            logger.error(
                "Entry execution failed for %s: %s",
                signal.symbol, result.errors,
            )

    async def _process_exit(self, signal: ExitSignal) -> None:
        """Process a single exit signal."""
        if signal.position_id == "ALL":
            # Circuit breaker — close all
            await self._execute_emergency_exit(signal)
            return

        pos = self._strategy.positions.get(signal.position_id)
        if pos is None:
            logger.warning("Position %s not found for exit", signal.position_id)
            return

        logger.info(
            "Processing exit: %s reason=%s urgency=%s partial=%.0f%%",
            pos.symbol, signal.reason, signal.urgency, signal.partial_pct,
        )

        spot_qty = pos.spot_quantity * (signal.partial_pct / 100)
        futures_qty = pos.futures_quantity * (signal.partial_pct / 100)

        result = await self._execution.execute_exit(
            symbol=pos.symbol,
            spot_quantity=spot_qty,
            futures_quantity=futures_qty,
        )

        if result.success:
            inst = self._strategy.instruments.get(pos.symbol)
            if self._paper_mode and inst:
                result.spot_fill_price = inst.spot_best_bid or inst.index_price
                result.futures_fill_price = inst.futures_best_ask or inst.mark_price

            if signal.partial_pct >= 100:
                # Full close
                trade_result = self._strategy.close_position(
                    signal.position_id,
                    result.spot_fill_price,
                    result.futures_fill_price,
                    result.total_fees,
                )
                if trade_result:
                    is_win = trade_result["total_pnl"] > 0
                    self._risk_manager.record_trade_result(
                        trade_result["total_pnl"], is_win,
                    )

                    if self._paper_mode:
                        self._wallet_manager.paper_remove_spot_asset(
                            pos.symbol, spot_qty, result.spot_fill_price,
                        )
                        self._wallet_manager.paper_release_futures_margin(
                            pos.futures_notional / 2,
                        )

                    self._shared_risk.record_position_change(
                        strategy_id=STRATEGY_ID,
                        symbol=pos.symbol,
                        direction="LONG",
                        size_usdt=pos.spot_notional,
                        is_open=False,
                    )

                    # Record in performance tracker
                    self._perf_tracker.record_trade({
                        "trade_id": signal.position_id,
                        "symbol": pos.symbol,
                        "side": "LONG",
                        "entry_price": pos.spot_entry_price,
                        "exit_price": result.spot_fill_price,
                        "quantity": spot_qty,
                        "pnl": trade_result["total_pnl"],
                        "pnl_pct": (trade_result["total_pnl"] / pos.spot_notional * 100) if pos.spot_notional > 0 else 0,
                        "fees": result.total_fees,
                        "entry_time_ms": pos.entry_time_ms,
                        "exit_time_ms": int(time.time() * 1000),
                    })

                    logger.info(
                        "Position closed: %s pnl=%.4f funding=%.4f reason=%s",
                        pos.symbol, trade_result["total_pnl"],
                        trade_result["funding_income"], signal.reason,
                    )
            else:
                # Partial close — adjust position
                pos.spot_quantity -= spot_qty
                pos.futures_quantity -= futures_qty
                pos.spot_notional = pos.spot_quantity * pos.spot_entry_price
                pos.futures_notional = pos.futures_quantity * pos.futures_entry_price
        else:
            logger.error(
                "Exit execution failed for %s: %s",
                pos.symbol, result.errors,
            )

    async def _execute_emergency_exit(self, signal: ExitSignal) -> None:
        """Emergency exit: close ALL positions."""
        logger.critical(
            "EMERGENCY EXIT: reason=%s — closing all positions",
            signal.reason,
        )

        positions_to_close = [
            {
                "symbol": p.symbol,
                "spot_quantity": p.spot_quantity,
                "futures_quantity": p.futures_quantity,
            }
            for p in self._strategy.positions.values()
        ]

        results = await self._execution.emergency_close_all(positions_to_close)

        # Close all strategy positions
        for pos_id in list(self._strategy.positions.keys()):
            pos = self._strategy.positions[pos_id]
            self._strategy.close_position(pos_id, 0, 0, 0)
            self._shared_risk.record_position_change(
                strategy_id=STRATEGY_ID,
                symbol=pos.symbol,
                direction="LONG",
                size_usdt=pos.spot_notional,
                is_open=False,
            )

        logger.critical(
            "Emergency exit complete: %d positions closed", len(results),
        )

    async def _emergency_exit_instrument(
        self, symbol: str, reason: str,
    ) -> None:
        """Emergency exit for a single instrument."""
        pos = None
        for p in self._strategy.positions.values():
            if p.symbol == symbol:
                pos = p
                break
        if pos is None:
            return

        signal = ExitSignal(
            position_id=pos.position_id,
            symbol=symbol,
            reason=reason,
            urgency="emergency",
        )
        await self._process_exit(signal)

    # ══════════════════════════════════════════════════════════════════════
    #  Kill switch
    # ══════════════════════════════════════════════════════════════════════

    async def _kill_switch(self, reason: str) -> Dict[str, Any]:
        """Execute kill switch: close ALL positions within 10 seconds."""
        logger.critical("KILL SWITCH: %s", reason)
        self._risk_manager.trigger_kill_switch(reason)

        signal = ExitSignal(
            position_id="ALL",
            symbol="ALL",
            reason=f"kill_switch: {reason}",
            urgency="emergency",
        )
        await self._execute_emergency_exit(signal)

        return {"status": "executed", "reason": reason}

    # ══════════════════════════════════════════════════════════════════════
    #  Background tasks
    # ══════════════════════════════════════════════════════════════════════

    async def _daily_reset_loop(self) -> None:
        """Reset daily drawdown at 00:00 UTC each day."""
        while self._running:
            try:
                now = datetime.now(timezone.utc)
                next_midnight = (now + timedelta(days=1)).replace(
                    hour=0, minute=0, second=0, microsecond=0,
                )
                wait = (next_midnight - now).total_seconds()
                await asyncio.sleep(wait)

                if not self._running:
                    break

                self._risk_manager.reset_daily_drawdown()
                logger.info("Daily drawdown reset at %s UTC", datetime.now(timezone.utc))

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in daily reset loop")
                await asyncio.sleep(3600)

    async def _weekly_rebalance_loop(self) -> None:
        """Rebalance allocations every Monday 00:00 UTC."""
        while self._running:
            try:
                now = datetime.now(timezone.utc)
                # Find next Monday 00:00
                days_until_monday = (7 - now.weekday()) % 7
                if days_until_monday == 0 and now.hour > 0:
                    days_until_monday = 7
                next_monday = (now + timedelta(days=days_until_monday)).replace(
                    hour=0, minute=0, second=0, microsecond=0,
                )
                wait = (next_monday - now).total_seconds()
                await asyncio.sleep(wait)

                if not self._running:
                    break

                # Reset weekly drawdown and rotation count
                self._risk_manager.reset_weekly_drawdown()
                self._strategy.reset_weekly_rotation_count()

                # Evaluate rebalancing
                snapshot = await self._wallet_manager.get_wallet_snapshot()
                exits, entries = self._strategy.evaluate_rebalancing(snapshot.total_equity)

                for exit_sig in exits:
                    await self._process_exit(exit_sig)

                for entry_sig in entries:
                    await self._process_entry(entry_sig, snapshot.total_equity)

                logger.info(
                    "Weekly rebalance: %d exits, %d entries",
                    len(exits), len(entries),
                )

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in weekly rebalance loop")
                await asyncio.sleep(3600)

    async def _delta_monitor_loop(self) -> None:
        """Monitor delta neutrality every 30 seconds."""
        while self._running:
            try:
                await asyncio.sleep(30)
                if not self._running or not self._strategy.positions:
                    continue

                snapshots = self._risk_manager.check_delta_neutrality()
                for snap in snapshots:
                    if not snap.within_tolerance and abs(snap.delta_pct) > 5.0:
                        logger.warning(
                            "Delta critical on %s: %.2f%% — corrective action needed",
                            snap.symbol, snap.delta_pct,
                        )

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in delta monitor loop")
                await asyncio.sleep(60)

    async def _daily_review_loop(self) -> None:
        """Run daily position reviews (Section 4.5)."""
        while self._running:
            try:
                # Wait until next day boundary
                now = datetime.now(timezone.utc)
                next_review = (now + timedelta(days=1)).replace(
                    hour=1, minute=0, second=0, microsecond=0,
                )
                wait = (next_review - now).total_seconds()
                await asyncio.sleep(wait)

                if not self._running:
                    break

                for pos in self._strategy.positions.values():
                    self._strategy.daily_review(pos.symbol)

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in daily review loop")
                await asyncio.sleep(3600)

    async def _time_sync_loop(self) -> None:
        """Sync clock with Binance every 60 seconds (Section 11.6)."""
        while self._running:
            try:
                await asyncio.sleep(60)
                if not self._running:
                    break
                await self._binance_client.ensure_time_synced()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Time sync failed")
                await asyncio.sleep(30)

    # ══════════════════════════════════════════════════════════════════════
    #  State persistence
    # ══════════════════════════════════════════════════════════════════════

    def _persist_state(self) -> None:
        """Save current state to disk."""
        if self._state is None:
            return

        custom = {
            "strategy": self._strategy.get_state(),
            "funding_tracker": self._funding_tracker.get_state(),
            "wallet_manager": self._wallet_manager.get_state(),
            "risk_manager": self._risk_manager.get_state(),
        }
        self._state.update_state("custom", custom)
        self._state.update_state("positions", self._strategy.get_state().get("positions", []))


# ══════════════════════════════════════════════════════════════════════════
#  Application entry point
# ══════════════════════════════════════════════════════════════════════════

def _setup_logging(log_level: str = "INFO", log_dir: str = "data/logs") -> None:
    """Configure logging with file and console handlers."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root.addHandler(console)

    # File handler
    fh = logging.FileHandler(
        os.path.join(log_dir, "strat_002.log"),
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root.addHandler(fh)

    # Trade logger
    trade_handler = logging.FileHandler(
        os.path.join(log_dir, "trades.log"),
        encoding="utf-8",
    )
    trade_handler.setFormatter(logging.Formatter(
        "%(asctime)s\t%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    trade_log = logging.getLogger("trade")
    trade_log.addHandler(trade_handler)
    trade_log.setLevel(logging.INFO)
    trade_log.propagate = False

    # System logger
    sys_handler = logging.FileHandler(
        os.path.join(log_dir, "system.log"),
        encoding="utf-8",
    )
    sys_handler.setFormatter(logging.Formatter(
        "%(asctime)s\t%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    sys_log = logging.getLogger("system")
    sys_log.addHandler(sys_handler)
    sys_log.setLevel(logging.INFO)
    sys_log.propagate = False


async def main() -> None:
    """Async entry point."""
    # Determine config path
    config_path = os.environ.get(
        "CONFIG_PATH",
        str(Path(__file__).parent.parent / "config.yaml"),
    )

    # Load .env if present
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_path)
        except ImportError:
            pass

    # Setup logging
    _setup_logging(
        log_level=os.environ.get("LOG_LEVEL", "INFO"),
        log_dir=os.environ.get("LOG_DIR", "data/logs"),
    )

    logger.info("Starting STRAT-002: Funding Rate Arbitrage Bot")

    bot = FundingArbBot(config_path=config_path)

    # Handle signals for graceful shutdown
    loop = asyncio.get_running_loop()
    shutdown_event = asyncio.Event()

    def _signal_handler():
        logger.info("Shutdown signal received")
        shutdown_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)

    try:
        await bot.start()
        await shutdown_event.wait()
    except KeyboardInterrupt:
        pass
    finally:
        await bot.stop()

    logger.info("STRAT-002 shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
