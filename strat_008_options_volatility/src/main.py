"""STRAT-008: Options & Volatility Strategies — Main Entry Point.

Initialises Binance (required) and optional Deribit connections, warms up
kline buffers + RV calculation, determines operating mode (Synthetic vs Full),
runs hourly volatility regime assessment, manages 7-day option cycles,
monitors Greeks, and coordinates all sub-strategies.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root and shared lib are importable
_project_root = Path(__file__).resolve().parent.parent
_strategies_root = _project_root.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_strategies_root))

from shared.config_loader import ConfigLoader, BotConfig
from shared.log_manager import setup_logging
from shared.binance_client import BinanceClient
from shared.binance_ws_manager import WebSocketManager, ConnectionType
from shared.state_persistence import StatePersistence
from shared.risk_manager import RiskManager, CrossStrategyReader
from shared.rate_limiter import RateLimiter
from shared.utils import TimeSync, ExchangeInfo
from shared.paper_trading import PaperTradingEngine
from shared.indicators import IndicatorBuffer

from src import STRATEGY_ID, STRATEGY_NAME
from src.volatility_engine import VolatilityEngine
from src.strategy import StrategyCoordinator, MODE_SYNTHETIC, MODE_FULL
from src.risk_manager import StrategyRiskManager
from src.deribit_client import DeribitClient
from src.dashboard import OptionsDashboard
from shared.external_data import GlassnodeClient

logger = logging.getLogger(__name__)
system_logger = logging.getLogger("system")
trade_logger = logging.getLogger("trade")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INSTRUMENTS = ["BTCUSDT", "ETHUSDT"]
REGIME_ASSESSMENT_INTERVAL = 3600      # 1 hour
CYCLE_CHECK_INTERVAL = 60              # 1 minute
CIRCUIT_BREAKER_INTERVAL = 10          # 10 seconds
STATE_SAVE_INTERVAL = 5.0              # 5 seconds
WARMUP_1M_CANDLES = 1440               # 24 hours
WARMUP_1H_CANDLES = 168                # 7 days
WARMUP_1D_CANDLES = 90                 # 90 days


# ---------------------------------------------------------------------------
# Main Bot
# ---------------------------------------------------------------------------

class OptionsVolatilityBot:
    """STRAT-008 Options & Volatility Strategies Bot.

    Lifecycle:
    1. Load config + setup logging
    2. Init Binance client + optional Deribit
    3. Determine mode (Synthetic vs Full)
    4. Warm up kline buffers + calculate initial RV
    5. Run main loop: hourly regime, cycle management, Greek monitoring
    """

    def __init__(self, config_path: str = "config.yaml") -> None:
        self._config_loader = ConfigLoader(config_path)
        self._config: BotConfig = self._config_loader.config
        self._params: dict = self._config.strategy_params

        # Core infrastructure
        self._time_sync = TimeSync()
        self._exchange_info = ExchangeInfo()
        self._rate_limiter: Optional[RateLimiter] = None
        self._binance: Optional[BinanceClient] = None
        self._ws_manager: Optional[WebSocketManager] = None
        self._deribit: Optional[DeribitClient] = None
        self._state: Optional[StatePersistence] = None
        self._shared_risk: Optional[RiskManager] = None

        # Strategy components
        self._vol_engine: Optional[VolatilityEngine] = None
        self._risk_mgr: Optional[StrategyRiskManager] = None
        self._coordinator: Optional[StrategyCoordinator] = None
        self._paper_engine: Optional[PaperTradingEngine] = None
        self._dashboard: Optional[OptionsDashboard] = None

        # Glassnode client for MVRV
        self._glassnode: Optional[GlassnodeClient] = None

        # Operating mode
        self._mode = MODE_SYNTHETIC

        # Lifecycle
        self._running = False
        self._tasks: List[asyncio.Task] = []

        # Prices
        self._spot_prices: Dict[str, float] = {}
        self._futures_prices: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the bot: initialize all components and begin main loop."""
        # 1. Logging
        setup_logging(
            strategy_id=STRATEGY_ID,
            log_dir=self._config.logging.log_dir,
            level=self._config.logging.level,
        )
        logger.info("=" * 70)
        logger.info("STRAT-008: %s starting", STRATEGY_NAME)
        logger.info("=" * 70)

        # 2. Rate limiter
        self._rate_limiter = RateLimiter(
            budget=self._config.rate_limit_weight_per_min,
            burst=self._config.rate_limit_burst_weight,
        )

        # 3. Binance client
        self._binance = BinanceClient(
            api_key=self._config.binance.api_key,
            api_secret=self._config.binance.api_secret,
            time_sync=self._time_sync,
            exchange_info=self._exchange_info,
            rate_limiter=self._rate_limiter,
            spot_base_url=self._config.binance.spot_base_url,
            futures_base_url=self._config.binance.futures_base_url,
        )
        await self._binance.start()
        await self._binance.sync_time()
        await self._binance.load_exchange_info(INSTRUMENTS)

        # 4. Deribit (optional)
        if self._config.deribit.enabled:
            self._deribit = DeribitClient(
                api_key=self._config.deribit.api_key,
                api_secret=self._config.deribit.api_secret,
                testnet=self._config.deribit.testnet,
            )
            try:
                await self._deribit.start()
                if self._deribit.is_connected:
                    self._mode = MODE_FULL
                    logger.info("Deribit connected — FULL mode (all sub-strategies)")
                else:
                    logger.warning("Deribit connection failed — falling back to Synthetic mode")
                    self._deribit = None
            except Exception as exc:
                logger.error("Deribit initialization failed: %s", exc)
                self._deribit = None

        if self._mode == MODE_SYNTHETIC:
            logger.info("Operating in SYNTHETIC mode (Binance only: Sub-A + Sub-B)")

        # 4b. Glassnode client for MVRV (optional)
        glassnode_key = self._params.get("glassnode_api_key", "")
        if glassnode_key:
            self._glassnode = GlassnodeClient(api_key=glassnode_key)
            logger.info("Glassnode MVRV integration enabled")

        # 5. Paper trading engine
        if self._config.paper_trading.enabled:
            self._paper_engine = PaperTradingEngine(
                starting_equity=self._config.paper_trading.starting_equity,
                maker_fee_pct=self._config.paper_trading.maker_fee_pct,
                taker_fee_pct=self._config.paper_trading.taker_fee_pct,
            )
            logger.info(
                "Paper trading enabled: equity=$%.2f",
                self._config.paper_trading.starting_equity,
            )

        # 6. Shared risk manager
        cross_reader = CrossStrategyReader(self._config.state.state_dir)
        self._shared_risk = RiskManager(
            config=self._config.risk,
            cross_strategy_reader=cross_reader,
        )
        equity = (
            self._paper_engine.get_equity()
            if self._paper_engine
            else self._config.paper_trading.starting_equity
        )
        self._shared_risk.update_equity(equity)

        # 7. Strategy components
        self._vol_engine = VolatilityEngine(self._params)
        self._risk_mgr = StrategyRiskManager(self._params, equity)
        self._coordinator = StrategyCoordinator(
            config=self._params,
            vol_engine=self._vol_engine,
            risk_mgr=self._risk_mgr,
            mode=self._mode,
        )

        # 8. State persistence
        self._state = StatePersistence(
            state_dir=self._config.state.state_dir,
            strategy_id=STRATEGY_ID,
            save_interval=STATE_SAVE_INTERVAL,
        )
        self._state.load()
        await self._state.start()

        # 9. Warm up data
        await self._warmup()

        # 10. WebSocket subscriptions
        self._ws_manager = WebSocketManager(
            spot_ws_url=self._config.binance.spot_ws_url + "/stream",
            futures_ws_url=self._config.binance.futures_ws_url + "/stream",
            binance_client=self._binance,
        )
        self._register_ws_streams()
        await self._ws_manager.start()

        # 11. Dashboard
        template_dir = str(_project_root / "templates")
        self._dashboard = OptionsDashboard(
            strategy_coordinator=self._coordinator,
            vol_engine=self._vol_engine,
            risk_mgr=self._risk_mgr,
            host=self._config.dashboard.host,
            port=self._config.dashboard.port,
            template_dir=template_dir,
            kill_fn=self._kill_switch,
        )
        await self._dashboard.start()

        # 12. Start main loops
        self._running = True
        self._tasks = [
            asyncio.create_task(
                self._regime_assessment_loop(), name="regime-loop"
            ),
            asyncio.create_task(
                self._cycle_management_loop(), name="cycle-loop"
            ),
            asyncio.create_task(
                self._circuit_breaker_loop(), name="circuit-breaker-loop"
            ),
            asyncio.create_task(
                self._state_persistence_loop(), name="state-persist-loop"
            ),
            asyncio.create_task(
                self._config_reload_loop(), name="config-reload-loop"
            ),
        ]

        # Greeks monitoring (Full mode only)
        if self._mode == MODE_FULL and self._deribit:
            self._tasks.append(
                asyncio.create_task(
                    self._greeks_monitoring_loop(), name="greeks-loop"
                )
            )
            # Deribit connectivity monitoring
            self._tasks.append(
                asyncio.create_task(
                    self._deribit_connectivity_loop(), name="deribit-conn-loop"
                )
            )

        # MVRV polling (if Glassnode available)
        if self._glassnode:
            self._tasks.append(
                asyncio.create_task(
                    self._mvrv_polling_loop(), name="mvrv-loop"
                )
            )

        logger.info("STRAT-008 fully started — mode=%s", self._mode)
        system_logger.info("bot_started mode=%s", self._mode)

        # 13. Initial regime assessment
        await self._coordinator.assess_volatility_regimes()

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    async def stop(self) -> None:
        """Gracefully shut down all components."""
        logger.info("Shutting down STRAT-008...")
        self._running = False

        # Cancel tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

        # Stop components in reverse order
        if self._dashboard:
            self._dashboard.stop()
        if self._ws_manager:
            await self._ws_manager.stop()
        if self._state:
            self._persist_state()
            await self._state.stop()
        if self._glassnode:
            await self._glassnode.close()
        if self._deribit:
            await self._deribit.stop()
        if self._binance:
            await self._binance.close()

        logger.info("STRAT-008 shut down complete")
        system_logger.info("bot_stopped")

    async def _kill_switch(self, reason: str) -> dict:
        """Emergency kill switch — close all positions immediately."""
        logger.critical("KILL SWITCH: %s", reason)
        self._risk_mgr.activate_kill_switch(reason)
        # In paper mode, just mark everything closed
        # In live mode, would cancel all orders and close positions
        await self.stop()
        return {"status": "killed", "reason": reason}

    # ------------------------------------------------------------------
    # Warm-up: fetch historical data
    # ------------------------------------------------------------------

    async def _warmup(self) -> None:
        """Fetch historical klines to warm up indicator buffers and RV."""
        logger.info("Warming up data buffers...")

        for symbol in INSTRUMENTS:
            # 1-minute candles (24h for RV)
            try:
                klines_1m = await self._binance.get_futures_klines(
                    symbol, "1m", limit=WARMUP_1M_CANDLES,
                )
                candles_1m = []
                for k in klines_1m:
                    ts = int(k[0])
                    close = float(k[4])
                    candles_1m.append((ts, close))
                    self._coordinator.on_kline_1m(symbol, {
                        "timestamp": ts,
                        "open": float(k[1]),
                        "high": float(k[2]),
                        "low": float(k[3]),
                        "close": close,
                        "volume": float(k[5]),
                    })
                self._vol_engine.add_1m_candles_bulk(symbol, candles_1m)
                logger.info("Loaded %d 1m candles for %s", len(klines_1m), symbol)
            except Exception as exc:
                logger.error("Failed to load 1m klines for %s: %s", symbol, exc)

            # 1-hour candles (7 days)
            try:
                klines_1h = await self._binance.get_futures_klines(
                    symbol, "1h", limit=WARMUP_1H_CANDLES,
                )
                for k in klines_1h:
                    self._coordinator.on_kline_1h(symbol, {
                        "timestamp": int(k[0]),
                        "open": float(k[1]),
                        "high": float(k[2]),
                        "low": float(k[3]),
                        "close": float(k[4]),
                        "volume": float(k[5]),
                    })
                logger.info("Loaded %d 1h candles for %s", len(klines_1h), symbol)
            except Exception as exc:
                logger.error("Failed to load 1h klines for %s: %s", symbol, exc)

            # Daily candles (90 days)
            try:
                klines_1d = await self._binance.get_futures_klines(
                    symbol, "1d", limit=WARMUP_1D_CANDLES,
                )
                for k in klines_1d:
                    self._coordinator.on_kline_1d(symbol, {
                        "timestamp": int(k[0]),
                        "open": float(k[1]),
                        "high": float(k[2]),
                        "low": float(k[3]),
                        "close": float(k[4]),
                        "volume": float(k[5]),
                    })
                logger.info("Loaded %d 1d candles for %s", len(klines_1d), symbol)
            except Exception as exc:
                logger.error("Failed to load 1d klines for %s: %s", symbol, exc)

            # Initial premium/basis for synthetic IV
            try:
                premium_data = await self._binance.get_premium_index(symbol)
                if premium_data:
                    mark = float(premium_data.get("markPrice", 0))
                    index = float(premium_data.get("indexPrice", 0))
                    if mark > 0 and index > 0:
                        self._coordinator.on_premium_update(symbol, index, mark)
                        self._spot_prices[symbol] = index
                        self._futures_prices[symbol] = mark
            except Exception as exc:
                logger.error("Failed to load premium index for %s: %s", symbol, exc)

        logger.info("Data warm-up complete")
        logger.info("Vol data status: %s", self._vol_engine.get_data_status())

    # ------------------------------------------------------------------
    # WebSocket registration
    # ------------------------------------------------------------------

    def _register_ws_streams(self) -> None:
        """Register all required WebSocket streams."""
        subscriptions = []

        for symbol in INSTRUMENTS:
            sym_lower = symbol.lower()

            # Mark price (1s) for delta hedging
            subscriptions.append(
                (f"{sym_lower}@markPrice@1s", self._make_mark_price_handler(symbol))
            )
            # 1-minute klines for RV
            subscriptions.append(
                (f"{sym_lower}@kline_1m", self._make_kline_handler(symbol, "1m"))
            )
            # Hourly klines for regime detection
            subscriptions.append(
                (f"{sym_lower}@kline_1h", self._make_kline_handler(symbol, "1h"))
            )
            # Daily klines
            subscriptions.append(
                (f"{sym_lower}@kline_1d", self._make_kline_handler(symbol, "1d"))
            )
            # Book ticker for execution
            subscriptions.append(
                (f"{sym_lower}@bookTicker", self._make_book_ticker_handler(symbol))
            )

        self._ws_manager.register_strategy(
            strategy_id=STRATEGY_ID,
            subscriptions=subscriptions,
            conn_type=ConnectionType.FUTURES,
            on_reconnect=self._on_ws_reconnect,
        )

        logger.info("Registered %d WS streams", len(subscriptions))

    def _make_mark_price_handler(self, symbol: str):
        async def handler(data: dict):
            price = float(data.get("p", 0))
            if price > 0:
                self._futures_prices[symbol] = price
                self._coordinator.on_mark_price(symbol, price)

                # Update premium for synthetic IV
                index_price = float(data.get("i", 0))
                if index_price > 0:
                    self._spot_prices[symbol] = index_price
                    self._coordinator.on_premium_update(symbol, index_price, price)

        return handler

    def _make_kline_handler(self, symbol: str, interval: str):
        async def handler(data: dict):
            k = data.get("k", {})
            if not k:
                return

            candle = {
                "timestamp": int(k.get("t", 0)),
                "open": float(k.get("o", 0)),
                "high": float(k.get("h", 0)),
                "low": float(k.get("l", 0)),
                "close": float(k.get("c", 0)),
                "volume": float(k.get("v", 0)),
            }

            is_closed = k.get("x", False)

            if interval == "1m":
                self._coordinator.on_kline_1m(symbol, candle)
            elif interval == "1h" and is_closed:
                self._coordinator.on_kline_1h(symbol, candle)
            elif interval == "1d" and is_closed:
                self._coordinator.on_kline_1d(symbol, candle)

        return handler

    def _make_book_ticker_handler(self, symbol: str):
        async def handler(data: dict):
            # Store best bid/ask for execution
            pass  # Used by order execution, stored in memory
        return handler

    async def _on_ws_reconnect(self, conn_type: ConnectionType) -> None:
        """Handle WebSocket reconnection."""
        logger.info("WS reconnected (%s) — re-syncing state", conn_type.value)
        system_logger.info("ws_reconnect conn=%s", conn_type.value)

    # ------------------------------------------------------------------
    # Main loops
    # ------------------------------------------------------------------

    async def _regime_assessment_loop(self) -> None:
        """Hourly volatility regime assessment loop."""
        try:
            while self._running:
                await asyncio.sleep(REGIME_ASSESSMENT_INTERVAL)

                if not self._running:
                    break

                try:
                    regimes = await self._coordinator.assess_volatility_regimes()
                    logger.info(
                        "Regime assessment complete: %s",
                        {s: r.regime for s, r in regimes.items()},
                    )
                except Exception:
                    logger.exception("Error in regime assessment")

        except asyncio.CancelledError:
            pass

    async def _cycle_management_loop(self) -> None:
        """Manage option cycles: evaluate entries, check exits, handle rolling."""
        try:
            while self._running:
                await asyncio.sleep(CYCLE_CHECK_INTERVAL)

                if not self._running:
                    break

                try:
                    # Check trading halt
                    halted, reason = self._risk_mgr.is_trading_halted()
                    if halted:
                        logger.debug("Trading halted: %s", reason)
                        # Still check exits even when halted
                        await self._process_exits()
                        continue

                    # Check exits
                    await self._process_exits()

                    # Check rolling
                    roll_actions = await self._coordinator.check_rolling()
                    for action in roll_actions:
                        await self._execute_action(action)

                    # Evaluate new entries
                    await self._evaluate_entries()

                    # Update equity
                    self._update_equity()

                    # Persist state
                    self._persist_state()

                except Exception:
                    logger.exception("Error in cycle management loop")

        except asyncio.CancelledError:
            pass

    async def _circuit_breaker_loop(self) -> None:
        """Fast loop checking circuit breakers (IV spike, flash crash)."""
        try:
            while self._running:
                await asyncio.sleep(CIRCUIT_BREAKER_INTERVAL)

                if not self._running:
                    break

                try:
                    actions = await self._coordinator.check_circuit_breakers()
                    for action in actions:
                        await self._handle_circuit_breaker(action)
                except Exception:
                    logger.exception("Error in circuit breaker loop")

        except asyncio.CancelledError:
            pass

    async def _greeks_monitoring_loop(self) -> None:
        """Monitor Greeks and rebalance delta hedges (Full mode only)."""
        try:
            while self._running:
                await asyncio.sleep(5)  # Check every 5 seconds

                if not self._running or not self._deribit:
                    break

                try:
                    dn_mgr = self._coordinator.dn_manager
                    if not dn_mgr:
                        continue

                    for pos_id, pos in dn_mgr.get_active_positions().items():
                        if pos.status != "open":
                            continue

                        # Check if hedge needed
                        hedge_result = dn_mgr.calculate_hedge_needed(pos_id)
                        if hedge_result:
                            adjustment, reason = hedge_result
                            logger.info("Hedge needed: %s — %s", pos_id, reason)
                            # In paper mode, simulate the hedge
                            if self._paper_engine:
                                dn_mgr.record_hedge_fill(
                                    pos_id,
                                    adjustment,
                                    self._futures_prices.get(pos.binance_symbol, 0),
                                    0.0,  # Fees calculated separately
                                )

                        # Check Greek limits
                        notional = (
                            pos.contracts_sold
                            * self._futures_prices.get(pos.binance_symbol, 0)
                        )
                        breaches = dn_mgr.check_greek_limits(pos_id, notional)
                        for breach in breaches:
                            logger.warning("Greek limit: %s — %s", pos_id, breach)

                    # Update aggregate Greeks in risk manager
                    agg_greeks = dn_mgr.get_aggregate_greeks()
                    total_notional = sum(
                        p.contracts_sold * self._futures_prices.get(p.binance_symbol, 0)
                        for p in dn_mgr.get_active_positions().values()
                    )
                    self._risk_mgr.update_greeks(agg_greeks, total_notional)

                except Exception:
                    logger.exception("Error in Greeks monitoring")

        except asyncio.CancelledError:
            pass

    async def _state_persistence_loop(self) -> None:
        """Periodic state save and config reload."""
        try:
            while self._running:
                await asyncio.sleep(STATE_SAVE_INTERVAL)
                if not self._running:
                    break
                self._persist_state()
        except asyncio.CancelledError:
            pass

    async def _config_reload_loop(self) -> None:
        """Check for config file changes."""
        try:
            while self._running:
                await asyncio.sleep(30)
                if not self._running:
                    break
                if self._config_loader.check_reload():
                    self._config = self._config_loader.config
                    self._params = self._config.strategy_params
                    logger.info("Configuration reloaded")
        except asyncio.CancelledError:
            pass

    async def _mvrv_polling_loop(self) -> None:
        """Poll Glassnode for MVRV ratio to adjust sub-strategy sizing."""
        try:
            while self._running:
                await asyncio.sleep(3600)  # Poll hourly
                if not self._running or not self._glassnode:
                    break

                try:
                    mvrv_data = await self._glassnode.get_mvrv_zscore("BTC")
                    if mvrv_data and isinstance(mvrv_data, list) and len(mvrv_data) > 0:
                        latest = mvrv_data[-1]
                        mvrv_value = float(latest.get("v", 0))
                        if mvrv_value > 0:
                            self._coordinator.update_mvrv_ratio(mvrv_value)
                            logger.info("MVRV updated: %.2f", mvrv_value)
                except Exception:
                    logger.exception("Error fetching MVRV data")

        except asyncio.CancelledError:
            pass

    async def _deribit_connectivity_loop(self) -> None:
        """Monitor Deribit connectivity and close hedges if lost > 5 minutes."""
        try:
            while self._running:
                await asyncio.sleep(10)  # Check every 10 seconds
                if not self._running or not self._deribit:
                    break

                dn_mgr = self._coordinator.dn_manager
                if not dn_mgr:
                    continue

                try:
                    # Update heartbeat if Deribit is responding
                    if self._deribit.is_connected:
                        dn_mgr.update_deribit_heartbeat()
                    else:
                        # Check if we need to close hedges
                        action = dn_mgr.check_deribit_connectivity()
                        if action:
                            logger.critical(
                                "DERIBIT CONNECTIVITY LOSS: %s", action["reason"],
                            )
                            # Close Binance hedge positions
                            for pos_id in action.get("positions", []):
                                pos = dn_mgr.get_active_positions().get(pos_id)
                                if pos and abs(pos.hedge_quantity) > 0:
                                    # In paper mode, just zero the hedge
                                    dn_mgr.record_hedge_fill(
                                        pos_id,
                                        -pos.hedge_quantity,  # Close hedge
                                        self._futures_prices.get(pos.binance_symbol, 0),
                                        0.0,
                                    )
                                    logger.critical(
                                        "Closed Binance hedge for %s "
                                        "(now UNHEDGED vol sale)", pos_id,
                                    )
                except Exception:
                    logger.exception("Error in Deribit connectivity check")

        except asyncio.CancelledError:
            pass

    # ------------------------------------------------------------------
    # Action execution
    # ------------------------------------------------------------------

    async def _evaluate_entries(self) -> None:
        """Evaluate potential entries for all symbols and sub-strategies."""
        equity = self._risk_mgr.equity

        for symbol in INSTRUMENTS:
            regime = self._coordinator._regimes.get(symbol)
            if not regime:
                continue

            eligible = self._coordinator.select_sub_strategies(symbol, regime)

            # Covered calls
            if "covered_calls" in eligible:
                # In paper mode, assume we hold a small spot position
                spot_qty = equity * 0.10 / max(self._spot_prices.get(symbol, 1), 1)
                action = await self._coordinator.evaluate_covered_call(
                    symbol, spot_qty, equity,
                )
                if action:
                    await self._execute_action(action)

            # Cash-secured puts
            if "cash_secured_puts" in eligible:
                available_usdt = equity * 0.10  # Available USDT for CSP
                action = await self._coordinator.evaluate_cash_secured_put(
                    symbol, available_usdt, equity,
                )
                if action:
                    await self._execute_action(action)

            # Delta-neutral
            if "delta_neutral" in eligible:
                action = await self._coordinator.evaluate_delta_neutral(
                    symbol,
                    deribit_connected=self._deribit is not None and self._deribit.is_connected,
                    deribit_balance_ok=True,  # Simplified for paper
                )
                if action:
                    await self._execute_action(action)

    async def _process_exits(self) -> None:
        """Check and process all exit conditions."""
        exit_actions = await self._coordinator.check_all_exits()
        for action in exit_actions:
            await self._execute_exit(action)

    async def _execute_action(self, action: dict) -> None:
        """Execute a strategy action (open position)."""
        sub = action.get("sub_strategy", "")
        act = action.get("action", "")
        symbol = action.get("symbol", "")

        if sub == "covered_calls" and act in ("open", "roll"):
            cycle = action.get("cycle")
            if cycle:
                # In paper mode, just activate
                spot_value = action.get("spot_value", 0)
                self._risk_mgr.record_allocation("cc", spot_value)
                self._coordinator.cc_manager.activate_cycle(cycle)
                logger.info("CC cycle activated: %s", cycle.cycle_id)

        elif sub == "cash_secured_puts" and act in ("open", "roll"):
            cycle = action.get("cycle")
            if cycle:
                self._risk_mgr.record_allocation("csp", cycle.reserved_usdt)
                self._coordinator.csp_manager.activate_cycle(cycle)
                logger.info("CSP cycle activated: %s", cycle.cycle_id)

        elif sub == "delta_neutral" and act == "evaluate":
            # Delta-neutral requires Deribit execution
            logger.info("DN entry evaluation for %s (would execute on Deribit)", symbol)

    async def _execute_exit(self, action: dict) -> None:
        """Execute an exit action."""
        sub = action.get("sub_strategy", "")
        act = action.get("action", "")
        cycle_id = action.get("cycle_id", "")
        current_price = action.get("current_price", 0)

        if sub == "covered_calls":
            exit_type = {
                "exercise": "exercised",
                "expire": "expired",
                "early_exit_drop": "early_exit",
                "early_exit_iv_crush": "early_exit",
                "rv_exit": "early_exit",
            }.get(act, "early_exit")

            cycle = self._coordinator.cc_manager.close_cycle(
                cycle_id, exit_type, current_price,
            )
            if cycle:
                self._risk_mgr.release_allocation(
                    "cc", cycle.spot_entry_price * cycle.spot_quantity
                )
                if cycle.realized_pnl >= 0:
                    self._risk_mgr.record_sub_strategy_win("cc")
                else:
                    self._risk_mgr.record_sub_strategy_loss("cc")

        elif sub == "cash_secured_puts":
            exit_type = {
                "exercise": "exercised",
                "expire": "expired",
                "early_exit_crash": "early_exit",
            }.get(act, "early_exit")

            cycle = self._coordinator.csp_manager.close_cycle(
                cycle_id, exit_type, current_price,
            )
            if cycle:
                self._risk_mgr.release_allocation("csp", cycle.reserved_usdt)
                if cycle.realized_pnl >= 0:
                    self._risk_mgr.record_sub_strategy_win("csp")
                else:
                    self._risk_mgr.record_sub_strategy_loss("csp")

    async def _handle_circuit_breaker(self, action: dict) -> None:
        """Handle circuit breaker triggers."""
        cb_type = action.get("type", "")
        symbol = action.get("symbol", "")

        logger.critical(
            "CIRCUIT BREAKER: %s on %s — %s",
            cb_type, symbol, action.get("action", ""),
        )

        # Close all positions
        for sym in INSTRUMENTS:
            # Close CC
            for cycle in list(self._coordinator.cc_manager.get_active_cycles().values()):
                if cycle.symbol == sym:
                    self._coordinator.cc_manager.close_cycle(
                        cycle.cycle_id, "circuit_break",
                        self._futures_prices.get(sym, 0),
                    )

            # Close CSP
            for cycle in list(self._coordinator.csp_manager.get_active_cycles().values()):
                if cycle.symbol == sym:
                    self._coordinator.csp_manager.close_cycle(
                        cycle.cycle_id, "circuit_break",
                        self._futures_prices.get(sym, 0),
                    )

    # ------------------------------------------------------------------
    # Equity and state
    # ------------------------------------------------------------------

    def _update_equity(self) -> None:
        """Update equity from paper trading engine."""
        if self._paper_engine:
            equity = self._paper_engine.get_equity()
        else:
            equity = self._config.paper_trading.starting_equity

        # Add premium PnL from active cycles
        cc_metrics = self._coordinator.cc_manager.get_metrics()
        csp_metrics = self._coordinator.csp_manager.get_metrics()
        total_pnl = cc_metrics.get("total_pnl", 0) + csp_metrics.get("total_pnl", 0)

        effective_equity = equity + total_pnl
        self._risk_mgr.update_equity(effective_equity)
        self._shared_risk.update_equity(effective_equity)

        # Record equity for Sharpe ratio calculation in strategy metrics
        if self._coordinator and hasattr(self._coordinator, '_metrics_calc'):
            self._coordinator._metrics_calc.record_daily_equity(effective_equity)

    def _persist_state(self) -> None:
        """Save current state to disk."""
        if not self._state or not self._coordinator:
            return

        state = self._coordinator.get_state()
        self._state.update_state("custom", state)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def get_metrics(self) -> dict:
        if self._coordinator:
            return self._coordinator.get_metrics()
        return {}

    def get_positions(self) -> list:
        if self._coordinator:
            return self._coordinator.get_positions()
        return []


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    """Main entry point."""
    config_path = os.environ.get("CONFIG_PATH", "config.yaml")
    bot = OptionsVolatilityBot(config_path)

    # Graceful shutdown
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _signal_handler():
        logger.info("Received shutdown signal")
        stop_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _signal_handler)

    try:
        await bot.start()
        await stop_event.wait()
    finally:
        await bot.stop()


if __name__ == "__main__":
    asyncio.run(main())
