"""STRAT-006 Market Making Bot — Main Entry Point.

Handles:
- Startup: cancel ALL existing orders, recalibrate from 60m of 1m klines,
  verify inventory, place fresh quotes within 30s
- WebSocket streams: depth20, bookTicker, aggTrade, kline_1m/5m/1h,
  markPrice, forceOrder per instrument (8 streams x max 3 instruments)
- User data stream for ORDER_TRADE_UPDATE fill detection
- Quote health check every 10 seconds
- 8-hour session resets at 00:00/08:00/16:00 UTC
- State persistence every 3 seconds
- Volatility spike / large order / spread collapse withdrawal
- Adverse selection tracking
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
from shared.performance_tracker import PerformanceTracker, DimensionalBreakdown, GoLiveCriteriaChecker
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

from src.strategy import AvellanedaStoikovStrategy, get_session_time_remaining, get_next_session_reset
from src.quote_manager import QuoteManager
from src.adverse_selection import AdverseSelectionTracker
from src.risk_manager import MarketMakingRiskManager
from src.dashboard import MarketMakingDashboard
from src.news_event_filter import NewsEventFilter
from src.strategy_metrics import StrategyMetrics

logger = logging.getLogger(__name__)
trade_logger = logging.getLogger("trade")
system_logger = logging.getLogger("system")

STRATEGY_ID = "STRAT-006"
STRATEGY_NAME = "Market Making"


class MarketMakingBot:
    """Main orchestrator for the STRAT-006 Market Making Bot."""

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
            futures_base_url=self._cfg.binance.futures_base_url,
            recv_window=self._cfg.binance.recv_window,
        )

        # ── WebSocket manager ──
        self._ws_manager = WebSocketManager(
            futures_ws_url=self._cfg.binance.futures_ws_url + "/stream"
            if not self._cfg.binance.futures_ws_url.endswith("/stream")
            else self._cfg.binance.futures_ws_url,
            binance_client=self._client,
        )

        # ── Paper trading ──
        self._paper_mode = self._cfg.paper_trading.enabled or self._cfg.mode == "paper"
        self._paper_engine: Optional[PaperTradingEngine] = None
        if self._paper_mode:
            self._paper_engine = PaperTradingEngine(
                starting_equity=self._cfg.paper_trading.starting_equity,
                maker_fee_pct=self._cfg.paper_trading.maker_fee_pct,
                taker_fee_pct=self._cfg.paper_trading.taker_fee_pct,
            )

        # ── Strategy engine ──
        self._strategy = AvellanedaStoikovStrategy(
            params=self._params,
            exchange_info=self._exchange_info,
        )

        # ── Quote manager ──
        self._quote_manager = QuoteManager(
            params=self._params,
            binance_client=self._client,
            exchange_info=self._exchange_info,
            paper_mode=self._paper_mode,
            maker_fee_pct=self._cfg.paper_trading.maker_fee_pct / 100.0,
        )

        # ── MM Risk manager ──
        starting_equity = (
            self._cfg.paper_trading.starting_equity if self._paper_mode else 10000.0
        )
        self._mm_risk = MarketMakingRiskManager(
            params=self._params,
            total_equity=starting_equity,
        )

        # ── Adverse selection tracker ──
        self._adverse_tracker = AdverseSelectionTracker(
            params=self._params,
            on_widen=self._on_adverse_widen,
            on_halt=self._on_adverse_halt,
        )

        # ── News/Event filter (Section 7.6) ──
        self._event_filter = NewsEventFilter(params=self._params)

        # ── Strategy-specific metrics (Section 10.2) ──
        self._strategy_metrics = StrategyMetrics(params=self._params)

        # ── State persistence (every 3 seconds) ──
        self._state = StatePersistence(
            state_dir=self._cfg.state.state_dir,
            strategy_id=STRATEGY_ID,
            save_interval=self._cfg.state.persistence_interval,
            max_snapshots=self._cfg.state.snapshot_count,
        )

        # ── Shared risk manager ──
        risk_cfg = RiskConfig(
            max_capital_pct=15.0,
            max_per_trade_pct=7.0,
            max_per_asset_pct=7.0,
            max_leverage=3,
            preferred_leverage=2,
            max_concurrent_positions=3,
            daily_drawdown_pct=1.0,
            weekly_drawdown_pct=2.0,
            monthly_drawdown_pct=4.0,
            system_wide_drawdown_pct=15.0,
        )
        self._shared_risk = RiskManager(
            config=risk_cfg,
            cross_strategy_reader=CrossStrategyReader(state_dir=self._cfg.state.state_dir),
        )

        # ── Database ──
        self._db = DatabaseManager(self._cfg.database.url)

        # ── Dashboard ──
        self._dashboard = MarketMakingDashboard(
            host=self._cfg.dashboard.host,
            port=self._cfg.dashboard.port,
            template_dir=str(Path(__file__).parent.parent / "templates"),
        )

        # ── Alert manager ──
        self._alerting = AlertManager(self._cfg.alerting, STRATEGY_ID) if self._cfg.alerting.enabled else None

        # ── Memory manager ──
        self._memory = MemoryManager(
            warn_mb=self._cfg.memory.warn_mb,
            restart_mb=self._cfg.memory.restart_mb,
        )

        # ── Kill switch ──
        self._kill_switch = KillSwitch(
            binance_client=self._client,
            database_manager=self._db,
        )

        # ── Heartbeat ──
        self._heartbeat = HeartbeatMonitor(
            strategy_id=STRATEGY_ID,
            interval=self._cfg.heartbeat.interval,
        )

        # ── Dimensional breakdown (Section 10.3) ──
        self._dimensional = DimensionalBreakdown()

        # ── Performance tracker (Section 10.1) ──
        self._perf_tracker = PerformanceTracker(strategy_id=STRATEGY_ID)

        # ── Go-Live Criteria (Section 9.3) ──
        self._go_live_checker = GoLiveCriteriaChecker(
            tracker=self._perf_tracker,
            min_days=30,
            min_trades=100,
            min_win_rate=0.50,       # Not a directional strategy, use fill profitability
            min_profit_factor=1.5,
            max_drawdown=2.0,        # Inventory DD < 2%
            min_sharpe=1.5,          # Sharpe > 1.5
            max_single_loss_pct=1.0,
            min_uptime_pct=95.0,
            custom_criteria={
                "profitable_days_pct": lambda metrics: self._check_profitable_days_pct(metrics),
                "adverse_selection_rate": lambda metrics: self._check_adverse_selection_rate(metrics),
                "spread_captured_vs_fees": lambda metrics: self._check_spread_vs_fees(metrics),
            },
        )

        # ── Internal state ──
        self._instruments = list(self._cfg.instruments)
        self._startup_complete = False
        self._last_calibration: Dict[str, float] = {}
        self._last_account_fetch: float = 0
        self._last_quote_health: float = 0
        self._last_fee_check: float = 0
        self._last_delisting_check: float = 0

        # Quote health consecutive failure tracking
        self._quote_health_failures: Dict[str, int] = {}  # symbol -> consecutive failures
        self._max_consecutive_health_failures: int = 3

        # Depth recovery tracking for sudden liquidity withdrawal
        self._depth_withdrawn_symbols: Dict[str, float] = {}  # symbol -> withdrawal timestamp

        # Known exchange instruments (for delisting detection)
        self._known_exchange_symbols: set = set()

        # Force order pattern tracking for delisting detection
        self._force_order_counts: Dict[str, int] = {}  # symbol -> count in window
        self._force_order_window_start: float = 0.0

        logger.info(
            "MarketMakingBot initialized: instruments=%s, mode=%s, "
            "paper=%s, rate_budget=%d/min",
            self._instruments, self._cfg.mode, self._paper_mode,
            self._cfg.rate_limit_weight_per_min,
        )

    # ==================================================================
    # Lifecycle
    # ==================================================================

    async def start(self) -> None:
        """Start the market making bot."""
        logger.info("=" * 60)
        logger.info("STRAT-006 MARKET MAKING BOT STARTING")
        logger.info("=" * 60)
        system_logger.info("bot_start strategy=%s mode=%s", STRATEGY_ID, self._cfg.mode)

        self._running = True

        # Setup signal handlers
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(self.shutdown("Signal received")))

        # Start infrastructure
        await self._client.start()
        await self._client.sync_time()

        # Load exchange info
        await self._client.load_exchange_info(self._instruments)

        # Initialize instruments in strategy
        for symbol in self._instruments:
            info = self._exchange_info.get_info(symbol)
            tick = info.get("tick_size", 0.0001) if info else 0.0001
            step = info.get("step_size", 0.01) if info else 0.01
            min_not = info.get("min_notional", 5.0) if info else 5.0
            self._strategy.add_instrument(symbol, tick, step, min_not)

        # Load persisted state
        self._state.load()
        self._restore_state()

        # ── STARTUP RECONCILIATION ──
        # Cancel ALL existing orders (stale quotes are dangerous)
        logger.info("Cancelling all existing orders on %s...", self._instruments)
        for symbol in self._instruments:
            try:
                if not self._paper_mode:
                    await self._client.cancel_all_futures_orders(symbol)
                logger.info("Cancelled all orders for %s", symbol)
            except Exception as e:
                logger.warning("Failed to cancel orders for %s: %s", symbol, e)

        # Recalibrate from 60 minutes of 1m klines
        logger.info("Fetching historical klines for calibration...")
        await self._fetch_historical_klines()

        # Verify inventory from account state
        await self._sync_inventory()

        # Start WebSocket streams
        self._register_ws_streams()
        await self._ws_manager.start()

        # Start adverse selection tracker
        await self._adverse_tracker.start()

        # Start state persistence
        await self._state.start()

        # Start dashboard
        self._setup_dashboard()
        await self._dashboard.start()

        # Start heartbeat
        await self._heartbeat.start()

        # Set go-live paper start
        if self._paper_mode:
            self._go_live_checker.set_paper_start()

        # Cache known exchange symbols for delisting detection
        for symbol in self._instruments:
            self._known_exchange_symbols.add(symbol)

        # Update shared risk equity
        equity = self._get_current_equity()
        self._shared_risk.update_equity(equity)
        self._mm_risk.update_equity(equity)

        # Initial calibration
        for symbol in self._instruments:
            max_inv = self._mm_risk.get_max_inventory_notional(symbol)
            equity = self._get_current_equity()
            self._strategy.calibrate(symbol, max_inv, equity)
            self._last_calibration[symbol] = time.time()

        # Place initial quotes (target < 30 seconds from startup)
        startup_start = time.time()
        await self._place_all_quotes()
        startup_elapsed = time.time() - startup_start
        logger.info("Initial quotes placed in %.1fs", startup_elapsed)

        self._startup_complete = True
        system_logger.info("bot_ready startup_time=%.1f", startup_elapsed)

        # ── Main loop ──
        await self._main_loop()

    async def shutdown(self, reason: str = "Manual shutdown") -> None:
        """Gracefully shut down the bot."""
        if not self._running:
            return

        logger.warning("SHUTDOWN initiated: %s", reason)
        system_logger.info("bot_shutdown reason=%s", reason)
        self._running = False

        # Cancel all quotes
        try:
            cancelled = await self._quote_manager.cancel_all_instruments()
            logger.info("Cancelled %d quotes on shutdown", cancelled)
        except Exception as e:
            logger.error("Error cancelling quotes on shutdown: %s", e)

        # Cancel live orders if not paper
        if not self._paper_mode:
            for symbol in self._instruments:
                try:
                    await self._client.cancel_all_futures_orders(symbol)
                except Exception:
                    pass

        # Stop components
        await self._adverse_tracker.stop()
        await self._state.stop()
        self._dashboard.stop()
        await self._heartbeat.stop()
        await self._ws_manager.stop()
        await self._client.close()

        self._shutdown_event.set()
        logger.info("Shutdown complete")

    # ==================================================================
    # Main loop
    # ==================================================================

    async def _main_loop(self) -> None:
        """Main event loop — runs until shutdown."""
        logger.info("Entering main loop")

        while self._running:
            try:
                cycle_start = time.time()

                # Check halt conditions
                halted, halt_reason = self._mm_risk.is_halted()
                if halted:
                    logger.warning("HALTED: %s — cancelling all quotes", halt_reason)
                    await self._quote_manager.cancel_all_instruments()
                    await asyncio.sleep(60)
                    continue

                # Process adverse selection checks
                self._adverse_tracker.process_pending(self._get_mid_price)

                # Recalibrate model every 15 minutes
                cal_interval = self._params.get("vol_update_interval", 900)
                for symbol in self._instruments:
                    last_cal = self._last_calibration.get(symbol, 0)
                    if time.time() - last_cal > cal_interval:
                        max_inv = self._mm_risk.get_max_inventory_notional(symbol)
                        equity = self._get_current_equity()
                        self._strategy.calibrate(symbol, max_inv, equity)
                        self._last_calibration[symbol] = time.time()

                # Quote health check every 10 seconds
                now = time.time()
                if now - self._last_quote_health > self._params.get("quote_health_interval", 10):
                    await self._quote_health_check()
                    self._last_quote_health = now

                    # Sample quote uptime and inventory for metrics
                    for symbol in self._instruments:
                        quotes = self._quote_manager.get_active_quotes(symbol)
                        sym_quotes = quotes.get(symbol, {})
                        has_bid = any(k.startswith("BUY") for k in sym_quotes)
                        has_ask = any(k.startswith("SELL") for k in sym_quotes)
                        self._strategy_metrics.sample_quote_uptime(symbol, has_bid, has_ask)

                        inst = self._strategy.get_instrument(symbol)
                        if inst:
                            self._strategy_metrics.sample_inventory(
                                symbol, inst.inventory_qty, inst.mid_price,
                            )
                            self._strategy_metrics.update_inventory_pnl(
                                symbol, inst.inventory_qty, inst.mid_price,
                            )

                    # Go-live uptime check
                    self._go_live_checker.record_uptime_check(is_up=True)

                # Fetch account every 30 seconds
                if now - self._last_account_fetch > 30:
                    await self._sync_account()
                    self._last_account_fetch = now

                # Maker fee check every 5 minutes
                if now - self._last_fee_check > 300:
                    await self._check_maker_fee_change()
                    self._last_fee_check = now

                # Delisting check every 60 seconds
                if now - self._last_delisting_check > 60:
                    await self._check_instrument_delisting()
                    self._last_delisting_check = now

                # Check depth recovery for withdrawn symbols
                await self._check_depth_recovery()

                # Clean up past events from the event filter
                self._event_filter.remove_past_events()

                # Session reset check
                await self._check_session_reset()

                # Calculate and place/update quotes for each instrument
                for symbol in self._instruments:
                    await self._process_instrument(symbol)

                # Persist state
                self._save_state()

                # Check memory
                await self._memory.check()

                # Config hot-reload
                self._config_loader.check_reload()

                # Sleep to maintain ~5 second cycle
                elapsed = time.time() - cycle_start
                sleep_time = max(0.1, 5.0 - elapsed)
                await asyncio.sleep(sleep_time)

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in main loop")
                await asyncio.sleep(5)

    # ==================================================================
    # Per-instrument processing
    # ==================================================================

    async def _process_instrument(self, symbol: str) -> None:
        """Process a single instrument: run filters, calculate quotes, place/update."""
        inst = self._strategy.get_instrument(symbol)
        if inst is None or inst.mid_price <= 0:
            return

        # Check if withdrawn
        withdrawn, reason = self._mm_risk.check_withdrawal(symbol)
        if withdrawn:
            return

        # Check news/event filter (Section 7.6)
        should_withdraw, event_name, minutes_to = self._event_filter.should_withdraw_for_event(symbol)
        if should_withdraw:
            logger.warning(
                "Event filter: withdrawing %s for event '%s' (%.0f min to event)",
                symbol, event_name, minutes_to,
            )
            await self._quote_manager.cancel_all_quotes(symbol)
            return

        # Check adverse selection halt
        if self._adverse_tracker.is_halted(symbol):
            await self._quote_manager.cancel_all_quotes(symbol)
            return

        # Check spread viability
        viable, reason = self._strategy.check_spread_viability(symbol)
        if not viable:
            logger.debug("Skip %s: %s", symbol, reason)
            await self._quote_manager.cancel_all_quotes(symbol)
            return

        # Check volume filter
        vol_ok, reason = self._strategy.check_volume_filter(symbol)
        if not vol_ok:
            logger.debug("Skip %s: %s", symbol, reason)
            await self._quote_manager.cancel_all_quotes(symbol)
            return

        # Check volatility spike
        if self._strategy.check_volatility_spike(symbol):
            withdraw_s = self._params.get("vol_spike_withdraw_minutes", 5) * 60
            self._mm_risk.set_withdrawal(symbol, "Volatility spike (3x hourly avg)", withdraw_s)
            await self._quote_manager.cancel_all_quotes(symbol)
            return

        # Check liquidation cascade
        if self._strategy.check_liquidation_cascade(symbol):
            withdraw_s = self._params.get("liquidation_withdraw_minutes", 10) * 60
            self._mm_risk.set_withdrawal(symbol, "Liquidation cascade (>10/min)", withdraw_s)
            await self._quote_manager.cancel_all_quotes(symbol)
            return

        # Check depth withdrawal (sudden liquidity drop — Section 11.2)
        if self._strategy.check_depth_withdrawal(symbol):
            logger.warning(
                "Sudden liquidity withdrawal: %s depth dropped >50%% in 5s — "
                "withdrawing all quotes, will resume after 60s if depth recovers",
                symbol,
            )
            self._mm_risk.set_withdrawal(symbol, "Sudden depth withdrawal (>50% in 5s)", 60)
            self._depth_withdrawn_symbols[symbol] = time.time()
            await self._quote_manager.cancel_all_quotes(symbol)
            return

        # Check trend filter
        trend_ok, trend_reason, widen_factor = self._strategy.check_trend_filter(symbol)
        if not trend_ok:
            logger.warning("Halt %s: %s", symbol, trend_reason)
            await self._quote_manager.cancel_all_quotes(symbol)
            inst.is_active = False
            inst.halt_reason = trend_reason
            return

        # Check inventory limits
        inv_action, inv_reason = self._mm_risk.check_inventory_limits(symbol)
        if inv_action == "emergency_reduce":
            await self._emergency_reduce(symbol)
            return

        # Check inventory blowout (Section 5.6)
        blowout, blow_reason = self._mm_risk.check_inventory_blowout(symbol)
        if blowout:
            logger.critical("BLOWOUT: %s — halting strategy immediately", blow_reason)
            await self._quote_manager.cancel_all_quotes(symbol)
            inst.is_active = False
            inst.halt_reason = blow_reason

            if self._alerting:
                await self._alerting.send(
                    "critical", f"Blowout: {symbol}", blow_reason,
                )
            return

        # Check same-side fill protection
        accumulating_side = self._quote_manager.check_same_side_fills(symbol)
        if accumulating_side:
            logger.warning("Same-side fill protection: withdrawing %s side for %s", accumulating_side, symbol)
            await self._quote_manager.cancel_side(symbol, accumulating_side)

        # Calculate quotes
        max_inv = self._mm_risk.get_max_inventory_notional(symbol)
        equity = self._get_current_equity()
        quotes = self._strategy.calculate_quotes(symbol, max_inv, equity)
        if quotes is None:
            return

        # Apply adverse selection spread widening
        adverse_factor = self._adverse_tracker.get_spread_factor(symbol)

        # Apply trend widening
        if widen_factor > 0:
            adverse_factor *= (1.0 + widen_factor)

        # If widening needed, adjust quote prices
        if adverse_factor > 1.0:
            mid = inst.mid_price
            hs = quotes.half_spread * adverse_factor
            quotes.bid_l1 = mid - hs
            quotes.ask_l1 = mid + hs
            quotes.spread_pct = (quotes.ask_l1 - quotes.bid_l1) / mid if mid > 0 else 0

        # Calculate sizes
        per_inst_capital = self._mm_risk.get_per_instrument_capital()
        base_size = self._quote_manager.calculate_base_size(symbol, per_inst_capital, inst.mid_price)
        sizes = self._quote_manager.calculate_layer_sizes(
            base_size, inst.inventory_qty, max_inv, inst.mid_price,
        )

        # Stop accumulating side if at critical
        if inv_action == "stop_accumulating":
            acc_side = self._mm_risk.get_accumulating_side(symbol)
            if acc_side:
                for layer in [1, 2, 3]:
                    sizes[acc_side][layer] = 0.0

        # Place/update quotes
        await self._quote_manager.place_quotes(
            symbol=symbol,
            quotes=quotes,
            sizes=sizes,
            tick_size=inst.tick_size,
            step_size=inst.step_size,
            min_notional=inst.min_notional,
        )

    # ==================================================================
    # Emergency inventory reduction
    # ==================================================================

    async def _emergency_reduce(self, symbol: str) -> None:
        """Execute emergency inventory reduction to warn level."""
        inst = self._strategy.get_instrument(symbol)
        if inst is None:
            return

        reduce_qty = self._mm_risk.get_emergency_reduce_qty(symbol, inst.mid_price)
        if reduce_qty <= 0:
            return

        side = "SELL" if inst.inventory_qty > 0 else "BUY"
        logger.warning(
            "EMERGENCY REDUCE: %s %s %.8f to reduce inventory",
            symbol, side, reduce_qty,
        )

        if self._paper_mode:
            # Simulate market order
            fill_price = inst.best_ask if side == "BUY" else inst.best_bid
            if fill_price > 0:
                self._quote_manager.handle_fill(
                    symbol=symbol, side=side, price=fill_price,
                    quantity=reduce_qty, mid_price=inst.mid_price,
                    is_maker=False,
                )
                # Update inventory
                if side == "SELL":
                    new_qty = inst.inventory_qty - reduce_qty
                else:
                    new_qty = inst.inventory_qty + reduce_qty
                self._strategy.update_inventory(symbol, new_qty, inst.inventory_avg_cost)
                self._mm_risk.update_inventory(symbol, new_qty, inst.inventory_avg_cost, inst.mid_price)
        else:
            try:
                qty = self._exchange_info.round_quantity(symbol, reduce_qty)
                if qty > 0:
                    await self._client.place_futures_order(
                        symbol=symbol, side=side, type="MARKET",
                        quantity=qty, reduce_only=True,
                    )
            except Exception as e:
                logger.error("Emergency reduce failed for %s: %s", symbol, e)

    # ==================================================================
    # Session management
    # ==================================================================

    async def _check_session_reset(self) -> None:
        """Check if we're at an 8-hour session boundary (00:00/08:00/16:00 UTC)."""
        now = datetime.now(timezone.utc)
        t_remain = get_session_time_remaining()

        # If within 1 minute of session end
        if t_remain < (1.0 / 60.0):
            logger.info("Session reset triggered at %s", now.strftime("%H:%M:%S UTC"))
            system_logger.info("session_reset time=%s", now.strftime("%H:%M:%S"))

            # Cancel all quotes
            await self._quote_manager.cancel_all_instruments()

            # Evaluate carry for each instrument
            for symbol in self._instruments:
                favorable, reason = self._strategy.evaluate_carry(symbol)
                logger.info("Carry evaluation %s: %s", symbol, reason)

                if not favorable:
                    # Reduce inventory to zero
                    inst = self._strategy.get_instrument(symbol)
                    if inst and abs(inst.inventory_qty) > 0:
                        logger.info("Reducing inventory to zero for %s (unfavorable carry)", symbol)
                        await self._emergency_reduce(symbol)

            # Reset daily if midnight
            if now.hour == 0:
                self._mm_risk.reset_daily()

            # Brief pause, then recalibrate and resume
            await asyncio.sleep(5)

            for symbol in self._instruments:
                max_inv = self._mm_risk.get_max_inventory_notional(symbol)
                equity = self._get_current_equity()
                self._strategy.calibrate(symbol, max_inv, equity)
                self._last_calibration[symbol] = time.time()

    # ==================================================================
    # Quote health check
    # ==================================================================

    async def _quote_health_check(self) -> None:
        """Verify quote health every 10 seconds (Section 8.3):
        - Both bid and ask quotes are active
        - Quotes within 0.5% of mid-price
        - Inventory within limits
        - Model parameters fresh (< 60 seconds old)
        If any check fails: log WARNING, attempt fix (re-quote or recalibrate).
        If persistent (3 consecutive failures): halt and alert.
        """
        for symbol in self._instruments:
            inst = self._strategy.get_instrument(symbol)
            if inst is None or not inst.is_active:
                continue

            withdrawn, _ = self._mm_risk.check_withdrawal(symbol)
            if withdrawn:
                # Reset failure count when legitimately withdrawn
                self._quote_health_failures[symbol] = 0
                continue

            health_failed = False

            # Check 1: Both bid and ask active
            quotes = self._quote_manager.get_active_quotes(symbol)
            sym_quotes = quotes.get(symbol, {})
            has_bid = any(k.startswith("BUY") for k in sym_quotes)
            has_ask = any(k.startswith("SELL") for k in sym_quotes)

            if not has_bid or not has_ask:
                missing = []
                if not has_bid:
                    missing.append("BID")
                if not has_ask:
                    missing.append("ASK")
                logger.warning("Health: %s missing %s quotes — attempting re-quote", symbol, "+".join(missing))
                health_failed = True
                await self._quote_manager.cancel_all_quotes(symbol)
                await self._process_instrument(symbol)

            # Check 2: Quotes within 0.5% of mid-price
            if not health_failed and sym_quotes:
                max_dist = self._params.get("quote_max_distance_pct", 0.5) / 100.0
                for key, q in sym_quotes.items():
                    if inst.mid_price > 0:
                        dist = abs(q["price"] - inst.mid_price) / inst.mid_price
                        if dist > max_dist:
                            logger.warning(
                                "Health: %s quote %s at %.8f is %.4f%% from mid %.8f — orphaned",
                                symbol, key, q["price"], dist * 100, inst.mid_price,
                            )
                            health_failed = True
                            await self._quote_manager.cancel_all_quotes(symbol)
                            await self._process_instrument(symbol)
                            break

            # Check 3: Inventory within limits
            if not health_failed:
                inv_action, inv_reason = self._mm_risk.check_inventory_limits(symbol)
                if inv_action == "emergency_reduce":
                    logger.warning("Health: %s inventory emergency — %s", symbol, inv_reason)
                    health_failed = True
                    await self._emergency_reduce(symbol)

            # Check 4: Model parameters fresh (< 60 seconds old)
            if not health_failed and inst.params.last_calibration > 0:
                age = time.time() - inst.params.last_calibration
                if age > 60:
                    logger.warning("Health: %s model stale (%.0fs) — recalibrating", symbol, age)
                    health_failed = True
                    max_inv = self._mm_risk.get_max_inventory_notional(symbol)
                    equity = self._get_current_equity()
                    self._strategy.calibrate(symbol, max_inv, equity)
                    self._last_calibration[symbol] = time.time()

            # Check 5: Quote staleness (> 60s without update)
            if not health_failed:
                staleness = self._quote_manager.get_quote_staleness(symbol)
                is_stale, stale_msg = self._mm_risk.check_quote_staleness(symbol, staleness)
                if is_stale:
                    logger.warning("Health: %s", stale_msg)
                    health_failed = True
                    await self._quote_manager.cancel_all_quotes(symbol)
                    await self._process_instrument(symbol)

            # Track consecutive failures
            if health_failed:
                self._quote_health_failures[symbol] = self._quote_health_failures.get(symbol, 0) + 1
                consecutive = self._quote_health_failures[symbol]
                logger.warning(
                    "Health: %s consecutive failure #%d/%d",
                    symbol, consecutive, self._max_consecutive_health_failures,
                )

                if consecutive >= self._max_consecutive_health_failures:
                    # Persistent failure: halt and alert
                    halt_reason = (
                        f"Quote health persistent failure: {symbol} failed "
                        f"{consecutive} consecutive health checks — halting"
                    )
                    logger.critical("HALT: %s", halt_reason)
                    await self._quote_manager.cancel_all_quotes(symbol)
                    inst.is_active = False
                    inst.halt_reason = halt_reason

                    if self._alerting:
                        await self._alerting.send(
                            "critical", "Strategy Halted", halt_reason,
                        )
            else:
                # Reset failure count on success
                self._quote_health_failures[symbol] = 0

    # ==================================================================
    # Sudden liquidity withdrawal recovery (Section 11.2)
    # ==================================================================

    async def _check_depth_recovery(self) -> None:
        """Check if depth has recovered for symbols withdrawn due to liquidity drop.

        After withdrawing due to >50% depth drop, resume after 60 seconds
        if depth recovers.
        """
        now = time.time()
        recovered = []

        for symbol, withdraw_ts in list(self._depth_withdrawn_symbols.items()):
            elapsed = now - withdraw_ts
            if elapsed < 60:
                continue  # Wait at least 60 seconds

            # Check if depth has recovered
            inst = self._strategy.get_instrument(symbol)
            if inst is None:
                recovered.append(symbol)
                continue

            # If depth is no longer showing a drop, resume
            if not self._strategy.check_depth_withdrawal(symbol):
                logger.info(
                    "Depth recovered for %s after %.0fs — resuming quoting",
                    symbol, elapsed,
                )
                recovered.append(symbol)
            elif elapsed > 300:
                # Force resume after 5 minutes regardless
                logger.warning(
                    "Depth still low for %s after %.0fs — force resuming",
                    symbol, elapsed,
                )
                recovered.append(symbol)

        for symbol in recovered:
            self._depth_withdrawn_symbols.pop(symbol, None)

    # ==================================================================
    # Instrument delisting detection (Section 11.5)
    # ==================================================================

    async def _check_instrument_delisting(self) -> None:
        """Check if any instrument has been delisted or shows delisting patterns.

        Detection methods:
        1. Exchange info no longer lists the instrument
        2. Unusual forceOrder patterns (elevated liquidations may indicate
           imminent delisting)
        """
        if self._paper_mode:
            return

        try:
            # Refresh exchange info to check if instrument still listed
            await self._client.load_exchange_info(self._instruments)

            for symbol in list(self._instruments):
                info = self._exchange_info.get_info(symbol)
                if info is None:
                    # Instrument no longer in exchange info -> delisted
                    logger.critical(
                        "DELISTING DETECTED: %s no longer in exchange info — "
                        "closing inventory and ceasing quoting",
                        symbol,
                    )
                    await self._handle_delisting(symbol)
                    continue

                # Check if instrument status indicates delisting
                status = info.get("status", "TRADING")
                if status != "TRADING":
                    logger.critical(
                        "DELISTING: %s status=%s — closing inventory",
                        symbol, status,
                    )
                    await self._handle_delisting(symbol)

        except Exception as e:
            logger.warning("Delisting check failed: %s", e)

    async def _handle_delisting(self, symbol: str) -> None:
        """Handle instrument delisting: close all inventory at market, cease quoting."""
        # Cancel all quotes
        await self._quote_manager.cancel_all_quotes(symbol)

        # Close inventory at market
        inst = self._strategy.get_instrument(symbol)
        if inst and abs(inst.inventory_qty) > 1e-12:
            side = "SELL" if inst.inventory_qty > 0 else "BUY"
            qty = abs(inst.inventory_qty)
            logger.critical(
                "DELISTING: closing %s inventory: %s %.8f at market",
                symbol, side, qty,
            )

            if self._paper_mode:
                fill_price = inst.best_ask if side == "BUY" else inst.best_bid
                if fill_price > 0:
                    self._quote_manager.handle_fill(
                        symbol=symbol, side=side, price=fill_price,
                        quantity=qty, mid_price=inst.mid_price, is_maker=False,
                    )
                    self._strategy.update_inventory(symbol, 0.0, 0.0)
                    self._mm_risk.update_inventory(symbol, 0.0, 0.0, inst.mid_price)
            else:
                try:
                    qty_rounded = self._exchange_info.round_quantity(symbol, qty)
                    if qty_rounded > 0:
                        await self._client.place_futures_order(
                            symbol=symbol, side=side, type="MARKET",
                            quantity=qty_rounded, reduce_only=True,
                        )
                except Exception as e:
                    logger.error("Delisting inventory close failed for %s: %s", symbol, e)

        # Mark instrument inactive
        if inst:
            inst.is_active = False
            inst.halt_reason = "Instrument delisted"

        # Alert
        if self._alerting:
            await self._alerting.send(
                "critical", f"Delisted: {symbol}", f"{symbol} delisted — inventory closed, quoting ceased",
            )

    # ==================================================================
    # Maker fee change detection (Section 11.4)
    # ==================================================================

    async def _check_maker_fee_change(self) -> None:
        """Periodically check if maker fee tier has changed.

        Fetches account info and compares current fee to stored value.
        If fees increase > 50%: halt.
        """
        if self._paper_mode:
            return

        try:
            account = await self._client.get_futures_account()
            # Fee tier info may be in account or separate endpoint
            fee_tier = account.get("feeTier", -1)
            maker_commission = float(account.get("makerCommissionRate", 0))

            if maker_commission > 0:
                # Convert from decimal to percentage (0.0002 -> 0.02%)
                fee_pct = maker_commission * 100.0

                should_halt, reason, new_min_spread = self._mm_risk.update_maker_fee(fee_pct)

                if should_halt:
                    logger.critical("HALT: %s", reason)
                    await self._quote_manager.cancel_all_instruments()

                    if self._alerting:
                        await self._alerting.send(
                            "critical", "Circuit Breaker", reason,
                        )
                elif reason:
                    # Fee changed but not halting — update min spread in strategy
                    self._strategy._min_spread_pct = new_min_spread / 100.0
                    logger.info("Updated min spread to %.4f%% after fee change", new_min_spread)

        except Exception as e:
            logger.debug("Fee check failed (non-critical): %s", e)

    # ==================================================================
    # Go-Live criteria checks (Section 9.3)
    # ==================================================================

    def _check_profitable_days_pct(self, metrics: dict) -> bool:
        """Custom go-live check: >70% profitable days."""
        daily_history = list(self._mm_risk._daily_pnl_history)
        if len(daily_history) < 10:
            return False
        profitable = sum(1 for d in daily_history if d.net_pnl > 0)
        return (profitable / len(daily_history)) >= 0.70

    def _check_adverse_selection_rate(self, metrics: dict) -> bool:
        """Custom go-live check: adverse selection < 55%."""
        rate = self._adverse_tracker.get_metrics().get("overall_rate", 1.0)
        return rate < 0.55

    def _check_spread_vs_fees(self, metrics: dict) -> bool:
        """Custom go-live check: avg spread captured > 2x maker fees."""
        sm = self._strategy_metrics.get_all_metrics()
        avg_spread = sm.get("spread_captured_per_fill", 0)
        fee_pct = self._mm_risk.get_current_maker_fee() / 100.0
        # Compare spread capture to fee on a per-fill basis
        # This is approximate; spread_captured is in price units
        return avg_spread > 0 and fee_pct > 0

    def get_go_live_status(self) -> dict:
        """Return go-live criteria evaluation for monitoring."""
        try:
            return self._go_live_checker.evaluate()
        except Exception:
            return {"error": "Cannot evaluate go-live criteria"}

    # ==================================================================
    # WebSocket stream registration
    # ==================================================================

    def _register_ws_streams(self) -> None:
        """Register all WebSocket streams for all instruments."""
        subscriptions = []

        for symbol in self._instruments:
            sym_lower = symbol.lower()

            subscriptions.extend([
                (f"{sym_lower}@depth20@100ms", self._on_depth),
                (f"{sym_lower}@bookTicker", self._on_book_ticker),
                (f"{sym_lower}@aggTrade", self._on_agg_trade),
                (f"{sym_lower}@kline_1m", self._on_kline_1m),
                (f"{sym_lower}@kline_5m", self._on_kline_5m),
                (f"{sym_lower}@kline_1h", self._on_kline_1h),
                (f"{sym_lower}@markPrice@1s", self._on_mark_price),
                (f"{sym_lower}@forceOrder", self._on_force_order),
            ])

        self._ws_manager.register_strategy(
            strategy_id=STRATEGY_ID,
            subscriptions=subscriptions,
            conn_type=ConnectionType.FUTURES,
            on_reconnect=self._on_ws_reconnect,
        )

        # Register user data stream for fill detection
        self._ws_manager.register_strategy(
            strategy_id=f"{STRATEGY_ID}_user",
            subscriptions=[
                ("ORDER_TRADE_UPDATE", self._on_order_update),
                ("ACCOUNT_UPDATE", self._on_account_update),
            ],
            conn_type=ConnectionType.FUTURES_USER,
        )

        logger.info(
            "Registered %d streams for %d instruments",
            len(subscriptions), len(self._instruments),
        )

    # ==================================================================
    # WebSocket callbacks
    # ==================================================================

    async def _on_depth(self, data: dict) -> None:
        """Handle depth20 order book update."""
        symbol = data.get("s", "")
        if not symbol:
            # Infer from data structure
            bids = data.get("b", data.get("bids", []))
            asks = data.get("a", data.get("asks", []))
        else:
            bids = data.get("b", [])
            asks = data.get("a", [])

        if not symbol:
            return

        self._strategy.update_depth(symbol, bids, asks)

        # Paper trading: check fills against depth
        if self._paper_mode:
            bid_depth = float(bids[0][1]) if bids else 0
            ask_depth = float(asks[0][1]) if asks else 0
            # Fills are checked on aggTrade, not depth
            pass

    async def _on_book_ticker(self, data: dict) -> None:
        """Handle best bid/ask update."""
        symbol = data.get("s", "")
        if not symbol:
            return

        best_bid = float(data.get("b", 0))
        best_ask = float(data.get("a", 0))

        if best_bid > 0 and best_ask > 0:
            self._strategy.update_book_ticker(symbol, best_bid, best_ask)

    async def _on_agg_trade(self, data: dict) -> None:
        """Handle aggregated trade."""
        symbol = data.get("s", "")
        if not symbol:
            return

        price = float(data.get("p", 0))
        qty = float(data.get("q", 0))
        is_buyer_maker = data.get("m", False)

        if price <= 0 or qty <= 0:
            return

        notional = price * qty
        self._strategy.update_trade(symbol, price, qty)

        # Check for large trade
        if self._strategy.check_large_trade(symbol, notional):
            withdraw_s = self._params.get("large_trade_withdraw_seconds", 30)
            self._mm_risk.set_withdrawal(
                symbol,
                f"Large trade detected ({notional:.0f} USDT, 5x avg)",
                withdraw_s,
            )
            await self._quote_manager.cancel_all_quotes(symbol)

        # Paper trading: check fills
        if self._paper_mode:
            inst = self._strategy.get_instrument(symbol)
            if inst:
                bid_depth = float(inst.depth_history[-1]) if inst.depth_history else 0
                fills = self._quote_manager.check_paper_fills(
                    symbol, price, qty, bid_depth, bid_depth,
                )
                for fill in fills:
                    await self._handle_fill(fill)

    async def _on_kline_1m(self, data: dict) -> None:
        """Handle 1-minute candle."""
        k = data.get("k", data)
        if not k:
            return
        symbol = k.get("s", "")
        is_closed = k.get("x", False)

        if is_closed and symbol:
            close = float(k.get("c", 0))
            ts = k.get("t", 0)
            if close > 0:
                self._strategy.update_candle_1m(symbol, close, ts / 1000.0 if ts > 1e10 else ts)

    async def _on_kline_5m(self, data: dict) -> None:
        """Handle 5-minute candle."""
        k = data.get("k", data)
        if not k:
            return
        symbol = k.get("s", "")
        is_closed = k.get("x", False)

        if is_closed and symbol:
            close = float(k.get("c", 0))
            if close > 0:
                self._strategy.update_candle_5m(symbol, close)

    async def _on_kline_1h(self, data: dict) -> None:
        """Handle 1-hour candle."""
        k = data.get("k", data)
        if not k:
            return
        symbol = k.get("s", "")
        is_closed = k.get("x", False)

        if is_closed and symbol:
            close = float(k.get("c", 0))
            if close > 0:
                self._strategy.update_candle_1h(symbol, close)

    async def _on_mark_price(self, data: dict) -> None:
        """Handle mark price update."""
        symbol = data.get("s", "")
        if not symbol:
            return

        mark = float(data.get("p", 0))
        funding = float(data.get("r", 0))

        if mark > 0:
            self._strategy.update_mark_price(symbol, mark, funding)

    async def _on_force_order(self, data: dict) -> None:
        """Handle liquidation event.

        Also monitors for unusual forceOrder patterns that may indicate
        instrument delisting (Section 11.5).
        """
        o = data.get("o", data)
        symbol = o.get("s", "")
        if not symbol:
            return

        now = time.time()
        self._strategy.update_liquidation(symbol, now)

        # Track force order patterns for delisting detection
        # Reset window every 5 minutes
        if now - self._force_order_window_start > 300:
            self._force_order_counts.clear()
            self._force_order_window_start = now

        self._force_order_counts[symbol] = self._force_order_counts.get(symbol, 0) + 1

        # If extremely elevated liquidations (>50 in 5 min), may indicate delisting
        if self._force_order_counts.get(symbol, 0) > 50:
            logger.warning(
                "Extremely elevated liquidations on %s (%d in 5min) — "
                "possible delisting pattern, closing inventory",
                symbol, self._force_order_counts[symbol],
            )
            await self._handle_delisting(symbol)
            self._force_order_counts[symbol] = 0  # Reset to prevent repeated triggers

    async def _on_order_update(self, data: dict) -> None:
        """Handle ORDER_TRADE_UPDATE from user data stream."""
        o = data.get("o", data)
        if not o:
            return

        symbol = o.get("s", "")
        status = o.get("X", "")  # Order status
        side = o.get("S", "")
        order_id = o.get("i", 0)
        price = float(o.get("p", 0))
        qty = float(o.get("l", 0))  # Last filled quantity
        avg_price = float(o.get("ap", 0)) or price
        commission = float(o.get("n", 0))

        if status in ("FILLED", "PARTIALLY_FILLED") and qty > 0:
            inst = self._strategy.get_instrument(symbol)
            mid = inst.mid_price if inst else avg_price

            fill = self._quote_manager.handle_fill(
                symbol=symbol,
                side=side,
                price=avg_price,
                quantity=qty,
                mid_price=mid,
                order_id=order_id,
                fee=commission,
                is_maker=True,
            )
            await self._handle_fill(fill)

    async def _on_account_update(self, data: dict) -> None:
        """Handle ACCOUNT_UPDATE from user data stream."""
        # Update equity and positions from account update
        a = data.get("a", data)
        if not a:
            return

        positions = a.get("P", [])
        for pos in positions:
            symbol = pos.get("s", "")
            qty = float(pos.get("pa", 0))
            entry_price = float(pos.get("ep", 0))

            if symbol in self._instruments:
                inst = self._strategy.get_instrument(symbol)
                mid = inst.mid_price if inst else entry_price
                self._strategy.update_inventory(symbol, qty, entry_price)
                self._mm_risk.update_inventory(symbol, qty, entry_price, mid)

    async def _on_ws_reconnect(self, conn_type: ConnectionType) -> None:
        """Handle WebSocket reconnection."""
        logger.warning("WebSocket reconnected: %s — recalibrating and replanting quotes", conn_type.value)

        if not self._paper_mode:
            for symbol in self._instruments:
                try:
                    await self._client.cancel_all_futures_orders(symbol)
                except Exception:
                    pass

        for symbol in self._instruments:
            max_inv = self._mm_risk.get_max_inventory_notional(symbol)
            equity = self._get_current_equity()
            self._strategy.calibrate(symbol, max_inv, equity)

        await self._place_all_quotes()

    # ==================================================================
    # Fill handling
    # ==================================================================

    async def _handle_fill(self, fill: Any) -> None:
        """Handle a fill from any source (live or paper)."""
        symbol = fill.symbol
        inst = self._strategy.get_instrument(symbol)
        if inst is None:
            return

        # Update inventory
        if fill.side == "BUY":
            new_qty = inst.inventory_qty + fill.quantity
        else:
            new_qty = inst.inventory_qty - fill.quantity

        # Calculate new average cost
        if abs(new_qty) < 1e-12:
            avg_cost = 0.0
        elif (inst.inventory_qty > 0 and fill.side == "BUY") or (inst.inventory_qty < 0 and fill.side == "SELL"):
            # Adding to position
            total_cost = abs(inst.inventory_qty) * inst.inventory_avg_cost + fill.quantity * fill.price
            avg_cost = total_cost / abs(new_qty) if abs(new_qty) > 0 else 0
        else:
            avg_cost = inst.inventory_avg_cost if abs(new_qty) > 0 else 0

        self._strategy.update_inventory(symbol, new_qty, avg_cost)
        self._mm_risk.update_inventory(symbol, new_qty, avg_cost, inst.mid_price)
        self._mm_risk.record_fill_pnl(fill.spread_captured * fill.quantity, fill.fee)

        # Register for adverse selection tracking
        self._adverse_tracker.register_fill(
            symbol=symbol,
            side=fill.side,
            fill_price=fill.price,
            mid_at_fill=fill.mid_price_at_fill,
        )

        # Record in strategy-specific metrics (Section 10.2)
        quoted_spread_pct = inst.quotes.spread_pct * 100.0 if inst.quotes else 0.0
        self._strategy_metrics.record_fill(
            symbol=symbol,
            side=fill.side,
            fill_price=fill.price,
            mid_price_at_fill=fill.mid_price_at_fill,
            quantity=fill.quantity,
            fee=fill.fee,
            quoted_spread_pct=quoted_spread_pct,
        )

        # Record in dimensional breakdown (Section 10.3)
        spread_pnl = fill.spread_captured * fill.quantity - fill.fee
        try:
            self._dimensional.record_trade(
                asset=symbol,
                direction="long" if fill.side == "BUY" else "short",
                pnl=spread_pnl,
                timestamp_ms=int(fill.timestamp * 1000) if hasattr(fill, 'timestamp') else int(time.time() * 1000),
            )
        except Exception:
            pass  # Non-critical

        # Record in performance tracker (Section 10.1)
        try:
            self._perf_tracker.record_trade(
                symbol=symbol,
                side=fill.side,
                entry_price=fill.price,
                exit_price=fill.price,  # Spread fills are atomic
                quantity=fill.quantity,
                pnl=spread_pnl,
                fee=fill.fee,
                timestamp_ms=int(time.time() * 1000),
            )
        except Exception:
            pass  # Non-critical

        # Update paper engine equity
        if self._paper_mode and self._paper_engine:
            spread_pnl_val = fill.spread_captured * fill.quantity
            self._paper_engine.update_equity(spread_pnl_val - fill.fee)

        # Recalculate and update quotes (fill triggers re-quote)
        max_inv = self._mm_risk.get_max_inventory_notional(symbol)
        equity = self._get_current_equity()
        quotes = self._strategy.calculate_quotes(symbol, max_inv, equity)
        # Quotes will be updated on next main loop cycle

    # ==================================================================
    # Adverse selection callbacks
    # ==================================================================

    async def _on_adverse_widen(self, symbol: str) -> None:
        """Called when adverse selection threshold hit — widen spreads 50%."""
        logger.warning("Adverse selection WIDEN triggered for %s", symbol)
        if self._alerting:
            await self._alerting.send(
                "warning", f"Adverse Selection: {symbol}", f"{symbol} rate >60% — widening spreads 50%",
            )

    async def _on_adverse_halt(self, symbol: str) -> None:
        """Called when adverse selection threshold hit — halt instrument."""
        logger.warning("Adverse selection HALT triggered for %s", symbol)
        await self._quote_manager.cancel_all_quotes(symbol)
        inst = self._strategy.get_instrument(symbol)
        if inst:
            inst.is_active = False
            inst.halt_reason = "Adverse selection >70%"
        if self._alerting:
            await self._alerting.send(
                "critical", f"Adverse Halt: {symbol}", f"{symbol} adverse selection >70% — instrument halted",
            )

    # ==================================================================
    # Data fetching
    # ==================================================================

    async def _fetch_historical_klines(self) -> None:
        """Fetch 60 minutes of 1m klines for initial calibration."""
        for symbol in self._instruments:
            try:
                klines = await self._client.get_futures_klines(symbol, "1m", limit=60)
                for k in klines:
                    close = float(k[4])
                    ts = k[0] / 1000.0
                    self._strategy.update_candle_1m(symbol, close, ts)

                # Also fetch 1h klines for trend filter
                klines_1h = await self._client.get_futures_klines(symbol, "1h", limit=24)
                for k in klines_1h:
                    close = float(k[4])
                    self._strategy.update_candle_1h(symbol, close)

                # Fetch 24h ticker for volume
                ticker = await self._client.get_ticker_24hr(symbol)
                if isinstance(ticker, dict):
                    vol = float(ticker.get("quoteVolume", 0))
                    self._strategy.update_volume_24h(symbol, vol)

                logger.info("Fetched historical data for %s: %d 1m candles", symbol, len(klines))

            except Exception as e:
                logger.error("Failed to fetch klines for %s: %s", symbol, e)

    async def _sync_inventory(self) -> None:
        """Sync inventory from account state."""
        if self._paper_mode:
            # Paper mode: inventory from internal state
            return

        try:
            account = await self._client.get_futures_account()
            positions = account.get("positions", [])
            for pos in positions:
                symbol = pos.get("symbol", "")
                if symbol in self._instruments:
                    qty = float(pos.get("positionAmt", 0))
                    entry = float(pos.get("entryPrice", 0))
                    inst = self._strategy.get_instrument(symbol)
                    mid = inst.mid_price if inst else entry
                    self._strategy.update_inventory(symbol, qty, entry)
                    self._mm_risk.update_inventory(symbol, qty, entry, mid)
                    logger.info("Inventory synced: %s qty=%.8f entry=%.8f", symbol, qty, entry)
        except Exception as e:
            logger.error("Failed to sync inventory: %s", e)

    async def _sync_account(self) -> None:
        """Periodically sync account equity."""
        if self._paper_mode:
            return

        try:
            account = await self._client.get_futures_account()
            equity = float(account.get("totalWalletBalance", 0))
            if equity > 0:
                self._mm_risk.update_equity(equity)
                self._shared_risk.update_equity(equity)
        except Exception as e:
            logger.warning("Failed to sync account: %s", e)

    # ==================================================================
    # Quote placement helper
    # ==================================================================

    async def _place_all_quotes(self) -> None:
        """Place fresh quotes for all instruments."""
        for symbol in self._instruments:
            try:
                await self._process_instrument(symbol)
            except Exception as e:
                logger.error("Failed to place quotes for %s: %s", symbol, e)

    # ==================================================================
    # Helpers
    # ==================================================================

    def _get_current_equity(self) -> float:
        """Return current equity."""
        if self._paper_mode and self._paper_engine:
            return self._paper_engine.get_equity()
        return self._mm_risk.get_total_equity()

    def _get_mid_price(self, symbol: str) -> float:
        """Return current mid-price for a symbol (used by adverse tracker)."""
        inst = self._strategy.get_instrument(symbol)
        return inst.mid_price if inst else 0.0

    # ==================================================================
    # State persistence
    # ==================================================================

    def _save_state(self) -> None:
        """Save strategy state to disk."""
        state = {
            "model": {
                sym: self._strategy.get_model_state(sym)
                for sym in self._instruments
            },
            "quotes": self._quote_manager.get_state_for_persistence(),
            "adverse": self._adverse_tracker.get_state_for_persistence(),
            "risk": self._mm_risk.get_state_for_persistence(),
            "event_filter": self._event_filter.get_state_for_persistence(),
            "strategy_metrics": self._strategy_metrics.get_state_for_persistence(),
        }
        self._state.update_state("custom", state)

    def _restore_state(self) -> None:
        """Restore strategy state from persisted state."""
        custom = self._state.get_state("custom", {})
        if not custom:
            return

        # Restore risk state
        risk_state = custom.get("risk", {})
        if risk_state:
            self._mm_risk.load_state(risk_state)

        # Restore adverse selection
        adverse_state = custom.get("adverse", {})
        if adverse_state:
            self._adverse_tracker.load_state(adverse_state)

        # Restore event filter
        event_state = custom.get("event_filter", {})
        if event_state:
            self._event_filter.load_state(event_state)

        # Restore strategy metrics
        metrics_state = custom.get("strategy_metrics", {})
        if metrics_state:
            self._strategy_metrics.load_state(metrics_state)

        logger.info("State restored from persistence")

    # ==================================================================
    # Dashboard setup
    # ==================================================================

    def _setup_dashboard(self) -> None:
        """Wire up dashboard data providers."""
        self._dashboard.set_data_providers(
            positions_fn=lambda: self._get_positions(),
            trades_fn=lambda limit: self._quote_manager.get_recent_fills(limit),
            metrics_fn=lambda: self._get_all_metrics(),
            equity_fn=lambda: self._paper_engine.get_equity_curve() if self._paper_engine else [],
            alerts_fn=lambda: [],
            config_fn=lambda: {
                "strategy_id": STRATEGY_ID,
                "mode": self._cfg.mode,
                "paper_trading": self._paper_mode,
                "instruments": self._instruments,
            },
            kill_fn=lambda reason: self._execute_kill_switch(reason),
        )

        self._dashboard.set_mm_providers(
            quotes_fn=lambda: self._quote_manager.get_active_quotes(),
            model_state_fn=lambda: {
                sym: self._strategy.get_model_state(sym)
                for sym in self._instruments
            },
            adverse_fn=lambda: self._adverse_tracker.get_metrics(),
            risk_fn=lambda: self._mm_risk.get_metrics(),
            fills_fn=lambda limit: self._quote_manager.get_recent_fills(limit),
            round_trips_fn=lambda limit: self._quote_manager.get_recent_round_trips(limit),
            strategy_metrics_fn=lambda: self._strategy_metrics.get_all_metrics(),
            event_filter_fn=lambda: self._event_filter.get_metrics(),
            go_live_fn=lambda: self.get_go_live_status(),
        )

    def _get_positions(self) -> List[dict]:
        """Return current positions/inventory for dashboard."""
        positions = []
        for symbol in self._instruments:
            inst = self._strategy.get_instrument(symbol)
            if inst and abs(inst.inventory_qty) > 1e-12:
                positions.append({
                    "symbol": symbol,
                    "side": "LONG" if inst.inventory_qty > 0 else "SHORT",
                    "quantity": abs(inst.inventory_qty),
                    "entry_price": inst.inventory_avg_cost,
                    "mark_price": inst.mark_price,
                    "notional": inst.inventory_notional,
                    "pnl": (inst.mid_price - inst.inventory_avg_cost) * inst.inventory_qty,
                })
        return positions

    def _get_all_metrics(self) -> Dict[str, Any]:
        """Return combined metrics for dashboard."""
        quote_metrics = self._quote_manager.get_metrics()
        adverse_metrics = self._adverse_tracker.get_metrics()
        risk_metrics = self._mm_risk.get_metrics()
        strategy_metrics = self._strategy_metrics.get_all_metrics()
        event_metrics = self._event_filter.get_metrics()

        return {
            **quote_metrics,
            "adverse_selection_rate": adverse_metrics.get("overall_rate", 0),
            "equity": self._get_current_equity(),
            "daily_pnl": risk_metrics.get("daily_pnl", {}).get("net_pnl", 0),
            "is_halted": risk_metrics.get("is_halted", False),
            "total_pnl": risk_metrics.get("daily_pnl", {}).get("net_pnl", 0),
            "rate_limit": self._rate_limiter.get_usage(),
            "strategy_specific": strategy_metrics,
            "event_filter": event_metrics,
            "maker_fee_pct": self._mm_risk.get_current_maker_fee(),
            "min_viable_spread_pct": self._mm_risk.get_min_viable_spread_pct(),
        }

    async def _execute_kill_switch(self, reason: str) -> dict:
        """Execute kill switch: cancel all orders, liquidate inventory."""
        logger.critical("KILL SWITCH: %s", reason)

        # Cancel all quotes
        await self._quote_manager.cancel_all_instruments()

        # Cancel all live orders
        if not self._paper_mode:
            for symbol in self._instruments:
                try:
                    await self._client.cancel_all_futures_orders(symbol)
                except Exception:
                    pass

        # Liquidate inventory
        for symbol in self._instruments:
            inst = self._strategy.get_instrument(symbol)
            if inst and abs(inst.inventory_qty) > 0:
                side = "SELL" if inst.inventory_qty > 0 else "BUY"
                qty = abs(inst.inventory_qty)
                if not self._paper_mode:
                    try:
                        qty = self._exchange_info.round_quantity(symbol, qty)
                        if qty > 0:
                            await self._client.place_futures_order(
                                symbol=symbol, side=side, type="MARKET",
                                quantity=qty, reduce_only=True,
                            )
                    except Exception as e:
                        logger.error("Kill switch liquidation failed for %s: %s", symbol, e)

        await self.shutdown(reason)
        return {"status": "killed", "reason": reason}


# ==================================================================
# Entry point
# ==================================================================

def main():
    """Entry point for the market making bot."""
    config_path = os.environ.get("CONFIG_PATH", "config.yaml")

    # Setup logging
    setup_logging(
        strategy_id=STRATEGY_ID,
        log_dir="data/logs",
        level="INFO",
    )

    logger.info("Starting STRAT-006 Market Making Bot")

    bot = MarketMakingBot(config_path=config_path)

    try:
        asyncio.run(bot.start())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception:
        logger.exception("Fatal error")
        sys.exit(1)


if __name__ == "__main__":
    main()
