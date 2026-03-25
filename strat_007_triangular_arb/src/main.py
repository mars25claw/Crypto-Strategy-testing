"""Entry point for STRAT-007: Cross-Exchange & Triangular Arbitrage Bot.

Orchestrates:
- Startup: check for unhedged positions (close immediately), verify wallet
  balances, calibrate latency, register 40-50 WS streams
- Mode A: spot+futures bookTicker + depth comparison loop
- Mode B: 10 triangle paths cross-rate scanning
- Wallet rebalancing every 4 hours
- Weekly threshold recalibration
- EOD reconciliation at 00:00 UTC
- Stuck position monitor (max 60s unhedged)
- State persistence every 5 seconds
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
from shared.paper_trading import PaperTradingEngine
from shared.utils import TimeSync, ExchangeInfo
from shared.log_manager import setup_logging
from shared.heartbeat import HeartbeatMonitor
from shared.kill_switch import KillSwitch

from strat_007_triangular_arb.src.strategy import (
    ArbMode,
    ArbOpportunity,
    ExecutionResult,
    TriangularArbStrategy,
)
from strat_007_triangular_arb.src.opportunity_scanner import OpportunityScanner
from strat_007_triangular_arb.src.execution import ExecutionEngine
from strat_007_triangular_arb.src.wallet_manager import WalletManager
from strat_007_triangular_arb.src.risk_manager import ArbRiskManager
from strat_007_triangular_arb.src.dashboard import ArbDashboard
from strat_007_triangular_arb.src.strategy_metrics import StrategyMetrics, TradeRecord

logger = logging.getLogger(__name__)
system_logger = logging.getLogger("system")
trade_logger = logging.getLogger("trade")

STRATEGY_ID = "STRAT-007"
STRATEGY_NAME = "Cross-Exchange Triangular Arbitrage"


class TriangularArbBot:
    """Main orchestrator for the Triangular Arbitrage bot."""

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
        self._strategy: Optional[TriangularArbStrategy] = None
        self._scanner: Optional[OpportunityScanner] = None
        self._execution: Optional[ExecutionEngine] = None
        self._wallet_manager: Optional[WalletManager] = None
        self._risk_manager: Optional[ArbRiskManager] = None
        self._shared_risk: Optional[SharedRiskManager] = None

        # Infrastructure
        self._state: Optional[StatePersistence] = None
        self._paper_engine: Optional[PaperTradingEngine] = None
        self._dashboard: Optional[ArbDashboard] = None
        self._heartbeat: Optional[HeartbeatMonitor] = None
        self._kill_switch: Optional[KillSwitch] = None
        self._strategy_metrics: Optional[StrategyMetrics] = None

        # Background tasks
        self._stuck_monitor_task: Optional[asyncio.Task] = None
        self._wallet_rebalance_task: Optional[asyncio.Task] = None
        self._latency_monitor_task: Optional[asyncio.Task] = None
        self._eod_reconciliation_task: Optional[asyncio.Task] = None
        self._weekly_recalibration_task: Optional[asyncio.Task] = None
        self._balance_refresh_task: Optional[asyncio.Task] = None
        self._volatility_monitor_task: Optional[asyncio.Task] = None
        self._maintenance_monitor_task: Optional[asyncio.Task] = None

        logger.info(
            "TriangularArbBot created: mode=%s instruments=%s",
            "paper" if self._paper_mode else "live",
            self._cfg.instruments,
        )

    # ══════════════════════════════════════════════════════════════════════
    #  Lifecycle
    # ══════════════════════════════════════════════════════════════════════

    async def start(self) -> None:
        """Initialize all components and start the bot."""
        system_logger.info("TriangularArbBot starting — mode=%s",
                           "paper" if self._paper_mode else "live")
        self._running = True

        # 1. Initialize infrastructure
        await self._init_infrastructure()

        # 2. Initialize strategy components
        self._init_strategy_components()

        # 3. Startup reconciliation
        await self._startup_reconciliation()

        # 4. Calibrate latency
        await self._calibrate_latency()

        # 5. Verify wallet balances and recalibrate thresholds on startup
        await self._wallet_manager.refresh_balances()
        bnb_ok = self._wallet_manager.has_bnb_for_discount()
        wallet_state = self._wallet_manager.get_state()
        total_bnb = wallet_state.spot_bnb + wallet_state.futures_bnb

        # Startup fee recalibration (query actual fee tier)
        spot_fee = self._params.get("spot_taker_fee_pct", 0.10)
        futures_fee = self._params.get("futures_taker_fee_pct", 0.04)
        if not self._paper_mode:
            try:
                account = await self._binance_client.get_futures_account()
                fee_tier = account.get("feeTier", 0)
                tier_fees = {
                    0: (0.10, 0.04), 1: (0.09, 0.035),
                    2: (0.08, 0.03), 3: (0.07, 0.025),
                }
                if fee_tier in tier_fees:
                    spot_fee, futures_fee = tier_fees[fee_tier]
                    logger.info("Startup fee tier %d: spot=%.2f%% futures=%.2f%%",
                                fee_tier, spot_fee, futures_fee)
            except Exception as exc:
                logger.warning("Could not fetch fee tier on startup: %s", exc)

        self._strategy.recalibrate_thresholds(spot_fee, futures_fee, bnb_balance=total_bnb)

        # 6. Register WebSocket streams
        self._register_ws_streams()

        # 7. Start WebSocket manager
        await self._ws_manager.start()

        # 8. Start background tasks
        self._start_background_tasks()

        # 9. Start scanning
        self._scanner.start_scanning()

        # 10. Start dashboard
        if self._dashboard:
            self._setup_dashboard_providers()
            await self._dashboard.start()

        # 11. Start heartbeat
        if self._heartbeat:
            await self._heartbeat.start()

        system_logger.info(
            "TriangularArbBot RUNNING — mode=%s equity=%.2f streams=%d",
            "paper" if self._paper_mode else "live",
            self._wallet_manager.get_available_balance(),
            len(self._ws_manager.get_registered_streams()),
        )

    async def stop(self) -> None:
        """Gracefully shut down all components."""
        system_logger.info("TriangularArbBot stopping")
        self._running = False

        # Stop scanning first
        if self._scanner:
            self._scanner.stop_scanning()

        # Cancel background tasks
        tasks = [
            self._stuck_monitor_task,
            self._wallet_rebalance_task,
            self._latency_monitor_task,
            self._eod_reconciliation_task,
            self._weekly_recalibration_task,
            self._balance_refresh_task,
            self._volatility_monitor_task,
            self._maintenance_monitor_task,
        ]
        for task in tasks:
            if task and not task.done():
                task.cancel()
        await asyncio.gather(*[t for t in tasks if t and not t.done()], return_exceptions=True)

        # Stop components
        if self._heartbeat:
            self._heartbeat.stop()
        if self._dashboard:
            self._dashboard.stop()
        if self._ws_manager:
            await self._ws_manager.stop()
        if self._binance_client:
            await self._binance_client.close()
        if self._state:
            await self._state.stop()

        system_logger.info("TriangularArbBot STOPPED")

    # ══════════════════════════════════════════════════════════════════════
    #  Initialization
    # ══════════════════════════════════════════════════════════════════════

    async def _init_infrastructure(self) -> None:
        """Initialize core infrastructure components."""
        # Rate limiter
        self._rate_limiter = RateLimiter(
            budget=self._cfg.rate_limit_weight_per_min,
            burst=self._cfg.rate_limit_burst_weight,
        )

        # Binance REST client
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
        if not self._paper_mode:
            await self._binance_client.sync_time()

        # Exchange info
        await self._binance_client.load_exchange_info(self._cfg.instruments)

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
        self._state.load()
        await self._state.start()

        # Paper trading engine
        if self._paper_mode:
            self._paper_engine = PaperTradingEngine(
                starting_equity=self._cfg.paper_trading.starting_equity,
                maker_fee_pct=self._cfg.paper_trading.maker_fee_pct,
                taker_fee_pct=self._cfg.paper_trading.taker_fee_pct,
            )

        # Heartbeat
        self._heartbeat = HeartbeatMonitor(
            strategy_id=STRATEGY_ID,
            interval=self._cfg.heartbeat.interval,
            timeout=self._cfg.heartbeat.timeout,
            max_restarts_per_hour=self._cfg.heartbeat.max_restarts_per_hour,
        )

        # Kill switch
        self._kill_switch = KillSwitch(
            binance_client=self._binance_client,
            state_persistence=self._state,
        )

        logger.info("Infrastructure initialized")

    def _init_strategy_components(self) -> None:
        """Initialize strategy-specific components."""
        equity = (
            self._paper_engine.get_equity()
            if self._paper_mode and self._paper_engine
            else self._cfg.paper_trading.starting_equity
        )

        # Strategy engine
        self._strategy = TriangularArbStrategy(self._params)

        # Risk manager
        self._risk_manager = ArbRiskManager(self._params, equity=equity)

        # Shared risk manager
        self._shared_risk = SharedRiskManager(
            config=self._cfg.risk,
        )
        self._shared_risk.update_equity(equity)

        # Wallet manager
        self._wallet_manager = WalletManager(
            binance_client=self._binance_client,
            params=self._params,
            paper_mode=self._paper_mode,
            starting_equity=equity,
        )

        # Opportunity scanner
        self._scanner = OpportunityScanner(
            strategy=self._strategy,
            params=self._params,
            on_opportunity=self._on_opportunity_detected,
        )

        # Execution engine
        self._execution = ExecutionEngine(
            binance_client=self._binance_client,
            params=self._params,
            paper_mode=self._paper_mode,
            paper_engine=self._paper_engine,
            scanner=self._scanner,
            exchange_info=self._exchange_info,
        )
        self._execution.set_callbacks(
            on_execution_complete=self._on_execution_complete,
        )

        # Strategy metrics (Section 10.2, 10.3, go-live criteria)
        self._strategy_metrics = StrategyMetrics()

        # Dashboard
        self._dashboard = ArbDashboard(
            host=self._cfg.dashboard.host,
            port=self._cfg.dashboard.port,
        )

        logger.info("Strategy components initialized")

    # ══════════════════════════════════════════════════════════════════════
    #  Startup reconciliation
    # ══════════════════════════════════════════════════════════════════════

    async def _startup_reconciliation(self) -> None:
        """Check for unhedged positions from previous session and close them."""
        logger.info("Running startup reconciliation...")

        if self._paper_mode:
            logger.info("Paper mode — no positions to reconcile")
            return

        try:
            # Check futures positions
            account = await self._binance_client.get_futures_account()
            for pos in account.get("positions", []):
                amt = float(pos.get("positionAmt", 0))
                if amt != 0:
                    symbol = pos["symbol"]
                    logger.critical(
                        "STARTUP: Found unhedged futures position %s: qty=%.6f — CLOSING",
                        symbol, amt,
                    )
                    close_side = "SELL" if amt > 0 else "BUY"
                    try:
                        await self._binance_client.place_futures_order(
                            symbol=symbol,
                            side=close_side,
                            type="MARKET",
                            quantity=abs(amt),
                            reduce_only=True,
                        )
                        logger.info("Closed startup position %s", symbol)
                    except Exception as exc:
                        logger.critical("FAILED to close startup position %s: %s", symbol, exc)

        except Exception as exc:
            logger.error("Startup reconciliation failed: %s", exc)

    async def _calibrate_latency(self) -> None:
        """Calibrate API latency on startup."""
        logger.info("Calibrating API latency...")
        latencies = []

        for _ in range(3):
            start = time.monotonic()
            try:
                if not self._paper_mode:
                    await self._binance_client.sync_time()
                else:
                    await asyncio.sleep(0.05)  # Simulate in paper mode
            except Exception:
                pass
            elapsed_ms = (time.monotonic() - start) * 1000.0
            latencies.append(elapsed_ms)

        avg_latency = sum(latencies) / len(latencies) if latencies else 100.0
        self._risk_manager.update_latency(avg_latency)

        logger.info(
            "Latency calibrated: avg=%.1fms min=%.1fms max=%.1fms",
            avg_latency,
            min(latencies) if latencies else 0,
            max(latencies) if latencies else 0,
        )

    # ══════════════════════════════════════════════════════════════════════
    #  WebSocket stream registration
    # ══════════════════════════════════════════════════════════════════════

    def _register_ws_streams(self) -> None:
        """Register all required WS streams for Mode A and Mode B."""

        # Mode A: Spot streams (bookTicker + depth per asset)
        spot_subs = []
        for stream_name in self._scanner.get_required_spot_streams():
            if "bookTicker" in stream_name:
                spot_subs.append((stream_name, self._scanner.handle_spot_book_ticker))
            elif "depth" in stream_name:
                spot_subs.append((stream_name, self._scanner.handle_spot_depth))

        if spot_subs:
            self._ws_manager.register_strategy(
                strategy_id=STRATEGY_ID,
                subscriptions=spot_subs,
                conn_type=ConnectionType.SPOT,
            )
            logger.info("Registered %d spot streams", len(spot_subs))

        # Mode A: Futures streams (bookTicker + depth per asset)
        futures_subs = []
        for stream_name in self._scanner.get_required_futures_streams():
            if "bookTicker" in stream_name:
                futures_subs.append((stream_name, self._scanner.handle_futures_book_ticker))
            elif "depth" in stream_name:
                futures_subs.append((stream_name, self._scanner.handle_futures_depth))

        if futures_subs:
            self._ws_manager.register_strategy(
                strategy_id=STRATEGY_ID,
                subscriptions=futures_subs,
                conn_type=ConnectionType.FUTURES,
            )
            logger.info("Registered %d futures streams", len(futures_subs))

        total = len(spot_subs) + len(futures_subs)
        logger.info("Total WS streams registered: %d", total)

    # ══════════════════════════════════════════════════════════════════════
    #  Opportunity handling
    # ══════════════════════════════════════════════════════════════════════

    async def _on_opportunity_detected(self, opp: ArbOpportunity) -> None:
        """Called when the scanner detects a viable opportunity."""
        # Record detection for metrics
        if self._strategy_metrics:
            self._strategy_metrics.record_detection()

        # Risk check
        can_trade, reason = self._risk_manager.check_can_trade()
        if not can_trade:
            self._scanner.record_skipped(opp, reason)
            if self._strategy_metrics:
                self._strategy_metrics.record_skip(opp.mode.value, opp.symbol, reason)
            return

        # Size check
        if opp.mode == ArbMode.MODE_A:
            size_ok, size_reason = self._risk_manager.check_mode_a_size(opp.trade_size_usdt)
        else:
            size_ok, size_reason = self._risk_manager.check_mode_b_size(opp.trade_size_usdt)

        if not size_ok:
            self._scanner.record_skipped(opp, size_reason)
            if self._strategy_metrics:
                self._strategy_metrics.record_skip(opp.mode.value, opp.symbol, size_reason)
            return

        # Execution capacity check
        if not self._execution.can_execute():
            self._scanner.record_skipped(opp, "max_concurrent_reached")
            if self._strategy_metrics:
                self._strategy_metrics.record_skip(opp.mode.value, opp.symbol, "max_concurrent_reached")
            return

        # Staleness check (double-check)
        if opp.is_stale(self._params.get("opportunity_max_age_s", 1.0)):
            self._scanner.record_skipped(opp, "stale_at_execution")
            if self._strategy_metrics:
                self._strategy_metrics.record_skip(opp.mode.value, opp.symbol, "stale_at_execution")
            return

        # Cross-strategy coordination (Section 11.4 / 12.3)
        arb_symbols = (
            [opp.symbol] if opp.mode == ArbMode.MODE_A
            else list(opp.triangle_path)
        )
        cross_ok, cross_reason = await self._risk_manager.check_cross_strategy(arb_symbols)
        if not cross_ok:
            self._scanner.record_skipped(opp, cross_reason)
            if self._strategy_metrics:
                self._strategy_metrics.record_skip(opp.mode.value, opp.symbol, cross_reason)
            return

        # Lock symbols for atomic arb execution
        locked = await self._risk_manager.lock_arb_position(arb_symbols)
        if not locked:
            self._scanner.record_skipped(opp, "arb_position_lock_conflict")
            if self._strategy_metrics:
                self._strategy_metrics.record_skip(opp.mode.value, opp.symbol, "arb_position_lock_conflict")
            return

        # Execute
        self._scanner.record_taken(opp)
        self._risk_manager.record_arb_start()

        try:
            result = await self._execution.execute(opp)
            await self._process_execution_result(result)
        except Exception as exc:
            logger.exception("Execution error for %s", opp.symbol)
            self._risk_manager.record_arb_end(0.0, False, 0, 0)
        finally:
            # Release lock after arb completes (atomic execution done)
            await self._risk_manager.unlock_arb_position(arb_symbols)

    async def _process_execution_result(self, result: ExecutionResult) -> None:
        """Process the result of an arb execution."""
        self._risk_manager.record_arb_end(
            profit_usdt=result.actual_profit_usdt,
            success=result.success,
            legs_filled=result.legs_filled,
            legs_total=result.legs_total,
        )

        # Record to strategy metrics (Section 10.2)
        if self._strategy_metrics:
            opp = result.opportunity
            # Compute per-leg slippages from leg_results
            leg_slippages = []
            for lr in result.leg_results:
                leg_slippages.append(abs(lr.get("slippage_bps", 0.0)))

            detection_to_order_ms = 0.0
            if opp.detected_at > 0:
                detection_to_order_ms = (time.time() - opp.detected_at) * 1000.0

            record = TradeRecord(
                timestamp=time.time(),
                mode=opp.mode.value,
                symbol=opp.symbol,
                success=result.success,
                profit_usdt=result.actual_profit_usdt,
                profit_pct=result.actual_profit_pct,
                fees_usdt=result.total_fees_usdt,
                volume_usdt=opp.trade_size_usdt,
                legs_filled=result.legs_filled,
                legs_total=result.legs_total,
                execution_time_ms=result.execution_time_ms,
                detection_to_order_ms=detection_to_order_ms,
                order_to_fill_ms=result.execution_time_ms,  # Approximation
                slippage_usdt=result.slippage_usdt,
                leg_slippages=leg_slippages,
                bnb_discount_active=(
                    self._strategy._has_bnb_discount if self._strategy else False
                ),
            )
            self._strategy_metrics.record_trade(record)

        # Update shared risk manager
        if self._shared_risk:
            is_win = result.success and result.actual_profit_usdt > 0
            self._shared_risk.record_trade_result(
                STRATEGY_ID, result.actual_profit_usdt, is_win,
            )

        # Update volatility filter
        if self._strategy:
            self._strategy.set_high_volatility(self._risk_manager.is_high_volatility)

        # Persist state
        if self._state:
            self._state.update_state("last_execution", result.to_dict())
            self._state.update_state("performance_counters", {
                "total_executions": self._execution.get_stats()["total_executions"],
                "win_rate": self._execution.get_win_rate(),
                "avg_profit_pct": self._execution.get_avg_profit_pct(),
                "total_profit": self._execution.get_stats()["total_profit_usdt"],
            })

    def _on_execution_complete(self, result: ExecutionResult) -> None:
        """Sync callback after execution completes (for logging)."""
        if result.success:
            logger.info(
                "Arb %s: %s profit=%.4f USDT (%.4f%%) in %.0fms",
                result.opportunity.mode.value,
                result.opportunity.symbol,
                result.actual_profit_usdt,
                result.actual_profit_pct,
                result.execution_time_ms,
            )
        else:
            logger.warning(
                "Arb FAILED %s: %s error=%s legs=%d/%d",
                result.opportunity.mode.value,
                result.opportunity.symbol,
                result.error,
                result.legs_filled,
                result.legs_total,
            )

    # ══════════════════════════════════════════════════════════════════════
    #  Background tasks
    # ══════════════════════════════════════════════════════════════════════

    def _start_background_tasks(self) -> None:
        """Start all background monitoring tasks."""
        self._stuck_monitor_task = asyncio.create_task(
            self._stuck_position_loop(), name="stuck-monitor"
        )
        self._wallet_rebalance_task = asyncio.create_task(
            self._wallet_rebalance_loop(), name="wallet-rebalance"
        )
        self._latency_monitor_task = asyncio.create_task(
            self._latency_monitor_loop(), name="latency-monitor"
        )
        self._eod_reconciliation_task = asyncio.create_task(
            self._eod_reconciliation_loop(), name="eod-reconciliation"
        )
        self._weekly_recalibration_task = asyncio.create_task(
            self._weekly_recalibration_loop(), name="weekly-recal"
        )
        self._balance_refresh_task = asyncio.create_task(
            self._balance_refresh_loop(), name="balance-refresh"
        )
        self._volatility_monitor_task = asyncio.create_task(
            self._volatility_monitor_loop(), name="volatility-monitor"
        )
        self._maintenance_monitor_task = asyncio.create_task(
            self._maintenance_monitor_loop(), name="maintenance-monitor"
        )
        logger.info("Background tasks started")

    async def _stuck_position_loop(self) -> None:
        """Monitor for stuck unhedged positions every second."""
        try:
            while self._running:
                await asyncio.sleep(1.0)
                if self._execution:
                    await self._execution.check_stuck_positions()
                    # Update risk manager with current exposure
                    unhedged = self._execution.get_unhedged_positions()
                    exposure = sum(
                        abs(p.get("qty", 0)) * 50000  # Approximate USDT value
                        for p in unhedged.values()
                    )
                    self._risk_manager.record_unhedged_change(exposure)
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Stuck position monitor error")

    async def _wallet_rebalance_loop(self) -> None:
        """Rebalance wallets every 4 hours if needed."""
        try:
            while self._running:
                interval = self._params.get("wallet_rebalance_interval_h", 4) * 3600
                await asyncio.sleep(interval)
                if not self._running:
                    break

                await self._wallet_manager.refresh_balances()
                rebalanced = await self._wallet_manager.rebalance()
                if rebalanced:
                    system_logger.info("WALLET_REBALANCE\tcompleted")

                # Update BNB discount status
                bnb_ok = self._wallet_manager.has_bnb_for_discount()
                self._strategy.update_bnb_discount(bnb_ok)

        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Wallet rebalance loop error")

    async def _latency_monitor_loop(self) -> None:
        """Measure API latency periodically."""
        try:
            interval = self._params.get("latency_check_interval_s", 60)
            while self._running:
                await asyncio.sleep(interval)
                if not self._running:
                    break

                start = time.monotonic()
                try:
                    if not self._paper_mode:
                        await self._binance_client.sync_time()
                    else:
                        await asyncio.sleep(0.05)
                except Exception:
                    pass
                latency_ms = (time.monotonic() - start) * 1000.0
                self._risk_manager.update_latency(latency_ms)

        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Latency monitor error")

    async def _eod_reconciliation_loop(self) -> None:
        """Run EOD reconciliation at 00:00 UTC daily."""
        try:
            while self._running:
                # Calculate sleep until next 00:00 UTC
                now = datetime.now(timezone.utc)
                target_hour = self._params.get("eod_reconciliation_hour_utc", 0)
                target_minute = self._params.get("eod_reconciliation_minute_utc", 0)

                target = now.replace(
                    hour=target_hour, minute=target_minute, second=0, microsecond=0
                )
                if target <= now:
                    target += timedelta(days=1)

                sleep_s = (target - now).total_seconds()
                await asyncio.sleep(sleep_s)

                if not self._running:
                    break

                logger.info("Running EOD reconciliation...")
                await self._run_eod_reconciliation()

                # Reset daily risk counters
                self._risk_manager.reset_daily()

                if self._shared_risk:
                    self._shared_risk.reset_daily_drawdown()

        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("EOD reconciliation loop error")

    async def _run_eod_reconciliation(self) -> None:
        """Verify zero net inventory and close any residual positions."""
        if self._paper_mode:
            logger.info("EOD reconciliation (paper): no residual positions expected")
            return

        try:
            account = await self._binance_client.get_futures_account()
            for pos in account.get("positions", []):
                amt = float(pos.get("positionAmt", 0))
                if amt != 0:
                    symbol = pos["symbol"]
                    logger.critical(
                        "EOD: Residual position found %s qty=%.6f — closing at market",
                        symbol, amt,
                    )
                    close_side = "SELL" if amt > 0 else "BUY"
                    try:
                        await self._binance_client.place_futures_order(
                            symbol=symbol, side=close_side, type="MARKET",
                            quantity=abs(amt), reduce_only=True,
                        )
                    except Exception as exc:
                        logger.critical("EOD: Failed to close %s: %s", symbol, exc)

            # Calculate daily P&L
            exec_stats = self._execution.get_stats() if self._execution else {}
            logger.info(
                "EOD summary: total_profit=%.4f total_fees=%.4f win_rate=%.1f%%",
                exec_stats.get("total_profit_usdt", 0),
                exec_stats.get("total_fees_usdt", 0),
                self._execution.get_win_rate() if self._execution else 0,
            )

        except Exception as exc:
            logger.error("EOD reconciliation error: %s", exc)

    async def _weekly_recalibration_loop(self) -> None:
        """Recalibrate thresholds weekly on Monday 00:00 UTC (Section 7.3)."""
        try:
            while self._running:
                # Sleep until next Monday 00:00 UTC
                now = datetime.now(timezone.utc)
                target_weekday = 0  # Monday
                days_ahead = (target_weekday - now.weekday()) % 7
                if days_ahead == 0 and (now.hour > 0 or now.minute > 0):
                    days_ahead = 7

                target = (now + timedelta(days=days_ahead)).replace(
                    hour=0, minute=0, second=0, microsecond=0,
                )
                sleep_s = (target - now).total_seconds()
                await asyncio.sleep(sleep_s)

                if not self._running:
                    break

                logger.info("Running weekly threshold recalibration (Monday 00:00 UTC)...")

                # Get current fee tier from exchange
                spot_fee = self._params.get("spot_taker_fee_pct", 0.10)
                futures_fee = self._params.get("futures_taker_fee_pct", 0.04)

                if not self._paper_mode:
                    try:
                        account = await self._binance_client.get_futures_account()
                        fee_tier = account.get("feeTier", 0)
                        tier_fees = {
                            0: (0.10, 0.04), 1: (0.09, 0.035),
                            2: (0.08, 0.03), 3: (0.07, 0.025),
                        }
                        if fee_tier in tier_fees:
                            spot_fee, futures_fee = tier_fees[fee_tier]
                            logger.info("Fee tier %d: spot=%.2f%% futures=%.2f%%",
                                        fee_tier, spot_fee, futures_fee)
                    except Exception as exc:
                        logger.warning("Could not fetch fee tier: %s", exc)

                # Refresh BNB balance
                await self._wallet_manager.refresh_balances()
                ws = self._wallet_manager.get_state()
                total_bnb = ws.spot_bnb + ws.futures_bnb

                self._strategy.recalibrate_thresholds(
                    spot_fee, futures_fee, bnb_balance=total_bnb,
                )

        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Weekly recalibration error")

    async def _mid_session_fee_check(self) -> None:
        """Check if fee tier changed mid-session and recalibrate immediately."""
        if self._paper_mode:
            return
        try:
            account = await self._binance_client.get_futures_account()
            fee_tier = account.get("feeTier", 0)
            tier_fees = {
                0: (0.10, 0.04), 1: (0.09, 0.035),
                2: (0.08, 0.03), 3: (0.07, 0.025),
            }
            if fee_tier in tier_fees:
                spot_fee, futures_fee = tier_fees[fee_tier]
                changed = self._strategy.check_fee_tier_change(spot_fee, futures_fee)
                if changed:
                    system_logger.info(
                        "FEE_TIER_CHANGE\ttier=%d\tspot=%.2f%%\tfutures=%.2f%%",
                        fee_tier, spot_fee, futures_fee,
                    )
        except Exception as exc:
            logger.debug("Mid-session fee check failed: %s", exc)

    async def _balance_refresh_loop(self) -> None:
        """Refresh balances every 30 seconds."""
        try:
            while self._running:
                await asyncio.sleep(30)
                if not self._running:
                    break

                state = await self._wallet_manager.refresh_balances()
                balance = self._wallet_manager.get_available_balance()
                equity = state.total_usdt

                # Update scanner with current balance
                self._scanner.update_balance(balance, equity)

                # Record wallet snapshot for drift metrics
                if self._strategy_metrics:
                    self._strategy_metrics.record_wallet_snapshot(
                        state.spot_usdt, state.futures_usdt, state.total_usdt,
                    )

                # Update risk manager
                self._risk_manager.update_equity(equity)
                if self._shared_risk:
                    self._shared_risk.update_equity(equity)

                # Mid-session fee tier check (Section 11.5)
                await self._mid_session_fee_check()

                # Update paper engine equity if applicable
                if self._paper_mode and self._paper_engine:
                    paper_equity = self._paper_engine.get_equity()
                    self._scanner.update_balance(paper_equity / 2, paper_equity)
                    self._risk_manager.update_equity(paper_equity)

        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Balance refresh error")

    async def _volatility_monitor_loop(self) -> None:
        """Monitor BTC volatility for threshold adjustment."""
        try:
            while self._running:
                await asyncio.sleep(5)
                if not self._running:
                    break

                # Get BTC price from cached book tickers
                if self._scanner:
                    btc_ticker = self._scanner.get_book_ticker("BTCUSDT", "spot")
                    if btc_ticker and btc_ticker.is_valid():
                        mid = (btc_ticker.bid + btc_ticker.ask) / 2.0
                        self._risk_manager.update_btc_price(mid)

                        # Propagate volatility state to strategy
                        if self._strategy:
                            self._strategy.set_high_volatility(
                                self._risk_manager.is_high_volatility
                            )

        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Volatility monitor error")

    async def _maintenance_monitor_loop(self) -> None:
        """Query Binance system status periodically. Halt 2h before maintenance.

        Section 7.5: If Binance announces maintenance for spot or futures,
        halt arb 2 hours before scheduled maintenance. Close any unhedged
        positions before maintenance.
        """
        try:
            check_interval = self._params.get("maintenance_check_interval_s", 300)  # 5 min
            while self._running:
                await asyncio.sleep(check_interval)
                if not self._running:
                    break

                if self._paper_mode:
                    continue  # No real maintenance in paper mode

                try:
                    # Query Binance system status endpoint
                    status = await self._binance_client._request(
                        "GET", "/sapi/v1/system/status", api_type="spot",
                    )
                    sys_status = status.get("status", 0)

                    if sys_status != 0:
                        # System is in maintenance or upcoming maintenance
                        # Halt 2 hours before
                        maint_msg = status.get("msg", "Maintenance")
                        logger.warning(
                            "Binance system status=%d msg=%s — halting arb",
                            sys_status, maint_msg,
                        )
                        halt_until = time.time() + 7200  # 2 hours
                        self._risk_manager.set_maintenance_halt(halt_until)

                        # Close any unhedged positions before maintenance
                        if self._execution and self._execution.has_unhedged_positions():
                            logger.critical(
                                "Closing unhedged positions before maintenance window",
                            )
                            await self._execution.check_stuck_positions()

                        system_logger.info(
                            "MAINTENANCE_HALT\tstatus=%d\tmsg=%s\thalt_hours=2",
                            sys_status, maint_msg,
                        )
                    else:
                        # System normal — clear any maintenance halt
                        self._risk_manager.clear_maintenance_halt()

                except Exception as exc:
                    logger.debug("Maintenance status check failed: %s", exc)

        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Maintenance monitor error")

    # ══════════════════════════════════════════════════════════════════════
    #  Dashboard providers
    # ══════════════════════════════════════════════════════════════════════

    def _setup_dashboard_providers(self) -> None:
        """Wire up dashboard data providers."""
        self._dashboard.set_data_providers(
            positions_fn=lambda: [],  # Arb has no held positions
            trades_fn=lambda limit: self._execution.get_execution_history(limit) if self._execution else [],
            metrics_fn=self._get_metrics,
            equity_fn=self._get_equity_curve,
            alerts_fn=lambda: [],
            config_fn=lambda: {
                "strategy_id": STRATEGY_ID,
                "mode": "paper" if self._paper_mode else "live",
                "instruments": self._cfg.instruments,
            },
            kill_fn=self._handle_kill_switch,
        )

        self._dashboard.set_arb_providers(
            scanner_stats_fn=lambda: self._scanner.get_stats() if self._scanner else {},
            execution_stats_fn=lambda: self._execution.get_stats() if self._execution else {},
            risk_state_fn=lambda: self._risk_manager.get_state() if self._risk_manager else {},
            wallet_stats_fn=lambda: self._wallet_manager.get_stats() if self._wallet_manager else {},
            opportunity_stream_fn=lambda: self._scanner.get_recent_opportunities() if self._scanner else [],
            execution_history_fn=lambda: self._execution.get_execution_history() if self._execution else [],
            strategy_stats_fn=lambda: self._strategy.get_stats() if self._strategy else {},
            strategy_metrics_fn=lambda: self._strategy_metrics.get_full_report() if self._strategy_metrics else {},
            go_live_fn=lambda: self._strategy_metrics.go_live_criteria() if self._strategy_metrics else {},
            latency_dist_fn=lambda: self._execution.get_paper_latency_distribution() if self._execution else {},
            partial_triangle_fn=lambda: self._execution.get_partial_triangle_events() if self._execution else [],
        )

    def _get_metrics(self) -> Dict[str, Any]:
        """Build metrics dict for dashboard."""
        exec_stats = self._execution.get_stats() if self._execution else {}
        scanner_stats = self._scanner.get_stats() if self._scanner else {}
        risk_state = self._risk_manager.get_state() if self._risk_manager else {}

        metrics = {
            "total_pnl": exec_stats.get("total_profit_usdt", 0),
            "total_trades": exec_stats.get("total_executions", 0),
            "win_rate": self._execution.get_win_rate() if self._execution else 0,
            "avg_profit_pct": self._execution.get_avg_profit_pct() if self._execution else 0,
            "opportunities_per_hour": scanner_stats.get("opportunities_per_hour", 0),
            "mode_a_detected": scanner_stats.get("mode_a_detected", 0),
            "mode_b_detected": scanner_stats.get("mode_b_detected", 0),
            "total_fees": exec_stats.get("total_fees_usdt", 0),
            "avg_latency_ms": exec_stats.get("avg_execution_ms", 0),
            "latency_p95_ms": exec_stats.get("latency_p95_ms", 0),
            "unhedged_count": exec_stats.get("unhedged_count", 0),
            "equity": risk_state.get("equity", 0),
            "daily_pnl": risk_state.get("daily_pnl", 0),
            "can_trade": risk_state.get("can_trade", False),
        }

        # Include strategy metrics dashboard summary (Section 10.2)
        if self._strategy_metrics:
            metrics["strategy_metrics_summary"] = (
                self._strategy_metrics.get_dashboard_summary()
            )

        # Include go-live criteria (Section 9.3)
        if self._strategy_metrics:
            metrics["go_live_criteria"] = self._strategy_metrics.go_live_criteria()

        return metrics

    def _get_equity_curve(self) -> List[Dict[str, Any]]:
        """Build equity curve for dashboard."""
        if self._paper_mode and self._paper_engine:
            return [
                {"timestamp_ms": ts, "equity": eq}
                for ts, eq in self._paper_engine.get_equity_curve()
            ]
        return []

    async def _handle_kill_switch(self, reason: str) -> Dict[str, Any]:
        """Handle kill switch activation."""
        logger.critical("KILL SWITCH: %s", reason)
        self._risk_manager.trigger_kill_switch()
        self._scanner.stop_scanning()

        if self._kill_switch:
            result = await self._kill_switch.execute(reason)
            return result.to_dict()
        return {"status": "killed", "reason": reason}


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

async def main() -> None:
    """Async entry point."""
    # Determine config path
    config_path = os.environ.get(
        "CONFIG_PATH",
        str(Path(__file__).resolve().parent.parent / "config.yaml"),
    )

    # Setup logging
    log_dir = os.environ.get("LOG_DIR", "data/logs")
    setup_logging(STRATEGY_ID, log_dir=log_dir)

    logger.info("=" * 70)
    logger.info("STRAT-007: Cross-Exchange Triangular Arbitrage Bot")
    logger.info("=" * 70)

    bot = TriangularArbBot(config_path=config_path)

    # Signal handling for graceful shutdown
    loop = asyncio.get_event_loop()
    stop_event = asyncio.Event()

    def signal_handler(sig: int, frame: Any) -> None:
        logger.info("Received signal %d — initiating shutdown", sig)
        stop_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        await bot.start()

        # Wait for shutdown signal
        await stop_event.wait()

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt — shutting down")
    except Exception:
        logger.exception("Fatal error in main loop")
    finally:
        await bot.stop()
        logger.info("Bot shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
