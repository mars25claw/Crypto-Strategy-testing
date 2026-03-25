"""Entry point for STRAT-010: ML & On-Chain Quantitative Models.

Startup sequence:
1. Load configuration
2. Initialise shared infrastructure (client, WS, rate limiter, etc.)
3. Load ML models (XGBoost + LSTM)
4. Warm up: 200 candles per TF + 30 days on-chain data (max 5 min)
5. Register WS streams for all instruments
6. Start hourly feature calculation + model inference cycle
7. Start model freshness checker
8. Start event loop
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT.parent))

from shared.binance_client import BinanceClient
from shared.binance_ws_manager import WebSocketManager, ConnectionType
from shared.config_loader import ConfigLoader
from shared.external_data import FearGreedClient, GlassnodeClient, SentimentClient
from shared.heartbeat import HeartbeatMonitor
from shared.kill_switch import KillSwitch
from shared.log_manager import setup_logging
from shared.memory_manager import MemoryManager
from shared.paper_trading import PaperTradingEngine
from shared.performance_tracker import PerformanceTracker
from shared.rate_limiter import RateLimiter
from shared.risk_manager import CrossStrategyReader, RiskManager
from shared.state_persistence import StatePersistence
from shared.utils import TimeSync, ExchangeInfo

from src import STRATEGY_ID, STRATEGY_NAME
from src.dashboard import MLDashboard
from src.feature_engine import FeatureEngine
from src.ml_models import ModelManager
from src.risk_manager import StrategyRiskManager
from src.strategy import MLStrategy

logger = logging.getLogger(__name__)


class MLOnChainBot:
    """Main bot orchestrator for STRAT-010."""

    def __init__(self, config_path: str = "config.yaml") -> None:
        # Configuration
        self._config_loader = ConfigLoader(config_path)
        self._cfg = self._config_loader.config
        self._params = self._cfg.strategy_params

        # Logging
        setup_logging(
            strategy_id=STRATEGY_ID,
            log_dir=self._cfg.logging.log_dir,
            level=self._cfg.logging.level,
        )

        logger.info("=" * 70)
        logger.info("  %s — %s", STRATEGY_ID, STRATEGY_NAME)
        logger.info("  Mode: %s", self._cfg.mode)
        logger.info("  Instruments: %s", self._cfg.instruments)
        logger.info("=" * 70)

        # Shared infrastructure
        self._time_sync = TimeSync()
        self._exchange_info = ExchangeInfo()
        self._rate_limiter = RateLimiter(
            weight_per_minute=self._cfg.rate_limit_weight_per_min,
        )

        # Binance client
        self._client = BinanceClient(
            api_key=self._cfg.binance.api_key,
            api_secret=self._cfg.binance.api_secret,
            time_sync=self._time_sync,
            exchange_info=self._exchange_info,
            rate_limiter=self._rate_limiter,
            futures_base_url=self._cfg.binance.futures_base_url,
        )

        # WebSocket manager
        self._ws = WebSocketManager(
            futures_ws_url=self._cfg.binance.futures_ws_url + "/stream",
            binance_client=self._client,
        )

        # Paper trading
        self._paper: Optional[PaperTradingEngine] = None
        if self._cfg.mode == "paper" or self._cfg.paper_trading.enabled:
            self._paper = PaperTradingEngine(
                starting_equity=self._cfg.paper_trading.starting_equity,
                maker_fee_pct=self._cfg.paper_trading.maker_fee_pct,
                taker_fee_pct=self._cfg.paper_trading.taker_fee_pct,
            )

        # State persistence
        self._state = StatePersistence(
            state_dir=self._cfg.state.state_dir,
            strategy_id=STRATEGY_ID,
            save_interval=self._cfg.state.persistence_interval,
        )

        # Risk management
        shared_risk_config = self._cfg.risk
        self._shared_risk = RiskManager(
            config=shared_risk_config,
            cross_strategy_reader=CrossStrategyReader(),
        )
        self._strategy_risk = StrategyRiskManager(
            config=shared_risk_config,
            strategy_params=self._params,
        )

        # ML models
        model_dir = str(PROJECT_ROOT / self._params.get("model_dir", "models/"))
        self._model_mgr = ModelManager(
            model_dir=model_dir,
            xgb_weight=self._params.get("xgboost_weight", 0.60),
            lstm_weight=self._params.get("lstm_weight", 0.40),
            long_threshold=self._params.get("long_threshold", 0.65),
            short_threshold=self._params.get("short_threshold", 0.35),
            high_confidence_long=self._params.get("high_confidence_long", 0.75),
            high_confidence_short=self._params.get("high_confidence_short", 0.25),
            lstm_timeout=self._params.get("lstm_timeout_seconds", 5.0),
            lstm_hidden_size=self._params.get("lstm_hidden_size", 64),
            lstm_num_layers=self._params.get("lstm_num_layers", 2),
            lstm_input_features=self._params.get("lstm_input_features", 35),
        )

        # Feature engines (one per instrument)
        self._engines: Dict[str, FeatureEngine] = {}
        for sym in self._cfg.instruments:
            self._engines[sym] = FeatureEngine(
                symbol=sym,
                z_score_window=self._params.get("z_score_window", 180),
                anomaly_sigma=self._params.get("anomaly_sigma_threshold", 2.0),
            )

        # Strategy
        self._strategy = MLStrategy(
            binance_client=self._client,
            model_manager=self._model_mgr,
            feature_engines=self._engines,
            risk_manager=self._strategy_risk,
            paper_engine=self._paper,
            params=self._params,
        )

        # Performance tracker
        self._perf = PerformanceTracker(strategy_id=STRATEGY_ID)

        # External data clients
        self._glassnode = GlassnodeClient(
            api_key=self._params.get("glassnode_api_key", "")
            or os.environ.get("GLASSNODE_API_KEY", ""),
        )
        self._fear_greed = FearGreedClient()
        self._sentiment = SentimentClient(
            lunarcrush_api_key=self._params.get("lunarcrush_api_key", "")
            or os.environ.get("LUNARCRUSH_API_KEY", ""),
            cryptocompare_api_key=self._params.get("cryptocompare_api_key", "")
            or os.environ.get("CRYPTOCOMPARE_API_KEY", ""),
        )

        # Heartbeat
        self._heartbeat = HeartbeatMonitor(
            strategy_id=STRATEGY_ID,
            interval=self._cfg.heartbeat.interval,
            timeout=self._cfg.heartbeat.timeout,
            max_restarts_per_hour=self._cfg.heartbeat.max_restarts_per_hour,
        )

        # Memory manager
        self._memory = MemoryManager(
            check_interval=self._cfg.memory.check_interval,
            warn_mb=self._cfg.memory.warn_mb,
            restart_mb=self._cfg.memory.restart_mb,
        )

        # Kill switch
        self._kill_switch = KillSwitch(
            binance_client=self._client,
            state_persistence=self._state,
        )

        # Dashboard
        self._dashboard = MLDashboard(
            host=self._cfg.dashboard.host,
            port=self._cfg.dashboard.port,
            template_dir=str(PROJECT_ROOT / "templates"),
        )

        self._running = False

    # ══════════════════════════════════════════════════════════════════
    #  Startup
    # ══════════════════════════════════════════════════════════════════

    async def start(self) -> None:
        """Full startup sequence."""
        self._running = True

        # 1. Start Binance client
        await self._client.start()
        await self._client.ensure_time_synced()
        await self._client.load_exchange_info(self._cfg.instruments)

        # 2. Load ML models
        logger.info("Loading ML models...")
        xgb_ok, lstm_ok = self._model_mgr.load_models(
            xgb_file=self._params.get("xgboost_model_file", "xgboost_model.json"),
            lstm_file=self._params.get("lstm_model_file", "lstm_model.pt"),
        )
        logger.info("Models loaded: XGBoost=%s, LSTM=%s", xgb_ok, lstm_ok)

        if not xgb_ok and not lstm_ok:
            logger.warning(
                "No ML models available -- will use fallback strategy until "
                "models are provided in %s/",
                self._params.get("model_dir", "models/"),
            )

        # 3. Warm up data
        await self._warmup()

        # 4. Load persisted state
        self._state.load()

        # 5. Register WS streams
        self._register_streams()

        # 6. Start WS manager
        await self._ws.start()

        # 7. Start background tasks
        await self._state.start()
        await self._heartbeat.start()

        # 8. Setup dashboard
        self._setup_dashboard()
        await self._dashboard.start()

        # 9. Start background loops
        asyncio.create_task(self._external_data_loop(), name="external-data")
        asyncio.create_task(self._model_freshness_loop(), name="model-freshness")
        asyncio.create_task(self._state_persistence_loop(), name="state-persist")
        asyncio.create_task(self._memory.start(), name="memory-monitor")

        # 10. Update equity
        if self._paper:
            self._shared_risk.update_equity(self._paper.get_equity())

        logger.info("STRAT-010 startup complete -- entering main loop")

    async def _warmup(self) -> None:
        """Warm up: fetch 200 candles per TF per instrument + on-chain data."""
        logger.info("Starting warm-up (max 5 minutes)...")
        start = time.monotonic()
        timeout = self._params.get("warmup_timeout_seconds", 300)
        candles_per_tf = self._params.get("warmup_candles", 200)

        timeframes = ["1m", "5m", "15m", "1h", "4h", "1d"]

        try:
            for symbol in self._cfg.instruments:
                engine = self._engines[symbol]
                for tf in timeframes:
                    elapsed = time.monotonic() - start
                    if elapsed > timeout:
                        logger.warning("Warm-up timeout reached at %.1f s", elapsed)
                        return

                    try:
                        klines = await self._client.get_futures_klines(
                            symbol=symbol,
                            interval=tf,
                            limit=candles_per_tf,
                        )
                        for k in klines:
                            candle = {
                                "timestamp": k[0],
                                "open": k[1],
                                "high": k[2],
                                "low": k[3],
                                "close": k[4],
                                "volume": k[5],
                            }
                            engine.on_kline(tf, candle)

                        logger.info(
                            "Warm-up: %s %s -- %d candles loaded",
                            symbol, tf, len(klines),
                        )
                        # Rate limit respect
                        await asyncio.sleep(0.2)

                    except Exception:
                        logger.exception("Warm-up failed for %s %s", symbol, tf)

            # Fetch on-chain data
            await self._fetch_onchain_data()
            await self._fetch_sentiment_data()

            # Run a test prediction
            for symbol, engine in self._engines.items():
                snap = engine.calculate_features()
                logger.info(
                    "Warm-up test: %s features=%d/%d",
                    symbol, snap.available_count, 35,
                )

        except Exception:
            logger.exception("Warm-up error")

        elapsed = time.monotonic() - start
        logger.info("Warm-up completed in %.1f seconds", elapsed)

    # ══════════════════════════════════════════════════════════════════
    #  WS stream registration
    # ══════════════════════════════════════════════════════════════════

    def _register_streams(self) -> None:
        """Register all WebSocket streams for each instrument."""
        for symbol in self._cfg.instruments:
            sym_lower = symbol.lower()
            engine = self._engines[symbol]

            subs = []

            # Kline streams
            for tf in ["1m", "5m", "15m", "1h", "4h", "1d"]:
                stream = f"{sym_lower}@kline_{tf}"
                subs.append((
                    stream,
                    lambda data, s=symbol, t=tf: self._strategy.on_kline(s, t, data),
                ))

            # Depth
            stream = f"{sym_lower}@depth20@100ms"
            subs.append((stream, lambda data, s=symbol: self._strategy.on_depth(s, data)))

            # AggTrade (not needed for features, but kept for flow analysis)
            # Mark price
            stream = f"{sym_lower}@markPrice@1s"
            subs.append((stream, self._strategy.on_mark_price))

            # Liquidation
            stream = f"{sym_lower}@forceOrder"
            subs.append((stream, lambda data, s=symbol: self._strategy.on_liquidation(s, data)))

            # Book ticker
            stream = f"{sym_lower}@bookTicker"
            subs.append((stream, self._strategy.on_book_ticker))

            self._ws.register_strategy(
                strategy_id=f"{STRATEGY_ID}-{symbol}",
                subscriptions=subs,
                conn_type=ConnectionType.FUTURES,
            )

        logger.info(
            "Registered %d streams for %d instruments",
            len(self._ws.get_registered_streams(ConnectionType.FUTURES)),
            len(self._cfg.instruments),
        )

    # ══════════════════════════════════════════════════════════════════
    #  Background loops
    # ══════════════════════════════════════════════════════════════════

    async def _external_data_loop(self) -> None:
        """Periodically fetch on-chain, derivatives, and sentiment data."""
        while self._running:
            try:
                # On-chain data (every 4 hours)
                await self._fetch_onchain_data()

                # Derivatives data (every 15 minutes)
                await self._fetch_derivatives_data()

                # Sentiment data (every hour)
                await self._fetch_sentiment_data()

            except Exception:
                logger.exception("External data loop error")

            await asyncio.sleep(900)  # 15 minutes

    async def _fetch_onchain_data(self) -> None:
        """Fetch on-chain metrics from Glassnode and update engines."""
        for symbol in self._cfg.instruments:
            asset = symbol.replace("USDT", "")
            engine = self._engines[symbol]

            try:
                # Exchange net flow
                flow_data = await self._glassnode.get_exchange_net_flow(asset)
                if flow_data and isinstance(flow_data, list) and len(flow_data) > 0:
                    latest = flow_data[-1]
                    engine.update_onchain(exchange_net_flow=float(latest.get("v", 0)))

                # MVRV Z-score
                mvrv_data = await self._glassnode.get_mvrv_zscore(asset)
                if mvrv_data and isinstance(mvrv_data, list) and len(mvrv_data) > 0:
                    engine.update_onchain(mvrv_zscore=float(mvrv_data[-1].get("v", 0)))

                # NUPL
                nupl_data = await self._glassnode.get_nupl(asset)
                if nupl_data and isinstance(nupl_data, list) and len(nupl_data) > 0:
                    engine.update_onchain(nupl=float(nupl_data[-1].get("v", 0)))

                # Active addresses
                addr_data = await self._glassnode.get_active_addresses(asset)
                if addr_data and isinstance(addr_data, list) and len(addr_data) > 1:
                    current = float(addr_data[-1].get("v", 0))
                    prev_7d = float(addr_data[-min(7, len(addr_data))].get("v", 1))
                    change = (current - prev_7d) / prev_7d if prev_7d > 0 else 0
                    engine.update_onchain(active_addresses_change=change)

                # Hash rate (BTC only)
                if asset == "BTC":
                    hr_data = await self._glassnode.get_hash_rate(asset)
                    if hr_data and isinstance(hr_data, list) and len(hr_data) > 1:
                        current = float(hr_data[-1].get("v", 0))
                        prev = float(hr_data[-min(7, len(hr_data))].get("v", 1))
                        change = (current - prev) / prev if prev > 0 else 0
                        engine.update_onchain(hash_rate_change=change)

                # Stablecoin reserves
                sc_data = await self._glassnode.get_stablecoin_reserves()
                if sc_data and isinstance(sc_data, list) and len(sc_data) > 0:
                    engine.update_onchain(stablecoin_reserves=float(sc_data[-1].get("v", 0)))

                self._strategy_risk.mark_data_source_healthy("glassnode")

            except Exception:
                logger.exception("On-chain data fetch failed for %s", symbol)
                self._strategy_risk.mark_data_source_stale("glassnode")

    async def _fetch_derivatives_data(self) -> None:
        """Fetch derivatives data from Binance REST API."""
        for symbol in self._cfg.instruments:
            engine = self._engines[symbol]
            try:
                # Open Interest
                oi_resp = await self._client._request(
                    "GET", "/fapi/v1/openInterest",
                    params={"symbol": symbol},
                    signed=False,
                )
                if oi_resp:
                    engine.update_derivatives(
                        oi_change_24h=float(oi_resp.get("openInterest", 0)),
                    )

                # Long/Short ratio
                ls_resp = await self._client._request(
                    "GET", "/futures/data/globalLongShortAccountRatio",
                    params={"symbol": symbol, "period": "1h", "limit": 1},
                    signed=False,
                )
                if ls_resp and isinstance(ls_resp, list) and len(ls_resp) > 0:
                    engine.update_derivatives(
                        long_short_ratio=float(ls_resp[-1].get("longShortRatio", 1)),
                    )

                # Taker buy/sell ratio
                taker_resp = await self._client._request(
                    "GET", "/futures/data/takerlongshortRatio",
                    params={"symbol": symbol, "period": "1h", "limit": 1},
                    signed=False,
                )
                if taker_resp and isinstance(taker_resp, list) and len(taker_resp) > 0:
                    engine.update_derivatives(
                        taker_buy_sell_ratio=float(taker_resp[-1].get("buySellRatio", 1)),
                    )

                await asyncio.sleep(0.1)

            except Exception:
                logger.exception("Derivatives data fetch failed for %s", symbol)

    async def _fetch_sentiment_data(self) -> None:
        """Fetch sentiment data from external APIs."""
        try:
            # Fear & Greed Index
            fg = await self._fear_greed.get_current()
            fg_value = fg.get("value", 50) / 100.0  # Normalise to 0-1

            for engine in self._engines.values():
                engine.update_sentiment(fear_greed_index=fg_value)

            # Social volume per asset
            for symbol in self._cfg.instruments:
                asset = symbol.replace("USDT", "")
                engine = self._engines[symbol]

                volume = await self._sentiment.get_social_volume(asset)
                score = await self._sentiment.get_sentiment_score(asset)

                # Convert to z-score vs 30-day (simplified)
                engine.update_sentiment(
                    social_volume_zscore=volume / 100.0 if volume else 0.0,
                    fear_greed_index=fg_value,
                )

            self._strategy_risk.mark_data_source_healthy("sentiment")

        except Exception:
            logger.exception("Sentiment data fetch failed")
            self._strategy_risk.mark_data_source_stale("sentiment")

    async def _model_freshness_loop(self) -> None:
        """Check model freshness every hour."""
        while self._running:
            try:
                mult = self._model_mgr.get_freshness_multiplier()
                self._strategy_risk.set_model_freshness(mult)

                if mult < 1.0:
                    logger.warning(
                        "Model freshness multiplier: %.2f (XGB age=%.1f d, LSTM age=%.1f d)",
                        mult, self._model_mgr.xgb_meta.age_days,
                        self._model_mgr.lstm_meta.age_days,
                    )

                # Check for new model files (hot-swap)
                model_dir = Path(self._params.get("model_dir", "models/"))
                new_xgb = model_dir / "xgboost_model_new.json"
                new_lstm = model_dir / "lstm_model_new.pt"

                if new_xgb.exists():
                    logger.info("New XGBoost model detected -- hot-swapping")
                    self._model_mgr.hot_swap_xgboost("xgboost_model_new.json")
                    new_xgb.rename(model_dir / "xgboost_model_new.json.deployed")

                if new_lstm.exists():
                    logger.info("New LSTM model detected -- hot-swapping")
                    self._model_mgr.hot_swap_lstm("lstm_model_new.pt")
                    new_lstm.rename(model_dir / "lstm_model_new.pt.deployed")

            except Exception:
                logger.exception("Model freshness check error")

            await asyncio.sleep(3600)  # Every hour

    async def _state_persistence_loop(self) -> None:
        """Persist strategy state every 5 seconds."""
        while self._running:
            try:
                state = self._strategy.get_state()
                state["risk"] = self._strategy_risk.get_status()
                state["model"] = self._model_mgr.get_status()
                self._state.update_state("custom", state)

                # Update equity
                if self._paper:
                    equity = self._paper.get_equity()
                    self._shared_risk.update_equity(equity)

                # Update heartbeat
                self._heartbeat.set_positions_count(self._strategy.get_position_count())

            except Exception:
                logger.exception("State persistence error")

            await asyncio.sleep(5)

    # ══════════════════════════════════════════════════════════════════
    #  Dashboard setup
    # ══════════════════════════════════════════════════════════════════

    def _setup_dashboard(self) -> None:
        """Wire up dashboard data providers."""
        self._dashboard.set_data_providers(
            positions_fn=self._strategy.get_positions,
            trades_fn=self._perf.get_recent_trades,
            metrics_fn=self._perf.get_metrics,
            equity_fn=lambda: self._paper.get_equity_curve() if self._paper else [],
            alerts_fn=lambda: [],
            config_fn=lambda: {"paper_trading": self._cfg.mode == "paper"},
            kill_fn=lambda reason: self._kill_switch.execute(reason),
        )

        self._dashboard.set_ml_providers(
            model_status_fn=self._model_mgr.get_status,
            feature_importance_fn=lambda: self._model_mgr.get_top_features(10),
            prediction_dist_fn=self._model_mgr.get_prediction_distribution,
            prediction_history_fn=lambda: self._model_mgr.get_prediction_history(50),
            risk_status_fn=self._strategy_risk.get_status,
            onchain_status_fn=lambda: {
                sym: eng.get_state().get("onchain", {})
                for sym, eng in self._engines.items()
            },
            data_source_health_fn=lambda: {
                "glassnode": self._glassnode.get_all_statuses(),
                "sentiment": self._sentiment.get_all_statuses(),
                "fear_greed": self._fear_greed.get_all_statuses(),
            },
            confidence_accuracy_fn=self._strategy_risk.get_confidence_vs_accuracy,
            ensemble_agreement_fn=lambda: {
                "agreement_rate": self._model_mgr.get_ensemble_agreement_rate(),
            },
            fallback_status_fn=lambda: self._strategy._fallback.get_status(),
        )

    # ══════════════════════════════════════════════════════════════════
    #  Shutdown
    # ══════════════════════════════════════════════════════════════════

    async def stop(self) -> None:
        """Graceful shutdown."""
        logger.info("Shutting down STRAT-010...")
        self._running = False

        self._dashboard.stop()
        self._heartbeat.stop()
        self._memory.stop()
        await self._state.stop()
        await self._ws.stop()
        await self._client.close()
        await self._glassnode.close()
        await self._sentiment.close()
        await self._fear_greed.close()

        logger.info("STRAT-010 shutdown complete")


# ══════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════

async def main() -> None:
    """Async entry point."""
    # Load .env if present
    try:
        from dotenv import load_dotenv
        env_path = PROJECT_ROOT / ".env"
        if env_path.exists():
            load_dotenv(str(env_path))
    except ImportError:
        pass

    config_path = str(PROJECT_ROOT / "config.yaml")
    bot = MLOnChainBot(config_path=config_path)

    # Signal handling
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(bot.stop()))

    try:
        await bot.start()
        # Run forever
        while bot._running:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        await bot.stop()


if __name__ == "__main__":
    asyncio.run(main())
