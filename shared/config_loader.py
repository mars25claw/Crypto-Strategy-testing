"""YAML configuration loader with validation and hot-reload support."""

import os
import yaml
import time
import logging
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class BinanceConfig:
    api_key: str = ""
    api_secret: str = ""
    testnet: bool = False
    spot_base_url: str = "https://api.binance.com"
    futures_base_url: str = "https://fapi.binance.com"
    spot_ws_url: str = "wss://stream.binance.com:9443"
    futures_ws_url: str = "wss://fstream.binance.com"
    recv_window: int = 5000
    time_sync_interval: int = 60


@dataclass
class DeribitConfig:
    enabled: bool = False
    api_key: str = ""
    api_secret: str = ""
    base_url: str = "https://www.deribit.com"
    ws_url: str = "wss://www.deribit.com/ws/api/v2"
    testnet: bool = False


@dataclass
class RiskConfig:
    max_capital_pct: float = 30.0
    max_per_trade_pct: float = 5.0
    risk_per_trade_pct: float = 1.5
    max_leverage: int = 5
    preferred_leverage: int = 3
    max_concurrent_positions: int = 5
    max_per_asset_pct: float = 10.0
    max_long_exposure_pct: float = 25.0
    max_short_exposure_pct: float = 25.0
    max_net_directional_pct: float = 20.0
    daily_drawdown_pct: float = 3.0
    weekly_drawdown_pct: float = 6.0
    monthly_drawdown_pct: float = 10.0
    system_wide_drawdown_pct: float = 15.0


@dataclass
class PaperTradingConfig:
    enabled: bool = True
    starting_equity: float = 10000.0
    maker_fee_pct: float = 0.02
    taker_fee_pct: float = 0.04
    slippage_model: str = "orderbook_walk"


@dataclass
class DashboardConfig:
    host: str = "0.0.0.0"
    port: int = 8080
    enable_kill_switch: bool = True


@dataclass
class AlertingConfig:
    enabled: bool = True
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    discord_webhook_url: str = ""
    email_smtp_host: str = ""
    email_smtp_port: int = 587
    email_from: str = ""
    email_to: str = ""
    email_password: str = ""


@dataclass
class DatabaseConfig:
    url: str = "sqlite:///data/bot.db"
    echo: bool = False


@dataclass
class LoggingConfig:
    level: str = "INFO"
    log_dir: str = "data/logs"
    rotate_days: int = 30
    trade_log_permanent: bool = True


@dataclass
class StateConfig:
    persistence_interval: float = 5.0
    snapshot_count: int = 3
    state_dir: str = "data/state"


@dataclass
class MemoryConfig:
    check_interval: int = 60
    warn_mb: int = 500
    restart_mb: int = 1000
    max_candles_per_tf: int = 500
    cache_clear_hours: int = 24


@dataclass
class HeartbeatConfig:
    interval: int = 10
    timeout: int = 30
    max_restarts_per_hour: int = 3


@dataclass
class BotConfig:
    strategy_id: str = "STRAT-000"
    strategy_name: str = "Unknown"
    mode: str = "paper"  # "paper" or "live"
    instruments: list = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])
    binance: BinanceConfig = field(default_factory=BinanceConfig)
    deribit: DeribitConfig = field(default_factory=DeribitConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    paper_trading: PaperTradingConfig = field(default_factory=PaperTradingConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    alerting: AlertingConfig = field(default_factory=AlertingConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    state: StateConfig = field(default_factory=StateConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    heartbeat: HeartbeatConfig = field(default_factory=HeartbeatConfig)
    rate_limit_weight_per_min: int = 200
    rate_limit_burst_weight: int = 400
    strategy_params: dict = field(default_factory=dict)


def _merge_dict(base: dict, override: dict) -> dict:
    """Deep merge override into base."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_dict(result[key], value)
        else:
            result[key] = value
    return result


def _dict_to_dataclass(cls, data: dict):
    """Convert a dict to a dataclass, ignoring unknown fields."""
    import dataclasses
    if not dataclasses.is_dataclass(cls):
        return data
    fieldnames = {f.name for f in dataclasses.fields(cls)}
    filtered = {}
    for k, v in data.items():
        if k in fieldnames:
            f = next(f for f in dataclasses.fields(cls) if f.name == k)
            if dataclasses.is_dataclass(f.type):
                filtered[k] = _dict_to_dataclass(f.type, v if isinstance(v, dict) else {})
            else:
                filtered[k] = v
    return cls(**filtered)


class ConfigLoader:
    """Loads and manages bot configuration from YAML files."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self._config: Optional[BotConfig] = None
        self._raw: dict = {}
        self._last_modified: float = 0
        self._load()

    def _load(self):
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            logger.warning(f"Config file not found: {self.config_path}, using defaults")
            self._config = BotConfig()
            return

        with open(self.config_path, "r") as f:
            raw = yaml.safe_load(f) or {}

        # Override with environment variables
        env_overrides = {}
        if os.environ.get("BINANCE_API_KEY"):
            env_overrides.setdefault("binance", {})["api_key"] = os.environ["BINANCE_API_KEY"]
        if os.environ.get("BINANCE_API_SECRET"):
            env_overrides.setdefault("binance", {})["api_secret"] = os.environ["BINANCE_API_SECRET"]
        if os.environ.get("DERIBIT_API_KEY"):
            env_overrides.setdefault("deribit", {})["api_key"] = os.environ["DERIBIT_API_KEY"]
        if os.environ.get("DERIBIT_API_SECRET"):
            env_overrides.setdefault("deribit", {})["api_secret"] = os.environ["DERIBIT_API_SECRET"]
        if os.environ.get("BOT_MODE"):
            env_overrides["mode"] = os.environ["BOT_MODE"]

        self._raw = _merge_dict(raw, env_overrides)
        self._config = _dict_to_dataclass(BotConfig, self._raw)
        self._last_modified = self.config_path.stat().st_mtime if self.config_path.exists() else 0
        logger.info(f"Configuration loaded: strategy={self._config.strategy_id}, mode={self._config.mode}")

    @property
    def config(self) -> BotConfig:
        """Get current configuration."""
        return self._config

    def check_reload(self) -> bool:
        """Check if config file changed and reload if so. Returns True if reloaded."""
        if not self.config_path.exists():
            return False
        current_mtime = self.config_path.stat().st_mtime
        if current_mtime > self._last_modified:
            logger.info("Configuration file changed, reloading...")
            old_mode = self._config.mode
            self._load()
            if self._config.mode != old_mode:
                logger.warning(f"Mode changed from {old_mode} to {self._config.mode} — takes effect on next trade")
            return True
        return False

    def get_strategy_param(self, key: str, default: Any = None) -> Any:
        """Get a strategy-specific parameter."""
        return self._config.strategy_params.get(key, default)
