"""Utility functions: time sync, rounding, lot size, notional checks."""

import time
import math
import hmac
import hashlib
import logging
from typing import Optional, Dict, Any
from decimal import Decimal, ROUND_DOWN, ROUND_UP

logger = logging.getLogger(__name__)


class TimeSync:
    """Manages time synchronization with Binance server."""

    def __init__(self):
        self._offset_ms: int = 0
        self._last_sync: float = 0
        self._sync_interval: int = 60

    @property
    def offset_ms(self) -> int:
        return self._offset_ms

    def update_offset(self, server_time_ms: int):
        """Update time offset from Binance server time."""
        local_time_ms = int(time.time() * 1000)
        self._offset_ms = server_time_ms - local_time_ms
        self._last_sync = time.time()
        if abs(self._offset_ms) > 1000:
            logger.warning(f"Large time offset detected: {self._offset_ms}ms")

    def get_timestamp(self) -> int:
        """Get current timestamp adjusted for server offset."""
        return int(time.time() * 1000) + self._offset_ms

    def needs_sync(self) -> bool:
        """Check if time sync is needed."""
        return (time.time() - self._last_sync) > self._sync_interval


class ExchangeInfo:
    """Manages exchange info for symbol precision and limits."""

    def __init__(self):
        self._symbols: Dict[str, Dict[str, Any]] = {}

    def update(self, symbol: str, info: Dict[str, Any]):
        """Update symbol info from exchange."""
        self._symbols[symbol] = info

    def update_from_filters(self, symbol: str, filters: list, base_precision: int = 8, quote_precision: int = 8):
        """Parse Binance exchange info filters."""
        parsed = {
            "base_precision": base_precision,
            "quote_precision": quote_precision,
        }
        for f in filters:
            if f["filterType"] == "PRICE_FILTER":
                parsed["tick_size"] = float(f["tickSize"])
                parsed["min_price"] = float(f["minPrice"])
                parsed["max_price"] = float(f["maxPrice"])
            elif f["filterType"] == "LOT_SIZE":
                parsed["step_size"] = float(f["stepSize"])
                parsed["min_qty"] = float(f["minQty"])
                parsed["max_qty"] = float(f["maxQty"])
            elif f["filterType"] == "MIN_NOTIONAL" or f["filterType"] == "NOTIONAL":
                parsed["min_notional"] = float(f.get("minNotional", f.get("notional", 5)))
            elif f["filterType"] == "MARKET_LOT_SIZE":
                parsed["market_step_size"] = float(f["stepSize"])
                parsed["market_min_qty"] = float(f["minQty"])
                parsed["market_max_qty"] = float(f["maxQty"])
        self._symbols[symbol] = parsed

    def get_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        return self._symbols.get(symbol)

    def round_price(self, symbol: str, price: float) -> float:
        """Round price to valid tick size."""
        info = self._symbols.get(symbol)
        if not info or "tick_size" not in info:
            return round(price, 8)
        tick = Decimal(str(info["tick_size"]))
        return float(Decimal(str(price)).quantize(tick, rounding=ROUND_DOWN))

    def round_quantity(self, symbol: str, quantity: float, round_up: bool = False) -> float:
        """Round quantity to valid step size (round DOWN by default for safety)."""
        info = self._symbols.get(symbol)
        if not info or "step_size" not in info:
            return round(quantity, 8)
        step = Decimal(str(info["step_size"]))
        rounding = ROUND_UP if round_up else ROUND_DOWN
        result = float(Decimal(str(quantity)).quantize(step, rounding=rounding))
        min_qty = info.get("min_qty", 0)
        if result < min_qty:
            return 0.0
        return result

    def check_notional(self, symbol: str, quantity: float, price: float) -> bool:
        """Check if order meets minimum notional value."""
        info = self._symbols.get(symbol)
        if not info:
            return True
        min_notional = info.get("min_notional", 5.0)
        return (quantity * price) >= min_notional

    def get_min_notional(self, symbol: str) -> float:
        info = self._symbols.get(symbol)
        return info.get("min_notional", 5.0) if info else 5.0


def sign_request(query_string: str, secret: str) -> str:
    """Create HMAC-SHA256 signature for Binance API."""
    return hmac.new(
        secret.encode("utf-8"),
        query_string.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()


def ms_to_human(ms: int) -> str:
    """Convert milliseconds to human-readable duration."""
    seconds = ms / 1000
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f}m"
    hours = minutes / 60
    if hours < 24:
        return f"{hours:.1f}h"
    days = hours / 24
    return f"{days:.1f}d"


def pct_change(old: float, new: float) -> float:
    """Calculate percentage change."""
    if old == 0:
        return 0.0
    return ((new - old) / abs(old)) * 100.0


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value between min and max."""
    return max(min_val, min(max_val, value))


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division that returns default on zero denominator."""
    if denominator == 0:
        return default
    return numerator / denominator


def annualize_return(period_return: float, periods_per_year: float) -> float:
    """Annualize a return given the number of periods per year."""
    if period_return <= -1.0:
        return -1.0
    return (1 + period_return) ** periods_per_year - 1


def bps_to_pct(bps: float) -> float:
    """Convert basis points to percentage."""
    return bps / 100.0


def pct_to_bps(pct: float) -> float:
    """Convert percentage to basis points."""
    return pct * 100.0
