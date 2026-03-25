"""Cross-strategy position awareness via shared filesystem.

Each running strategy writes its current positions to a JSON file in a
shared directory.  Other strategies can read all files to compute aggregate
exposure, detect correlated risk, and clean up stale entries.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hard-coded correlation groups for major crypto assets.
# Assets within the same group are treated as correlated.
# ---------------------------------------------------------------------------

_CORRELATION_GROUPS: List[set] = [
    # Large-cap L1s — highly correlated with BTC
    {"BTCUSDT", "BTCBUSD", "WBTCUSDT"},
    # ETH ecosystem
    {"ETHUSDT", "ETHBUSD", "STETHUSDT"},
    # Alt-L1s — tend to move together
    {"SOLUSDT", "AVAXUSDT", "ADAUSDT", "DOTUSDT", "NEARUSDT", "APTUSDT", "SUIUSDT"},
    # DeFi tokens
    {"UNIUSDT", "AAVEUSDT", "MKRUSDT", "LINKUSDT", "SNXUSDT"},
    # Meme coins
    {"DOGEUSDT", "SHIBUSDT", "PEPEUSDT", "FLOKIUSDT", "BONKUSDT"},
    # L2 tokens
    {"ARBUSDT", "OPUSDT", "MATICUSDT", "STRKUSDT"},
]


def _find_correlated_symbols(symbol: str, threshold: float = 0.75) -> set:
    """Return symbols in the same correlation group as *symbol*.

    The *threshold* parameter is accepted for API compatibility but the
    current implementation uses fixed groups (implying correlation >= 0.75
    within each group).
    """
    correlated: set = set()
    symbol_upper = symbol.upper()
    for group in _CORRELATION_GROUPS:
        if symbol_upper in group:
            correlated |= group
    correlated.discard(symbol_upper)
    return correlated


class CrossStrategyManager:
    """Read/write per-strategy position files for cross-strategy awareness.

    Parameters:
        strategy_id: Unique identifier for this strategy (e.g. "STRAT-001").
        shared_dir: Directory where position files are stored.  All running
            strategies must share the same directory.
    """

    def __init__(self, strategy_id: str, shared_dir: str = "/app/data/shared") -> None:
        self.strategy_id = strategy_id
        self.shared_dir = Path(shared_dir)
        self.shared_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write_positions(self, positions: List[dict]) -> None:
        """Atomically write this strategy's current positions.

        Each dict in *positions* should contain:
            symbol, direction, size_usdt, entry_price, strategy_id, timestamp_ms

        The file is written atomically via rename to prevent readers from
        seeing partial content.
        """
        payload = {
            "strategy_id": self.strategy_id,
            "timestamp_ms": int(time.time() * 1000),
            "positions": [
                {
                    "symbol": p.get("symbol", ""),
                    "direction": p.get("direction", "flat"),
                    "size_usdt": float(p.get("size_usdt", 0)),
                    "entry_price": float(p.get("entry_price", 0)),
                    "strategy_id": p.get("strategy_id", self.strategy_id),
                    "timestamp_ms": int(p.get("timestamp_ms", time.time() * 1000)),
                }
                for p in positions
            ],
        }

        target = self.shared_dir / f"{self.strategy_id}_positions.json"

        # Atomic write: write to temp file then rename
        fd, tmp_path = tempfile.mkstemp(dir=str(self.shared_dir), suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp_path, str(target))
            logger.debug("Wrote %d positions for %s", len(positions), self.strategy_id)
        except Exception:
            # Clean up temp file on failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def read_all_positions(self) -> Dict[str, List[dict]]:
        """Read all strategy position files.

        Returns:
            {strategy_id: [position_dict, ...]} for every *_positions.json
            file found in the shared directory.
        """
        result: Dict[str, List[dict]] = {}
        for path in self.shared_dir.glob("*_positions.json"):
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                sid = data.get("strategy_id", path.stem.replace("_positions", ""))
                result[sid] = data.get("positions", [])
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Could not read position file %s: %s", path.name, exc)
        return result

    # ------------------------------------------------------------------
    # Exposure queries
    # ------------------------------------------------------------------

    def get_exposure_for_symbol(self, symbol: str) -> dict:
        """Compute aggregate exposure across all strategies for *symbol*.

        Returns:
            dict with keys: long_usdt, short_usdt, net_usdt, strategy_count.
        """
        all_pos = self.read_all_positions()
        long_usdt = 0.0
        short_usdt = 0.0
        strategies_with_exposure: set = set()
        symbol_upper = symbol.upper()

        for sid, positions in all_pos.items():
            for p in positions:
                if p.get("symbol", "").upper() != symbol_upper:
                    continue
                size = abs(float(p.get("size_usdt", 0)))
                direction = p.get("direction", "").lower()
                if direction == "long":
                    long_usdt += size
                elif direction == "short":
                    short_usdt += size
                strategies_with_exposure.add(sid)

        return {
            "symbol": symbol_upper,
            "long_usdt": round(long_usdt, 2),
            "short_usdt": round(short_usdt, 2),
            "net_usdt": round(long_usdt - short_usdt, 2),
            "strategy_count": len(strategies_with_exposure),
        }

    def get_correlated_exposure(self, symbol: str, correlation_threshold: float = 0.75) -> float:
        """Compute total absolute exposure in assets correlated with *symbol*.

        Uses fixed correlation groups.  The *correlation_threshold* is for
        API compatibility (groups imply >= 0.75 correlation).

        Returns:
            Total absolute exposure (USDT) across correlated assets and all
            strategies, **excluding** the symbol itself.
        """
        correlated = _find_correlated_symbols(symbol, correlation_threshold)
        if not correlated:
            return 0.0

        all_pos = self.read_all_positions()
        total = 0.0
        for _sid, positions in all_pos.items():
            for p in positions:
                if p.get("symbol", "").upper() in correlated:
                    total += abs(float(p.get("size_usdt", 0)))
        return round(total, 2)

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def cleanup_stale(self, max_age_seconds: int = 300) -> int:
        """Remove position files that haven't been updated recently.

        Returns:
            Number of files removed.
        """
        removed = 0
        now_ms = int(time.time() * 1000)
        cutoff_ms = now_ms - (max_age_seconds * 1000)

        for path in self.shared_dir.glob("*_positions.json"):
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                ts = int(data.get("timestamp_ms", 0))
                if ts < cutoff_ms:
                    path.unlink()
                    removed += 1
                    logger.info("Removed stale position file: %s (age=%ds)", path.name, (now_ms - ts) / 1000)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Error checking stale file %s: %s", path.name, exc)
        return removed
