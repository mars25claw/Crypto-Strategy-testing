"""Atomic JSON state persistence with rotating snapshots and background save loop.

Writes strategy state to disk as JSON atomically (temp file + os.rename).
Maintains a configurable number of rolling snapshots for corruption recovery.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class StatePersistence:
    """Atomic JSON state persistence with rotating snapshots.

    Parameters
    ----------
    state_dir : str
        Directory where state files are written.
    strategy_id : str
        Identifier used to namespace state files on disk.
    save_interval : float
        Seconds between automatic background saves.
    max_snapshots : int
        Number of rotating snapshots to keep (e.g. 3 keeps
        ``state_001.json`` through ``state_003.json``).
    """

    def __init__(
        self,
        state_dir: str = "data/state",
        strategy_id: str = "STRAT-000",
        save_interval: float = 5.0,
        max_snapshots: int = 3,
    ) -> None:
        self._state_dir = Path(state_dir)
        self._strategy_id = strategy_id
        self._save_interval = save_interval
        self._max_snapshots = max_snapshots

        # Internal mutable state dict
        self._state: Dict[str, Any] = self._empty_state()

        # Background task handle
        self._save_task: Optional[asyncio.Task] = None
        self._running = False

        # Current snapshot index (1-based, wraps at max_snapshots)
        self._snapshot_index = 1

        # Dirty flag — skip writes when nothing changed
        self._dirty = False

    # ------------------------------------------------------------------
    # Default state template
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_state() -> Dict[str, Any]:
        return {
            "positions": [],
            "orders": [],
            "indicators": {},
            "performance_counters": {},
            "drawdown_state": {},
            "pending_signals": [],
            "custom": {},
            "last_save_timestamp_ms": 0,
        }

    # ------------------------------------------------------------------
    # Public API — state access
    # ------------------------------------------------------------------

    def update_state(self, key: str, value: Any) -> None:
        """Update a top-level key in the state dict."""
        self._state[key] = value
        self._dirty = True

    def get_state(self, key: str, default: Any = None) -> Any:
        """Return the value for *key*, or *default* if missing."""
        return self._state.get(key, default)

    def get_full_state(self) -> Dict[str, Any]:
        """Return a shallow copy of the full state dict."""
        return dict(self._state)

    # ------------------------------------------------------------------
    # Persistence — save
    # ------------------------------------------------------------------

    def save_now(self) -> None:
        """Force an immediate synchronous save and snapshot rotation."""
        self._state["last_save_timestamp_ms"] = int(time.time() * 1000)
        self._ensure_state_dir()
        self._rotate_snapshots()
        self._dirty = False

    def _ensure_state_dir(self) -> None:
        self._state_dir.mkdir(parents=True, exist_ok=True)

    def _snapshot_path(self, index: int) -> Path:
        """Return the path for snapshot *index* (1-based)."""
        return self._state_dir / f"{self._strategy_id}_state_{index:03d}.json"

    def _atomic_write(self, filepath: Path, data: Dict[str, Any]) -> None:
        """Write *data* as JSON to *filepath* atomically.

        Writes to a temporary file in the same directory, then uses
        ``os.rename`` (which is atomic on POSIX when src and dst are on
        the same filesystem) to replace the target.
        """
        filepath_str = str(filepath)
        dir_name = os.path.dirname(filepath_str)
        try:
            fd, tmp_path = tempfile.mkstemp(
                suffix=".tmp",
                prefix=".state_",
                dir=dir_name,
            )
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(data, f, separators=(",", ":"), default=str)
                    f.flush()
                    os.fsync(f.fileno())
                os.rename(tmp_path, filepath_str)
            except BaseException:
                # Clean up temp file on failure
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except Exception:
            logger.exception("Atomic write failed for %s", filepath)
            raise

    def _rotate_snapshots(self) -> None:
        """Write state to the next snapshot slot, wrapping around."""
        filepath = self._snapshot_path(self._snapshot_index)
        self._atomic_write(filepath, self._state)
        logger.debug(
            "Snapshot %d written: %s",
            self._snapshot_index,
            filepath,
        )
        # Advance index, wrapping 1 -> 2 -> 3 -> 1
        self._snapshot_index = (self._snapshot_index % self._max_snapshots) + 1

    # ------------------------------------------------------------------
    # Persistence — load
    # ------------------------------------------------------------------

    def load(self) -> Dict[str, Any]:
        """Load the latest valid state from disk.

        Tries snapshots in reverse-recency order (most recent first).
        Falls back to an empty state if every snapshot is missing or
        corrupt.

        Returns
        -------
        dict
            The loaded (or empty) state dict, also stored internally.
        """
        self._ensure_state_dir()

        # Build a list of candidate files sorted by modification time (newest first)
        candidates: list[tuple[float, Path]] = []
        for idx in range(1, self._max_snapshots + 1):
            p = self._snapshot_path(idx)
            if p.exists():
                try:
                    candidates.append((p.stat().st_mtime, p))
                except OSError:
                    continue

        candidates.sort(key=lambda t: t[0], reverse=True)

        for _mtime, path in candidates:
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                if not isinstance(data, dict):
                    logger.warning("Snapshot %s is not a dict, skipping", path)
                    continue
                # Merge loaded data into a clean template so missing keys
                # get their defaults.
                state = self._empty_state()
                state.update(data)
                self._state = state
                logger.info(
                    "Loaded state from %s (last_save=%d)",
                    path,
                    state.get("last_save_timestamp_ms", 0),
                )
                # Set snapshot index to follow the one we just loaded
                # so the next write goes to a *different* slot.
                idx = self._index_from_path(path)
                if idx is not None:
                    self._snapshot_index = (idx % self._max_snapshots) + 1
                self._dirty = False
                return dict(self._state)
            except (json.JSONDecodeError, ValueError) as exc:
                logger.warning("Corrupt snapshot %s: %s — trying next", path, exc)
                continue
            except OSError as exc:
                logger.warning("Cannot read snapshot %s: %s — trying next", path, exc)
                continue

        logger.warning("No valid snapshots found in %s — starting with empty state", self._state_dir)
        self._state = self._empty_state()
        self._dirty = False
        return dict(self._state)

    def _index_from_path(self, path: Path) -> Optional[int]:
        """Extract the 1-based snapshot index from a filename, or None."""
        name = path.stem  # e.g. "STRAT-001_state_002"
        try:
            return int(name.rsplit("_", 1)[-1])
        except (ValueError, IndexError):
            return None

    # ------------------------------------------------------------------
    # Background save loop
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the background save loop."""
        if self._running:
            return
        self._running = True
        self._save_task = asyncio.create_task(self._save_loop(), name=f"state_save_{self._strategy_id}")
        logger.info(
            "StatePersistence started: dir=%s  interval=%.1fs  max_snapshots=%d",
            self._state_dir,
            self._save_interval,
            self._max_snapshots,
        )

    async def stop(self) -> None:
        """Stop the background save loop and perform a final save."""
        self._running = False
        if self._save_task is not None:
            self._save_task.cancel()
            try:
                await self._save_task
            except asyncio.CancelledError:
                pass
            self._save_task = None
        # Final save regardless of dirty flag
        try:
            self._state["last_save_timestamp_ms"] = int(time.time() * 1000)
            self._ensure_state_dir()
            self._rotate_snapshots()
            logger.info("StatePersistence stopped — final snapshot written")
        except Exception:
            logger.exception("Final save on stop() failed")

    async def _save_loop(self) -> None:
        """Background coroutine that periodically saves state to disk."""
        while self._running:
            try:
                await asyncio.sleep(self._save_interval)
                if self._dirty:
                    self.save_now()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in state save loop")
                # Continue running — transient FS errors should not kill the loop
                await asyncio.sleep(1.0)
