"""Memory monitoring and management."""

import gc
import os
import time
try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    _PSUTIL_AVAILABLE = False
import logging
import asyncio
from typing import Optional, Callable, Awaitable

logger = logging.getLogger("system")


class MemoryManager:
    """
    Monitors process memory usage and triggers actions:
    - Check every 60 seconds
    - Warn at 500MB → trigger GC
    - Restart at 1GB → graceful restart (persist state first)
    - Clear indicator caches every 24 hours
    """

    def __init__(
        self,
        check_interval: int = 60,
        warn_mb: int = 500,
        restart_mb: int = 1000,
        cache_clear_hours: int = 24,
        on_restart: Optional[Callable[[], Awaitable[None]]] = None,
    ):
        self.check_interval = check_interval
        self.warn_mb = warn_mb
        self.restart_mb = restart_mb
        self.cache_clear_hours = cache_clear_hours
        self.on_restart = on_restart
        self._process = psutil.Process(os.getpid()) if _PSUTIL_AVAILABLE else None
        self._last_cache_clear = time.time()
        self._cache_clear_callbacks: list = []
        self._running = False

    def add_cache_clear_callback(self, callback: Callable):
        """Register a callback to clear caches (e.g., indicator buffers)."""
        self._cache_clear_callbacks.append(callback)

    def get_memory_mb(self) -> float:
        """Get current process memory usage in MB."""
        if self._process is None:
            return 0.0
        return self._process.memory_info().rss / (1024 * 1024)

    async def start(self):
        """Start the memory monitoring loop."""
        self._running = True
        logger.info(
            f"Memory manager started: warn={self.warn_mb}MB, restart={self.restart_mb}MB, "
            f"check_interval={self.check_interval}s, cache_clear={self.cache_clear_hours}h"
        )
        while self._running:
            try:
                await self._check()
            except Exception as e:
                logger.error(f"Memory check error: {e}")
            await asyncio.sleep(self.check_interval)

    async def _check(self):
        mem_mb = self.get_memory_mb()

        # Check if cache clear is needed (every 24 hours)
        if time.time() - self._last_cache_clear > self.cache_clear_hours * 3600:
            logger.info(f"Clearing indicator caches (scheduled every {self.cache_clear_hours}h)")
            for callback in self._cache_clear_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback()
                    else:
                        callback()
                except Exception as e:
                    logger.error(f"Cache clear callback error: {e}")
            self._last_cache_clear = time.time()
            gc.collect()
            logger.info(f"Cache cleared. Memory: {self.get_memory_mb():.1f}MB")

        # Memory threshold checks
        if mem_mb >= self.restart_mb:
            logger.critical(
                f"Memory usage {mem_mb:.1f}MB exceeds restart threshold {self.restart_mb}MB. "
                f"Triggering graceful restart."
            )
            gc.collect()
            mem_after_gc = self.get_memory_mb()
            if mem_after_gc >= self.restart_mb:
                if self.on_restart:
                    await self.on_restart()
                else:
                    logger.critical("No restart handler configured. Manual intervention required.")
        elif mem_mb >= self.warn_mb:
            logger.warning(f"Memory usage {mem_mb:.1f}MB exceeds warning threshold {self.warn_mb}MB. Running GC.")
            gc.collect()
            mem_after = self.get_memory_mb()
            logger.info(f"Post-GC memory: {mem_after:.1f}MB (freed {mem_mb - mem_after:.1f}MB)")

    def stop(self):
        """Stop the monitoring loop."""
        self._running = False

    def get_status(self) -> dict:
        """Get current memory status for heartbeat."""
        mem_mb = self.get_memory_mb()
        return {
            "memory_mb": round(mem_mb, 1),
            "warn_threshold_mb": self.warn_mb,
            "restart_threshold_mb": self.restart_mb,
            "status": "critical" if mem_mb >= self.restart_mb else "warning" if mem_mb >= self.warn_mb else "ok",
            "last_cache_clear": self._last_cache_clear,
        }

    async def check(self) -> None:
        """Compatibility shim — run a memory check cycle."""
        await self._check()
