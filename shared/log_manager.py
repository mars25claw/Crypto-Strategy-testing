"""Log management with rotation and separate log files."""

import os
import logging
import logging.handlers
from pathlib import Path
from typing import Optional


def setup_logging(
    strategy_id: str,
    log_dir: str = "data/logs",
    level: str = "INFO",
    rotate_days: int = 30,
) -> dict:
    """
    Set up logging with 4 separate log files:
    - trade.log: permanent append-only (entries, exits, fills) — NEVER deleted
    - error.log: errors and exceptions, rotated
    - performance.log: performance metrics, rotated
    - system.log: system health, heartbeats, connections, rotated

    Returns dict of logger names for convenience.
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console_fmt = logging.Formatter(
        f"[%(asctime)s] [{strategy_id}] %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console.setFormatter(console_fmt)
    root_logger.addHandler(console)

    file_fmt = logging.Formatter(
        "%(asctime)s\t%(levelname)s\t%(name)s\t%(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S"
    )

    # Trade log — permanent, append-only, NO rotation
    trade_logger = logging.getLogger("trade")
    trade_handler = logging.FileHandler(log_path / "trade.log", mode="a")
    trade_handler.setFormatter(file_fmt)
    trade_handler.setLevel(logging.INFO)
    trade_logger.addHandler(trade_handler)
    trade_logger.propagate = False

    # Error log — rotated daily
    error_logger = logging.getLogger("error")
    error_handler = logging.handlers.TimedRotatingFileHandler(
        log_path / "error.log", when="midnight", backupCount=rotate_days, utc=True
    )
    error_handler.setFormatter(file_fmt)
    error_handler.setLevel(logging.WARNING)
    error_logger.addHandler(error_handler)
    error_logger.propagate = True

    # Performance log — rotated daily
    perf_logger = logging.getLogger("performance")
    perf_handler = logging.handlers.TimedRotatingFileHandler(
        log_path / "performance.log", when="midnight", backupCount=rotate_days, utc=True
    )
    perf_handler.setFormatter(file_fmt)
    perf_handler.setLevel(logging.INFO)
    perf_logger.addHandler(perf_handler)
    perf_logger.propagate = False

    # System log — rotated daily
    system_logger = logging.getLogger("system")
    system_handler = logging.handlers.TimedRotatingFileHandler(
        log_path / "system.log", when="midnight", backupCount=rotate_days, utc=True
    )
    system_handler.setFormatter(file_fmt)
    system_handler.setLevel(logging.DEBUG)
    system_logger.addHandler(system_handler)
    system_logger.propagate = False

    # Also capture all WARNING+ into error.log from any logger
    root_error_handler = logging.handlers.TimedRotatingFileHandler(
        log_path / "error.log", when="midnight", backupCount=rotate_days, utc=True
    )
    root_error_handler.setFormatter(file_fmt)
    root_error_handler.setLevel(logging.WARNING)
    root_logger.addHandler(root_error_handler)

    logging.info(f"Logging initialized for {strategy_id} at {log_path}")

    return {
        "trade": "trade",
        "error": "error",
        "performance": "performance",
        "system": "system",
    }


def log_trade(
    action: str,
    symbol: str,
    side: str,
    quantity: float,
    price: float,
    **kwargs
):
    """Log a trade event to the permanent trade log."""
    logger = logging.getLogger("trade")
    extras = " ".join(f"{k}={v}" for k, v in kwargs.items())
    logger.info(f"{action}\t{symbol}\t{side}\t{quantity}\t{price}\t{extras}")


def log_performance(metric: str, value: float, **kwargs):
    """Log a performance metric."""
    logger = logging.getLogger("performance")
    extras = " ".join(f"{k}={v}" for k, v in kwargs.items())
    logger.info(f"{metric}\t{value}\t{extras}")


def log_system(event: str, **kwargs):
    """Log a system event."""
    logger = logging.getLogger("system")
    extras = " ".join(f"{k}={v}" for k, v in kwargs.items())
    logger.info(f"{event}\t{extras}")
