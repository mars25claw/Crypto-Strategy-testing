"""STRAT-002: Funding Rate Arbitrage (Cash-and-Carry) Strategy.

Captures funding rate premium by holding delta-neutral positions:
LONG spot + SHORT perpetual futures on the same asset.
"""

__version__ = "1.0.0"
STRATEGY_ID = "STRAT-002"
STRATEGY_NAME = "Funding Rate Arbitrage"
