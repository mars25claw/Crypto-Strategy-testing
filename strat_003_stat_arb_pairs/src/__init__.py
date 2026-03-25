"""STRAT-003: Statistical Arbitrage Pairs Trading Bot.

Identifies cointegrated cryptocurrency pairs and trades temporary
divergences from statistical equilibrium using Z-score signals on
1-hour spreads. Market-neutral by construction (long one leg, short
the other), with daily Engle-Granger requalification.
"""

__version__ = "1.0.0"
STRATEGY_ID = "STRAT-003"
STRATEGY_NAME = "Statistical Arbitrage Pairs Trading"
