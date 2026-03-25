"""STRAT-005 Grid Trading Bot.

Captures profit from price oscillation within a defined range by placing
a mesh of buy and sell limit orders at geometric (or arithmetic) intervals.
Fill-triggered order placement creates a self-sustaining grid.
"""

__version__ = "1.0.0"
STRATEGY_ID = "STRAT-005"
STRATEGY_NAME = "Grid Trading"
