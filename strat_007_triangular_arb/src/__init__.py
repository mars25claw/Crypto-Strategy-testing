"""STRAT-007: Cross-Exchange & Triangular Arbitrage Strategy.

Exploits price discrepancies between Binance spot and futures markets (Mode A)
and triangular pricing inconsistencies across related trading pairs (Mode B).

Modules:
    strategy          -- Core evaluation logic (Mode A & B)
    opportunity_scanner -- Real-time bookTicker/depth processing
    execution         -- Ultra-low-latency order execution engine
    risk_manager      -- Strategy-specific risk & circuit breakers
    wallet_manager    -- Spot/futures balance allocation
    strategy_metrics  -- Section 10.2/10.3 metrics & go-live criteria
    dashboard         -- Real-time dashboard integration
"""

__version__ = "1.1.0"
STRATEGY_ID = "STRAT-007"
STRATEGY_NAME = "Cross-Exchange Triangular Arbitrage"
