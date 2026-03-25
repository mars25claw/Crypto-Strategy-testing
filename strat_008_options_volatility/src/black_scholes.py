"""Full Black-Scholes implementation with Greeks and IV solver.

Provides European option pricing, all first-order Greeks, and
Newton-Raphson implied volatility solver from market prices.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

from scipy.stats import norm

logger = logging.getLogger(__name__)

# Default risk-free rate (annualized)
DEFAULT_RISK_FREE_RATE = 0.05


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class OptionGreeks:
    """Container for all Black-Scholes Greeks."""
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0       # per day (annualized theta / 365)
    vega: float = 0.0        # per 1% IV move
    rho: float = 0.0


@dataclass
class BSResult:
    """Full result from Black-Scholes pricing."""
    price: float
    greeks: OptionGreeks
    d1: float
    d2: float
    intrinsic: float
    time_value: float


# ---------------------------------------------------------------------------
# Core Black-Scholes functions
# ---------------------------------------------------------------------------

def _d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Calculate d1 in the Black-Scholes formula.

    d1 = [ln(S/K) + (r + sigma^2/2) * T] / (sigma * sqrt(T))
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    return (math.log(S / K) + (r + sigma ** 2 / 2.0) * T) / (sigma * math.sqrt(T))


def _d2(d1_val: float, sigma: float, T: float) -> float:
    """Calculate d2 = d1 - sigma * sqrt(T)."""
    if T <= 0 or sigma <= 0:
        return 0.0
    return d1_val - sigma * math.sqrt(T)


def call_price(
    S: float,
    K: float,
    T: float,
    r: float = DEFAULT_RISK_FREE_RATE,
    sigma: float = 0.5,
) -> float:
    """European call option price via Black-Scholes.

    Parameters
    ----------
    S : float
        Current underlying price.
    K : float
        Strike price.
    T : float
        Time to expiration in years (e.g. 7/365 for 7 days).
    r : float
        Risk-free rate (annualized).
    sigma : float
        Implied volatility (annualized, decimal e.g. 0.60 for 60%).

    Returns
    -------
    float
        Theoretical call price.
    """
    if T <= 0:
        return max(S - K, 0.0)
    if sigma <= 0:
        return max(S - K * math.exp(-r * T), 0.0)

    d1_val = _d1(S, K, T, r, sigma)
    d2_val = _d2(d1_val, sigma, T)

    price = S * norm.cdf(d1_val) - K * math.exp(-r * T) * norm.cdf(d2_val)
    return max(price, 0.0)


def put_price(
    S: float,
    K: float,
    T: float,
    r: float = DEFAULT_RISK_FREE_RATE,
    sigma: float = 0.5,
) -> float:
    """European put option price via Black-Scholes.

    Put = K * e^(-rT) * N(-d2) - S * N(-d1)
    """
    if T <= 0:
        return max(K - S, 0.0)
    if sigma <= 0:
        return max(K * math.exp(-r * T) - S, 0.0)

    d1_val = _d1(S, K, T, r, sigma)
    d2_val = _d2(d1_val, sigma, T)

    price = K * math.exp(-r * T) * norm.cdf(-d2_val) - S * norm.cdf(-d1_val)
    return max(price, 0.0)


# ---------------------------------------------------------------------------
# Greeks
# ---------------------------------------------------------------------------

def delta(
    S: float, K: float, T: float, r: float, sigma: float,
    option_type: str = "call",
) -> float:
    """Option delta: dC/dS for calls, dP/dS for puts."""
    if T <= 0 or sigma <= 0:
        if option_type == "call":
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0

    d1_val = _d1(S, K, T, r, sigma)
    if option_type == "call":
        return norm.cdf(d1_val)
    else:
        return norm.cdf(d1_val) - 1.0


def gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Option gamma: d^2C/dS^2 (same for calls and puts)."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0

    d1_val = _d1(S, K, T, r, sigma)
    return norm.pdf(d1_val) / (S * sigma * math.sqrt(T))


def theta(
    S: float, K: float, T: float, r: float, sigma: float,
    option_type: str = "call",
) -> float:
    """Option theta: dC/dT (per day).

    Returns a negative value for long options (time decay).
    """
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0

    d1_val = _d1(S, K, T, r, sigma)
    d2_val = _d2(d1_val, sigma, T)
    sqrt_T = math.sqrt(T)

    # First term: common to both calls and puts
    first_term = -(S * norm.pdf(d1_val) * sigma) / (2.0 * sqrt_T)

    if option_type == "call":
        theta_annual = first_term - r * K * math.exp(-r * T) * norm.cdf(d2_val)
    else:
        theta_annual = first_term + r * K * math.exp(-r * T) * norm.cdf(-d2_val)

    # Convert to per-day
    return theta_annual / 365.0


def vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Option vega: dC/dsigma per 1% IV move (same for calls and puts).

    Returns the dollar change for a 1 percentage point change in IV.
    """
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0

    d1_val = _d1(S, K, T, r, sigma)
    # Vega per 1% = S * N'(d1) * sqrt(T) / 100
    return S * norm.pdf(d1_val) * math.sqrt(T) / 100.0


def rho(
    S: float, K: float, T: float, r: float, sigma: float,
    option_type: str = "call",
) -> float:
    """Option rho: dC/dr per 1% rate move."""
    if T <= 0 or sigma <= 0:
        return 0.0

    d1_val = _d1(S, K, T, r, sigma)
    d2_val = _d2(d1_val, sigma, T)

    if option_type == "call":
        return K * T * math.exp(-r * T) * norm.cdf(d2_val) / 100.0
    else:
        return -K * T * math.exp(-r * T) * norm.cdf(-d2_val) / 100.0


# ---------------------------------------------------------------------------
# Full pricing with Greeks
# ---------------------------------------------------------------------------

def price_option(
    S: float,
    K: float,
    T: float,
    r: float = DEFAULT_RISK_FREE_RATE,
    sigma: float = 0.5,
    option_type: str = "call",
) -> BSResult:
    """Price a European option and compute all Greeks.

    Parameters
    ----------
    S : underlying price
    K : strike price
    T : time to expiration in years
    r : risk-free rate (annualized)
    sigma : implied volatility (annualized, decimal)
    option_type : "call" or "put"

    Returns
    -------
    BSResult with price and all Greeks.
    """
    if option_type == "call":
        opt_price = call_price(S, K, T, r, sigma)
        intrinsic = max(S - K, 0.0)
    else:
        opt_price = put_price(S, K, T, r, sigma)
        intrinsic = max(K - S, 0.0)

    d1_val = _d1(S, K, T, r, sigma) if T > 0 and sigma > 0 else 0.0
    d2_val = _d2(d1_val, sigma, T) if T > 0 and sigma > 0 else 0.0

    greeks = OptionGreeks(
        delta=delta(S, K, T, r, sigma, option_type),
        gamma=gamma(S, K, T, r, sigma),
        theta=theta(S, K, T, r, sigma, option_type),
        vega=vega(S, K, T, r, sigma),
        rho=rho(S, K, T, r, sigma, option_type),
    )

    return BSResult(
        price=opt_price,
        greeks=greeks,
        d1=d1_val,
        d2=d2_val,
        intrinsic=intrinsic,
        time_value=opt_price - intrinsic,
    )


# ---------------------------------------------------------------------------
# Implied Volatility Solver (Newton-Raphson)
# ---------------------------------------------------------------------------

def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float = DEFAULT_RISK_FREE_RATE,
    option_type: str = "call",
    max_iterations: int = 100,
    tolerance: float = 1e-8,
    initial_guess: float = 0.5,
) -> Optional[float]:
    """Solve for implied volatility using Newton-Raphson method.

    Parameters
    ----------
    market_price : float
        Observed market price of the option.
    S : float
        Underlying price.
    K : float
        Strike price.
    T : float
        Time to expiration in years.
    r : float
        Risk-free rate.
    option_type : str
        "call" or "put".
    max_iterations : int
        Maximum Newton-Raphson iterations.
    tolerance : float
        Convergence tolerance on price difference.
    initial_guess : float
        Initial IV guess (annualized decimal).

    Returns
    -------
    float or None
        Implied volatility if solver converges, else None.
    """
    if market_price <= 0 or T <= 0 or S <= 0 or K <= 0:
        return None

    # Check that market price is within valid bounds
    if option_type == "call":
        intrinsic = max(S - K * math.exp(-r * T), 0.0)
        upper_bound = S
    else:
        intrinsic = max(K * math.exp(-r * T) - S, 0.0)
        upper_bound = K * math.exp(-r * T)

    if market_price > upper_bound * 1.01:
        logger.debug(
            "Market price %.4f exceeds theoretical upper bound %.4f",
            market_price, upper_bound,
        )
        return None

    sigma = initial_guess

    for i in range(max_iterations):
        if sigma <= 0.001:
            sigma = 0.001

        if option_type == "call":
            theo_price = call_price(S, K, T, r, sigma)
        else:
            theo_price = put_price(S, K, T, r, sigma)

        diff = theo_price - market_price

        if abs(diff) < tolerance:
            return sigma

        # Vega for Newton step (raw vega, not per-percent)
        d1_val = _d1(S, K, T, r, sigma)
        raw_vega = S * norm.pdf(d1_val) * math.sqrt(T)

        if raw_vega < 1e-12:
            # Vega too small, try bisection fallback
            logger.debug("Vega too small at sigma=%.4f, falling back", sigma)
            return _bisection_iv(market_price, S, K, T, r, option_type)

        # Newton step
        sigma = sigma - diff / raw_vega

        # Bound sigma
        if sigma < 0.001:
            sigma = 0.001
        if sigma > 10.0:
            sigma = 10.0

    logger.debug(
        "IV solver did not converge after %d iterations (last sigma=%.4f)",
        max_iterations, sigma,
    )
    return _bisection_iv(market_price, S, K, T, r, option_type)


def _bisection_iv(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str,
    lo: float = 0.01,
    hi: float = 5.0,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
) -> Optional[float]:
    """Bisection fallback for IV solving when Newton-Raphson fails."""
    pricer = call_price if option_type == "call" else put_price

    p_lo = pricer(S, K, T, r, lo)
    p_hi = pricer(S, K, T, r, hi)

    if market_price < p_lo or market_price > p_hi:
        return None

    for _ in range(max_iterations):
        mid = (lo + hi) / 2.0
        p_mid = pricer(S, K, T, r, mid)

        if abs(p_mid - market_price) < tolerance:
            return mid

        if p_mid > market_price:
            hi = mid
        else:
            lo = mid

    return (lo + hi) / 2.0


# ---------------------------------------------------------------------------
# Convenience: compute portfolio Greeks
# ---------------------------------------------------------------------------

def aggregate_greeks(
    positions: list[dict],
) -> OptionGreeks:
    """Aggregate Greeks across a portfolio of option positions.

    Each dict in *positions* should have:
        - quantity: int (positive=long, negative=short)
        - delta, gamma, theta, vega: float (per-unit Greeks)
    """
    agg = OptionGreeks()
    for pos in positions:
        qty = pos.get("quantity", 0)
        agg.delta += qty * pos.get("delta", 0.0)
        agg.gamma += qty * pos.get("gamma", 0.0)
        agg.theta += qty * pos.get("theta", 0.0)
        agg.vega += qty * pos.get("vega", 0.0)
        agg.rho += qty * pos.get("rho", 0.0)
    return agg
