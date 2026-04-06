"""
Monte Carlo Option Pricing Engine
==================================
Core simulation module using NumPy for vectorised stochastic computation.

Implements:
  - Geometric Brownian Motion (GBM) path simulation
  - European option pricing via Monte Carlo (standard, antithetic, control variate)
  - Black-Scholes analytical pricing
  - Sensitivity analysis (price vs spot, vol, time)
  - Convergence tracking for all MC methods

All heavy computation is vectorised with NumPy — no Python loops over simulations.
"""

import numpy as np
from scipy.stats import norm
from dataclasses import dataclass, field
from typing import Literal

# ──────────────────────────────────────────────────────────────
# Black-Scholes Analytical Pricing
# ──────────────────────────────────────────────────────────────

def black_scholes(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: Literal["call", "put"] = "call",
) -> float:
    """
    Compute the Black-Scholes price for a European option.

    Parameters
    ----------
    S0    : Current spot price
    K     : Strike price
    T     : Time to maturity (years)
    r     : Risk-free interest rate (annualised)
    sigma : Volatility (annualised)
    option_type : 'call' or 'put'

    Returns
    -------
    Analytical option price.
    """
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        return float(S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    else:
        return float(K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1))


# ──────────────────────────────────────────────────────────────
# GBM Path Simulation (for visualisation)
# ──────────────────────────────────────────────────────────────

def simulate_gbm_paths(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int = 252,
    n_paths: int = 30,
    seed: int | None = None,
) -> dict:
    """
    Simulate GBM price paths for visualisation.

    Returns dict with:
      - paths: (n_paths, n_steps+1) array of price trajectories
      - time_grid: (n_steps+1,) array of time points
      - mean_path: (n_steps+1,) array of mean across paths
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    drift = (r - 0.5 * sigma**2) * dt
    vol = sigma * np.sqrt(dt)

    # Vectorised: generate all random increments at once
    Z = rng.standard_normal((n_paths, n_steps))
    log_returns = drift + vol * Z  # (n_paths, n_steps)

    # Cumulative sum of log-returns, prepend 0 for initial price
    log_prices = np.concatenate(
        [np.zeros((n_paths, 1)), np.cumsum(log_returns, axis=1)], axis=1
    )
    paths = S0 * np.exp(log_prices)  # (n_paths, n_steps+1)

    time_grid = np.linspace(0, T, n_steps + 1)
    mean_path = paths.mean(axis=0)

    return {
        "paths": paths.tolist(),
        "time_grid": time_grid.tolist(),
        "mean_path": mean_path.tolist(),
    }


# ──────────────────────────────────────────────────────────────
# Monte Carlo Pricing Methods
# ──────────────────────────────────────────────────────────────

@dataclass
class MCResult:
    """Result container for a Monte Carlo pricing run."""
    price: float
    stderr: float
    payoffs: list[float]
    convergence: list[dict]  # [{n, price, stderr}, ...]

    def to_dict(self) -> dict:
        return {
            "price": self.price,
            "stderr": self.stderr,
            "payoffs": self.payoffs,
            "convergence": self.convergence,
        }


def _convergence_tracker(
    cumsum: np.ndarray,
    cumsum2: np.ndarray,
    disc: float,
    n_sim: int,
    n_points: int = 500,
) -> list[dict]:
    """
    Build convergence data from cumulative sums.
    Sample at ~n_points evenly spaced indices.
    """
    indices = np.unique(
        np.concatenate([
            np.linspace(0, n_sim - 1, min(n_points, n_sim), dtype=int),
            [n_sim - 1],
        ])
    )
    conv = []
    for i in indices:
        n = int(i + 1)
        mean = cumsum[i] / n
        mean2 = cumsum2[i] / n
        var = max(mean2 - mean**2, 0.0)
        se = np.sqrt(var / n) * disc if n > 1 else 0.0
        conv.append({"n": n, "price": float(mean * disc), "stderr": float(se)})
    return conv


def mc_standard(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_sim: int = 50000,
    option_type: str = "call",
    seed: int | None = None,
) -> MCResult:
    """Standard Monte Carlo pricing — fully vectorised."""
    rng = np.random.default_rng(seed)
    disc = np.exp(-r * T)
    drift = (r - 0.5 * sigma**2) * T
    vol = sigma * np.sqrt(T)

    Z = rng.standard_normal(n_sim)
    ST = S0 * np.exp(drift + vol * Z)

    if option_type == "call":
        payoffs = np.maximum(ST - K, 0.0)
    else:
        payoffs = np.maximum(K - ST, 0.0)

    # Convergence tracking
    cumsum = np.cumsum(payoffs)
    cumsum2 = np.cumsum(payoffs**2)
    convergence = _convergence_tracker(cumsum, cumsum2, disc, n_sim)

    price = float(payoffs.mean() * disc)
    var = float(payoffs.var())
    stderr = float(np.sqrt(var / n_sim) * disc)

    return MCResult(
        price=price,
        stderr=stderr,
        payoffs=(payoffs * disc).tolist(),
        convergence=convergence,
    )


def mc_antithetic(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_sim: int = 50000,
    option_type: str = "call",
    seed: int | None = None,
) -> MCResult:
    """
    Antithetic variates Monte Carlo.
    For each Z, also use -Z — paired paths are negatively correlated.
    """
    rng = np.random.default_rng(seed)
    disc = np.exp(-r * T)
    drift = (r - 0.5 * sigma**2) * T
    vol = sigma * np.sqrt(T)
    half = n_sim // 2

    Z = rng.standard_normal(half)
    ST_pos = S0 * np.exp(drift + vol * Z)
    ST_neg = S0 * np.exp(drift + vol * (-Z))

    if option_type == "call":
        pay_pos = np.maximum(ST_pos - K, 0.0)
        pay_neg = np.maximum(ST_neg - K, 0.0)
    else:
        pay_pos = np.maximum(K - ST_pos, 0.0)
        pay_neg = np.maximum(K - ST_neg, 0.0)

    # Average of each antithetic pair
    avg_payoffs = (pay_pos + pay_neg) / 2.0

    cumsum = np.cumsum(avg_payoffs)
    cumsum2 = np.cumsum(avg_payoffs**2)
    convergence = _convergence_tracker(cumsum, cumsum2, disc, half)
    # Rescale n in convergence to reflect total sims (each pair = 2 sims)
    for c in convergence:
        c["n"] = c["n"] * 2

    price = float(avg_payoffs.mean() * disc)
    var = float(avg_payoffs.var())
    stderr = float(np.sqrt(var / half) * disc)

    # Full payoff list (both sides interleaved)
    all_payoffs = np.empty(n_sim)
    all_payoffs[0::2] = pay_pos * disc
    all_payoffs[1::2] = pay_neg * disc

    return MCResult(
        price=price,
        stderr=stderr,
        payoffs=all_payoffs.tolist(),
        convergence=convergence,
    )


def mc_control_variate(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_sim: int = 50000,
    option_type: str = "call",
    seed: int | None = None,
) -> MCResult:
    """
    Control variate Monte Carlo.
    Uses S(T) as the control — E[S(T)] = S0 * exp(rT) is known analytically.
    """
    rng = np.random.default_rng(seed)
    disc = np.exp(-r * T)
    drift = (r - 0.5 * sigma**2) * T
    vol = sigma * np.sqrt(T)
    expected_ST = S0 * np.exp(r * T)

    Z = rng.standard_normal(n_sim)
    ST = S0 * np.exp(drift + vol * Z)

    if option_type == "call":
        raw_payoffs = np.maximum(ST - K, 0.0)
    else:
        raw_payoffs = np.maximum(K - ST, 0.0)

    # Compute optimal beta via covariance
    cov_matrix = np.cov(raw_payoffs, ST)
    beta = cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] > 0 else 0.0

    # Adjusted payoffs
    adjusted = raw_payoffs - beta * (ST - expected_ST)

    cumsum = np.cumsum(adjusted)
    cumsum2 = np.cumsum(adjusted**2)
    convergence = _convergence_tracker(cumsum, cumsum2, disc, n_sim)

    price = float(adjusted.mean() * disc)
    var = float(adjusted.var())
    stderr = float(np.sqrt(var / n_sim) * disc)

    return MCResult(
        price=price,
        stderr=stderr,
        payoffs=(adjusted * disc).tolist(),
        convergence=convergence,
    )


# ──────────────────────────────────────────────────────────────
# Sensitivity Analysis
# ──────────────────────────────────────────────────────────────

def sensitivity_spot(
    S0: float, K: float, T: float, r: float, sigma: float, option_type: str = "call"
) -> dict:
    """BS price as a function of spot price."""
    spots = np.linspace(S0 * 0.5, S0 * 1.5, 50)
    prices = [black_scholes(s, K, T, r, sigma, option_type) for s in spots]
    if option_type == "call":
        intrinsic = [max(s - K, 0) for s in spots]
    else:
        intrinsic = [max(K - s, 0) for s in spots]
    return {
        "spots": spots.tolist(),
        "prices": prices,
        "intrinsic": intrinsic,
    }


def sensitivity_vol(
    S0: float, K: float, T: float, r: float, sigma: float, option_type: str = "call"
) -> dict:
    """BS price as a function of volatility."""
    vols = np.linspace(0.05, 0.80, 50)
    prices = [black_scholes(S0, K, T, r, v, option_type) for v in vols]
    return {"vols": (vols * 100).tolist(), "prices": prices}


def sensitivity_time(
    S0: float, K: float, T: float, r: float, sigma: float, option_type: str = "call"
) -> dict:
    """BS price as a function of time-to-maturity."""
    times = np.linspace(0.01, T, 50)
    prices = [black_scholes(S0, K, t, r, sigma, option_type) for t in times]
    return {"times": times.tolist(), "prices": prices}


# ──────────────────────────────────────────────────────────────
# Histogram Builder
# ──────────────────────────────────────────────────────────────

def build_histogram(payoffs: list[float], bins: int = 60) -> list[dict]:
    """Build histogram data for the payoff distribution."""
    arr = np.array(payoffs)
    non_zero = arr[arr > 0]

    if len(non_zero) == 0:
        return [{"midpoint": 0.0, "range": "0 (OTM)", "count": int(len(arr))}]

    lo = float(non_zero.min() * 0.9)
    hi = float(non_zero.max() * 1.1)
    counts, edges = np.histogram(non_zero, bins=bins, range=(lo, hi))

    zero_count = int(np.sum(arr <= 0))
    hist = [{"midpoint": 0.0, "range": "0 (OTM)", "count": zero_count}]
    for i in range(len(counts)):
        mid = (edges[i] + edges[i + 1]) / 2
        hist.append({
            "midpoint": float(mid),
            "range": f"{edges[i]:.1f}",
            "count": int(counts[i]),
        })
    return hist
