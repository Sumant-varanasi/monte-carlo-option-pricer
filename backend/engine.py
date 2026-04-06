# engine.py — core pricing engine
# all the heavy math lives here, numpy does the grunt work

import numpy as np
from scipy.stats import norm
from dataclasses import dataclass
from typing import Literal


def black_scholes(S0, K, T, r, sigma, option_type="call"):
    """Closed-form BS price. Nothing fancy, just the textbook formula."""
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        return float(S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    else:
        return float(K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1))


def simulate_gbm_paths(S0, r, sigma, T, n_steps=252, n_paths=30, seed=None):
    """Generate sample GBM trajectories for the frontend chart."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps

    # risk-neutral drift with Ito correction (the -sigma^2/2 matters, trust me)
    drift = (r - 0.5 * sigma**2) * dt
    vol = sigma * np.sqrt(dt)

    Z = rng.standard_normal((n_paths, n_steps))
    log_returns = drift + vol * Z

    # cumsum trick to avoid looping over timesteps
    log_prices = np.concatenate(
        [np.zeros((n_paths, 1)), np.cumsum(log_returns, axis=1)], axis=1
    )
    paths = S0 * np.exp(log_prices)

    time_grid = np.linspace(0, T, n_steps + 1)
    mean_path = paths.mean(axis=0)

    return {
        "paths": paths.tolist(),
        "time_grid": time_grid.tolist(),
        "mean_path": mean_path.tolist(),
    }


@dataclass
class MCResult:
    price: float
    stderr: float
    payoffs: list[float]
    convergence: list[dict]

    def to_dict(self):
        return {
            "price": self.price,
            "stderr": self.stderr,
            "payoffs": self.payoffs,
            "convergence": self.convergence,
        }


def _convergence_tracker(cumsum, cumsum2, disc, n_sim, n_points=500):
    """Sample the running average at ~500 points for the convergence plot."""
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
        var = max(mean2 - mean**2, 0.0)  # clamp to avoid floating point nonsense
        se = np.sqrt(var / n) * disc if n > 1 else 0.0
        conv.append({"n": n, "price": float(mean * disc), "stderr": float(se)})
    return conv


def mc_standard(S0, K, T, r, sigma, n_sim=50000, option_type="call", seed=None):
    """Plain vanilla MC — generate terminal prices, compute payoffs, average."""
    rng = np.random.default_rng(seed)
    disc = np.exp(-r * T)

    # Ito-corrected drift under Q measure
    drift = (r - 0.5 * sigma**2) * T
    vol = sigma * np.sqrt(T)

    Z = rng.standard_normal(n_sim)
    ST = S0 * np.exp(drift + vol * Z)

    if option_type == "call":
        payoffs = np.maximum(ST - K, 0.0)
    else:
        payoffs = np.maximum(K - ST, 0.0)

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


def mc_antithetic(S0, K, T, r, sigma, n_sim=50000, option_type="call", seed=None):
    """
    Antithetic variates — for every path Z, also run -Z.
    Negatively correlated pairs cancel out some noise for free.
    """
    rng = np.random.default_rng(seed)
    disc = np.exp(-r * T)
    drift = (r - 0.5 * sigma**2) * T
    vol = sigma * np.sqrt(T)
    half = n_sim // 2

    Z = rng.standard_normal(half)
    ST_pos = S0 * np.exp(drift + vol * Z)
    ST_neg = S0 * np.exp(drift + vol * (-Z))  # the mirror path

    if option_type == "call":
        pay_pos = np.maximum(ST_pos - K, 0.0)
        pay_neg = np.maximum(ST_neg - K, 0.0)
    else:
        pay_pos = np.maximum(K - ST_pos, 0.0)
        pay_neg = np.maximum(K - ST_neg, 0.0)

    # average each pair — this is where the variance reduction kicks in
    avg_payoffs = (pay_pos + pay_neg) / 2.0

    cumsum = np.cumsum(avg_payoffs)
    cumsum2 = np.cumsum(avg_payoffs**2)
    convergence = _convergence_tracker(cumsum, cumsum2, disc, half)
    # each pair uses 2 draws, so rescale for the x-axis
    for c in convergence:
        c["n"] = c["n"] * 2

    price = float(avg_payoffs.mean() * disc)
    var = float(avg_payoffs.var())
    stderr = float(np.sqrt(var / half) * disc)

    # interleave both sides for the histogram
    all_payoffs = np.empty(n_sim)
    all_payoffs[0::2] = pay_pos * disc
    all_payoffs[1::2] = pay_neg * disc

    return MCResult(
        price=price,
        stderr=stderr,
        payoffs=all_payoffs.tolist(),
        convergence=convergence,
    )


def mc_control_variate(S0, K, T, r, sigma, n_sim=50000, option_type="call", seed=None):
    """
    Control variate using S(T) — we know E[S(T)] = S0*exp(rT) analytically,
    so we can use that to correct the MC estimate. Biggest variance reduction
    of the three methods, typically 50%+ improvement.
    """
    rng = np.random.default_rng(seed)
    disc = np.exp(-r * T)
    drift = (r - 0.5 * sigma**2) * T
    vol = sigma * np.sqrt(T)
    expected_ST = S0 * np.exp(r * T)  # known under Q

    Z = rng.standard_normal(n_sim)
    ST = S0 * np.exp(drift + vol * Z)

    if option_type == "call":
        raw_payoffs = np.maximum(ST - K, 0.0)
    else:
        raw_payoffs = np.maximum(K - ST, 0.0)

    # OLS beta — how much does the payoff move with the stock?
    cov_matrix = np.cov(raw_payoffs, ST)
    beta = cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] > 0 else 0.0

    # if simulated stock prices ran high, correct the option price downward
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


# --- sensitivity sweeps for the frontend charts ---

def sensitivity_spot(S0, K, T, r, sigma, option_type="call"):
    spots = np.linspace(S0 * 0.5, S0 * 1.5, 50)
    prices = [black_scholes(s, K, T, r, sigma, option_type) for s in spots]
    if option_type == "call":
        intrinsic = [max(s - K, 0) for s in spots]
    else:
        intrinsic = [max(K - s, 0) for s in spots]
    return {"spots": spots.tolist(), "prices": prices, "intrinsic": intrinsic}


def sensitivity_vol(S0, K, T, r, sigma, option_type="call"):
    vols = np.linspace(0.05, 0.80, 50)
    prices = [black_scholes(S0, K, T, r, v, option_type) for v in vols]
    return {"vols": (vols * 100).tolist(), "prices": prices}


def sensitivity_time(S0, K, T, r, sigma, option_type="call"):
    times = np.linspace(0.01, T, 50)
    prices = [black_scholes(S0, K, t, r, sigma, option_type) for t in times]
    return {"times": times.tolist(), "prices": prices}


def build_histogram(payoffs, bins=60):
    """Bin the payoffs for the distribution chart. Handles the OTM spike at zero."""
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
