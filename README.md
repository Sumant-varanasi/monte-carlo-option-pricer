# Monte Carlo Option Pricing Engine

A full-stack quantitative finance application with a **Python/NumPy backend** and **React frontend**.

## Architecture

```
┌─────────────────────────┐       HTTP/JSON        ┌──────────────────────────┐
│     React Frontend      │ ◄───────────────────► │    Python Backend         │
│                         │   POST /api/simulate    │                          │
│  • Parameter controls   │   POST /api/sensitivity │  • NumPy vectorised GBM  │
│  • Recharts viz         │   GET  /api/health      │  • SciPy (norm.cdf)      │
│  • Convergence plots    │                         │  • Flask REST API        │
│  • Sensitivity analysis │                         │  • Black-Scholes exact   │
└─────────────────────────┘                         └──────────────────────────┘
```

## Backend (Python)

All computation is in `backend/engine.py` — fully vectorised with NumPy:

- **`black_scholes()`** — Analytical European option pricing via the BS formula
- **`simulate_gbm_paths()`** — Generate N price trajectories under GBM
- **`mc_standard()`** — Standard Monte Carlo with convergence tracking
- **`mc_antithetic()`** — Antithetic variates (pair Z with −Z)
- **`mc_control_variate()`** — Control variates using known E[S(T)]
- **`sensitivity_*()`** — BS price vs spot, volatility, time-to-maturity
- **`build_histogram()`** — Payoff distribution binning

The Flask server (`backend/app.py`) exposes these as REST endpoints.

### Setup

```bash
cd backend
pip install -r requirements.txt
python app.py
```

Server runs on `http://localhost:5000`.

### API Endpoints

#### `POST /api/simulate`

Request:
```json
{
  "S0": 100,
  "K": 105,
  "T": 1.0,
  "r": 0.05,
  "sigma": 0.2,
  "n_sim": 50000,
  "n_paths": 30,
  "option_type": "call"
}
```

Response includes: `bs_price`, `mc_standard`, `mc_antithetic`, `mc_control`,
`gbm_paths`, `histogram`, `convergence`, `stats`, `elapsed_ms`.

#### `POST /api/sensitivity`

Same request body. Returns BS price sensitivity data for spot, volatility, and time.

#### `GET /api/health`

Returns `{"status": "ok"}`.

## Frontend (React)

The frontend (`frontend/monte_carlo_frontend.jsx`) is a React component that:

1. Sends parameters to the Python backend via fetch
2. Renders the results with Recharts
3. Provides interactive parameter controls
4. Shows four tabbed views: GBM Paths, Payoff Distribution, Convergence, Sensitivity

### Dependencies (via CDN in artifact)
- React (hooks)
- Recharts
- Lodash
- Google Fonts: IBM Plex Mono + Newsreader

## Key Concepts

| Concept | Implementation |
|---------|---------------|
| GBM simulation | `np.exp(drift + vol * Z)` — vectorised over all paths |
| Risk-neutral pricing | Drift = `r - σ²/2` (not μ) |
| Itô correction | The `- σ²/2` term in the drift |
| Box-Muller (JS) / `rng.standard_normal` (Python) | Normal random number generation |
| Antithetic variates | For each Z, also simulate −Z |
| Control variates | Adjust using known E[S(T)] = S₀eʳᵀ |
| Convergence | Running average of discounted payoffs → BS price |

## Performance

With NumPy vectorisation, 500,000 simulations across all three MC methods
typically complete in **< 500ms** on the backend.
