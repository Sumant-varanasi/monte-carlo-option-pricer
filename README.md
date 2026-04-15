# 📈 Monte Carlo Option Pricing Engine

A full-stack **quantitative finance** application that prices European options using Monte Carlo simulation, validated against the Black–Scholes analytical solution.

**Python/NumPy backend** · **Flask REST API** · **Chart.js frontend**

---

## 🎯 What It Does

Simulates **50,000 stochastic stock price paths** under Geometric Brownian Motion (GBM), computes discounted payoffs at maturity, and estimates option prices using three Monte Carlo methods — comparing each against the exact Black–Scholes formula.

### Pricing Results (S₀=100, K=105, T=1yr, r=5%, σ=20%)

| Method | Price | Std Error | vs Black–Scholes |
|--------|-------|-----------|------------------|
| **Black–Scholes (exact)** | **8.0214** | — | — |
| MC Standard | 8.0076 | ±0.0590 | −0.17% |
| MC Antithetic | 8.0433 | ±0.0464 | +0.27% |
| MC Control Variate | 8.0216 | ±0.0269 | +0.003% |

### Variance Reduction

| Technique | Error Reduction |
|-----------|----------------|
| Antithetic Variates | **−21.4%** |
| Control Variates | **−54.4%** |

> Control variates achieves the accuracy of ~110k standard simulations using only 50k — a **2.2× speedup** for free.

---

## 🏗️ Architecture

```
┌─────────────────────────┐       HTTP/JSON         ┌──────────────────────────┐
│   Chart.js Frontend     │ ◄───────────────────►   │    Python Backend        │
│                         │   POST /api/simulate    │                          │
│  • Parameter controls   │   POST /api/sensitivity │  • NumPy vectorised GBM  │
│  • GBM path charts      │   GET  /api/health      │  • SciPy (norm.cdf)      │
│  • Convergence plots    │                         │  • Flask REST API        │
│  • Sensitivity analysis │                         │  • Black-Scholes exact   │
└─────────────────────────┘                         └──────────────────────────┘
```

---

## 📊 Features

- **GBM Path Simulation** — 30 sample trajectories visualised with mean path overlay
- **4 Pricing Methods** — Black–Scholes, Standard MC, Antithetic Variates MC, Control Variate MC
- **Payoff Distribution** — Histogram showing ITM vs OTM frequency with statistics
- **Convergence Analysis** — Watch all three MC methods converge toward the analytical price
- **Sensitivity Analysis** — BS price vs spot, volatility, and time-to-maturity
- **Variance Reduction Comparison** — Side-by-side effectiveness of each technique
- **Interactive Controls** — Adjust spot, strike, maturity, rate, vol, simulations in real-time
- **Call & Put Support** — Toggle between European call and put options

---

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/Sumant-varanasi/monte-carlo-option-pricer.git
cd monte-carlo-option-pricer/backend

# Install dependencies
pip install flask flask-cors numpy scipy

# Run
python app.py
```

Open **http://127.0.0.1:5000** in your browser. That's it.

---

## 📁 Project Structure

```
monte-carlo-option-pricer/
├── backend/
│   ├── engine.py           # Core simulation engine (NumPy vectorised)
│   ├── app.py              # Flask API server + embedded frontend
│   └── requirements.txt
├── frontend/
│   └── index.html          # Standalone Chart.js dashboard
└── README.md
```

### Backend Modules (engine.py)

- `black_scholes()` — Analytical European option pricing
- `simulate_gbm_paths()` — GBM path generation for visualisation
- `mc_standard()` — Standard Monte Carlo with convergence tracking
- `mc_antithetic()` — Antithetic variates (pair Z with −Z)
- `mc_control_variate()` — Control variates using known E[S(T)]
- `sensitivity_*()` — BS price vs spot, vol, time-to-maturity

---

## 🔬 Key Concepts

| Concept | Implementation |
|---------|---------------|
| **GBM simulation** | S(T) = S₀ · exp[(r − σ²/2)T + σ√T · Z] vectorised over all paths |
| **Risk-neutral pricing** | Drift = r − σ²/2 (not real-world μ) |
| **Itô correction** | The −σ²/2 term prevents systematic overestimation |
| **Antithetic variates** | For each Z, also simulate −Z → negatively correlated pairs |
| **Control variates** | Adjust using known E[S(T)] = S₀·eʳᵀ to correct estimate |
| **Convergence** | Running average → BS price as N → ∞ (Law of Large Numbers) |

---

## ⚡ Performance

All computation is **vectorised with NumPy** — no Python loops over simulations:

- 50,000 simulations across 3 MC methods: **~90ms**
- 30 GBM paths (252 steps each): **~5ms**
- Sensitivity analysis (150 BS evaluations): **~2ms**
- **Total backend response: < 120ms**

---

## 📚 References

- Glasserman, *Monte Carlo Methods in Financial Engineering*
- Hull, *Options, Futures, and Other Derivatives*
- Black & Scholes (1973), *The Pricing of Options and Corporate Liabilities*

---

## 🛠️ Tech Stack

**Backend:** Python 3 · Flask · NumPy · SciPy
**Frontend:** Chart.js · Vanilla JavaScript · HTML/CSS
**No build step** — pure CDN dependencies, single `python app.py` to run

---

## 📄 License

MIT
