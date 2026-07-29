# Portfolio Optimizer

A constrained portfolio optimization engine built from scratch in Python (numpy/scipy for all core computation, scikit-learn used only for the standard Ledoit-Wolf shrinkage estimator), completing a fifth project alongside the Options Pricing Engine, VaR Calculator, Statistical Arbitrage BackTester, and HF Market Simulator by asking a construction question none of the other four address: **given a set of assets, what is the best possible portfolio — and how does "best" actually change once real-world constraints (leverage, turnover, sector limits) are applied?**

**Live app:** [portfoliooptimisation.streamlit.app](https://portfoliooptimisation.streamlit.app/)

---

## Overview

The other four projects each ask a different core quant question — what should an asset be worth, how much could a portfolio lose, can you exploit a mispricing, what happens when a strategy actually tries to trade. This project asks a construction question: starting from a set of assets' historical returns, what portfolio maximizes risk-adjusted return, and what happens to that answer once it has to survive contact with the constraints a real fund actually operates under? This project builds the unconstrained closed-form solution first, adds constraints one at a time, stress-tests the underlying statistics for stability, and finally backtests both versions walk-forward to see which one actually performs better once it's traded for real.

---

## 1. Unconstrained Tangency (Max-Sharpe) Portfolio

### What it does
Computes the exact, closed-form maximum-Sharpe-ratio portfolio given a vector of expected returns (μ) and a covariance matrix (Σ):

$$w \propto \Sigma^{-1}(\mu - r_f\mathbf{1})$$

normalized so weights sum to 1. No numerical optimizer is needed — this is a direct matrix inversion, the same "instantaneous, no-simulation" role Black-Scholes plays in the Options Pricing Engine.

### Why it's important
This is the theoretical ceiling every other module in this project is measured against. Every constraint added later necessarily gives up some of this Sharpe ratio — the whole point of the project is to quantify exactly how much, and why.

### Validation
Every result was checked against a fundamental guarantee: the tangency portfolio's Sharpe ratio must equal or exceed the Sharpe ratio of every individual asset in the universe, since a 100%-single-asset holding is itself a valid portfolio. This check passed on final validation (portfolio Sharpe 2.66 vs. best individual asset — JPM — at 1.23), and was also independently cross-validated against a numerical optimizer (`scipy.optimize.minimize`) searching for the same answer via a completely different method; both converged to the identical Sharpe ratio (2.6611).

### Real bug found and corrected during development
A first attempt at this module failed its own sanity check — the closed-form Sharpe (0.48) came in *below* the best individual asset's Sharpe (1.23), which is mathematically impossible if the code is correct. Two hypotheses were tested and ruled out before the real cause was found: the covariance matrix's condition number (84.1) was healthy, ruling out numerical instability from matrix inversion, and the sign of the raw pre-normalization weight sum was positive, ruling out a normalization-direction error. The actual bug was simpler than either: the weights table had been sorted for display purposes, but the sorted (reordered) version was then silently used in the downstream Sharpe calculation — pairing each weight with the wrong ticker's return and risk. Keeping a separate, never-reordered array for all math fixed it immediately, and the closed-form and numerical-optimizer Sharpe ratios matched exactly afterward (2.6611 = 2.6611).

---

## 2. Constrained Optimization

### What it does
Takes the same max-Sharpe objective from Module 1 and adds three constraints simultaneously via `scipy.optimize.minimize` (SLSQP): a leverage cap (gross exposure, Σ|wᵢ| ≤ L), a turnover cap relative to a starting portfolio (Σ|wᵢ - wᵢ,prev| ≤ T), and per-sector net exposure limits. Once inequality constraints are introduced, the closed-form solution no longer applies — this becomes a genuine constrained quadratic program with no shortcut.

### Why it's important
Real funds cannot actually hold the theoretical max-Sharpe portfolio: unlimited leverage, unlimited turnover, and unlimited sector concentration are not available to almost any real allocator. This module quantifies the actual cost of the constraints that make a portfolio implementable.

### Real finding from validation
With leverage capped at 1.5x, turnover capped at 0.5, and sector exposure capped at ±35%, the constrained portfolio's Sharpe fell from 2.66 (unconstrained) to 1.58 — a real, sizable cost of roughly 1.09 Sharpe from constraints alone. The turnover constraint bound exactly at its limit (0.500), identifying it as the dominant constraint in this scenario, not leverage or sector limits.

### A diagnostic that turned out to be a false alarm, documented rather than assumed
The first constrained result showed 13 of 19 weights landing at exactly the equal-weight starting value (1/19 ≈ 0.0526) — a classic warning sign that a numerical optimizer may have gotten stuck near its own starting point rather than genuinely finding the best answer. Rather than accepting or dismissing this, the optimization was re-run from a completely different (random) starting point as an independent check. Both runs converged to the identical Sharpe ratio (1.5751 = 1.5751), confirming the result was genuinely optimal — the clustering at equal-weight was a real consequence of the binding turnover constraint, not a stuck search.

---

## 3. Estimation Risk & Covariance Shrinkage

### What it does
Tests whether the "optimal" portfolio from Modules 1–2 is actually a stable, trustworthy result or largely an artifact of sampling noise, by splitting the historical return data into two halves and independently computing the tangency portfolio on each. It then applies Ledoit-Wolf shrinkage — blending the raw sample covariance matrix with a simpler, more stable structured target — and re-runs the identical stability test to measure whether shrinkage actually helps.

### Why it's important
The Markowitz framework is famously sensitive to estimation error in its inputs, especially the covariance matrix — a well-known practical weakness of mean-variance optimization that doesn't show up in the clean math of Module 1, but shows up immediately once the same math is applied to two different historical windows.

### Real finding from validation
The instability was large and concrete, not theoretical: XOM's optimal weight swung from 0.38 in the first half of the historical window to 1.77 in the second half — a difference of nearly 1.4, on a single asset, from the exact same underlying data source just split differently. Averaged across all 19 assets, the average absolute weight swing between the two halves was 0.363.

Ledoit-Wolf shrinkage genuinely reduced this instability, but only partially: an 8.8% reduction in average weight swing (0.363 → 0.331). The modest size of the improvement is itself informative rather than a shortcoming: the optimal shrinkage intensity chosen by the estimator was small (0.05 out of a possible 0–1 range), which is expected with a large-enough sample (19 assets, 375+ days per half) — shrinkage dampens estimation risk here, it does not eliminate it, an honest "partial fix" rather than a claimed cure.

---

## 4. Walk-Forward Backtest

### What it does
Runs both the constrained (Module 2) and unconstrained (Module 1) strategies forward through time with monthly rolling re-optimization, using only historical data available at each rebalance point (no lookahead), and charges realistic transaction costs (10 bps) on the actual turnover incurred at each rebalance.

### Why it's important
This is the module that tests whether the theoretical Sharpe advantage of the unconstrained portfolio (2.66 vs. 1.58 in Module 2) survives actually being traded over time, or whether the estimation instability found in Module 3 turns into real, compounding costs.

### Real finding from validation
It did not survive intact. Over the backtest period (Aug 2024–Jul 2026, 23 monthly rebalances):

- The **unconstrained** strategy's average turnover was 2.95 — meaning close to 300% of the portfolio was traded away and replaced *every single month*, a direct, concrete manifestation of the estimation instability documented in Module 3.
- That turnover was expensive: 6.78% of total portfolio value was consumed by transaction costs over the backtest, versus only 1.13% for the constrained version — roughly 6x more cost drag purely from chasing an unstable "optimal" target.
- The unconstrained strategy still finished with a **higher raw total return** (+111.66% vs. +62.14%), but a **lower risk-adjusted return** (realized Sharpe 1.06 vs. 1.63). The extra raw return came bundled with enough extra volatility and whipsaw trading that the unconstrained strategy was not, in fact, the better strategy once risk is accounted for.

This is the project's central, honest conclusion: the theoretically optimal portfolio and the practically best portfolio are not the same thing, and the gap between them is driven precisely by the instability that constraints exist to control.

---

## What We Achieved

- **A complete, four-module portfolio construction pipeline** built from first principles: a closed-form unconstrained tangency portfolio, a constrained optimizer handling leverage/turnover/sector limits simultaneously, an estimation-risk stress test with a standard shrinkage correction, and a walk-forward backtest comparing both approaches under realistic trading conditions
- **Formal correctness discipline throughout**: every closed-form result cross-validated against an independently-implemented numerical optimizer; every "does this look unstable?" instinct (weight clustering, condition number) tested with an explicit diagnostic rather than assumed
- **Real bugs found and corrected during development — documented, not hidden:**
  - A closed-form tangency portfolio that impossibly underperformed the best individual asset, traced (after ruling out numerical instability and a normalization-sign error) to a display-sorting operation silently desyncing weights from tickers in the downstream Sharpe calculation
  - A ticker universe omission during notebook-to-app consolidation (SQ was accidentally dropped from the Payments sector list), caught only by cross-validating the deployed app's output against the same-day notebook re-run and noticing the app's universe count didn't match — a direct application of the discipline that caught the HF Market Simulator's RNG-API mismatch
  - A Streamlit deployment crash traced to a one-line ternary (`st.success(...) if ... else st.error(...)`) tripping up Streamlit's source-inspection "magic" formatting feature — fixed by rewriting as a standard `if/else` block
- **A genuine, ruled-out false alarm**: a suspicious weight-clustering pattern in the constrained optimizer's output (13 of 19 weights landing exactly at the equal-weight starting point) was tested via an independent random-restart, not assumed to be either a bug or a coincidence — confirmed genuine via matching Sharpe ratios (1.5751 = 1.5751) from two different starting points
- **An honest, statistically grounded set of conclusions**: constraints cost roughly 1.09 Sharpe in theory (Module 2); the underlying covariance estimate is genuinely unstable across historical sub-periods, with shrinkage providing only a partial (8.8%) fix (Module 3); and in a realistic walk-forward backtest, the constrained portfolio achieves a materially better risk-adjusted return (1.63 vs. 1.06 Sharpe) than the theoretically "optimal" unconstrained portfolio, once real turnover and transaction costs are accounted for (Module 4)
- **Full app-vs-notebook cross-validation**: all four modules' live app output checked directly against the validated notebook. Module 4 (the most complex, longest-running module) reproduced notebook figures to 2-3 decimal places; Modules 1-3 showed small, explained differences attributable to routine day-to-day live data pull variance (`yfinance`'s rolling lookback window shifts slightly depending on the exact day the app is run)
- **Deployed as a live, interactive web application** (Streamlit Community Cloud) with all four modules navigable from a single sidebar, adjustable constraint parameters, and on-demand buttons for the more expensive optimization and backtest runs

---

## Tech Stack

- **Core computation**: Python, numpy, scipy (optimize)
- **Covariance shrinkage**: scikit-learn (`LedoitWolf`) — the one standard, non-first-principles component in this project, used because shrinkage estimation has a well-established reference implementation and isn't the numerically interesting part of this project; the interesting part is what it fixes and by how much
- **Visualization**: matplotlib (non-interactive `Agg` backend for server-side rendering stability)
- **Live market data**: yfinance
- **Web interface**: Streamlit
- **Deployment**: Streamlit Community Cloud
- **Development environment**: Google Colab

## Repository Structure
```
├── app.py # Consolidated Streamlit application (all 4 modules)
├── requirements.txt # Python dependencies
└── Portfolio_Optimiser.ipynb # Development notebook: all 4 modules, diagnostics, and validation
```

The notebook documents the full development and validation process across all four modules, including every bug found and corrected along the way. `app.py` consolidates the final, validated logic into a single deployable application, cross-checked module-by-module against the notebook's own results after deployment — including catching and fixing a ticker universe omission that the notebook alone would not have surfaced.