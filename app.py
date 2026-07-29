# ============================================================
# Portfolio Optimizer - Streamlit App
# Consolidates Modules 1-4 from the development notebook
# ============================================================
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

st.set_page_config(page_title="Portfolio Optimizer", layout="wide")

# ------------------------------------------------------------
# Sidebar - universe & shared parameters
# ------------------------------------------------------------
st.sidebar.title("Portfolio Optimizer")
module = st.sidebar.radio(
    "Module",
    ["1. Unconstrained Tangency Portfolio",
     "2. Constrained Optimization",
     "3. Estimation Risk & Shrinkage",
     "4. Walk-Forward Backtest"]
)

TICKERS = {
    'Tech':       ['AAPL', 'MSFT', 'NVDA', 'GOOGL'],
    'Financials': ['JPM', 'BAC', 'MA', 'V'],
    'Energy':     ['XOM', 'CVX', 'COP', 'SLB'],
    'Retail':     ['WMT', 'TGT', 'HD', 'COST'],
    'Payments':   ['PYPL', 'FIS', 'GPN'],
}
ALL_TICKERS = [t for group in TICKERS.values() for t in group]
SECTOR_MAP = {t: sector for sector, group in TICKERS.items() for t in group}

st.sidebar.subheader("Shared Settings")
rf_rate = st.sidebar.number_input("Risk-free rate (annual)", value=0.04, step=0.005, format="%.3f")
lookback_years = st.sidebar.slider("Lookback (years)", 1, 5, 3)

@st.cache_data(ttl=3600)
def load_data(tickers, years):
    raw = yf.download(tickers, period=f"{years}y", auto_adjust=True)['Close']
    raw = raw.dropna(axis=1, how='all').dropna(axis=0, how='any')
    daily_returns = raw.pct_change().dropna()
    return raw, daily_returns

with st.spinner("Pulling market data..."):
    raw_prices, daily_returns = load_data(ALL_TICKERS, lookback_years)

kept_tickers = list(raw_prices.columns)
dropped = set(ALL_TICKERS) - set(kept_tickers)
n = len(kept_tickers)
sector_names = sorted(set(SECTOR_MAP.values()))
sector_matrix = np.zeros((len(sector_names), n))
for j, ticker in enumerate(kept_tickers):
    sector_matrix[sector_names.index(SECTOR_MAP[ticker]), j] = 1

mu = daily_returns.mean() * 252
Sigma = daily_returns.cov() * 252

if dropped:
    st.sidebar.warning(f"Dropped (no data): {', '.join(dropped)}")
st.sidebar.caption(f"Universe: {n} tickers | {len(daily_returns)} trading days | "
                    f"{daily_returns.index[0].date()} to {daily_returns.index[-1].date()}")

# ------------------------------------------------------------
# Shared helper functions
# ------------------------------------------------------------
def tangency_weights_from(mu_vec, Sigma_mat, rf):
    excess = mu_vec.values - rf
    raw = np.linalg.inv(Sigma_mat.values) @ excess
    return raw / raw.sum()

def neg_sharpe(w, mu_vals, Sigma_vals, rf):
    ret = w @ mu_vals
    vol = np.sqrt(w @ Sigma_vals @ w)
    return -(ret - rf) / vol

def portfolio_stats(w, mu_vals, Sigma_vals, rf):
    ret = w @ mu_vals
    vol = np.sqrt(w @ Sigma_vals @ w)
    return ret, vol, (ret - rf) / vol

def plot_weights_bar(weights_series, title):
    fig, ax = plt.subplots(figsize=(10, 5))
    sorted_w = weights_series.sort_values(ascending=False)
    colors = ['#2ca02c' if v >= 0 else '#d62728' for v in sorted_w.values]
    ax.bar(sorted_w.index, sorted_w.values, color=colors)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_title(title)
    ax.set_ylabel("Weight")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig

# ============================================================
# MODULE 1: Unconstrained Tangency Portfolio
# ============================================================
if module.startswith("1"):
    st.title("Module 1: Unconstrained Tangency Portfolio")
    st.markdown("Closed-form max-Sharpe portfolio: $w \\propto \\Sigma^{-1}(\\mu - r_f\\mathbf{1})$, no numerical optimizer needed.")

    w = pd.Series(tangency_weights_from(mu, Sigma, rf_rate), index=kept_tickers)
    ret, vol, sharpe = portfolio_stats(w.values, mu.values, Sigma.values, rf_rate)

    col1, col2, col3 = st.columns(3)
    col1.metric("Expected Return (ann.)", f"{ret:.2%}")
    col2.metric("Volatility (ann.)", f"{vol:.2%}")
    col3.metric("Sharpe Ratio", f"{sharpe:.3f}")

    st.pyplot(plot_weights_bar(w, "Tangency Portfolio Weights"))

    st.subheader("Sanity Check: Portfolio vs. Individual Assets")
    indiv_sharpe = ((mu - rf_rate) / np.sqrt(np.diag(Sigma))).sort_values(ascending=False)
    st.write(f"Best individual asset Sharpe: **{indiv_sharpe.max():.3f}** ({indiv_sharpe.idxmax()})")
    st.write(f"Tangency portfolio Sharpe: **{sharpe:.3f}**")
    passed = sharpe >= indiv_sharpe.max()
    st.success("✓ Check passed: portfolio beats every individual asset") if passed else \
        st.error("✗ Check FAILED — portfolio underperforms best individual asset. Something is wrong.")

    with st.expander("Individual asset Sharpe ratios"):
        st.dataframe(indiv_sharpe.round(4))

# ============================================================
# MODULE 2: Constrained Optimization
# ============================================================
elif module.startswith("2"):
    st.title("Module 2: Constrained Optimization")
    st.markdown("Same max-Sharpe objective, with leverage, turnover, and sector limits applied.")

    st.subheader("Constraints")
    c1, c2, c3 = st.columns(3)
    max_leverage = c1.slider("Max leverage (gross exposure)", 1.0, 3.0, 1.5, 0.1)
    max_turnover = c2.slider("Max turnover", 0.1, 2.0, 0.5, 0.1)
    max_sector = c3.slider("Max sector weight (+/-)", 0.1, 1.0, 0.35, 0.05)

    w_previous = np.ones(n) / n  # starting point: equal-weight

    constraints = [
        {'type': 'eq', 'fun': lambda w: w.sum() - 1},
        {'type': 'ineq', 'fun': lambda w: max_leverage - np.sum(np.abs(w))},
        {'type': 'ineq', 'fun': lambda w: max_turnover - np.sum(np.abs(w - w_previous))},
    ]
    for i in range(len(sector_names)):
        constraints.append({'type': 'ineq', 'fun': (lambda w, i=i: max_sector - sector_matrix[i] @ w)})
        constraints.append({'type': 'ineq', 'fun': (lambda w, i=i: sector_matrix[i] @ w + max_sector)})

    if st.button("Run Constrained Optimization"):
        with st.spinner("Optimizing..."):
            result = minimize(neg_sharpe, w_previous, args=(mu.values, Sigma.values, rf_rate),
                               method='SLSQP', constraints=constraints,
                               options={'maxiter': 1000, 'ftol': 1e-12})
        w_constrained = pd.Series(result.x, index=kept_tickers)
        ret_c, vol_c, sharpe_c = portfolio_stats(result.x, mu.values, Sigma.values, rf_rate)

        w_unc = tangency_weights_from(mu, Sigma, rf_rate)
        _, _, sharpe_unc = portfolio_stats(w_unc, mu.values, Sigma.values, rf_rate)

        col1, col2, col3 = st.columns(3)
        col1.metric("Constrained Sharpe", f"{sharpe_c:.3f}")
        col2.metric("Unconstrained Sharpe", f"{sharpe_unc:.3f}")
        col3.metric("Sharpe Given Up", f"{sharpe_unc - sharpe_c:.3f}")

        st.pyplot(plot_weights_bar(w_constrained, "Constrained Portfolio Weights"))

        st.subheader("Constraint Verification")
        gross_lev = np.sum(np.abs(result.x))
        turnover = np.sum(np.abs(result.x - w_previous))
        st.write(f"Gross leverage: {gross_lev:.3f} / {max_leverage} — "
                 f"{'✓ OK' if gross_lev <= max_leverage + 1e-6 else '✗ VIOLATED'}")
        st.write(f"Turnover: {turnover:.3f} / {max_turnover} — "
                 f"{'✓ OK' if turnover <= max_turnover + 1e-6 else '✗ VIOLATED'}")
        for i, sector in enumerate(sector_names):
            sw = sector_matrix[i] @ result.x
            ok = -max_sector - 1e-6 <= sw <= max_sector + 1e-6
            st.write(f"{sector}: {sw:+.3f} (limit ±{max_sector}) — {'✓ OK' if ok else '✗ VIOLATED'}")

# ============================================================
# MODULE 3: Estimation Risk & Shrinkage
# ============================================================
elif module.startswith("3"):
    st.title("Module 3: Estimation Risk & Covariance Shrinkage")
    st.markdown("Testing whether the 'optimal' portfolio is stable — or just noise — by splitting history in half.")

    if st.button("Run Stability Test"):
        half = len(daily_returns) // 2
        ret_1, ret_2 = daily_returns.iloc[:half], daily_returns.iloc[half:]
        mu_1, Sigma_1 = ret_1.mean() * 252, ret_1.cov() * 252
        mu_2, Sigma_2 = ret_2.mean() * 252, ret_2.cov() * 252

        w1 = pd.Series(tangency_weights_from(mu_1, Sigma_1, rf_rate), index=kept_tickers)
        w2 = pd.Series(tangency_weights_from(mu_2, Sigma_2, rf_rate), index=kept_tickers)
        diff = (w1 - w2).abs()

        st.subheader("Without Shrinkage")
        st.write(f"Average weight swing between halves: **{diff.mean():.4f}**")
        st.write(f"Largest swing: **{diff.max():.4f}** ({diff.idxmax()})")

        lw1 = pd.DataFrame(LedoitWolf().fit(ret_1.values).covariance_ * 252, index=kept_tickers, columns=kept_tickers)
        lw2 = pd.DataFrame(LedoitWolf().fit(ret_2.values).covariance_ * 252, index=kept_tickers, columns=kept_tickers)
        w1s = pd.Series(tangency_weights_from(mu_1, lw1, rf_rate), index=kept_tickers)
        w2s = pd.Series(tangency_weights_from(mu_2, lw2, rf_rate), index=kept_tickers)
        diff_s = (w1s - w2s).abs()

        st.subheader("With Ledoit-Wolf Shrinkage")
        st.write(f"Average weight swing between halves: **{diff_s.mean():.4f}**")
        improvement = (1 - diff_s.mean() / diff.mean()) * 100
        st.metric("Instability Reduction", f"{improvement:.1f}%")

        comparison = pd.DataFrame({'First Half': w1, 'Second Half': w2, 'Abs Diff': diff}).sort_values('Abs Diff', ascending=False)
        st.dataframe(comparison.round(4))

# ============================================================
# MODULE 4: Walk-Forward Backtest
# ============================================================
elif module.startswith("4"):
    st.title("Module 4: Walk-Forward Backtest")
    st.markdown("Rolling re-optimization over time, with transaction costs — constrained vs. unconstrained.")

    c1, c2, c3 = st.columns(3)
    est_window = c1.slider("Estimation window (days)", 60, 252, 252, 21)
    rebal_freq = c2.slider("Rebalance frequency (days)", 5, 63, 21, 1)
    tc_bps = c3.slider("Transaction cost (bps)", 0, 50, 10, 1)

    max_leverage_bt = st.slider("Max leverage (constrained)", 1.0, 3.0, 1.5, 0.1)
    max_turnover_bt = st.slider("Max turnover (constrained)", 0.1, 2.0, 0.5, 0.1)
    max_sector_bt = st.slider("Max sector weight (constrained)", 0.1, 1.0, 0.35, 0.05)

    if st.button("Run Backtest"):
        dates = daily_returns.index
        n_days = len(dates)
        pv_c, pv_u = [1.0], [1.0]
        w_c_prev, w_u_prev = np.ones(n) / n, np.ones(n) / n
        turnovers_c, turnovers_u, rebal_dates = [], [], []

        t = est_window
        progress = st.progress(0)
        while t + rebal_freq < n_days:
            window = daily_returns.iloc[t - est_window:t]
            mu_t = window.mean() * 252
            Sigma_t = pd.DataFrame(LedoitWolf().fit(window.values).covariance_ * 252,
                                    index=kept_tickers, columns=kept_tickers)

            w_u = tangency_weights_from(mu_t, Sigma_t, rf_rate)

            cons_t = [
                {'type': 'eq', 'fun': lambda w: w.sum() - 1},
                {'type': 'ineq', 'fun': lambda w: max_leverage_bt - np.sum(np.abs(w))},
                {'type': 'ineq', 'fun': lambda w: max_turnover_bt - np.sum(np.abs(w - w_c_prev))},
            ]
            for i in range(len(sector_names)):
                cons_t.append({'type': 'ineq', 'fun': (lambda w, i=i: max_sector_bt - sector_matrix[i] @ w)})
                cons_t.append({'type': 'ineq', 'fun': (lambda w, i=i: sector_matrix[i] @ w + max_sector_bt)})

            res_t = minimize(neg_sharpe, w_c_prev, args=(mu_t.values, Sigma_t.values, rf_rate),
                              method='SLSQP', constraints=cons_t, options={'maxiter': 1000, 'ftol': 1e-10})
            w_c = res_t.x

            turn_c = np.sum(np.abs(w_c - w_c_prev))
            turn_u = np.sum(np.abs(w_u - w_u_prev))
            cost_c = turn_c * (tc_bps / 10000)
            cost_u = turn_u * (tc_bps / 10000)
            turnovers_c.append(turn_c)
            turnovers_u.append(turn_u)
            rebal_dates.append(dates[t])

            hold_returns = daily_returns.iloc[t:t + rebal_freq]
            pv_c.append(pv_c[-1] * (1 + (hold_returns.values @ w_c).sum() - cost_c))
            pv_u.append(pv_u[-1] * (1 + (hold_returns.values @ w_u).sum() - cost_u))

            w_c_prev, w_u_prev = w_c, w_u
            t += rebal_freq
            progress.progress(min(t / n_days, 1.0))

        col1, col2 = st.columns(2)
        col1.metric("Constrained Total Return", f"{(pv_c[-1]-1):+.2%}")
        col2.metric("Unconstrained Total Return", f"{(pv_u[-1]-1):+.2%}")

        col3, col4 = st.columns(2)
        col3.metric("Avg Turnover (Constrained)", f"{np.mean(turnovers_c):.3f}")
        col4.metric("Avg Turnover (Unconstrained)", f"{np.mean(turnovers_u):.3f}")

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(rebal_dates, pv_c[1:], label="Constrained", linewidth=2)
        ax.plot(rebal_dates, pv_u[1:], label="Unconstrained", linewidth=2)
        ax.set_title("Portfolio Value Over Time")
        ax.set_ylabel("Portfolio Value ($1 start)")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)

        periods_per_year = 252 / rebal_freq
        r_c = np.diff(pv_c) / np.array(pv_c[:-1])
        r_u = np.diff(pv_u) / np.array(pv_u[:-1])
        sharpe_c = (np.mean(r_c) * periods_per_year) / (np.std(r_c) * np.sqrt(periods_per_year))
        sharpe_u = (np.mean(r_u) * periods_per_year) / (np.std(r_u) * np.sqrt(periods_per_year))

        col5, col6 = st.columns(2)
        col5.metric("Realized Sharpe (Constrained)", f"{sharpe_c:.3f}")
        col6.metric("Realized Sharpe (Unconstrained)", f"{sharpe_u:.3f}")

        st.info("Note: unconstrained often shows higher raw return but lower risk-adjusted "
                "return once real turnover and transaction costs are applied — the central "
                "lesson of this project.")