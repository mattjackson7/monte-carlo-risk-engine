"""
risk_engine.py — Monte Carlo Risk Engine
=========================================
Core quantitative machinery:
  1. Cholesky decomposition (implemented from scratch)
  2. Correlated multivariate return simulation
  3. VaR and CVaR (both MC and historical)
  4. Portfolio-level aggregation
  5. Stress testing under alternative correlation regimes

Mathematical conventions
------------------------
- Returns are daily log-returns (dimensionless)
- Weights w ∈ R^N, sum(w) = 1, long-only unless noted
- Portfolio return:  R_p = w · r  (dot product)
- VaR_α   = -(1-α) quantile of R_p distribution   [positive = loss]
- CVaR_α  = -E[R_p | R_p < -VaR_α]                [positive = loss]
"""

import numpy as np
import pandas as pd
from typing import Optional


# ══════════════════════════════════════════════════════════════════
# 1. CHOLESKY DECOMPOSITION — from scratch
# ══════════════════════════════════════════════════════════════════

def cholesky_decompose(Sigma: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """
    Compute the lower-triangular Cholesky factor L of a real SPD matrix Σ
    such that  Σ = L Lᵀ.

    Algorithm (outer-product / column-by-column form):
    ───────────────────────────────────────────────────
    For j = 0, 1, …, N-1:

        L[j, j] = sqrt( Σ[j, j] − Σ_{k<j} L[j,k]² )          (diagonal)

        L[i, j] = ( Σ[i, j] − Σ_{k<j} L[i,k]·L[j,k] ) / L[j,j]
                  for i = j+1, …, N-1                          (sub-diagonal)

    This is the standard Cholesky–Banachiewicz algorithm.
    Complexity: O(N³/3) FLOPs — roughly 2× faster than LU for SPD matrices
    because we exploit symmetry and only compute the lower triangle.

    Parameters
    ----------
    Sigma : (N, N) symmetric positive definite matrix
    tol   : tolerance for detecting non-PD matrices (negative diagonal pivot)

    Returns
    -------
    L : (N, N) lower triangular, with L @ L.T ≈ Sigma
    """
    Sigma = np.array(Sigma, dtype=float)
    N = Sigma.shape[0]

    if Sigma.shape != (N, N):
        raise ValueError(f"Matrix must be square; got shape {Sigma.shape}")
    if not np.allclose(Sigma, Sigma.T, atol=1e-8):
        raise ValueError("Matrix must be symmetric.")

    L = np.zeros_like(Sigma)

    for j in range(N):
        # ── Diagonal element ─────────────────────────────────────────
        # Accumulated squared sum from previously computed columns
        accumulated = np.dot(L[j, :j], L[j, :j])   # Σ_{k<j} L[j,k]²
        pivot = Sigma[j, j] - accumulated

        if pivot < tol:
            raise np.linalg.LinAlgError(
                f"Matrix is not positive definite: "
                f"pivot at [{j},{j}] = {pivot:.6e} < tol={tol}. "
                f"Try adding a small regulariser (e.g. Σ + εI)."
            )
        L[j, j] = np.sqrt(pivot)

        # ── Sub-diagonal elements ────────────────────────────────────
        # For each row i below j: dot-subtract already-computed contributions
        if j + 1 < N:
            cross = np.dot(L[j+1:, :j], L[j, :j])   # shape (N-j-1,)
            L[j+1:, j] = (Sigma[j+1:, j] - cross) / L[j, j]

    return L


def regularise_covariance(Sigma: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """
    Add a small multiple of the identity to ensure positive definiteness.

    Σ_reg = Σ + ε·I

    This is the simplest form of ridge regularisation / Tikhonov regularisation.
    Needed when the sample covariance is near-singular (e.g. more assets than
    observations, or highly correlated assets).

    ε = 1e-6 shifts the smallest eigenvalue by ε, negligible for typical
    daily variance magnitudes (~1e-4).
    """
    N = Sigma.shape[0]
    return Sigma + epsilon * np.eye(N)


# ══════════════════════════════════════════════════════════════════
# 2. CORRELATED RETURN SIMULATION
# ══════════════════════════════════════════════════════════════════

def simulate_correlated_returns(
    mu: np.ndarray,
    Sigma: np.ndarray,
    horizon: int,
    n_simulations: int,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    Simulate n_simulations paths of N-asset log-returns over a `horizon`-day
    window using the Cholesky method.

    Mathematical derivation
    -----------------------
    We want  X ~ N(μ_H, Σ_H)  where:
        μ_H = μ · horizon            (mean scales linearly with horizon)
        Σ_H = Σ · horizon            (variance scales linearly → std ∝ √horizon)

    Step 1:  Z ~ N(0, I_N)  — draw N independent standard normals per simulation
    Step 2:  X = μ_H + L · Z · √horizon

    Proof that Cov(X) = Σ_H:
        Cov(X) = L · Cov(Z·√h) · Lᵀ
               = L · (h · I) · Lᵀ
               = h · L Lᵀ
               = h · Σ  = Σ_H  ✓

    Parameters
    ----------
    mu           : (N,) array of daily mean log-returns
    Sigma        : (N, N) daily covariance matrix
    horizon      : number of trading days (e.g. 1 for daily VaR, 10 for Basel)
    n_simulations: number of Monte Carlo paths (e.g. 100_000)
    random_state : seed for reproducibility

    Returns
    -------
    sim_returns : (n_simulations, N) array of horizon-period log-returns
    """
    if random_state is not None:
        np.random.seed(random_state)

    N = len(mu)
    mu    = np.asarray(mu,    dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)

    # Regularise and decompose
    Sigma_reg = regularise_covariance(Sigma)
    L = cholesky_decompose(Sigma_reg)

    # Z ~ N(0, I_N),  shape (N, n_simulations)
    Z = np.random.standard_normal((N, n_simulations))

    # Scale by sqrt(horizon) and apply Cholesky factor
    # Shape: (N, n_sims) → transpose to (n_sims, N)
    scaled_returns = (mu * horizon)[:, np.newaxis] + L @ Z * np.sqrt(horizon)

    return scaled_returns.T   # (n_simulations, N)


# ══════════════════════════════════════════════════════════════════
# 3. PORTFOLIO RETURNS
# ══════════════════════════════════════════════════════════════════

def portfolio_returns(
    asset_returns: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """
    Compute portfolio-level log-returns from simulated asset returns.

    R_p = w · r  (dot product per simulation)

    Note: for small returns, w·r ≈ log-return of the portfolio.
    For exact portfolio log-returns you'd need to compute the weighted
    sum of simple returns, but this approximation is standard in
    short-horizon VaR estimation.

    Parameters
    ----------
    asset_returns : (n_simulations, N) array
    weights       : (N,) portfolio weights, must sum to 1

    Returns
    -------
    (n_simulations,) array of portfolio returns
    """
    weights = np.asarray(weights, dtype=float)
    if not np.isclose(weights.sum(), 1.0, atol=1e-6):
        raise ValueError(f"Weights must sum to 1; got {weights.sum():.6f}")

    return asset_returns @ weights   # (n_sims,)


# ══════════════════════════════════════════════════════════════════
# 4. VaR AND CVaR ESTIMATION
# ══════════════════════════════════════════════════════════════════

def compute_var_cvar_mc(
    portfolio_rets: np.ndarray,
    confidence_levels: list[float] = [0.95, 0.99],
) -> dict:
    """
    Monte Carlo VaR and CVaR from simulated portfolio return distribution.

    VaR_α  = -quantile_{1-α}(R_p)
           = the loss not exceeded with probability α.
           Signs: positive VaR means a loss.

    CVaR_α = -E[R_p | R_p ≤ -VaR_α]
           = average loss in the worst (1-α) fraction of scenarios.
           CVaR is a *coherent* risk measure (Artzner et al. 1999):
             - Monotone, sub-additive, homogeneous, translation-invariant.
           VaR is NOT sub-additive in general → diversification can
           apparently increase VaR, which is a flaw CVaR doesn't share.

    Parameters
    ----------
    portfolio_rets    : (n_simulations,) array
    confidence_levels : list of α values, e.g. [0.95, 0.99]

    Returns
    -------
    dict with keys like 'VaR_0.95', 'CVaR_0.95', 'VaR_0.99', 'CVaR_0.99'
    """
    results = {}
    for alpha in confidence_levels:
        q = np.quantile(portfolio_rets, 1 - alpha)   # (1-α) lower quantile
        var  = -q
        # CVaR: mean of returns that are ≤ the VaR threshold
        tail_returns = portfolio_rets[portfolio_rets <= q]
        cvar = -tail_returns.mean() if len(tail_returns) > 0 else var

        results[f"VaR_{alpha:.2f}_MC"]  = float(var)
        results[f"CVaR_{alpha:.2f}_MC"] = float(cvar)

    return results


def compute_var_cvar_historical(
    historical_returns: np.ndarray,
    weights: np.ndarray,
    confidence_levels: list[float] = [0.95, 0.99],
) -> dict:
    """
    Historical simulation VaR and CVaR.

    Uses the empirical return distribution directly — no parametric
    assumptions about the shape. The portfolio return series is
    constructed from actual historical asset returns and the given weights.

    Advantages over MC:
      - Captures actual fat tails, skewness, and autocorrelation
    Disadvantages:
      - Limited by the length of historical record
      - Cannot extrapolate to regimes not seen in the data

    Parameters
    ----------
    historical_returns : (T, N) array of actual daily log-returns
    weights            : (N,) portfolio weights
    confidence_levels  : list of α values

    Returns
    -------
    dict with keys like 'VaR_0.95_Hist', 'CVaR_0.95_Hist'
    """
    port_rets = historical_returns @ weights   # (T,)
    results = {}

    for alpha in confidence_levels:
        q    = np.quantile(port_rets, 1 - alpha)
        var  = -q
        tail = port_rets[port_rets <= q]
        cvar = -tail.mean() if len(tail) > 0 else var

        results[f"VaR_{alpha:.2f}_Hist"]  = float(var)
        results[f"CVaR_{alpha:.2f}_Hist"] = float(cvar)

    return results


# ══════════════════════════════════════════════════════════════════
# 5. STRESS TESTING — alternative correlation regimes
# ══════════════════════════════════════════════════════════════════

def build_stress_covariance(
    base_cov: np.ndarray,
    stress_corr: np.ndarray,
) -> np.ndarray:
    """
    Construct a stressed covariance matrix by replacing the correlation
    structure while preserving the individual asset volatilities.

    Decomposition:  Σ = D · ρ · D
    where D = diag(σ_1, …, σ_N) is the diagonal volatility matrix.

    Stressed:  Σ_stress = D · ρ_stress · D

    This separates vol estimation (from history) from correlation
    assumptions (which we shock), giving a clean stress test.

    Parameters
    ----------
    base_cov    : (N, N) historical covariance
    stress_corr : (N, N) stressed correlation matrix (must be PSD, diagonal = 1)

    Returns
    -------
    (N, N) stressed covariance matrix
    """
    vols = np.sqrt(np.diag(base_cov))   # σ_i = sqrt(Σ_ii)
    D = np.diag(vols)
    return D @ stress_corr @ D


def crisis_correlation_matrix(N: int, high_corr: float = 0.80) -> np.ndarray:
    """
    Stylised 'crisis' correlation matrix: all off-diagonal elements equal
    to `high_corr`, representing the empirical observation that correlations
    converge toward 1 during market stress (the 'correlation breakdown'
    phenomenon, well-documented in 2008 and COVID-2020 selloffs).

    Σ_crisis[i,j] = high_corr  for i ≠ j
    Σ_crisis[i,i] = 1.0

    This is valid (PSD) as long as high_corr ≥ -1/(N-1), which holds
    for any positive high_corr.
    """
    corr = np.full((N, N), high_corr)
    np.fill_diagonal(corr, 1.0)
    return corr


def normal_correlation_matrix(N: int, low_corr: float = 0.20) -> np.ndarray:
    """
    Stylised 'normal' market correlation: modest uniform off-diagonal
    correlation, representing calm-market co-movement.
    """
    corr = np.full((N, N), low_corr)
    np.fill_diagonal(corr, 1.0)
    return corr


def run_stress_test(
    mu: np.ndarray,
    base_cov: np.ndarray,
    weights: np.ndarray,
    stress_scenarios: dict,   # name → (N,N) correlation matrix
    horizon: int = 1,
    n_simulations: int = 100_000,
    confidence_levels: list[float] = [0.95, 0.99],
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Run VaR/CVaR under multiple correlation regimes and return a comparison table.

    Parameters
    ----------
    stress_scenarios : dict mapping scenario name → (N,N) correlation matrix
                       e.g. {'Historical': base_corr,
                              'Crisis (ρ=0.8)': crisis_corr,
                              'Calm (ρ=0.2)': calm_corr}
    All other args as in simulate_correlated_returns.

    Returns
    -------
    DataFrame with scenarios as rows and VaR/CVaR columns.
    """
    rows = []

    for scenario_name, stress_corr in stress_scenarios.items():
        stressed_cov = build_stress_covariance(base_cov, stress_corr)
        sim = simulate_correlated_returns(
            mu, stressed_cov, horizon, n_simulations, random_state
        )
        port = portfolio_returns(sim, weights)
        metrics = compute_var_cvar_mc(port, confidence_levels)
        metrics["Scenario"] = scenario_name
        rows.append(metrics)

    df = pd.DataFrame(rows).set_index("Scenario")
    return df


# ══════════════════════════════════════════════════════════════════
# 6. RETURN DISTRIBUTION HELPER (for plotting)
# ══════════════════════════════════════════════════════════════════

def get_simulation_data(
    mu: np.ndarray,
    Sigma: np.ndarray,
    weights: np.ndarray,
    horizon: int = 1,
    n_simulations: int = 100_000,
    confidence_levels: list[float] = [0.95, 0.99],
    random_state: int = 42,
) -> dict:
    """
    One-stop function: simulate, compute portfolio returns, compute all metrics.
    Returns a dict with simulation results and risk metrics for the dashboard.
    """
    sim_asset = simulate_correlated_returns(mu, Sigma, horizon, n_simulations, random_state)
    sim_port  = portfolio_returns(sim_asset, weights)
    mc_metrics   = compute_var_cvar_mc(sim_port, confidence_levels)

    return {
        "sim_asset_returns":  sim_asset,    # (n_sims, N)
        "sim_port_returns":   sim_port,     # (n_sims,)
        "mc_metrics":         mc_metrics,
    }


# ══════════════════════════════════════════════════════════════════
# 7. MARGINAL AND COMPONENT VaR
# ══════════════════════════════════════════════════════════════════

def compute_component_var(
    sim_asset_returns: np.ndarray,
    weights: np.ndarray,
    confidence_levels: list[float] = [0.95, 0.99],
    tickers: list[str] = None,
    band_width: float = 0.005,
) -> dict:
    """
    Compute Marginal VaR, Component VaR, and percentage contribution
    for each asset in the portfolio.

    Mathematical derivation
    -----------------------
    The Marginal VaR is the partial derivative of portfolio VaR with
    respect to weight w_i, evaluated via the Euler decomposition:

        MVaR_i = dVaR_p / dw_i
               = -E[r_i | R_p = -VaR_p]

    The key subtlety is the conditioning: we want the expectation of
    r_i conditional on the portfolio return being *exactly* at the VaR
    threshold, not anywhere in the tail. Using the full tail mean
    (-E[r_i | R_p <= -VaR]) introduces a bias because it averages over
    the entire tail rather than the boundary.

    In practice we approximate the point condition with a narrow band:
        E[r_i | R_p ≈ -VaR_p] ≈ E[r_i | -VaR_p - δ < R_p < -VaR_p + δ]

    Component VaR is then the Euler decomposition:
        CVaR_i = w_i * MVaR_i

    By Euler's theorem for homogeneous risk measures:
        VaR_p = sum_i CVaR_i   (exact, recovers total VaR)

    Percentage contribution:
        %Contrib_i = CVaR_i / VaR_p * 100

    Parameters
    ----------
    sim_asset_returns : (n_simulations, N) simulated asset returns
    weights           : (N,) portfolio weights
    confidence_levels : list of alpha values
    tickers           : optional list of asset names for labelling
    band_width        : half-width of quantile band as fraction of
                        portfolio return std (default 0.5%)

    Returns
    -------
    dict keyed by alpha string, each value a DataFrame with columns:
        Weight, Marginal VaR, Component VaR, % Contribution
    """
    weights = np.asarray(weights, dtype=float)
    N = len(weights)
    tickers = tickers or [f"Asset {i+1}" for i in range(N)]

    sim_port = sim_asset_returns @ weights
    port_std = sim_port.std()
    delta    = band_width * port_std      # narrow band around the VaR quantile
    results  = {}

    for alpha in confidence_levels:
        q   = np.quantile(sim_port, 1 - alpha)   # VaR threshold (negative)
        var = -q

        # Narrow band around the quantile boundary
        band_mask = (sim_port >= q - delta) & (sim_port <= q + delta)

        # Fall back to wider tail if band is too sparse (< 50 obs)
        if band_mask.sum() < 50:
            band_mask = sim_port <= q

        band_asset = sim_asset_returns[band_mask]     # (n_band, N)
        marginal_var  = -band_asset.mean(axis=0)      # MVaR_i = -E[r_i | R_p ≈ -VaR]
        component_var = weights * marginal_var         # CVaR_i = w_i * MVaR_i

        # Rescale so components sum exactly to VaR_p
        # (removes residual simulation noise while preserving relative contributions)
        component_sum = component_var.sum()
        if abs(component_sum) > 1e-10:
            component_var = component_var * (var / component_sum)

        pct_contribution = component_var / var * 100 if var > 0 else np.zeros(N)

        df = pd.DataFrame({
            "Ticker":         tickers,
            "Weight":         weights,
            "Marginal VaR":   marginal_var,
            "Component VaR":  component_var,
            "% Contribution": pct_contribution,
        }).set_index("Ticker")

        results[f"{alpha:.2f}"] = df

    return results


# ══════════════════════════════════════════════════════════════════
# 8. VAR BACKTESTING
# ══════════════════════════════════════════════════════════════════

def rolling_var_backtest(
    returns: np.ndarray,
    weights: np.ndarray,
    window: int = 252,
    confidence_levels: list[float] = [0.95, 0.99],
    method: str = "historical",
    n_simulations: int = 10_000,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Rolling-window VaR backtest.

    For each day t = window, window+1, ..., T-1:
      1. Fit VaR using returns[t-window : t]  (ex-ante, no lookahead)
      2. Record actual portfolio return on day t  (ex-post)
      3. Flag a violation if actual_return < -VaR

    Parameters
    ----------
    returns           : (T, N) array of daily log-returns
    weights           : (N,) portfolio weights
    window            : rolling estimation window in days (Basel = 250)
    confidence_levels : list of alpha values
    method            : 'historical' uses empirical quantile;
                        'mc' uses Cholesky Monte Carlo each step
                        (historical is much faster for backtesting)
    n_simulations     : only used if method='mc'

    Returns
    -------
    DataFrame indexed by day number (t = window..T-1) with columns:
      actual_return, VaR_0.95, VaR_0.99, violation_0.95, violation_0.99
    """
    T, N = returns.shape
    port_returns = returns @ weights   # (T,) portfolio return series

    records = []
    rng = np.random.default_rng(random_state)

    for t in range(window, T):
        window_returns = returns[t - window : t]   # (window, N) — ex-ante only
        actual         = port_returns[t]            # ex-post realisation

        row = {"actual_return": actual}

        for alpha in confidence_levels:
            if method == "historical":
                port_window = window_returns @ weights
                q   = np.quantile(port_window, 1 - alpha)
                var = -q

            elif method == "mc":
                mu_w    = window_returns.mean(axis=0)
                Sigma_w = np.cov(window_returns.T)
                Sigma_w = regularise_covariance(Sigma_w)
                L       = cholesky_decompose(Sigma_w)
                Z       = rng.standard_normal((N, n_simulations))
                sim     = (mu_w[:, None] + L @ Z).T   # (n_sims, N)
                port_sim = sim @ weights
                q   = np.quantile(port_sim, 1 - alpha)
                var = -q

            key = f"{alpha:.2f}"
            row[f"VaR_{key}"]       = var
            row[f"violation_{key}"] = int(actual < -var)

        records.append(row)

    df = pd.DataFrame(records)
    df.index = np.arange(window, T)
    return df


def kupiec_pof_test(
    violations: np.ndarray,
    alpha: float,
) -> dict:
    """
    Kupiec (1995) Proportion of Failures (POF) likelihood ratio test.

    H0: the true violation probability equals (1 - alpha)
    H1: it differs

    Test statistic:
        LR = -2 * ln[ p^V * (1-p)^(T-V) / (p_hat^V * (1-p_hat)^(T-V)) ]
           ~ chi^2(1) under H0

    where:
        T     = total number of observations
        V     = number of violations
        p     = 1 - alpha  (theoretical violation rate)
        p_hat = V / T      (observed violation rate)

    A p-value > 0.05 means we fail to reject H0 — the model is
    well-calibrated at that confidence level.

    Parameters
    ----------
    violations : binary array, 1 = violation on that day
    alpha      : confidence level (e.g. 0.95)

    Returns
    -------
    dict with keys: T, V, p_theoretical, p_observed, LR_stat, p_value, result
    """
    from scipy.stats import chi2

    violations = np.asarray(violations)
    T   = len(violations)
    V   = int(violations.sum())
    p   = 1 - alpha          # theoretical rate
    p_hat = V / T            # observed rate

    # Guard against p_hat = 0 or 1 (log(0) undefined)
    eps = 1e-10
    p_hat_safe = np.clip(p_hat, eps, 1 - eps)

    # Log-likelihood under H0 (theoretical p)
    ll_null = V * np.log(p + eps) + (T - V) * np.log(1 - p + eps)
    # Log-likelihood under H1 (observed p_hat)
    ll_alt  = V * np.log(p_hat_safe) + (T - V) * np.log(1 - p_hat_safe)

    LR     = -2 * (ll_null - ll_alt)
    p_value = 1 - chi2.cdf(LR, df=1)

    return {
        "T":             T,
        "Violations":    V,
        "p_theoretical": round(p, 4),
        "p_observed":    round(p_hat, 4),
        "LR_statistic":  round(float(LR), 4),
        "p_value":       round(float(p_value), 4),
        "result":        "PASS" if p_value > 0.05 else "FAIL",
    }


def run_full_backtest(
    returns: np.ndarray,
    weights: np.ndarray,
    window: int = 252,
    confidence_levels: list[float] = [0.95, 0.99],
) -> tuple[pd.DataFrame, dict]:
    """
    Convenience wrapper: run rolling backtest + Kupiec test for all
    confidence levels. Returns (backtest_df, kupiec_results_dict).
    """
    bt = rolling_var_backtest(returns, weights, window, confidence_levels)

    kupiec = {}
    for alpha in confidence_levels:
        key = f"{alpha:.2f}"
        kupiec[f"alpha={key}"] = kupiec_pof_test(
            bt[f"violation_{key}"].values, alpha
        )

    return bt, kupiec


if __name__ == "__main__":
    print("=" * 60)
    print("RISK ENGINE SANITY CHECKS")
    print("=" * 60)

    # ── 1. Cholesky correctness ───────────────────────────────
    print("\n[1] Cholesky decomposition")
    np.random.seed(0)
    A = np.random.randn(4, 4)
    Sigma_test = A @ A.T + 0.1 * np.eye(4)   # guaranteed SPD

    L = cholesky_decompose(Sigma_test)
    residual = np.max(np.abs(L @ L.T - Sigma_test))
    lower_triangular = np.allclose(np.triu(L, k=1), 0)
    print(f"  Max |LLᵀ - Σ|: {residual:.2e}  (should be ~0)")
    print(f"  L is lower triangular: {lower_triangular}")
    print(f"  vs np.linalg.cholesky: max diff = {np.max(np.abs(L - np.linalg.cholesky(Sigma_test))):.2e}")

    # ── 2. Simulation recovers target covariance ──────────────
    print("\n[2] Simulated covariance ≈ target covariance")
    mu_test = np.array([0.0005, 0.0008, 0.0003, 0.0001])
    N = 4
    sim = simulate_correlated_returns(mu_test, Sigma_test, horizon=1, n_simulations=500_000, random_state=1)
    cov_recovered = np.cov(sim.T)
    max_cov_err = np.max(np.abs(cov_recovered - regularise_covariance(Sigma_test)))
    print(f"  Max |Cov(sim) - Σ_reg|: {max_cov_err:.6f}  (converges to 0 with more sims)")

    # ── 3. VaR ordering ──────────────────────────────────────
    print("\n[3] VaR ordering:  VaR_99 ≥ VaR_95  and  CVaR ≥ VaR")
    weights_test = np.array([0.4, 0.3, 0.2, 0.1])
    port = portfolio_returns(sim, weights_test)
    metrics = compute_var_cvar_mc(port, [0.95, 0.99])
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")
    assert metrics["VaR_0.99_MC"] >= metrics["VaR_0.95_MC"],  "VaR ordering violated!"
    assert metrics["CVaR_0.99_MC"] >= metrics["CVaR_0.95_MC"], "CVaR ordering violated!"
    assert metrics["CVaR_0.95_MC"] >= metrics["VaR_0.95_MC"],  "CVaR < VaR !"
    assert metrics["CVaR_0.99_MC"] >= metrics["VaR_0.99_MC"],  "CVaR < VaR !"
    print("  ✓ All ordering constraints satisfied")

    # ── 4. Crisis scenario increases risk ──────────────────────
    print("\n[4] Crisis correlation → higher risk than calm")
    N = 4
    base_corr = normal_correlation_matrix(N, low_corr=0.2)
    cris_corr  = crisis_correlation_matrix(N, high_corr=0.8)
    stress_scenarios = {
        "Calm (ρ=0.2)":   base_corr,
        "Crisis (ρ=0.8)": cris_corr,
    }
    results = run_stress_test(mu_test, Sigma_test, weights_test, stress_scenarios,
                              horizon=1, n_simulations=200_000, random_state=7)
    print(results.round(6).to_string())
    assert results.loc["Crisis (ρ=0.8)", "VaR_0.99_MC"] >= results.loc["Calm (ρ=0.2)", "VaR_0.99_MC"], \
        "Crisis VaR should be ≥ calm VaR"
    print("  ✓ Crisis scenario produces higher or equal tail risk")

    # ── 5. Component VaR sums to total VaR ───────────────────
    print("\n[5] Component VaR decomposition: sum(CVaR_i) ≈ VaR_p")
    tickers_test = ["A", "B", "C", "D"]
    port_rets    = portfolio_returns(sim, weights_test)
    comp = compute_component_var(sim, weights_test, [0.95, 0.99], tickers_test)
    all_var = compute_var_cvar_mc(port_rets, [0.95, 0.99])
    for alpha_key, df in comp.items():
        port_var = all_var[f"VaR_{alpha_key}_MC"]
        sum_comp = df["Component VaR"].sum()
        rel_err  = abs(sum_comp - port_var) / port_var
        print(f"  alpha={alpha_key}: VaR_p={port_var:.6f}, "
              f"sum(Component VaR)={sum_comp:.6f}, rel_err={rel_err:.2e}")
        assert rel_err < 0.02, f"Component VaR decomposition error too large: {rel_err:.2e}"
        print(df.round(6).to_string())
    print("  ✓ Component VaR sums to portfolio VaR within simulation noise")

    print("\nAll sanity checks passed ✓")