"""
data.py — Market Data Pipeline
================================
Fetches historical price data via yfinance, computes log-returns,
and estimates the covariance/correlation structure needed by the
Monte Carlo engine.

Mathematical background
-----------------------
Log-returns:  r_t = ln(P_t / P_{t-1})
  - Additive over time: r_{0→T} = sum of daily r_t
  - Approximately normally distributed (CLT kicks in)
  - More symmetric than simple returns for large moves

Sample covariance (MLE with Bessel correction):
  Σ̂_ij = (1 / (T-1)) * Σ_t (r_it - r̄_i)(r_jt - r̄_j)

Correlation:
  ρ̂_ij = Σ̂_ij / sqrt(Σ̂_ii * Σ̂_jj)
"""

import numpy as np
import pandas as pd
import yfinance as yf
from typing import Optional


# ──────────────────────────────────────────────
# Core data fetching
# ──────────────────────────────────────────────

def fetch_prices(
    tickers: list[str],
    start: str,
    end: Optional[str] = None,
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """
    Download adjusted closing prices for a list of tickers.

    Parameters
    ----------
    tickers   : e.g. ['AAPL', 'MSFT', 'GLD', 'TLT', 'SPY']
    start     : ISO date string, e.g. '2018-01-01'
    end       : ISO date string; defaults to today if None
    auto_adjust: use split/dividend-adjusted prices (recommended)

    Returns
    -------
    DataFrame of shape (T, N) — dates × tickers
    Drops any dates with NaN in *any* ticker (inner join on dates).
    """
    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=auto_adjust,
        progress=False,
    )

    # yfinance returns multi-level columns when >1 ticker
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]]
        prices.columns = tickers

    # Ensure column ordering matches the input list
    prices = prices[tickers]

    # Drop any rows with missing data (conservative approach)
    prices = prices.dropna()

    if prices.empty:
        raise ValueError(
            f"No price data returned for tickers {tickers} "
            f"between {start} and {end}. Check ticker symbols."
        )

    return prices


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily log-returns from a price DataFrame.

    r_t = ln(P_t) - ln(P_{t-1})
       = ln(P_t / P_{t-1})

    The first row is dropped (NaN after differencing).
    Returns DataFrame of shape (T-1, N).
    """
    log_prices = np.log(prices)
    log_returns = log_prices.diff().dropna()
    return log_returns


def compute_covariance_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Sample covariance matrix of log-returns.

    Uses pandas .cov() which applies Bessel's correction (divides by T-1,
    not T), giving an unbiased estimator of the population covariance.

    Shape: (N, N), symmetric positive semi-definite.
    """
    return returns.cov()


def compute_correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Sample Pearson correlation matrix.

    ρ_ij = Σ_ij / sqrt(Σ_ii * Σ_jj)

    All diagonal elements = 1; off-diagonal ∈ [-1, 1].
    """
    return returns.corr()


def annualised_stats(returns: pd.DataFrame, trading_days: int = 252) -> pd.DataFrame:
    """
    Annualised mean return and volatility for each asset.

    For daily log-returns r_t:
      Annual mean  = E[r] * T
      Annual vol   = std(r) * sqrt(T)

    (The sqrt(T) scaling comes from the variance being additive for iid
    returns, so Var(sum) = T * Var(daily), hence std scales as sqrt(T).)
    """
    stats = pd.DataFrame(index=returns.columns)
    stats["Annual Mean Return"] = returns.mean() * trading_days
    stats["Annual Volatility"]  = returns.std() * np.sqrt(trading_days)
    stats["Sharpe (rf=0)"]      = stats["Annual Mean Return"] / stats["Annual Volatility"]
    return stats


# ──────────────────────────────────────────────
# Sanity checks
# ──────────────────────────────────────────────

def validate_covariance_matrix(cov: np.ndarray, tol: float = 1e-8) -> dict:
    """
    Verify that the covariance matrix has the right mathematical properties:
      1. Symmetric:              Σ = Σᵀ
      2. Positive semi-definite: all eigenvalues ≥ 0
      3. No NaNs / Infs
    Returns a dict of check results.
    """
    results = {}
    arr = np.array(cov)

    results["symmetric"]   = np.allclose(arr, arr.T, atol=tol)
    results["no_nan_inf"]  = np.all(np.isfinite(arr))
    eigenvalues            = np.linalg.eigvalsh(arr)   # symmetric → use eigvalsh
    results["min_eigenvalue"] = float(eigenvalues.min())
    results["psd"]         = bool(eigenvalues.min() >= -tol)

    return results


if __name__ == "__main__":
    # ── Quick smoke test ──────────────────────────────────────
    TICKERS = ["AAPL", "MSFT", "GLD", "TLT", "SPY"]
    START   = "2018-01-01"

    print("Fetching prices...")
    prices = fetch_prices(TICKERS, start=START)
    print(f"  Price matrix shape: {prices.shape}  ({prices.index[0].date()} → {prices.index[-1].date()})")

    print("\nComputing log-returns...")
    returns = compute_log_returns(prices)
    print(f"  Returns matrix shape: {returns.shape}")

    print("\nAnnualised statistics:")
    print(annualised_stats(returns).round(4).to_string())

    print("\nCovariance matrix (daily):")
    cov = compute_covariance_matrix(returns)
    print(cov.round(6).to_string())

    print("\nCovariance matrix validation:")
    checks = validate_covariance_matrix(cov.values)
    for k, v in checks.items():
        status = "✓" if v is True or (isinstance(v, float) and v > -1e-8) else "✗"
        print(f"  {status} {k}: {v}")

    print("\nCorrelation matrix:")
    print(compute_correlation_matrix(returns).round(3).to_string())
