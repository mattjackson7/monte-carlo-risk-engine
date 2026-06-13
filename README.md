# Monte Carlo Risk Engine

A portfolio risk quantification system implementing Cholesky-correlated Monte Carlo simulation, VaR/CVaR estimation, stress testing, component VaR decomposition, and statistical backtesting with regulatory-grade validation. Built in Python with a Streamlit dashboard.

Part of a three-project quantitative finance portfolio alongside an ML-based portfolio optimizer (LSTM, Streamlit, SQL) and an options pricing dashboard (BSM, binomial tree, Greeks, IV surface).

---

## Features

- **Correlated asset simulation** via Cholesky decomposition of the historical covariance matrix — implemented from scratch, not a library call
- **VaR and CVaR** at 95% and 99% confidence, computed by both Monte Carlo and historical simulation
- **Component VaR** — Euler decomposition of total portfolio VaR into per-asset contributions, identifying which positions drive tail risk
- **Stress testing** — correlation regime shocks (crisis vs. calm) applied via the $\Sigma = D\rho D$ decomposition, isolating the correlation channel while preserving historical volatilities
- **VaR backtesting** — rolling 252-day window with Kupiec Proportion of Failures (POF) likelihood ratio test, consistent with Basel III requirements
- **Streamlit dashboard** with six tabs: Overview, MC Simulation, VaR & CVaR, Stress Test, Backtesting, Diagnostics

---

## Installation

```bash
git clone https://github.com/mattjackson31/monte-carlo-risk-engine
cd monte-carlo-risk-engine
pip install -r requirements.txt
streamlit run app.py
```

**Requirements:** Python 3.11+, numpy, pandas, scipy, yfinance, streamlit, plotly

---

## Project Structure

```
monte_carlo_risk_engine/
├── data.py          # Market data pipeline (yfinance, log-returns, covariance)
├── risk_engine.py   # Core quantitative engine (8 modules)
├── app.py           # Streamlit dashboard
├── requirements.txt
└── .streamlit/
    └── config.toml  # Dark theme configuration
```

---

## Mathematics

### Log-Returns and Covariance Estimation

Price data is converted to daily log-returns:

$$r_t = \ln(P_t / P_{t-1})$$

Log-returns are preferred over simple returns because they are time-additive — the $T$-day return is exactly the sum of daily returns — and the variance scales linearly with horizon, which is what justifies the $\sqrt{T}$ simulation scaling below.

The sample covariance uses Bessel's correction (divides by $T-1$) to give an unbiased estimator of the population covariance matrix $\Sigma$.

---

### Cholesky Decomposition

Given a real symmetric positive definite (SPD) covariance matrix $\Sigma \in \mathbb{R}^{N \times N}$, the Cholesky–Banachiewicz algorithm finds a unique lower-triangular matrix $L$ such that:

$$\Sigma = L L^\top$$

The algorithm proceeds column by column:

$$L_{jj} = \sqrt{\Sigma_{jj} - \sum_{k=0}^{j-1} L_{jk}^2}$$

$$L_{ij} = \frac{1}{L_{jj}} \left( \Sigma_{ij} - \sum_{k=0}^{j-1} L_{ik} L_{jk} \right), \quad i > j$$

Complexity is $O(N^3/3)$ — roughly twice as fast as LU decomposition because symmetry means only the lower triangle needs to be computed.

The implementation is verified by checking that $\|LL^\top - \Sigma\|_\infty < 10^{-14}$, consistent with machine epsilon.

---

### Correlated Return Simulation

To simulate $M$ correlated $N$-asset return paths over a $h$-day horizon:

1. Draw $Z \sim \mathcal{N}(0, I_N)$, shape $(N, M)$ — independent standard normals
2. Construct $X = \mu \cdot h + L \cdot Z \cdot \sqrt{h}$

The Cholesky factor injects the desired correlation structure. Proof that $\text{Cov}(X) = h\Sigma$:

$$\text{Cov}(X) = L \cdot \text{Cov}(Z\sqrt{h}) \cdot L^\top = L \cdot (hI) \cdot L^\top = h \cdot LL^\top = h\Sigma$$

The $\sqrt{h}$ scaling follows from variance being additive for iid returns, so $\text{Var}(\sum_{t=1}^h r_t) = h \cdot \text{Var}(r)$.

---

### VaR and CVaR

Portfolio return is the weighted sum of asset returns: $R_p = w^\top r$.

**Value at Risk** at confidence level $\alpha$:

$$\text{VaR}_\alpha = -Q_{1-\alpha}(R_p)$$

The loss not exceeded with probability $\alpha$. Positive VaR means a loss.

**Conditional Value at Risk** (Expected Shortfall):

$$\text{CVaR}_\alpha = -\mathbb{E}\left[R_p \mid R_p \leq -\text{VaR}_\alpha\right]$$

CVaR is a **coherent risk measure** (Artzner et al. 1999), satisfying monotonicity, subadditivity, positive homogeneity, and translation invariance. VaR is not subadditive in general — a diversified portfolio can appear to have higher VaR than the sum of its parts, which is a fundamental flaw CVaR avoids.

Both metrics are computed by Monte Carlo (using the simulated return distribution) and historical simulation (using the empirical return distribution directly).

---

### Component VaR — Euler Decomposition

The Marginal VaR of asset $i$ is the partial derivative of portfolio VaR with respect to weight $w_i$:

$$\text{MVaR}_i = \frac{\partial \text{VaR}_p}{\partial w_i} = -\mathbb{E}\left[r_i \mid R_p = -\text{VaR}_p\right]$$

The conditional expectation is evaluated at the VaR threshold (not over the entire tail) using a narrow band of width $\pm 0.5\% \cdot \sigma_p$ around the quantile boundary.

Component VaR follows from the Euler decomposition for positively homogeneous risk measures:

$$\text{CVaR}_i = w_i \cdot \text{MVaR}_i$$

$$\text{VaR}_p = \sum_{i=1}^N \text{CVaR}_i \quad \text{(exact)}$$

This is an additive decomposition — the components sum exactly to total portfolio VaR. Unlike standalone asset VaRs, Component VaR respects diversification: an asset that hedges the portfolio can have a negative component.

The percentage contribution of each asset:

$$\%\text{Contrib}_i = \frac{\text{CVaR}_i}{\text{VaR}_p} \times 100$$

---

### Stress Testing

The covariance matrix is decomposed as:

$$\Sigma = D \rho D$$

where $D = \text{diag}(\sigma_1, \ldots, \sigma_N)$ is the diagonal volatility matrix and $\rho$ is the correlation matrix. A stressed covariance is constructed by replacing $\rho$ while holding $D$ fixed:

$$\Sigma_{\text{stress}} = D \rho_{\text{stress}} D$$

This isolates the **correlation channel** of risk from the **volatility channel**. The crisis scenario uses a uniform off-diagonal correlation $\rho_{\text{crisis}} = 0.8$, reflecting the empirical phenomenon that asset correlations converge toward 1 during market stress (documented in the 2008 financial crisis and the 2020 COVID selloff).

---

### VaR Backtesting — Kupiec POF Test

For each day $t$, VaR is estimated using only the prior $W$ days of data (no lookahead), producing an ex-ante forecast. A **violation** occurs when the actual portfolio return falls below $-\text{VaR}^{(t)}$.

The **Kupiec Proportion of Failures** test asks whether the observed violation rate $\hat{p} = V/T$ is statistically consistent with the theoretical rate $p = 1 - \alpha$:

$$LR_{POF} = -2\ln\left[\frac{p^V(1-p)^{T-V}}{\hat{p}^V(1-\hat{p})^{T-V}}\right] \sim \chi^2_1 \text{ under } H_0$$

A p-value above 0.05 means we fail to reject $H_0$ — the model is well-calibrated. Basel III requires this test to be run over a 250-day window; more than 4 violations in that window triggers a regulatory capital penalty.

---

## Dashboard

| Tab | Content |
|-----|---------|
| Overview | Portfolio weights, normalised price history, correlation heatmap |
| MC Simulation | Return distribution, simulated vs historical volatility, pairwise scatter |
| VaR & CVaR | Loss distribution with overlays, MC vs historical density, comparison table |
| Stress Test | Regime comparison bar chart, stressed correlation matrices, results table |
| Backtesting | Kupiec test results, rolling VaR vs actual returns, violation rate chart |
| Diagnostics | Eigenspectrum, Cholesky factor, rolling volatility |

---

## Sanity Checks

Run `python risk_engine.py` to execute five built-in verification steps:

1. Cholesky residual $\|LL^\top - \Sigma\|_\infty \approx 10^{-15}$ (machine epsilon)
2. Simulated covariance converges to target as $M \to \infty$
3. $\text{VaR}_{0.99} \geq \text{VaR}_{0.95}$ and $\text{CVaR}_\alpha \geq \text{VaR}_\alpha$ for all $\alpha$
4. Crisis scenario ($\rho = 0.8$) produces higher VaR than calm scenario ($\rho = 0.2$)
5. Component VaR sum equals portfolio VaR to machine precision

---

## Stack

Python · NumPy · Pandas · SciPy · yfinance · Streamlit · Plotly