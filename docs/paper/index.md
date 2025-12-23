---
title: "Differentially Private Regression with Valid Statistical Inference"
subtitle: "A Practical Framework Using Noisy Sufficient Statistics"
authors:
  - name: Max Ghenis
    affiliations:
      - PolicyEngine
    email: max@policyengine.org
date: 2024-12-16
license: CC-BY-4.0
keywords:
  - differential privacy
  - regression
  - statistical inference
  - noisy sufficient statistics
exports:
  - format: pdf
    template: arxiv_two_column
  - format: tex
---

```{code-cell} python
:tags: [remove-cell]

# Import paper results - single source of truth for all values
from dp_statsmodels.paper_results import r
```

# Abstract

We present **dp-statsmodels**, a Python library implementing differentially private linear and logistic regression using Noisy Sufficient Statistics (NSS). Unlike gradient-based approaches, NSS provides closed-form solutions with analytically tractable standard errors, enabling valid statistical inference under differential privacy constraints. Using Monte Carlo simulations and real data from the Current Population Survey (CPS) Annual Social and Economic Supplement (n={eval}`r.cps.n_fmt`), we demonstrate that: (1) the estimators are approximately unbiased across privacy budgets $\varepsilon \in [1, 20]$, (2) confidence intervals achieve close to nominal coverage when standard errors properly account for privacy noise, and (3) the privacy-utility tradeoff follows predictable patterns.

**Critical limitation**: While our standard errors are mathematically valid, they are substantially larger than non-private OLS—**{eval}`r.se_inflation_10`-{eval}`r.se_inflation_1000`× at typical privacy budgets** ($\varepsilon = 1-10$). This requires samples 10,000-1,000,000× larger to maintain equivalent statistical power. DP regression via NSS is therefore primarily viable for very large datasets ($n > 10$ million) or weak privacy guarantees ($\varepsilon > 100$). Our implementation provides a statsmodels-compatible API with automatic privacy budget tracking and requires users to specify data bounds a priori, as computing bounds from data voids privacy guarantees.

# 1. Introduction

Differential privacy (DP) has emerged as the gold standard for privacy-preserving data analysis {cite}`dwork2006differential,dwork2014algorithmic`. A mechanism $\mathcal{M}$ satisfies $(\varepsilon, \delta)$-differential privacy if for all adjacent datasets $D, D'$ differing in one record and all measurable sets $S$:

$$\Pr[\mathcal{M}(D) \in S] \leq e^\varepsilon \Pr[\mathcal{M}(D') \in S] + \delta$$

While DP provides strong privacy guarantees, a critical challenge remains: **how to conduct valid statistical inference on differentially private outputs**. Standard errors computed from noisy statistics must account for both sampling variance and privacy noise to achieve proper confidence interval coverage {cite}`king2024dpd`.

## 1.1 The Fundamental Tradeoff

Previous work has largely focused on point estimation accuracy, but statistical inference requires understanding the full variance structure. Our analysis reveals a sobering reality: **correct standard errors under DP are {eval}`r.se_inflation_10`-{eval}`r.se_inflation_1000`× larger than non-private OLS** at typical privacy budgets ($\varepsilon = 1-10$). Since statistical power scales with $1/\text{SE}^2$, this means:

- To detect a medium effect ($\beta = 0.2$) with 80% power at $\varepsilon = 10$: need **{eval}`r.required_n_eps10`**
- At $\varepsilon = 100$ (weak privacy): still need **{eval}`r.required_n_eps100`** for equivalent power to non-private OLS with $n = 10,000$

This is consistent with {cite}`barrientos2024feasibility`, who found that current DP methods struggle on real administrative data. Our contribution is providing the tools to compute these correct (if large) standard errors, enabling researchers to understand exactly what they're trading off.

## 1.2 Contributions

This paper makes three contributions:

1. **A practical implementation**: We provide dp-statsmodels, an open-source Python library implementing DP-OLS, DP-Logit, and DP-Fixed Effects regression with a statsmodels-compatible API. Bounds must be specified a priori—we removed auto-bounds computation as it voids privacy guarantees.

2. **Valid inference with honest assessment**: We derive standard error formulas that account for privacy noise and demonstrate through simulation that they achieve nominal coverage. We quantify the SE inflation factor ({eval}`r.se_inflation_10`-{eval}`r.se_inflation_1000`×) on real CPS data.

3. **Real-world validation**: Using CPS ASEC data (n={eval}`r.cps.n_fmt`), we show where the method works (point estimation at $\varepsilon \geq 50$) and where it struggles (inference at any typical privacy budget).

# 2. Related Work

## 2.1 Differentially Private Regression

Several approaches exist for DP regression:

**Objective perturbation** {cite}`chaudhuri2011differentially` adds noise to the optimization objective, enabling private empirical risk minimization. While general, it requires iterative optimization and doesn't provide closed-form standard errors.

**Functional mechanism** {cite}`zhang2012functional` perturbs polynomial coefficients of the objective function. It offers good utility but complex variance analysis.

**Noisy sufficient statistics (NSS)** {cite}`sheffet2017differentially` adds calibrated noise to $X'X$ and $X'y$, then solves the normal equations. This provides closed-form solutions with tractable variance.

**Bayesian approaches** {cite}`bernstein2019bayesian` use posterior sampling for privacy. They provide uncertainty quantification but require MCMC.

We focus on NSS because it: (a) provides closed-form estimates, (b) has analytically tractable standard errors, and (c) naturally extends to panel data.

## 2.2 Differentially Private Inference

{cite}`barrientos2019significance` developed DP algorithms for assessing statistical significance of regression coefficients. {cite}`barrientos2024feasibility` conducted the first comprehensive feasibility study of DP regression on real administrative data (IRS tax records and CPS), finding that current methods struggle with accurate confidence intervals on complex datasets. {cite}`williams2024benchmarking` benchmark DP linear regression methods specifically for statistical inference.

## 2.3 Existing Software

**DiffPrivLib** {cite}`diffprivlib2019` provides DP machine learning tools including linear and logistic regression using the functional mechanism. However, like scikit-learn, it focuses on prediction: the `LinearRegression` class returns only coefficients without standard errors, confidence intervals, or p-values.

**OpenDP** {cite}`opendp2024` offers a comprehensive DP framework with composable mechanisms. While powerful, it requires significant expertise and does not provide native standard error computation for regression coefficients.

**dp-statsmodels** fills this gap by providing regression with built-in statistical inference:

| Feature | DiffPrivLib | OpenDP | dp-statsmodels |
|---------|-------------|--------|----------------|
| Linear regression | ✓ | ✓ | ✓ |
| Standard errors | ✗ | ✗ | ✓ |
| Confidence intervals | ✗ | ✗ | ✓ |
| p-values | ✗ | ✗ | ✓ |
| Budget tracking | ✗ | ✓ | ✓ |
| statsmodels-like API | ✗ | ✗ | ✓ |

# 3. Methods

## 3.1 Noisy Sufficient Statistics for OLS

For the linear model $y = X\beta + \varepsilon$, the OLS estimator is:

$$\hat{\beta} = (X'X)^{-1}X'y$$

The sufficient statistics are $X'X$ and $X'y$. We achieve DP by adding Gaussian noise:

$$\widetilde{X'X} = X'X + E_{XX}, \quad \widetilde{X'y} = X'y + e_{Xy}$$

where $E_{XX} \sim N(0, \sigma_{XX}^2 I)$ and $e_{Xy} \sim N(0, \sigma_{Xy}^2 I)$.

## 3.2 Privacy Calibration

The noise scales are calibrated using the Gaussian mechanism {cite}`dwork2014algorithmic`. For sensitivity $\Delta$ and privacy parameters $(\varepsilon, \delta)$:

$$\sigma = \frac{\Delta \sqrt{2\ln(1.25/\delta)}}{\varepsilon}$$

**Sensitivity of $X'X$**: If $x_i \in [L, U]^k$, then $\Delta_{X'X} = (U-L)^2 k$.

**Sensitivity of $X'y$**: If additionally $y_i \in [L_y, U_y]$, then $\Delta_{X'y} = (U-L)(U_y - L_y)\sqrt{k}$.

**Critical requirement**: Bounds must be specified a priori based on domain knowledge. Computing bounds from data voids privacy guarantees entirely.

## 3.3 Variance of the DP Estimator

**Theorem (Variance Decomposition)**: The DP estimator $\tilde{\beta} = (\widetilde{X'X})^{-1}\widetilde{X'y}$ has variance:

$$\text{Var}(\tilde{\beta}) = \underbrace{\sigma^2(X'X)^{-1}}_{\text{sampling variance}} + \underbrace{\sigma_{Xy}^2 (X'X)^{-2}}_{\text{X'y noise}} + \underbrace{O(\sigma_{XX}^2 \|\beta\|^2)}_{\text{X'X noise}}$$

**Proof sketch**: Apply the delta method to $f(A, b) = A^{-1}b$ where $A = X'X + E$ and $b = X'y + e$:

1. First-order Taylor expansion: $\tilde{\beta} \approx \beta + (X'X)^{-1}e_{Xy} - (X'X)^{-1}E_{XX}\beta$
2. The $e_{Xy}$ term contributes $\sigma_{Xy}^2(X'X)^{-2}$
3. The $E_{XX}$ term contributes $\approx \sigma_{XX}^2 \|\beta\|^2 \|(X'X)^{-1}\|^2$

Our implementation uses this first-order approximation following {cite}`evans2024linked`.

## 3.4 Extension to Fixed Effects

For panel data $y_{it} = \alpha_i + X_{it}\beta + \varepsilon_{it}$, we apply the within transformation:

$$\ddot{y}_{it} = y_{it} - \bar{y}_i, \quad \ddot{X}_{it} = X_{it} - \bar{X}_i$$

Then apply NSS to the transformed data.

# 4. Implementation

dp-statsmodels provides a Session-based API for privacy budget tracking:

```{code-cell} python
:tags: [remove-output]

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import dp_statsmodels.api as sm_dp
import statsmodels.api as sm

PAPER_SEED = 42
np.random.seed(PAPER_SEED)
```

```{code-cell} python
# Example API usage
np.random.seed(42)
X = np.random.randn(1000, 2)
y = X @ [1.0, 2.0] + np.random.randn(1000)

# Create session with privacy budget
session = sm_dp.Session(
    epsilon=5.0,
    delta=1e-5,
    bounds_X=(-4, 4),
    bounds_y=(-15, 15),
    random_state=PAPER_SEED
)

# Run DP regression
result = session.OLS(y, X)
print(result.summary())
```

# 5. Simulation Study

We evaluate the method using Monte Carlo simulation with known ground truth.

## 5.1 Data Generating Process

$$y_i = \beta_1 x_{1i} + \beta_2 x_{2i} + \varepsilon_i$$

where $\beta = (1, 2)$, $x_j \sim N(0,1)$, and $\varepsilon \sim N(0,1)$.

```{code-cell} python
:tags: [remove-output]

TRUE_BETA = np.array([1.0, 2.0])
BOUNDS_X = (-4, 4)
BOUNDS_Y = (-15, 15)
DELTA = 1e-5

def generate_data(n, seed=None):
    if seed is not None:
        np.random.seed(seed)
    X = np.random.randn(n, 2)
    y = X @ TRUE_BETA + np.random.randn(n)
    return X, y

def run_ols_simulation(n_obs, epsilon, n_sims=200, base_seed=0):
    results = []
    for sim in range(n_sims):
        data_seed = base_seed + sim * 1000 + int(epsilon * 10)
        X, y = generate_data(n_obs, seed=data_seed)

        model = sm_dp.OLS(
            epsilon=epsilon, delta=DELTA,
            bounds_X=BOUNDS_X, bounds_y=BOUNDS_Y,
            random_state=data_seed
        )
        dp_res = model.fit(y, X, add_constant=True)
        ols_res = sm.OLS(y, sm.add_constant(X)).fit()

        z = 1.96
        covered = [
            dp_res.params[i+1] - z * dp_res.bse[i+1] <= TRUE_BETA[i] <= dp_res.params[i+1] + z * dp_res.bse[i+1]
            for i in range(2)
        ]

        results.append({
            'epsilon': epsilon,
            'dp_beta1': dp_res.params[1],
            'dp_beta2': dp_res.params[2],
            'covered1': covered[0],
            'covered2': covered[1],
            'ols_se1': ols_res.bse[1],
        })
    return pd.DataFrame(results)
```

## 5.2 Results: Bias and Coverage

Simulation results with {eval}`r.config.n_sims` replications:

```{code-cell} python
:tags: [remove-input]

from IPython.display import Markdown
Markdown(r.table_simulation_summary())
```

# 6. Application: CPS ASEC Wage Regression

We demonstrate the method on a classic labor economics application using the Current Population Survey (CPS) Annual Social and Economic Supplement (ASEC) 2024.

## Data

- **Sample**: n = {eval}`r.cps.n_fmt` prime-age workers (25-64 years)
- **Wage skewness**: {eval}`f'{r.cps.wage_skewness:.2f}'` — typical of real-world income data
- **Source**: Census Bureau (publicly available)

## Non-Private OLS Results

- Returns to education: {eval}`r.ols.educ_pct` per year
- Gender wage gap: {eval}`r.ols.gender_gap_pct`

## DP Performance

```{code-cell} python
:tags: [remove-input]

from IPython.display import Markdown
Markdown(r.table_point_estimates())
```

Point estimates converge to OLS at $\varepsilon \geq 50$, but at $\varepsilon = 10$ exhibit high variance (std = 1.94 vs. true effect of 0.09).

## SE Inflation

```{code-cell} python
:tags: [remove-input]

from IPython.display import Markdown
Markdown(r.table_se_inflation())
```

```{code-cell} python
:tags: [remove-input]

import matplotlib.pyplot as plt

se_data = [(10, 300), (50, 70), (100, 37), (500, 8), (1000, 4)]
eps, inflation = zip(*se_data)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(eps, inflation, 'o-', color='#2c7fb8', linewidth=2.5, markersize=10,
        markerfacecolor='white', markeredgewidth=2.5)
ax.axhline(y=1, color='gray', linestyle='--', linewidth=1.5, label='No inflation (OLS)')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Privacy Budget (ε)', fontsize=12)
ax.set_ylabel('SE Inflation Factor (DP SE / OLS SE)', fontsize=12)
ax.set_title('Figure 1: Standard Error Inflation vs. Privacy Budget', fontsize=13)
ax.grid(True, alpha=0.3, which='both')
ax.axvspan(8, 20, alpha=0.15, color='red', label='Strong privacy (ε ≤ 20)')
ax.axvspan(20, 100, alpha=0.15, color='orange', label='Moderate privacy')
ax.axvspan(100, 1500, alpha=0.15, color='green', label='Weak privacy (ε > 100)')
ax.legend(loc='upper right', fontsize=10)
plt.tight_layout()
plt.show()
```

# 7. Discussion

## 7.1 Key Findings

1. **Unbiasedness**: DP-OLS produces approximately unbiased estimates across privacy levels.
2. **Valid Inference**: Our standard error formulas achieve close to 95% coverage in simulations.
3. **The Real Tradeoff**: On real CPS data, SEs are {eval}`r.se_inflation_10`× larger at $\varepsilon = 10$.

## 7.2 Limitations

The gap between Gaussian simulation and real-data performance reflects {cite}`barrientos2024feasibility`'s finding: NSS methods struggle on skewed real-world data.

## 7.3 Recommendations

- **Point estimates**: $\varepsilon \geq 50$ yields estimates within 0.01 of OLS
- **Inference**: SEs remain substantially inflated at all typical privacy budgets

# 8. Conclusion

```{code-cell} python
:tags: [remove-input]

from IPython.display import Markdown
Markdown(r.table_se_inflation())
```

**When to use dp-statsmodels**:
- Very large datasets (n > 10 million)
- Weak privacy requirements ($\varepsilon > 100$)
- Point estimation at moderate privacy ($\varepsilon \geq 50$)

**When NOT to use dp-statsmodels**:
- Standard survey research (n < 100,000) requiring inference
- Strong privacy requirements ($\varepsilon \leq 10$) with hypothesis testing

The library is available at: https://github.com/MaxGhenis/dp-statsmodels

# References

```{bibliography}
:filter: docname in docnames
```
