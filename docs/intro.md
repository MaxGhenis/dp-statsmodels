# dp-econometrics

**Differentially private econometric models with valid statistical inference.**

## Overview

`dp-econometrics` is a Python library for running standard econometric analyses on sensitive data with formal differential privacy guarantees. Unlike existing DP libraries that only provide point estimates, this library provides:

- **Coefficients** with formal (ε,δ)-differential privacy guarantees
- **Standard errors** properly adjusted for privacy noise
- **Privacy budget tracking** across multiple analyses
- **Familiar API** similar to statsmodels

## Motivation

Researchers analyzing sensitive data (tax returns, health records, survey microdata) face a fundamental tradeoff: access to detailed data enables rigorous analysis, but risks individual privacy. Differential privacy provides a mathematical framework to quantify and bound this risk.

However, existing DP tools have critical gaps for econometric research:

| Tool | Coefficients | Std Errors | Fixed Effects | Budget Tracking |
|------|--------------|------------|---------------|-----------------|
| diffprivlib | ✅ | ❌ | ❌ | ❌ |
| OpenDP | Basic stats | ❌ | ❌ | ✅ |
| **dp-econometrics** | ✅ | ✅ | ✅ | ✅ |

## Use Case: Validation Servers

This library is designed for **validation server** architectures where:

1. Researchers develop and test code on synthetic/public data
2. Code is submitted to run on confidential data
3. Each query consumes privacy budget
4. Results include proper uncertainty quantification

See the [Validation Servers Guide](guides/validation_servers.md) for detailed workflow documentation.

## Practical Guides

- **[Choosing Epsilon](guides/choosing_epsilon.md)**: How to select privacy parameters based on data sensitivity and sample size
- **[Setting Bounds](guides/setting_bounds.md)**: How to determine appropriate data bounds for valid privacy guarantees

## Quick Example

```python
import dp_econometrics as dpe
import numpy as np

# Create a privacy session with total budget
session = dpe.PrivacySession(epsilon=1.0, delta=1e-5)

# Generate example data
np.random.seed(42)
n, k = 1000, 3
X = np.random.randn(n, k)
y = X @ [1, 2, 3] + np.random.randn(n)

# Fit OLS with differential privacy
result = session.ols(y, X)

# Results include coefficients AND valid standard errors
print(result.summary())
#            coef    std_err    t_stat    p_value    [0.025    0.975]
# const     0.032      0.045     0.711      0.477    -0.056     0.120
# x1        0.987      0.048    20.563      0.000     0.893     1.081
# x2        2.014      0.051    39.490      0.000     1.914     2.114
# x3        2.978      0.047    63.362      0.000     2.886     3.070

# Check remaining privacy budget
print(f"Privacy spent: ε = {session.epsilon_spent:.3f}")
print(f"Remaining: ε = {session.epsilon_remaining:.3f}")
```

## Installation

```bash
pip install dp-econometrics
```

Or install from source:

```bash
git clone https://github.com/MaxGhenis/dp-econometrics.git
cd dp-econometrics
pip install -e ".[dev]"
```

## Supported Models

### Currently Implemented
- **OLS** (Ordinary Least Squares) via Noisy Sufficient Statistics
- **WLS** (Weighted Least Squares)

### Planned
- **Logit** (Logistic Regression)
- **Fixed Effects** (Panel data with entity effects)
- **IV/2SLS** (Instrumental Variables)

## How It Works

The library uses **Noisy Sufficient Statistics (NSS)** {cite}`sheffet2017differentially` for linear models:

1. Compute sufficient statistics: $X'X$ and $X'y$
2. Add calibrated Gaussian noise for (ε,δ)-DP
3. Solve: $\hat{\beta} = (X'X + \text{noise})^{-1}(X'y + \text{noise})$
4. Compute variance using analytical formulas {cite}`evans2024linked`

This approach is more efficient than DP-SGD for linear models and has well-understood variance formulas.

## Citation

```bibtex
@software{ghenis2024dp_econometrics,
  author = {Ghenis, Max},
  title = {dp-econometrics: Differentially Private Econometrics with Valid Inference},
  year = {2024},
  url = {https://github.com/MaxGhenis/dp-econometrics}
}
```

## License

MIT License. See [LICENSE](https://github.com/MaxGhenis/dp-econometrics/blob/main/LICENSE) for details.

```{tableofcontents}
```
