# dp-econometrics

**Differentially private econometric models with valid statistical inference.**

[![CI](https://github.com/MaxGhenis/dp-econometrics/actions/workflows/ci.yml/badge.svg)](https://github.com/MaxGhenis/dp-econometrics/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

`dp-econometrics` provides differentially private implementations of standard econometric models that return both:
- **Coefficient estimates** with formal (ε,δ)-differential privacy guarantees
- **Standard errors** properly adjusted for privacy noise

This enables valid statistical inference (hypothesis tests, confidence intervals) on sensitive data while maintaining rigorous privacy protections.

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

## Quick Start

```python
import dp_econometrics as dpe
import numpy as np

# Create a privacy session with your budget
session = dpe.PrivacySession(
    epsilon=1.0,
    delta=1e-5,
    bounds_X=(-10, 10),
    bounds_y=(-100, 100)
)

# Generate example data
np.random.seed(42)
n = 1000
X = np.random.randn(n, 3)
y = X @ [1, 2, 3] + np.random.randn(n)

# Run DP OLS
result = session.ols(y, X)

# View results (coefficients + standard errors)
print(result.summary())

# Check privacy budget
print(f"Privacy spent: ε = {session.epsilon_spent:.3f}")
print(f"Remaining: ε = {session.epsilon_remaining:.3f}")
```

## Features

- **Privacy Budget Tracking**: Automatically tracks cumulative privacy loss across queries
- **Valid Inference**: Standard errors account for both sampling and privacy noise
- **Familiar API**: Similar to statsmodels for easy adoption
- **Explicit Bounds**: Requires explicit data bounds for proper DP guarantees
- **Reproducibility**: Optional `random_state` parameter for reproducible results

## Guides

New to differential privacy? Start here:

- **[Choosing Epsilon](docs/guides/choosing_epsilon.md)**: How to select privacy parameters for your analysis
- **[Setting Bounds](docs/guides/setting_bounds.md)**: How to determine appropriate data bounds
- **[Validation Servers](docs/guides/validation_servers.md)**: Workflow guide for secure computing environments

## Supported Models

| Model | Status | Method |
|-------|--------|--------|
| OLS | ✅ Implemented | Noisy Sufficient Statistics |
| WLS | ✅ Implemented | Noisy Sufficient Statistics |
| Logit | 🚧 Planned | Objective Perturbation |
| Fixed Effects | 🚧 Planned | Within Transform + NSS |

## Use Case: Validation Servers

This library is designed for **validation server** architectures where researchers:
1. Develop code on synthetic/public data
2. Submit code to run on confidential data
3. Receive results with privacy guarantees
4. Privacy budget accumulates across queries

See the [Validation Servers Guide](docs/guides/validation_servers.md) for detailed workflow documentation.

## Documentation

Full documentation: [https://maxghenis.github.io/dp-econometrics](https://maxghenis.github.io/dp-econometrics)

## References

- Sheffet, O. (2017). [Differentially Private Ordinary Least Squares](https://proceedings.mlr.press/v70/sheffet17a.html). ICML.
- Evans, G., King, G., et al. (2024). [Differentially Private Linear Regression with Linked Data](https://hdsr.mitpress.mit.edu/pub/4if53bjq). Harvard Data Science Review.
- Dwork, C. & Roth, A. (2014). [The Algorithmic Foundations of Differential Privacy](https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf).

## License

MIT License. See [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please see our [contributing guidelines](CONTRIBUTING.md).
