# Reproducibility Guide

This guide explains how to ensure reproducible results when using dp-statsmodels.

## Why Reproducibility Matters

Differential privacy adds random noise to protect individuals. While essential for privacy, this randomness can make results vary across runs. Reproducibility is crucial for:

- **Publication**: Meeting scientific standards
- **Debugging**: Consistently reproducing errors
- **Collaboration**: Sharing exact results with others

## The `random_state` Parameter

All dp-statsmodels operations support `random_state` for deterministic results.

### Basic Usage

```python
from dp_statsmodels import Session

# For reproducible results, set random_state to any integer
session = Session(
    epsilon=1.0,
    delta=1e-5,
    bounds_X=(-5, 5),
    bounds_y=(-20, 20),
    random_state=42  # This makes results reproducible
)

result = session.OLS(y, X)
```

### What Gets Fixed?

When you set `random_state`, the following become deterministic:
- Privacy noise added to sufficient statistics (X'X, X'y, y'y)
- Coefficient estimates
- Standard errors
- Confidence intervals
- All derived statistics

### Verification Example

```python
import numpy as np
from dp_statsmodels import Session

# Generate data
np.random.seed(100)
X = np.random.randn(1000, 2)
y = X @ [1.0, 2.0] + np.random.randn(1000)

# Run 1
session1 = Session(epsilon=1.0, delta=1e-5, random_state=42,
                   bounds_X=(-5, 5), bounds_y=(-20, 20))
result1 = session1.OLS(y, X)

# Run 2 - Same seed
session2 = Session(epsilon=1.0, delta=1e-5, random_state=42,
                   bounds_X=(-5, 5), bounds_y=(-20, 20))
result2 = session2.OLS(y, X)

# Verify identical
assert np.array_equal(result1.params, result2.params)
print("Results are reproducible!")
```

## Random vs. Deterministic

### Deterministic (Recommended for Publication)

```python
session = Session(epsilon=1.0, delta=1e-5, random_state=42,
                  bounds_X=(-5, 5), bounds_y=(-20, 20))
```

Use when:
- Publishing research
- Running production analyses
- Debugging code
- Need exact reproducibility

### Random (Default)

```python
session = Session(epsilon=1.0, delta=1e-5,
                  bounds_X=(-5, 5), bounds_y=(-20, 20))
# random_state defaults to None
```

Use when:
- Exploring data interactively
- Want to see natural variation
- Sensitivity analysis

## Best Practices

### 1. Always Set Seeds for Published Work

```python
# Good - reproducible
session = Session(epsilon=1.0, delta=1e-5, random_state=42, ...)

# Bad - will vary
session = Session(epsilon=1.0, delta=1e-5, ...)
```

### 2. Document Your Seed

```python
RANDOM_SEED = 42  # For reproducibility

session = Session(epsilon=1.0, delta=1e-5, random_state=RANDOM_SEED, ...)
```

### 3. Separate Data and Analysis Seeds

```python
# Data generation seed
np.random.seed(100)
X = np.random.randn(1000, 3)
y = X @ beta + np.random.randn(1000)

# DP analysis seed (independent)
session = Session(epsilon=1.0, delta=1e-5, random_state=42, ...)
```

## Troubleshooting

### "My results keep changing!"

1. **Check random_state**: Ensure you set it explicitly
2. **Check data generation**: Use `np.random.seed()` for data
3. **Same parameters**: Verify epsilon, delta, bounds are identical

### "Results differ between machines"

Ensure:
- Same Python version
- Same NumPy version
- Same dp-statsmodels version

```python
import sys, numpy as np, dp_statsmodels
print(f"Python: {sys.version}")
print(f"NumPy: {np.__version__}")
print(f"dp-statsmodels: {dp_statsmodels.__version__}")
```

## Complete Example

```python
"""Fully reproducible DP analysis."""
import numpy as np
from dp_statsmodels import Session

# Configuration
DATA_SEED = 100
DP_SEED = 42
EPSILON = 1.0

# Generate data (reproducible)
np.random.seed(DATA_SEED)
X = np.random.randn(1000, 2)
true_beta = [1.0, 2.0]
y = X @ true_beta + np.random.randn(1000)

# DP analysis (reproducible)
session = Session(
    epsilon=EPSILON, delta=1e-5, random_state=DP_SEED,
    bounds_X=(-5, 5), bounds_y=(-15, 15)
)
result = session.OLS(y, X)

print(f"Estimates: {result.params}")
print(f"Standard errors: {result.bse}")

# Verify reproducibility
session2 = Session(
    epsilon=EPSILON, delta=1e-5, random_state=DP_SEED,
    bounds_X=(-5, 5), bounds_y=(-15, 15)
)
result2 = session2.OLS(y, X)
assert np.array_equal(result.params, result2.params)
print("Reproducibility verified!")
```
