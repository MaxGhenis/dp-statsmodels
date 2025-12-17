# Choosing Epsilon: A Practical Guide

This guide helps you select an appropriate privacy parameter (epsilon) for your differentially private analysis.

## What is Epsilon?

Epsilon (ε) quantifies the privacy-utility tradeoff in differential privacy:

- **Lower ε** = Stronger privacy, more noise, wider confidence intervals
- **Higher ε** = Weaker privacy, less noise, tighter confidence intervals

Mathematically, ε bounds how much the output distribution can change when any single individual's data is added or removed.

## Quick Reference

| Epsilon | Privacy Level | Typical Use Case | Expected Accuracy |
|---------|--------------|------------------|-------------------|
| ε ≤ 1 | Very strong | Highly sensitive data (health, genetics) | Requires large n (10,000+) |
| ε = 1-3 | Strong | Government statistics, census | Good with n > 1,000 |
| ε = 3-5 | Moderate | Research with sensitive data | Near-OLS accuracy |
| ε = 5-10 | Light | Internal analytics, low sensitivity | Minimal accuracy loss |
| ε ≥ 10 | Minimal | Development/testing | Essentially non-private |

## Decision Framework

### Step 1: Assess Data Sensitivity

Consider the **worst-case harm** if an individual's participation were revealed:

| Sensitivity Level | Examples | Suggested ε Range |
|------------------|----------|-------------------|
| **Critical** | HIV status, genetic data, addiction records | ε ≤ 1 |
| **High** | Income, health conditions, location history | ε = 1-3 |
| **Moderate** | Education, employment, survey responses | ε = 3-5 |
| **Low** | Aggregated/semi-public data | ε = 5-10 |

### Step 2: Consider Sample Size

Privacy noise has a **fixed magnitude** regardless of sample size, while sampling variance decreases with n. This means:

```
Effective accuracy ≈ f(n, ε, k)
```

Where k is the number of parameters. Rules of thumb:

| Sample Size | Minimum Practical ε |
|-------------|---------------------|
| n < 500 | ε ≥ 5 |
| n = 500-2,000 | ε ≥ 2 |
| n = 2,000-10,000 | ε ≥ 1 |
| n > 10,000 | ε ≥ 0.5 |

### Step 3: Account for Multiple Queries

If you plan multiple analyses, divide your total budget:

```python
# Total budget for the project
total_epsilon = 3.0

# Plan your queries
n_queries = 5
epsilon_per_query = total_epsilon / n_queries  # 0.6 per query
```

With dp-statsmodels, the Session automatically tracks cumulative spending:

```python
session = Session(epsilon=3.0, delta=1e-5, bounds_X=(-5, 5), bounds_y=(-50, 50))

# Each query uses part of the budget
result1 = session.OLS(y, X, epsilon=0.6)  # 0.6 spent
result2 = session.OLS(y, X2, epsilon=0.6)  # 1.2 total

print(f"Remaining: {session.epsilon_remaining}")  # 1.8 remaining
```

## Domain-Specific Guidance

### Economics/Policy Research

For academic research with survey or administrative data:

- **Exploratory analysis**: ε = 5-10 (develop models, check specifications)
- **Main results**: ε = 1-3 (publishable with strong privacy)
- **Robustness checks**: Reserve ε = 1-2 for additional specifications

Example budget allocation:
```python
session = Session(epsilon=5.0, delta=1e-5, bounds_X=(-5, 5), bounds_y=(-100, 100))

# Exploratory (use more budget since results may not be published)
explore = session.OLS(y, X_full, epsilon=2.0)

# Main specification
main = session.OLS(y, X_main, epsilon=2.0)

# Robustness
robust = session.OLS(y, X_alt, epsilon=1.0)
```

### Health/Medical Data

Health data often requires stronger privacy:

- **HIPAA-adjacent contexts**: ε ≤ 1
- **Clinical research**: ε = 1-2
- **Population health studies**: ε = 2-3

### Financial Data

For tax or financial records:

- **Individual-level**: ε = 1-3
- **Business records**: ε = 2-5
- **Aggregated statistics**: ε = 5-10

## Understanding the Accuracy Impact

### Confidence Interval Width

With dp-statsmodels, confidence intervals widen to account for privacy noise:

| Epsilon | Relative CI Width (approx.) |
|---------|----------------------------|
| ε = 10 | 1.0x (baseline) |
| ε = 5 | 1.2x |
| ε = 2 | 1.8x |
| ε = 1 | 2.5x |
| ε = 0.5 | 4x |

### Power Analysis

For hypothesis testing, privacy noise reduces statistical power. Plan accordingly:

```python
# Non-private: n=500 gives 80% power to detect effect=0.2
# With ε=2: need n≈1,200 for same power
# With ε=1: need n≈2,500 for same power
```

## Common Pitfalls

### 1. Choosing ε Post-Hoc

**Wrong**: Run analysis, see noisy results, increase ε
**Right**: Set ε based on sensitivity assessment before seeing data

### 2. Ignoring Composition

**Wrong**: Run 10 queries at ε=1, claim ε=1 privacy
**Right**: Total privacy loss is ε=10 (basic composition)

### 3. Comparing to Non-Private Results

**Wrong**: "My DP results don't match OLS, something is broken"
**Right**: DP results should differ; the noise is the privacy protection

## Practical Example

```python
import dp_statsmodels.api as sm_dp
import numpy as np

# Scenario: Analyzing income data (moderate-high sensitivity)
# Sample size: n=5,000
# Plan: 1 main regression + 2 robustness checks

# Budget decision: ε=3 total (strong privacy for income data)
session = sm_dp.Session(
    epsilon=3.0,
    delta=1e-5,
    bounds_X=(-5, 5),      # Standardized covariates
    bounds_y=(0, 500000),  # Income in dollars
    random_state=42        # For reproducibility
)

# Main analysis (use 60% of budget)
main_result = session.OLS(y, X, epsilon=1.8)
print(main_result.summary())

# Robustness checks (20% each)
robust1 = session.OLS(y, X_alt1, epsilon=0.6)
robust2 = session.OLS(y, X_alt2, epsilon=0.6)

# Verify budget
print(f"Total spent: ε = {session.epsilon_spent}")  # Should be 3.0
```

## References

- Dwork, C. & Roth, A. (2014). The Algorithmic Foundations of Differential Privacy. Foundations and Trends in Theoretical Computer Science.
- Abowd, J. M. (2018). The U.S. Census Bureau Adopts Differential Privacy. KDD '18.
- Garfinkel, S. et al. (2019). Understanding Database Reconstruction Attacks on Public Data. Communications of the ACM.
