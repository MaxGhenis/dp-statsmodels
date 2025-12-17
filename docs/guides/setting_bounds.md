# Setting Data Bounds: A Practical Guide

Data bounds are **critical** for differential privacy. This guide explains why bounds matter and how to set them correctly.

## Why Bounds Are Required

Differential privacy requires knowing the **sensitivity** of your computation - how much the output can change when one person's data changes. For regression:

- Sensitivity of X'X depends on maximum possible ||x||
- Sensitivity of X'y depends on maximum possible ||x|| and |y|

Without bounds, sensitivity is infinite, and no finite amount of noise provides privacy.

**dp-statsmodels enforces this by default:**

```python
session = Session(epsilon=1.0, delta=1e-5)
result = session.OLS(y, X)  # Raises ValueError!

# Error: bounds_X is required for differential privacy guarantees.
# Computing bounds from data completely breaks privacy.
```

## The Bounds-Noise Tradeoff

Bounds directly affect noise magnitude:

| Bounds | Sensitivity | Noise Added | Privacy |
|--------|-------------|-------------|---------|
| Tight (correct) | Low | Low | Valid |
| Wide (conservative) | High | High | Valid but noisy |
| Too tight (incorrect) | Low | Low | **INVALID** (clipping changes results) |
| From data | N/A | N/A | **NO PRIVACY** |

**Key insight**: Wider bounds are always "safe" for privacy but increase noise. The goal is bounds that are tight but still cover your data.

## Setting Bounds: Three Approaches

### Approach 1: Domain Knowledge (Recommended)

Use what you know about the data:

```python
# Income data: legal minimum is 0, practical max ~$500k for most analyses
bounds_y = (0, 500000)

# Age: 18-100 for adult studies
bounds_X_age = (18, 100)

# Education years: 0-25
bounds_X_edu = (0, 25)

# Standardized variables: typically within [-5, 5]
bounds_X_std = (-5, 5)
```

### Approach 2: Theoretical Ranges

Use the theoretical range of your variables:

| Variable Type | Theoretical Bounds |
|--------------|-------------------|
| Binary (0/1) | (0, 1) |
| Percentage | (0, 100) |
| Likert scale (1-5) | (1, 5) |
| Log-transformed | Based on original range |
| Z-scores | (-4, 4) or (-5, 5) |

### Approach 3: Public Data Reference (Use with Care)

If similar public data exists, use its range:

```python
# From published summary statistics of similar data
# "Income ranges from $0 to $450,000 in the ACS"
bounds_y = (0, 500000)  # Slightly wider for safety
```

**Warning**: Don't use the confidential data itself to set bounds - this leaks privacy!

## Variable-by-Variable Guide

### Continuous Outcome Variables (y)

```python
# Income (dollars)
bounds_y = (0, 500000)  # Or use log(income)

# Log income
bounds_y = (0, 15)  # log(1) to log(~3.3M)

# Test scores (0-100 scale)
bounds_y = (0, 100)

# Health expenditure
bounds_y = (0, 100000)  # Or use log transform
```

### Continuous Predictors (X)

```python
# Standardized variables (z-scores)
bounds_X = (-5, 5)  # Covers 99.9999% of normal distribution

# Age (years)
bounds_X = (0, 120)  # Or restrict to study population (18, 85)

# Years of education
bounds_X = (0, 25)

# Number of children
bounds_X = (0, 15)
```

### Binary/Categorical Predictors

```python
# Binary indicator
bounds_X = (0, 1)

# Dummy variables from one-hot encoding
bounds_X = (0, 1)  # Each dummy is 0 or 1
```

### Interaction Terms

For interactions, bounds multiply:

```python
# age * education
# If age in (18, 85) and edu in (0, 25)
bounds_interaction = (18 * 0, 85 * 25)  # (0, 2125)

# Better: standardize before interacting
# age_std * edu_std, each in (-5, 5)
bounds_interaction = (-25, 25)
```

## Common Patterns

### Pattern 1: Standardize First

Standardizing variables simplifies bounds:

```python
import numpy as np

# On PUBLIC/SYNTHETIC data, compute standardization parameters
X_mean = X_public.mean(axis=0)
X_std = X_public.std(axis=0)

# Apply to confidential data
X_standardized = (X_confidential - X_mean) / X_std

# Now use simple bounds
session = Session(
    epsilon=1.0,
    delta=1e-5,
    bounds_X=(-5, 5),  # Works for any standardized variable
    bounds_y=(-5, 5)   # If y is also standardized
)
```

### Pattern 2: Log Transform Skewed Variables

```python
# Income is highly skewed
# Instead of bounds_y = (0, 10000000)  # Very wide!

# Use log transform
y_log = np.log1p(y)  # log(1 + y) handles zeros
bounds_y = (0, 17)   # log1p(10M) ≈ 16.1
```

### Pattern 3: Winsorize Outliers (Pre-DP)

If extreme outliers exist, consider winsorizing on public data:

```python
# On PUBLIC data, find reasonable percentiles
p1, p99 = np.percentile(y_public, [1, 99])

# Set bounds slightly beyond
bounds_y = (p1 * 0.9, p99 * 1.1)

# Note: The actual DP analysis will CLIP values outside bounds
# This is privacy-safe but may introduce bias if many values clipped
```

## What Happens If Bounds Are Wrong?

### Too Tight (Data Exceeds Bounds)

Data is **clipped** to bounds before computing statistics:

```python
# If bounds_y = (0, 100) but actual max is 150
# Values > 100 become 100
# This biases results but maintains privacy
```

**Signs of too-tight bounds:**
- Results seem biased toward bound values
- Coefficients are attenuated

### Too Wide (Bounds Far Exceed Data)

More noise is added than necessary:

```python
# If bounds_y = (0, 1000000) but actual max is 50000
# Sensitivity is 20x higher than needed
# Results are valid but unnecessarily noisy
```

**Signs of too-wide bounds:**
- Very wide confidence intervals
- Low statistical significance despite large effects

## Multi-Dimensional Bounds

When X has multiple columns with different scales:

```python
# Option 1: Single bounds (simpler but may be loose)
bounds_X = (-100, 100)  # Must cover ALL columns

# Option 2: Standardize everything (recommended)
X_std = (X - X.mean(axis=0)) / X.std(axis=0)
bounds_X = (-5, 5)  # Works for all standardized columns
```

## Complete Example

```python
import dp_statsmodels.api as sm_dp
import numpy as np

# Scenario: Wage regression with education, experience, gender
# Variables:
#   y: hourly wage ($5-$200 for most workers)
#   X1: years of education (0-25)
#   X2: years of experience (0-50)
#   X3: female indicator (0 or 1)

# Standardize continuous variables (on public/synthetic data)
edu_mean, edu_std = 13.5, 3.0  # From public statistics
exp_mean, exp_std = 20.0, 12.0

# Prepare data
X = np.column_stack([
    (education - edu_mean) / edu_std,  # Standardized education
    (experience - exp_mean) / exp_std,  # Standardized experience
    female  # Already 0/1
])

# For y, use log wage
y = np.log(hourly_wage)

# Set bounds
session = sm_dp.Session(
    epsilon=2.0,
    delta=1e-5,
    bounds_X=(-5, 5),  # Standardized vars and 0/1 dummies
    bounds_y=(1.5, 6),  # log(5) to log(400)
    random_state=42
)

result = session.OLS(y, X)
print(result.summary())
```

## Checklist Before Running DP Analysis

- [ ] All bounds set using domain knowledge or public data
- [ ] Bounds are NOT computed from confidential data
- [ ] Continuous variables standardized if scales differ
- [ ] Binary variables bounded at (0, 1)
- [ ] Skewed variables log-transformed
- [ ] Bounds are slightly wider than expected range
- [ ] Interaction terms have appropriate bounds

## Common Mistakes

### Mistake 1: Computing Bounds from Data

```python
# WRONG - This leaks privacy!
bounds_y = (y.min(), y.max())

# RIGHT - Use domain knowledge
bounds_y = (0, 500000)  # Known range for income
```

### Mistake 2: Forgetting Intercept Column

If your X includes a column of 1s for intercept:

```python
# The intercept column is always 1
# bounds_X must accommodate this
# If using (-5, 5), intercept is clipped but still valid

# Better: let dp-statsmodels handle intercept automatically
result = session.OLS(y, X, add_constant=True)
```

### Mistake 3: Using Different Bounds for Same Variable

```python
# Inconsistent across analyses
result1 = session.OLS(y, X, bounds_y=(0, 100000))
result2 = session.OLS(y, X2, bounds_y=(0, 200000))  # Different!

# Better: set once at session level
session = Session(epsilon=1.0, delta=1e-5,
                  bounds_X=(-5, 5), bounds_y=(0, 200000))
```

## References

- Dwork, C. & Roth, A. (2014). The Algorithmic Foundations of Differential Privacy. Section 3 on Sensitivity.
- Sheffet, O. (2017). Differentially Private Ordinary Least Squares. ICML. Sensitivity calculations for OLS.
