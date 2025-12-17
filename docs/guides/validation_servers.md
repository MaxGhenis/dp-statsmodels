# Validation Server Workflow Guide

This guide explains how to use dp-statsmodels in a **validation server** architecture, where researchers submit code to run on confidential data.

## What is a Validation Server?

A validation server is a secure computing environment that:

1. Holds confidential microdata (tax records, health data, census responses)
2. Accepts code submissions from approved researchers
3. Runs analyses with differential privacy protections
4. Returns privatized results to researchers
5. Tracks cumulative privacy budget

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Researcher    │     │  Validation      │     │  Confidential   │
│   Workstation   │────▶│  Server          │────▶│  Data           │
│                 │     │                  │     │                 │
│ - Synthetic data│     │ - Runs DP code   │     │ - Never leaves  │
│ - Develop code  │◀────│ - Tracks budget  │     │   secure env    │
│ - View results  │     │ - Returns output │     │                 │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

## Workflow Overview

### Phase 1: Development (On Your Machine)

Work with synthetic or public data that mimics the confidential data:

```python
import dp_statsmodels.api as sm_dp
import numpy as np

# Create synthetic data matching expected structure
np.random.seed(42)
n = 5000
synthetic_X = np.column_stack([
    np.random.normal(13, 3, n),      # Education years
    np.random.normal(20, 10, n),     # Experience years
    np.random.binomial(1, 0.5, n),   # Female indicator
])
synthetic_y = synthetic_X @ [0.08, 0.02, -0.15] + np.random.normal(2.5, 0.4, n)

# Develop your analysis
session = sm_dp.Session(
    epsilon=2.0,
    delta=1e-5,
    bounds_X=(-5, 30),  # Cover education, experience, binary
    bounds_y=(1, 5),    # Log wage bounds
    random_state=42
)

result = session.OLS(synthetic_y, synthetic_X)
print(result.summary())
```

### Phase 2: Submission

Package your code for the validation server:

```python
# analysis_submission.py
"""
Wage regression analysis for validation server submission.

Privacy Budget Request: epsilon=2.0, delta=1e-5
Expected Queries: 1 OLS regression
"""

import dp_statsmodels.api as sm_dp
import numpy as np

def run_analysis(X, y, epsilon=2.0, delta=1e-5):
    """
    Main analysis function.

    Parameters
    ----------
    X : ndarray
        Feature matrix [education, experience, female]
    y : ndarray
        Log hourly wage
    epsilon : float
        Privacy budget for this analysis
    delta : float
        Privacy parameter

    Returns
    -------
    dict
        Results including coefficients, standard errors, summary
    """
    session = sm_dp.Session(
        epsilon=epsilon,
        delta=delta,
        bounds_X=(-5, 30),
        bounds_y=(1, 5),
        random_state=42  # For reproducibility
    )

    result = session.OLS(y, X)

    return {
        'params': result.params.tolist(),
        'std_errors': result.bse.tolist(),
        'conf_int': result.conf_int().tolist(),
        'summary': result.summary(),
        'epsilon_used': session.epsilon_spent,
        'nobs': result.nobs
    }

# For local testing
if __name__ == '__main__':
    # Load synthetic data
    data = np.load('synthetic_data.npz')
    results = run_analysis(data['X'], data['y'])
    print(results['summary'])
```

### Phase 3: Review and Results

After submission, the server runs your code and returns results:

```
=== Analysis Results ===
Privacy Used: ε=2.0, δ=1e-5
Cumulative Budget: ε=2.0 of 5.0 remaining

                 DP OLS Results
================================================
Dep. Variable:                  y   R-squared:     N/A
Method:               DP-OLS (NSS)  Adj. R-sq:     N/A
Privacy:         ε=2.00, δ=1.0e-05  F-stat:        N/A
No. Observations:            5000
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
x1             0.0823      0.008     10.288      0.000       0.067       0.098
x2             0.0198      0.004      4.950      0.000       0.012       0.028
x3            -0.1534      0.022     -6.973      0.000      -0.196      -0.110
==============================================================================
```

## Budget Planning

### Allocating Your Total Budget

Most validation servers assign a total privacy budget per project:

```python
# Example: Total budget ε=5.0 for a research project

# Plan your queries
budget_plan = {
    'exploratory': 1.0,     # Initial exploration
    'main_spec': 2.0,       # Main regression
    'robustness_1': 1.0,    # Alternative specification
    'robustness_2': 1.0,    # Subgroup analysis
}
# Total: 5.0
```

### Tracking Across Sessions

dp-statsmodels tracks budget automatically within a session:

```python
session = sm_dp.Session(epsilon=5.0, delta=1e-5,
                        bounds_X=(-5, 5), bounds_y=(-5, 5))

# Query 1
result1 = session.OLS(y, X1, epsilon=1.0)
print(f"After Q1: {session.epsilon_spent:.1f} spent")  # 1.0

# Query 2
result2 = session.OLS(y, X2, epsilon=2.0)
print(f"After Q2: {session.epsilon_spent:.1f} spent")  # 3.0

# Query 3
result3 = session.OLS(y, X3, epsilon=1.5)
print(f"After Q3: {session.epsilon_spent:.1f} spent")  # 4.5

# Remaining
print(f"Remaining: {session.epsilon_remaining:.1f}")   # 0.5
```

### What Happens When Budget is Exhausted?

```python
# With 0.5 remaining...
result4 = session.OLS(y, X4, epsilon=1.0)  # Raises ValueError!

# Error: Privacy budget exceeded.
# Requested ε=1.0. Available ε=0.5
```

## Best Practices

### 1. Develop on Synthetic Data First

Create synthetic data that matches the confidential data's structure:

```python
def create_synthetic_data(n=5000, seed=42):
    """Create synthetic data matching CPS structure."""
    np.random.seed(seed)

    # Match variable distributions from public documentation
    education = np.clip(np.random.normal(13.5, 3.0, n), 0, 25)
    experience = np.clip(np.random.normal(20, 12, n), 0, 50)
    female = np.random.binomial(1, 0.47, n)

    # Plausible relationship
    log_wage = (
        2.0 +
        0.08 * education +
        0.02 * experience +
        -0.15 * female +
        np.random.normal(0, 0.4, n)
    )

    return {
        'X': np.column_stack([education, experience, female]),
        'y': log_wage
    }
```

### 2. Test Budget Exhaustion

Verify your code handles budget limits gracefully:

```python
def safe_query(session, y, X, epsilon, fallback_epsilon=None):
    """Run query with budget check."""
    if not session._accountant.can_afford(epsilon, session.delta):
        if fallback_epsilon and session._accountant.can_afford(fallback_epsilon, session.delta):
            print(f"Using fallback budget: {fallback_epsilon}")
            epsilon = fallback_epsilon
        else:
            raise RuntimeError(f"Insufficient budget. Have {session.epsilon_remaining:.2f}")

    return session.OLS(y, X, epsilon=epsilon)
```

### 3. Document Your Bounds

Include clear documentation of bound choices:

```python
"""
Data Bounds Documentation
=========================

bounds_X = (-5, 30):
- Education (0-25 years): Covered
- Experience (0-50 years): Need to standardize or widen
- Female (0-1): Covered

bounds_y = (1, 5):
- Log hourly wage: log($3) to log($150)
- Based on minimum wage floor and 99th percentile from ACS

Note: Experience standardized as (exp - 20) / 12 before analysis.
"""
```

### 4. Save Intermediate Results

Request that intermediate results be saved:

```python
def run_analysis_with_checkpoints(X, y, session):
    """Analysis with saved checkpoints."""
    results = {}

    # Main model
    results['main'] = session.OLS(y, X)
    checkpoint_1 = {
        'epsilon_spent': session.epsilon_spent,
        'params': results['main'].params.tolist()
    }

    # Continue only if budget allows
    if session.epsilon_remaining >= 1.0:
        results['robust'] = session.OLS(y, X[:, :2])

    return results, checkpoint_1
```

## Server Operator Guide

### Setting Up dp-statsmodels for a Validation Server

```python
# server_config.py

from dp_statsmodels import Session
from dataclasses import dataclass
from typing import Dict
import json

@dataclass
class ProjectBudget:
    """Track budget for a research project."""
    project_id: str
    total_epsilon: float
    total_delta: float
    spent_epsilon: float = 0.0
    spent_delta: float = 0.0
    queries: list = None

    def __post_init__(self):
        self.queries = self.queries or []

    def can_afford(self, epsilon: float, delta: float) -> bool:
        return (self.spent_epsilon + epsilon <= self.total_epsilon and
                self.spent_delta + delta <= self.total_delta)

    def record_query(self, epsilon: float, delta: float, query_type: str):
        self.spent_epsilon += epsilon
        self.spent_delta += delta
        self.queries.append({
            'epsilon': epsilon,
            'delta': delta,
            'type': query_type,
            'cumulative_epsilon': self.spent_epsilon
        })

def run_submitted_analysis(code_path: str, data: Dict, budget: ProjectBudget):
    """
    Execute submitted analysis code with budget tracking.
    """
    # Verify budget
    requested_epsilon = extract_epsilon_from_code(code_path)
    if not budget.can_afford(requested_epsilon, 1e-5):
        return {'error': 'Insufficient budget'}

    # Create session with remaining budget
    session = Session(
        epsilon=budget.total_epsilon - budget.spent_epsilon,
        delta=budget.total_delta - budget.spent_delta,
        bounds_X=data['bounds_X'],
        bounds_y=data['bounds_y']
    )

    # Execute code (in sandbox)
    result = execute_in_sandbox(code_path, data, session)

    # Update budget
    budget.record_query(session.epsilon_spent, session.delta_spent, 'OLS')

    return result
```

### Audit Logging

```python
import logging
from datetime import datetime

def log_query(project_id: str, query_result, budget: ProjectBudget):
    """Log privacy-relevant query information."""
    logging.info(json.dumps({
        'timestamp': datetime.now().isoformat(),
        'project_id': project_id,
        'query_type': 'OLS',
        'epsilon_used': query_result.epsilon_used,
        'delta_used': query_result.delta_used,
        'cumulative_epsilon': budget.spent_epsilon,
        'remaining_epsilon': budget.total_epsilon - budget.spent_epsilon,
        'nobs': query_result.nobs,
        'n_params': len(query_result.params)
    }))
```

## FAQ

### Q: Can I see the raw data to set bounds?

**No.** Bounds must be set using domain knowledge or public data. Seeing the confidential data to set bounds breaks privacy.

### Q: What if my results are too noisy?

Options:
- Request a larger privacy budget (if available)
- Simplify your model (fewer parameters = less noise)
- Focus on variables with larger effect sizes
- Use a larger sample (if subsampling)

### Q: Can I run the same analysis twice?

Each run costs privacy budget. Running the same analysis twice uses 2x the budget. Results will differ due to random noise.

### Q: How do I know if my bounds are correct?

Test on synthetic data with known true values. If coefficients are biased toward bounds, bounds may be too tight. If results are very noisy, bounds may be too wide.

### Q: What's the minimum sample size?

Depends on epsilon and number of parameters. Rules of thumb:
- ε ≥ 5: n ≥ 500
- ε = 2-5: n ≥ 1,000
- ε ≤ 2: n ≥ 5,000
