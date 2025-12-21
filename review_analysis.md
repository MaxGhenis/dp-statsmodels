# Statistical Correctness Re-Review: dp-statsmodels

## Review Date
2025-12-16

## Files Reviewed
- `/Users/maxghenis/dp-statsmodels/dp_statsmodels/models/ols.py`
- `/Users/maxghenis/dp-statsmodels/tests/test_ols.py`
- `/Users/maxghenis/dp-statsmodels/dp_statsmodels/privacy/noisy_stats.py`
- `/Users/maxghenis/dp-statsmodels/dp_statsmodels/privacy/sensitivity.py`

## Assessment of Each Fix

### 1. Residual Variance Computation (FIXED ✓)

**Previous Issue**: Residual variance was computed using true y, leaking privacy.

**Current Implementation** (lines 346-350 in ols.py):
```python
# Estimate residual variance using ONLY noisy sufficient statistics
# RSS = y'y - 2β'X'y + β'X'Xβ = y'y - β'X'y (since β = (X'X)^{-1}X'y)
# This is computed entirely from noisy statistics, fixing the privacy leak
noisy_rss = noisy_yty - params @ noisy_xty
resid_var = max(noisy_rss / (n - k), 1e-10)
```

**Assessment**: ✓ **CORRECT**
- Computes residual variance using only noisy sufficient statistics (noisy_yty, noisy_xty, params)
- Uses the algebraic identity: RSS = y'y - β'X'y (valid when β = (X'X)^{-1}X'y)
- No access to true data
- 20% of privacy budget allocated to y'y (lines 299-318)
- The noisy_yty is computed in `noisy_stats.py` lines 219-277 with proper sensitivity

**Privacy Guarantee**: The residual variance is now properly private. Good fix.

---

### 2. Standard Error Formula (MOSTLY CORRECT, with caveats ⚠️)

**Previous Issue**: Standard error formula had errors in variance propagation.

**Current Implementation** (lines 352-386 in ols.py):
```python
# Compute standard errors using proper variance formula from Evans et al. (2024)
# Var(β̂) = (X'X)^{-1} [σ² X'X + Var(N_Xy)] (X'X)^{-1} + higher order terms
#
# For the X'y noise: Var(β̂)_xty = (X'X)^{-1} Σ_Xy (X'X)^{-1}
# where Σ_Xy = σ²_xty * I (independent noise on each component)
#
# For the X'X noise: This is more complex, but we use a first-order
# approximation: Var(β̂)_xtx ≈ ||β||² σ²_xtx * diag((X'X)^{-2})

sens_xtx = compute_xtx_sensitivity(bounds_X, k)
sens_xty = compute_xty_sensitivity(bounds_X, bounds_y, k)

sigma_xtx = sens_xtx * np.sqrt(2 * np.log(1.25 / delta_each)) / eps_xtx
sigma_xty = sens_xty * np.sqrt(2 * np.log(1.25 / delta_each)) / eps_xty

# Sampling variance component: σ² (X'X)^{-1}
var_sampling = resid_var * np.diag(xtx_inv)

# Privacy noise variance from X'y:
# Var(β̂) from X'y noise = (X'X)^{-1} * σ²_xty * I * (X'X)^{-1}
# = σ²_xty * (X'X)^{-1} * (X'X)^{-1}
# Diagonal elements: σ²_xty * sum_j (xtx_inv[i,j])^2
var_xty_noise = sigma_xty ** 2 * np.sum(xtx_inv ** 2, axis=1)

# Privacy noise variance from X'X (first-order approximation):
# Uses the fact that d(β)/d(X'X) involves β and (X'X)^{-1}
# Approximation: σ²_xtx * ||β||² * diag((X'X)^{-2})
# This is conservative (tends to overestimate variance)
beta_norm_sq = np.sum(params ** 2)
xtx_inv_sq_diag = np.diag(xtx_inv @ xtx_inv)
var_xtx_noise = sigma_xtx ** 2 * beta_norm_sq * xtx_inv_sq_diag

# Total variance
var_total = var_sampling + var_xty_noise + var_xtx_noise
bse = np.sqrt(np.maximum(var_total, 1e-10))
```

**Assessment**: ⚠️ **PARTIALLY CORRECT but with important issues**

**What's Correct:**
1. **Sampling variance**: `σ² * diag((X'X)^{-1})` - standard OLS formula ✓
2. **X'y noise variance**: `σ²_xty * rowSums((X'X)^{-1}²)` - correct delta method ✓
3. **X'X noise variance**: First-order approximation using ||β||² - reasonable but approximate

**Concerns:**

**A. X'X Noise Variance Formula**:
The approximation `σ²_xtx * ||β||² * diag((X'X)^{-2})` is a simplified first-order approximation.

- The true variance from X'X noise requires the full delta method with tensor products
- The current formula uses ||β||² as a crude upper bound on the perturbation effect
- This is **conservative** (tends to overestimate), which is acceptable for DP but not theoretically exact

**Reference Check**: Evans et al. (2024) use a more sophisticated formula involving the full covariance structure. The current implementation is a simplification.

**Recommendation**: The formula is acceptable as a conservative approximation, but should be documented as such. For high-dimensional cases or when X'X noise is large relative to X'y noise, this could be overly conservative.

**B. Sensitivity Calculations**:

Looking at `sensitivity.py`:

```python
def compute_xtx_sensitivity(bounds_X: Tuple[float, float], n_features: int) -> float:
    x_min, x_max = bounds_X
    max_abs = max(abs(x_min), abs(x_max))
    max_row_norm_sq = n_features * max_abs ** 2
    sensitivity = max_row_norm_sq
    return sensitivity
```

This computes L2 (Frobenius) sensitivity as ||x||₂² = k × B². This is **correct** for the Gaussian mechanism.

```python
def compute_xty_sensitivity(bounds_X, bounds_y, n_features) -> float:
    max_abs_x = max(abs(x_min), abs(x_max))
    max_abs_y = max(abs(y_min), abs(y_max))
    max_row_norm = np.sqrt(n_features) * max_abs_x
    sensitivity = max_row_norm * max_abs_y
    return sensitivity
```

This computes ||x||₂ × |y| = √k × B_x × B_y. This is **correct**.

**CRITICAL ISSUE**: When bounds are large or k is large, these sensitivities can become enormous, leading to:
- Numerical overflow (as seen in warnings)
- Extremely large noise
- Useless estimates

This is not a bug in the formula, but a **fundamental limitation** of the NSS approach with bounded data. The library should:
1. Warn when sensitivities are very large
2. Document the importance of tight bounds
3. Consider using clipping or preprocessing to reduce sensitivity

---

### 3. predict() Method (CORRECT ✓)

**Previous Issue**: Missing predict() method.

**Current Implementation** (lines 87-114 in ols.py):
```python
def predict(self, X: np.ndarray) -> np.ndarray:
    """
    Generate predictions for new data.

    Note: Predictions use the noisy coefficients, so they inherit
    the privacy guarantee. No additional privacy budget is consumed.
    """
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    # Add constant if needed
    n_features = len(self.params) - 1 if self._add_constant else len(self.params)
    if X.shape[1] == n_features and self._add_constant:
        X = np.column_stack([np.ones(X.shape[0]), X])

    return X @ self.params
```

**Assessment**: ✓ **CORRECT**
- Properly handles 1D and 2D inputs
- Correctly adds constant when needed
- Uses noisy parameters (post-processing, no additional privacy cost)
- Clear documentation about privacy properties

**Tests**: Lines 487-525 in test_ols.py verify:
- Returns correct shape
- Works with new data
- Both tests pass

---

### 4. Confidence Interval Coverage (NEEDS ATTENTION ⚠️)

**Test Implementation** (lines 191-224 in test_ols.py):
```python
def test_confidence_interval_coverage(self):
    """95% CIs should contain true parameters ~95% of the time."""
    n = 2000
    true_coef = np.array([1.0, 2.0])
    true_intercept = 0.0

    coverage_count = 0
    n_simulations = 100

    for sim in range(n_simulations):
        # Generate new data with fixed seed for reproducibility
        rng = np.random.default_rng(sim)
        X = rng.standard_normal((n, 2))
        y = true_intercept + X @ true_coef + rng.standard_normal(n)

        # Fit DP OLS
        session = PrivacySession(
            epsilon=5.0, delta=1e-5,
            bounds_X=DEFAULT_BOUNDS_X, bounds_y=(-15, 15),
            random_state=sim + 10000
        )
        result = session.ols(y, X)
        ci = result.conf_int(alpha=0.05)

        # Check if true coefficients are in CIs
        true_params = np.concatenate([[true_intercept], true_coef])
        in_ci = (ci[:, 0] <= true_params) & (true_params <= ci[:, 1])
        if all(in_ci):
            coverage_count += 1

    coverage_rate = coverage_count / n_simulations
    # Should be close to 95% - updated threshold to be more stringent
    # Allow some slack due to finite sample approximations
    assert coverage_rate >= 0.85, f"Coverage {coverage_rate:.2%} too low"
```

**Issues**:

1. **Data Generation Problem**: The test generates X ~ N(0,1), but DEFAULT_BOUNDS_X = (-5, 5). With normal data, there's a small probability of values outside bounds, which get clipped. This creates a mismatch between the true data generating process and the clipped data.

2. **Weak Threshold**: The test requires ≥85% coverage for nominal 95% CIs. This is very lenient. Could indicate:
   - Under-coverage of CIs (SEs too small)
   - Conservative SEs but high Monte Carlo error
   - Need more simulations

3. **Finite-Sample vs. Asymptotic**: The test uses t-distribution with df=n-k. For DP OLS with noisy sufficient statistics, the finite-sample distribution is more complex. The t-distribution approximation may not hold well.

**Recommendation**:
- The 85% threshold is concerning. Should aim for ≥90% with proper implementation
- Need to investigate if under-coverage is real or due to Monte Carlo noise
- Consider running test with more simulations (500-1000) to reduce MC error

---

## Additional Statistical Concerns

### 5. Residual Variance Bias

**Code** (line 350):
```python
resid_var = max(noisy_rss / (n - k), 1e-10)
```

**Issue**: When noisy_rss can be negative (due to noise), taking max(., 1e-10) creates a biased estimator:
- E[max(noisy_rss, 1e-10)] ≠ E[noisy_rss]
- This truncation introduces positive bias
- Standard errors will be systematically too large when true RSS is small

**Impact**: Conservative inference (wider CIs than necessary), but still valid for coverage.

**Alternatives**:
- Use E[noisy_rss | noisy_rss > 0] with probability adjustment
- Use a bias-corrected estimator
- Current approach is safe but not optimal

---

### 6. Budget Allocation

**Code** (lines 299-303):
```python
# Split privacy budget: 40% X'X, 40% X'y, 20% y'y (for residual variance)
eps_xtx = self.epsilon * 0.4
eps_xty = self.epsilon * 0.4
eps_yty = self.epsilon * 0.2
delta_each = self.delta / 3
```

**Assessment**: Reasonable but arbitrary
- X'X and X'y get equal shares (40% each)
- y'y gets 20% for residual variance
- This split is not proven optimal

**Alternative**: Could use adaptive allocation based on sensitivity ratios, but current split is reasonable for most applications.

---

## Test Coverage Analysis

**From test results**: 25/25 tests pass ✓

**Test categories**:
1. Basic functionality (4 tests) - ✓ PASS
2. Accuracy (4 tests) - ✓ PASS
3. Statistical inference (2 tests) - ✓ PASS (but coverage threshold is weak)
4. Privacy budget (4 tests) - ✓ PASS
5. Weighted OLS (2 tests) - ✓ PASS
6. Edge cases (4 tests) - ✓ PASS
7. Data bounds (3 tests) - ✓ PASS
8. Prediction (2 tests) - ✓ PASS

**Missing tests**:
- No test verifying residual variance is computed from noisy stats only
- No test checking SE decomposition (sampling vs. privacy noise)
- No test for numerical stability with extreme bounds
- No test verifying privacy guarantee (composition)

---

## Summary

### Fixed Issues ✓
1. **Residual variance**: Now computed using only noisy sufficient statistics (y'y, X'y, X'X). No privacy leak. ✓
2. **predict() method**: Implemented correctly with proper constant handling. ✓

### Partially Fixed Issues ⚠️
3. **Standard error formula**:
   - Sampling variance: Correct ✓
   - X'y noise variance: Correct ✓
   - X'X noise variance: Conservative first-order approximation (not exact) ⚠️

### Remaining Concerns ⚠️
4. **Confidence interval coverage**: Test threshold (85%) is too weak. May indicate under-coverage or high MC error. Needs investigation.
5. **Numerical overflow**: With large bounds or many features, sensitivities explode. Not a bug, but a fundamental limitation that should be documented.
6. **Residual variance truncation**: Creates positive bias when truncating negative noisy_rss.

### Overall Assessment

The fixes address the **critical privacy leak** in residual variance computation and provide reasonable (if approximate) standard errors. The implementation is **suitable for use** with these caveats:

1. Bounds should be as tight as possible (loose bounds → large noise → useless estimates)
2. Confidence intervals may have slightly lower than nominal coverage
3. Standard errors are conservative approximations, not exact

**Grade**: B+ (was D-, significant improvement)

The library is now **statistically sound for practical use**, though some refinements would improve accuracy.
