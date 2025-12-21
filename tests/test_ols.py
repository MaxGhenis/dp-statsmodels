"""
Tests for OLS regression with Noisy Sufficient Statistics.

Following TDD: These tests are written BEFORE the implementation.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

# These imports will fail until we implement the modules
from dp_statsmodels import PrivacySession
from dp_statsmodels.models import DPOLS

# Standard bounds for test data (normal data typically in [-5, 5])
DEFAULT_BOUNDS_X = (-5, 5)
DEFAULT_BOUNDS_Y = (-20, 20)


class TestDPOLSBasic:
    """Basic functionality tests for DP OLS."""

    @pytest.fixture
    def simple_data(self):
        """Generate simple regression data with known coefficients."""
        np.random.seed(42)
        n = 1000
        X = np.random.randn(n, 3)
        true_coef = np.array([1.0, 2.0, 3.0])
        y = X @ true_coef + np.random.randn(n) * 0.5
        return X, y, true_coef

    def test_ols_returns_coefficients(self, simple_data):
        """OLS should return coefficient estimates."""
        X, y, _ = simple_data
        session = PrivacySession(
            epsilon=10.0, delta=1e-5,
            bounds_X=DEFAULT_BOUNDS_X, bounds_y=DEFAULT_BOUNDS_Y
        )
        result = session.ols(y, X)

        assert hasattr(result, "params")
        assert len(result.params) == X.shape[1] + 1  # +1 for intercept

    def test_ols_returns_standard_errors(self, simple_data):
        """OLS should return standard errors for all coefficients."""
        X, y, _ = simple_data
        session = PrivacySession(
            epsilon=10.0, delta=1e-5,
            bounds_X=DEFAULT_BOUNDS_X, bounds_y=DEFAULT_BOUNDS_Y
        )
        result = session.ols(y, X)

        assert hasattr(result, "bse")  # Standard errors (like statsmodels)
        assert len(result.bse) == len(result.params)
        assert all(se > 0 for se in result.bse)  # SEs must be positive

    def test_ols_returns_confidence_intervals(self, simple_data):
        """OLS should return confidence intervals."""
        X, y, _ = simple_data
        session = PrivacySession(
            epsilon=10.0, delta=1e-5,
            bounds_X=DEFAULT_BOUNDS_X, bounds_y=DEFAULT_BOUNDS_Y
        )
        result = session.ols(y, X)

        ci = result.conf_int(alpha=0.05)
        assert ci.shape == (len(result.params), 2)
        assert all(ci[:, 0] < ci[:, 1])  # Lower < Upper

    def test_ols_returns_t_stats_and_pvalues(self, simple_data):
        """OLS should return t-statistics and p-values."""
        X, y, _ = simple_data
        session = PrivacySession(
            epsilon=10.0, delta=1e-5,
            bounds_X=DEFAULT_BOUNDS_X, bounds_y=DEFAULT_BOUNDS_Y
        )
        result = session.ols(y, X)

        assert hasattr(result, "tvalues")
        assert hasattr(result, "pvalues")
        assert len(result.tvalues) == len(result.params)
        assert all(0 <= p <= 1 for p in result.pvalues)


class TestDPOLSAccuracy:
    """Tests for accuracy of DP OLS estimates."""

    @pytest.fixture
    def regression_data(self):
        """Generate regression data for accuracy tests."""
        np.random.seed(123)
        n = 5000  # Larger sample for accuracy
        X = np.random.randn(n, 2)
        true_coef = np.array([1.5, -2.0])
        true_intercept = 0.5
        y = true_intercept + X @ true_coef + np.random.randn(n) * 1.0
        return X, y, true_intercept, true_coef

    def test_coefficients_close_to_true_high_epsilon(self, regression_data):
        """With high epsilon (low privacy), coefficients should be close to OLS."""
        X, y, true_intercept, true_coef = regression_data
        session = PrivacySession(
            epsilon=100.0, delta=1e-5,
            bounds_X=DEFAULT_BOUNDS_X, bounds_y=(-20, 20)
        )
        result = session.ols(y, X)

        # Intercept (allow some slack due to noise)
        assert_allclose(result.params[0], true_intercept, atol=0.3)
        # Coefficients (allow some slack due to noise)
        assert_allclose(result.params[1:], true_coef, atol=0.2)

    def test_coefficients_noisier_with_low_epsilon(self, regression_data):
        """With low epsilon (high privacy), coefficients should be noisier."""
        X, y, _, _ = regression_data

        # Run multiple times with same data, different privacy levels
        results_high_eps = []
        results_low_eps = []

        for seed in range(10):
            session_high = PrivacySession(
                epsilon=50.0, delta=1e-5,
                bounds_X=DEFAULT_BOUNDS_X, bounds_y=(-20, 20),
                random_state=seed
            )
            session_low = PrivacySession(
                epsilon=0.5, delta=1e-5,
                bounds_X=DEFAULT_BOUNDS_X, bounds_y=(-20, 20),
                random_state=seed + 1000
            )

            result_high = session_high.ols(y, X)
            result_low = session_low.ols(y, X)

            results_high_eps.append(result_high.params[1])
            results_low_eps.append(result_low.params[1])

        # Low epsilon should have higher variance
        var_high = np.var(results_high_eps)
        var_low = np.var(results_low_eps)
        assert var_low > var_high

    def test_different_runs_give_different_results(self, regression_data):
        """DP mechanism should give different results on different runs."""
        X, y, _, _ = regression_data

        session1 = PrivacySession(
            epsilon=1.0, delta=1e-5,
            bounds_X=DEFAULT_BOUNDS_X, bounds_y=(-20, 20),
            random_state=1
        )
        result1 = session1.ols(y, X)

        session2 = PrivacySession(
            epsilon=1.0, delta=1e-5,
            bounds_X=DEFAULT_BOUNDS_X, bounds_y=(-20, 20),
            random_state=2
        )
        result2 = session2.ols(y, X)

        # Results should differ (privacy noise)
        assert not np.allclose(result1.params, result2.params)

    def test_reproducible_with_same_random_state(self, regression_data):
        """Same random_state should give identical results."""
        X, y, _, _ = regression_data

        session1 = PrivacySession(
            epsilon=1.0, delta=1e-5,
            bounds_X=DEFAULT_BOUNDS_X, bounds_y=(-20, 20),
            random_state=42
        )
        result1 = session1.ols(y, X)

        session2 = PrivacySession(
            epsilon=1.0, delta=1e-5,
            bounds_X=DEFAULT_BOUNDS_X, bounds_y=(-20, 20),
            random_state=42
        )
        result2 = session2.ols(y, X)

        # Results should be identical with same seed
        assert np.allclose(result1.params, result2.params)


class TestDPOLSInference:
    """Tests for statistical inference validity."""

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

    def test_standard_errors_increase_with_lower_epsilon(self):
        """Standard errors should increase as privacy increases (lower epsilon)."""
        np.random.seed(42)
        n = 1000
        X = np.random.randn(n, 2)
        y = X @ [1, 2] + np.random.randn(n)

        session_high = PrivacySession(
            epsilon=10.0, delta=1e-5,
            bounds_X=DEFAULT_BOUNDS_X, bounds_y=(-15, 15)
        )
        session_low = PrivacySession(
            epsilon=0.1, delta=1e-5,
            bounds_X=DEFAULT_BOUNDS_X, bounds_y=(-15, 15)
        )

        result_high = session_high.ols(y, X)
        result_low = session_low.ols(y, X)

        # Lower epsilon = more noise = larger mean SE
        # Use mean SE to be more robust to randomness
        mean_se_high = np.mean(result_high.bse)
        mean_se_low = np.mean(result_low.bse)
        assert mean_se_low > mean_se_high, (
            f"Mean SE with low ε ({mean_se_low:.4f}) should exceed "
            f"mean SE with high ε ({mean_se_high:.4f})"
        )


class TestPrivacyBudget:
    """Tests for privacy budget tracking."""

    def test_privacy_budget_initialized(self):
        """Session should track initial privacy budget."""
        session = PrivacySession(
            epsilon=1.0, delta=1e-5,
            bounds_X=DEFAULT_BOUNDS_X, bounds_y=DEFAULT_BOUNDS_Y
        )
        assert session.epsilon == 1.0
        assert session.delta == 1e-5
        assert session.epsilon_spent == 0.0

    def test_privacy_budget_consumed_by_query(self):
        """Running a query should consume privacy budget."""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = X @ [1, 2] + np.random.randn(100)

        session = PrivacySession(
            epsilon=1.0, delta=1e-5,
            bounds_X=DEFAULT_BOUNDS_X, bounds_y=(-15, 15)
        )
        assert session.epsilon_spent == 0.0

        session.ols(y, X)
        assert session.epsilon_spent > 0.0
        assert session.epsilon_remaining < 1.0

    def test_privacy_budget_accumulates(self):
        """Multiple queries should accumulate privacy cost."""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = X @ [1, 2] + np.random.randn(100)

        session = PrivacySession(
            epsilon=2.0, delta=1e-5,
            bounds_X=DEFAULT_BOUNDS_X, bounds_y=(-15, 15)
        )

        session.ols(y, X)
        spent_after_1 = session.epsilon_spent

        session.ols(y, X)
        spent_after_2 = session.epsilon_spent

        assert spent_after_2 > spent_after_1

    def test_privacy_budget_exhausted_raises_error(self):
        """Should raise error when privacy budget exhausted."""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = X @ [1, 2] + np.random.randn(100)

        session = PrivacySession(
            epsilon=0.1, delta=1e-5,
            bounds_X=DEFAULT_BOUNDS_X, bounds_y=(-15, 15)
        )

        # Request more than available - should raise
        with pytest.raises(ValueError, match="[Pp]rivacy budget"):
            # Request epsilon=0.2 when only 0.1 available
            session.ols(y, X, epsilon=0.2)


class TestDPOLSWeighted:
    """Tests for weighted least squares."""

    def test_weighted_ols_accepts_weights(self):
        """WLS should accept sample weights."""
        np.random.seed(42)
        n = 500
        X = np.random.randn(n, 2)
        y = X @ [1, 2] + np.random.randn(n)
        weights = np.random.uniform(0.5, 2.0, n)

        session = PrivacySession(
            epsilon=5.0, delta=1e-5,
            bounds_X=DEFAULT_BOUNDS_X, bounds_y=(-15, 15)
        )
        result = session.ols(y, X, weights=weights)

        assert hasattr(result, "params")
        assert len(result.params) == 3  # intercept + 2 coefs

    def test_weights_affect_estimates(self):
        """Different weights should give different estimates."""
        np.random.seed(42)
        n = 500
        X = np.random.randn(n, 2)
        y = X @ [1, 2] + np.random.randn(n)

        weights1 = np.ones(n)
        weights2 = np.random.uniform(0.1, 10.0, n)

        session1 = PrivacySession(
            epsilon=10.0, delta=1e-5,
            bounds_X=DEFAULT_BOUNDS_X, bounds_y=(-15, 15),
            random_state=99
        )
        session2 = PrivacySession(
            epsilon=10.0, delta=1e-5,
            bounds_X=DEFAULT_BOUNDS_X, bounds_y=(-15, 15),
            random_state=99
        )

        result1 = session1.ols(y, X, weights=weights1)
        result2 = session2.ols(y, X, weights=weights2)

        # Results should differ due to different weights
        assert not np.allclose(result1.params, result2.params)


class TestDPOLSEdgeCases:
    """Edge case tests."""

    def test_single_feature(self):
        """Should work with single feature (simple linear regression)."""
        np.random.seed(42)
        n = 500
        X = np.random.randn(n, 1)
        y = 2 * X.flatten() + 1 + np.random.randn(n) * 0.5

        session = PrivacySession(
            epsilon=5.0, delta=1e-5,
            bounds_X=DEFAULT_BOUNDS_X, bounds_y=(-15, 15)
        )
        result = session.ols(y, X)

        assert len(result.params) == 2  # intercept + 1 coef

    def test_many_features(self):
        """Should work with many features."""
        np.random.seed(42)
        n = 1000
        k = 20
        X = np.random.randn(n, k)
        y = X @ np.arange(1, k + 1) + np.random.randn(n)

        session = PrivacySession(
            epsilon=5.0, delta=1e-5,
            bounds_X=DEFAULT_BOUNDS_X, bounds_y=(-200, 200)
        )
        result = session.ols(y, X)

        assert len(result.params) == k + 1

    def test_no_intercept(self):
        """Should support regression without intercept."""
        np.random.seed(42)
        n = 500
        X = np.random.randn(n, 2)
        y = X @ [1, 2] + np.random.randn(n)

        session = PrivacySession(
            epsilon=5.0, delta=1e-5,
            bounds_X=DEFAULT_BOUNDS_X, bounds_y=(-15, 15)
        )
        result = session.ols(y, X, add_constant=False)

        assert len(result.params) == 2  # No intercept

    def test_collinear_features_handled(self):
        """Should handle collinear features gracefully (with regularization)."""
        np.random.seed(42)
        n = 500
        X1 = np.random.randn(n)
        X = np.column_stack([X1, X1 * 2])  # Perfectly collinear
        y = X1 + np.random.randn(n)

        session = PrivacySession(
            epsilon=5.0, delta=1e-5,
            bounds_X=(-10, 10), bounds_y=(-15, 15)
        )

        # With DP noise, collinear features may not cause singularity
        # Just check it doesn't crash and produces valid results
        result = session.ols(y, X)
        assert result.params is not None
        assert len(result.params) == 3  # const + 2 features
        assert not np.any(np.isnan(result.params))


class TestDPOLSDataBounds:
    """Tests for data bounds handling."""

    def test_explicit_bounds_accepted(self):
        """Should accept explicit data bounds."""
        np.random.seed(42)
        n = 500
        X = np.random.randn(n, 2)
        y = X @ [1, 2] + np.random.randn(n)

        session = PrivacySession(
            epsilon=5.0, delta=1e-5,
            bounds_X=(-5, 5), bounds_y=(-10, 10)
        )
        result = session.ols(y, X)

        assert hasattr(result, "params")

    def test_missing_bounds_raises_error(self):
        """Missing bounds should raise error - bounds are REQUIRED for valid DP."""
        # Session without bounds should raise TypeError (required args)
        with pytest.raises(TypeError):
            PrivacySession(epsilon=5.0, delta=1e-5)

    def test_bounds_required_enforced(self):
        """Bounds must be provided - auto-bounds feature removed (voids privacy)."""
        np.random.seed(42)
        n = 500
        X = np.random.randn(n, 2)
        y = X @ [1, 2] + np.random.randn(n)

        # With bounds provided, should work
        session = PrivacySession(
            epsilon=5.0, delta=1e-5,
            bounds_X=(-4, 4), bounds_y=(-15, 15)
        )
        result = session.ols(y, X)
        assert hasattr(result, "params")


class TestDPOLSPredict:
    """Tests for predict functionality."""

    def test_predict_returns_values(self):
        """predict() should return predicted values."""
        np.random.seed(42)
        n = 500
        X = np.random.randn(n, 2)
        y = X @ [1, 2] + np.random.randn(n)

        session = PrivacySession(
            epsilon=5.0, delta=1e-5,
            bounds_X=DEFAULT_BOUNDS_X, bounds_y=(-15, 15)
        )
        result = session.ols(y, X)

        # Predict on same data
        y_pred = result.predict(X)
        assert len(y_pred) == n

    def test_predict_with_new_data(self):
        """predict() should work with new data."""
        np.random.seed(42)
        n_train = 500
        n_test = 100
        X_train = np.random.randn(n_train, 2)
        y_train = X_train @ [1, 2] + np.random.randn(n_train)
        X_test = np.random.randn(n_test, 2)

        session = PrivacySession(
            epsilon=5.0, delta=1e-5,
            bounds_X=DEFAULT_BOUNDS_X, bounds_y=(-15, 15)
        )
        result = session.ols(y_train, X_train)

        # Predict on new data
        y_pred = result.predict(X_test)
        assert len(y_pred) == n_test
