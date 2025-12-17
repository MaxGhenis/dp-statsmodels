"""
Tests for reproducibility of dp-statsmodels.

Verifies that:
1. Results are deterministic with fixed random_state
2. Results differ with different random_state values
3. All derived quantities (params, SE, CIs) are reproducible
"""

import numpy as np
import pytest

from dp_statsmodels import Session

# Standard bounds
BOUNDS_X = (-5, 5)
BOUNDS_Y = (-20, 20)


def generate_test_data(n=1000, p=2, seed=100):
    """Generate synthetic test data with known parameters."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(n, p))
    true_beta = np.array([1.0, 2.0])
    y = X @ true_beta + rng.normal(0, 0.5, size=n)
    return X, y, true_beta


class TestReproducibilityWithFixedSeed:
    """Test that results are identical with same random_state."""

    def test_params_reproducible(self):
        """Coefficients should be identical with same random_state."""
        X, y, _ = generate_test_data()

        session1 = Session(
            epsilon=1.0, delta=1e-5, random_state=42,
            bounds_X=BOUNDS_X, bounds_y=BOUNDS_Y
        )
        result1 = session1.OLS(y, X)

        session2 = Session(
            epsilon=1.0, delta=1e-5, random_state=42,
            bounds_X=BOUNDS_X, bounds_y=BOUNDS_Y
        )
        result2 = session2.OLS(y, X)

        assert np.array_equal(result1.params, result2.params), \
            f"Parameters differ: {result1.params} vs {result2.params}"

    def test_standard_errors_reproducible(self):
        """Standard errors should be identical with same random_state."""
        X, y, _ = generate_test_data()

        session1 = Session(
            epsilon=1.0, delta=1e-5, random_state=42,
            bounds_X=BOUNDS_X, bounds_y=BOUNDS_Y
        )
        result1 = session1.OLS(y, X)

        session2 = Session(
            epsilon=1.0, delta=1e-5, random_state=42,
            bounds_X=BOUNDS_X, bounds_y=BOUNDS_Y
        )
        result2 = session2.OLS(y, X)

        assert np.array_equal(result1.bse, result2.bse), \
            f"Standard errors differ: {result1.bse} vs {result2.bse}"

    def test_confidence_intervals_reproducible(self):
        """Confidence intervals should be identical with same random_state."""
        X, y, _ = generate_test_data()

        session1 = Session(
            epsilon=1.0, delta=1e-5, random_state=42,
            bounds_X=BOUNDS_X, bounds_y=BOUNDS_Y
        )
        result1 = session1.OLS(y, X)
        ci1 = result1.conf_int()

        session2 = Session(
            epsilon=1.0, delta=1e-5, random_state=42,
            bounds_X=BOUNDS_X, bounds_y=BOUNDS_Y
        )
        result2 = session2.OLS(y, X)
        ci2 = result2.conf_int()

        assert np.array_equal(ci1, ci2), \
            f"Confidence intervals differ: {ci1} vs {ci2}"

    def test_multiple_queries_reproducible(self):
        """Multiple queries in sequence should be reproducible."""
        X1, y1, _ = generate_test_data(seed=100)
        X2, y2, _ = generate_test_data(seed=101)

        # Session 1: two queries
        session1 = Session(
            epsilon=2.0, delta=1e-5, random_state=42,
            bounds_X=BOUNDS_X, bounds_y=BOUNDS_Y
        )
        result1a = session1.OLS(y1, X1, epsilon=1.0)
        result1b = session1.OLS(y2, X2, epsilon=1.0)

        # Session 2: same queries, same seed
        session2 = Session(
            epsilon=2.0, delta=1e-5, random_state=42,
            bounds_X=BOUNDS_X, bounds_y=BOUNDS_Y
        )
        result2a = session2.OLS(y1, X1, epsilon=1.0)
        result2b = session2.OLS(y2, X2, epsilon=1.0)

        assert np.array_equal(result1a.params, result2a.params)
        assert np.array_equal(result1b.params, result2b.params)


class TestDifferentSeedsProduceDifferentResults:
    """Test that different seeds give different results."""

    def test_different_seeds_different_params(self):
        """Different random_state values should give different results."""
        X, y, _ = generate_test_data()

        session1 = Session(
            epsilon=1.0, delta=1e-5, random_state=42,
            bounds_X=BOUNDS_X, bounds_y=BOUNDS_Y
        )
        result1 = session1.OLS(y, X)

        session2 = Session(
            epsilon=1.0, delta=1e-5, random_state=999,
            bounds_X=BOUNDS_X, bounds_y=BOUNDS_Y
        )
        result2 = session2.OLS(y, X)

        # Results should differ (with overwhelming probability)
        assert not np.array_equal(result1.params, result2.params), \
            "Different seeds should produce different results"

    def test_no_seed_varies_results(self):
        """random_state=None should give different results across runs."""
        X, y, _ = generate_test_data()

        results = []
        for _ in range(5):
            session = Session(
                epsilon=1.0, delta=1e-5, random_state=None,
                bounds_X=BOUNDS_X, bounds_y=BOUNDS_Y
            )
            result = session.OLS(y, X)
            results.append(result.params.copy())

        # At least some should differ
        all_identical = all(np.array_equal(results[0], r) for r in results[1:])
        assert not all_identical, \
            "Without random_state, results should vary"


class TestReproducibilityAcrossEpsilon:
    """Test reproducibility holds for different privacy budgets."""

    @pytest.mark.parametrize("epsilon", [0.5, 1.0, 5.0, 10.0])
    def test_reproducibility_at_epsilon(self, epsilon):
        """Reproducibility should hold at various epsilon values."""
        X, y, _ = generate_test_data()

        session1 = Session(
            epsilon=epsilon, delta=1e-5, random_state=42,
            bounds_X=BOUNDS_X, bounds_y=BOUNDS_Y
        )
        result1 = session1.OLS(y, X)

        session2 = Session(
            epsilon=epsilon, delta=1e-5, random_state=42,
            bounds_X=BOUNDS_X, bounds_y=BOUNDS_Y
        )
        result2 = session2.OLS(y, X)

        assert np.array_equal(result1.params, result2.params), \
            f"Reproducibility failed for epsilon={epsilon}"


class TestReproducibilityDocumentation:
    """Test that random_state parameter is properly accessible."""

    def test_random_state_parameter_exists(self):
        """Session should accept random_state parameter."""
        session = Session(
            epsilon=1.0, delta=1e-5, random_state=42,
            bounds_X=BOUNDS_X, bounds_y=BOUNDS_Y
        )
        assert session.random_state == 42

    def test_random_state_none_is_valid(self):
        """random_state=None should be valid."""
        session = Session(
            epsilon=1.0, delta=1e-5, random_state=None,
            bounds_X=BOUNDS_X, bounds_y=BOUNDS_Y
        )
        assert session.random_state is None
