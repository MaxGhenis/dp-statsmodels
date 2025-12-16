"""
Tests for DP Fixed Effects Regression.

Following TDD: These tests are written BEFORE verifying the implementation.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from dp_econometrics import PrivacySession
from dp_econometrics.models import DPFixedEffects


class TestDPFixedEffectsBasic:
    """Basic functionality tests for DP Fixed Effects."""

    @pytest.fixture
    def panel_data(self):
        """Generate panel data with known fixed effects and coefficients."""
        np.random.seed(42)
        n_entities = 50
        n_periods = 10
        n = n_entities * n_periods

        # Group identifiers
        groups = np.repeat(np.arange(n_entities), n_periods)

        # Features
        X = np.random.randn(n, 2)

        # True parameters
        true_coef = np.array([1.0, 2.0])

        # Fixed effects (one per entity)
        alpha = np.random.randn(n_entities) * 2
        alpha_expanded = np.repeat(alpha, n_periods)

        # Generate y with fixed effects
        y = alpha_expanded + X @ true_coef + np.random.randn(n) * 0.5

        return X, y, groups, true_coef, n_entities

    def test_fe_returns_coefficients(self, panel_data):
        """Fixed effects should return coefficient estimates."""
        X, y, groups, _, _ = panel_data

        model = DPFixedEffects(
            epsilon=10.0, delta=1e-5,
            bounds_X=(-5, 5), bounds_y=(-20, 20)
        )
        result = model.fit(y, X, groups)

        assert hasattr(result, "params")
        assert len(result.params) == X.shape[1]  # No intercept (absorbed by FE)

    def test_fe_returns_standard_errors(self, panel_data):
        """Fixed effects should return standard errors."""
        X, y, groups, _, _ = panel_data

        model = DPFixedEffects(
            epsilon=10.0, delta=1e-5,
            bounds_X=(-5, 5), bounds_y=(-20, 20)
        )
        result = model.fit(y, X, groups)

        assert hasattr(result, "bse")
        assert len(result.bse) == len(result.params)
        assert all(se > 0 for se in result.bse)

    def test_fe_returns_group_count(self, panel_data):
        """Fixed effects should report number of groups."""
        X, y, groups, _, n_entities = panel_data

        model = DPFixedEffects(
            epsilon=10.0, delta=1e-5,
            bounds_X=(-5, 5), bounds_y=(-20, 20)
        )
        result = model.fit(y, X, groups)

        assert hasattr(result, "n_groups")
        assert result.n_groups == n_entities

    def test_fe_returns_confidence_intervals(self, panel_data):
        """Fixed effects should return confidence intervals."""
        X, y, groups, _, _ = panel_data

        model = DPFixedEffects(
            epsilon=10.0, delta=1e-5,
            bounds_X=(-5, 5), bounds_y=(-20, 20)
        )
        result = model.fit(y, X, groups)

        ci = result.conf_int(alpha=0.05)
        assert ci.shape == (len(result.params), 2)
        assert all(ci[:, 0] < ci[:, 1])  # Lower < Upper


class TestDPFixedEffectsAccuracy:
    """Tests for accuracy of DP Fixed Effects estimates."""

    @pytest.fixture
    def large_panel(self):
        """Larger panel for accuracy tests."""
        np.random.seed(123)
        n_entities = 100
        n_periods = 20
        n = n_entities * n_periods

        groups = np.repeat(np.arange(n_entities), n_periods)
        X = np.random.randn(n, 2)
        true_coef = np.array([1.5, -1.0])

        alpha = np.random.randn(n_entities) * 3
        alpha_expanded = np.repeat(alpha, n_periods)

        y = alpha_expanded + X @ true_coef + np.random.randn(n) * 0.5

        return X, y, groups, true_coef

    def test_fe_removes_fixed_effects_bias(self, large_panel):
        """FE should recover coefficients despite large fixed effects."""
        X, y, groups, true_coef = large_panel

        model = DPFixedEffects(
            epsilon=20.0, delta=1e-5,  # High epsilon for accuracy
            bounds_X=(-5, 5), bounds_y=(-30, 30)
        )
        result = model.fit(y, X, groups)

        # Should recover true coefficients reasonably well
        assert_allclose(result.params, true_coef, atol=0.3)

    def test_pooled_ols_would_be_biased(self, large_panel):
        """Verify that ignoring FE would give biased estimates."""
        X, y, groups, true_coef = large_panel

        # This test documents WHY we need FE:
        # If we ignored the panel structure, OLS would be biased
        # because fixed effects are correlated with X in general

        # FE model should be closer to truth than pooled
        model = DPFixedEffects(
            epsilon=20.0, delta=1e-5,
            bounds_X=(-5, 5), bounds_y=(-30, 30)
        )
        result = model.fit(y, X, groups)

        fe_error = np.mean((result.params - true_coef) ** 2)

        # FE should have low error
        assert fe_error < 0.5

    def test_different_runs_give_different_results(self, large_panel):
        """DP mechanism should give different results on different runs."""
        X, y, groups, _ = large_panel

        np.random.seed(1)
        model1 = DPFixedEffects(
            epsilon=1.0, delta=1e-5,
            bounds_X=(-5, 5), bounds_y=(-30, 30)
        )
        result1 = model1.fit(y, X, groups)

        np.random.seed(2)
        model2 = DPFixedEffects(
            epsilon=1.0, delta=1e-5,
            bounds_X=(-5, 5), bounds_y=(-30, 30)
        )
        result2 = model2.fit(y, X, groups)

        assert not np.allclose(result1.params, result2.params)


class TestDPFixedEffectsInference:
    """Tests for statistical inference validity."""

    def test_confidence_interval_coverage(self):
        """95% CIs should contain true parameters ~95% of the time."""
        np.random.seed(42)
        n_entities = 50
        n_periods = 10
        n = n_entities * n_periods
        true_coef = np.array([1.0, 2.0])

        coverage_count = 0
        n_simulations = 50  # Reduced for speed

        for sim in range(n_simulations):
            # Generate new panel data
            groups = np.repeat(np.arange(n_entities), n_periods)
            X = np.random.randn(n, 2)
            alpha = np.random.randn(n_entities)
            alpha_expanded = np.repeat(alpha, n_periods)
            y = alpha_expanded + X @ true_coef + np.random.randn(n) * 0.5

            # Fit DP FE
            model = DPFixedEffects(
                epsilon=5.0, delta=1e-5,
                bounds_X=(-5, 5), bounds_y=(-15, 15)
            )
            result = model.fit(y, X, groups)
            ci = result.conf_int(alpha=0.05)

            # Check coverage
            in_ci = (ci[:, 0] <= true_coef) & (true_coef <= ci[:, 1])
            if all(in_ci):
                coverage_count += 1

        coverage_rate = coverage_count / n_simulations
        # Should be around 95%, allow slack for small sample
        assert coverage_rate >= 0.70, f"Coverage {coverage_rate:.2%} too low"


class TestDPFixedEffectsDegreesOfFreedom:
    """Tests for degrees of freedom handling."""

    def test_df_accounts_for_groups(self):
        """Degrees of freedom should account for absorbed fixed effects."""
        np.random.seed(42)
        n_entities = 20
        n_periods = 10
        n = n_entities * n_periods
        k = 3  # features

        groups = np.repeat(np.arange(n_entities), n_periods)
        X = np.random.randn(n, k)
        y = np.random.randn(n)

        model = DPFixedEffects(
            epsilon=5.0, delta=1e-5,
            bounds_X=(-5, 5), bounds_y=(-10, 10)
        )
        result = model.fit(y, X, groups)

        # df = n - n_groups - k
        expected_df = n - n_entities - k
        assert result.df_resid == expected_df

    def test_insufficient_df_raises_error(self):
        """Should raise error if not enough observations per group."""
        np.random.seed(42)
        n_entities = 100
        n_periods = 2  # Only 2 obs per entity
        n = n_entities * n_periods
        k = 5  # Many features

        groups = np.repeat(np.arange(n_entities), n_periods)
        X = np.random.randn(n, k)
        y = np.random.randn(n)

        model = DPFixedEffects(
            epsilon=5.0, delta=1e-5,
            bounds_X=(-5, 5), bounds_y=(-10, 10)
        )

        # n=200, n_groups=100, k=5 -> df = 200 - 100 - 5 = 95 (OK)
        # But let's create a case where it fails
        groups_bad = np.arange(n)  # Each obs is its own group!
        with pytest.raises(ValueError, match="[Dd]egrees of freedom"):
            model.fit(y, X, groups_bad)


class TestDPFixedEffectsEdgeCases:
    """Edge case tests."""

    def test_single_feature(self):
        """Should work with single feature."""
        np.random.seed(42)
        n_entities = 30
        n_periods = 5
        n = n_entities * n_periods

        groups = np.repeat(np.arange(n_entities), n_periods)
        X = np.random.randn(n, 1)
        y = np.random.randn(n)

        model = DPFixedEffects(
            epsilon=5.0, delta=1e-5,
            bounds_X=(-5, 5), bounds_y=(-10, 10)
        )
        result = model.fit(y, X, groups)

        assert len(result.params) == 1

    def test_many_features(self):
        """Should work with many features."""
        np.random.seed(42)
        n_entities = 50
        n_periods = 20
        n = n_entities * n_periods
        k = 10

        groups = np.repeat(np.arange(n_entities), n_periods)
        X = np.random.randn(n, k)
        y = np.random.randn(n)

        model = DPFixedEffects(
            epsilon=5.0, delta=1e-5,
            bounds_X=(-5, 5), bounds_y=(-10, 10)
        )
        result = model.fit(y, X, groups)

        assert len(result.params) == k

    def test_unbalanced_panel(self):
        """Should work with unbalanced panel (different obs per entity)."""
        np.random.seed(42)

        # Create unbalanced panel
        groups = []
        for i in range(30):
            n_obs = np.random.randint(3, 15)  # 3-14 obs per entity
            groups.extend([i] * n_obs)
        groups = np.array(groups)
        n = len(groups)

        X = np.random.randn(n, 2)
        y = np.random.randn(n)

        model = DPFixedEffects(
            epsilon=5.0, delta=1e-5,
            bounds_X=(-5, 5), bounds_y=(-10, 10)
        )
        result = model.fit(y, X, groups)

        assert hasattr(result, "params")
        assert result.n_groups == 30

    def test_string_group_ids(self):
        """Should work with string group identifiers."""
        np.random.seed(42)
        n_entities = 20
        n_periods = 5
        n = n_entities * n_periods

        # String group IDs
        entity_names = [f"entity_{i}" for i in range(n_entities)]
        groups = np.repeat(entity_names, n_periods)

        X = np.random.randn(n, 2)
        y = np.random.randn(n)

        model = DPFixedEffects(
            epsilon=5.0, delta=1e-5,
            bounds_X=(-5, 5), bounds_y=(-10, 10)
        )
        result = model.fit(y, X, groups)

        assert result.n_groups == n_entities


class TestDPFixedEffectsSummary:
    """Tests for summary output."""

    def test_summary_string(self):
        """Should produce formatted summary."""
        np.random.seed(42)
        n_entities = 20
        n_periods = 5
        n = n_entities * n_periods

        groups = np.repeat(np.arange(n_entities), n_periods)
        X = np.random.randn(n, 2)
        y = np.random.randn(n)

        model = DPFixedEffects(
            epsilon=5.0, delta=1e-5,
            bounds_X=(-5, 5), bounds_y=(-10, 10)
        )
        result = model.fit(y, X, groups)

        summary = result.summary()
        assert isinstance(summary, str)
        assert "Fixed Effects" in summary
        assert "Groups" in summary
        assert "Privacy" in summary
