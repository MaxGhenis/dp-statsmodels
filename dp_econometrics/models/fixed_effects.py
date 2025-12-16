"""
Differentially Private Fixed Effects Models.

Implements panel data fixed effects regression with (ε,δ)-differential privacy
using the within transformation combined with Noisy Sufficient Statistics.
"""

import numpy as np
from scipy import stats
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import warnings

from dp_econometrics.privacy import (
    compute_noisy_xtx,
    compute_noisy_xty,
    compute_xtx_sensitivity,
    compute_xty_sensitivity,
)


@dataclass
class DPFixedEffectsResults:
    """
    Results from DP Fixed Effects Regression.

    Attributes
    ----------
    params : np.ndarray
        Coefficient estimates (excluding fixed effects).
    bse : np.ndarray
        Standard errors of coefficients.
    tvalues : np.ndarray
        t-statistics.
    pvalues : np.ndarray
        Two-sided p-values.
    nobs : int
        Number of observations.
    n_groups : int
        Number of groups (entities).
    df_resid : int
        Residual degrees of freedom.
    epsilon_used : float
        Privacy budget consumed.
    delta_used : float
        Delta parameter used.
    resid_var : float
        Estimated residual variance.
    """
    params: np.ndarray
    bse: np.ndarray
    tvalues: np.ndarray
    pvalues: np.ndarray
    nobs: int
    n_groups: int
    df_resid: int
    epsilon_used: float
    delta_used: float
    resid_var: float

    def conf_int(self, alpha: float = 0.05) -> np.ndarray:
        """
        Compute confidence intervals.

        Parameters
        ----------
        alpha : float
            Significance level.

        Returns
        -------
        np.ndarray of shape (k, 2)
            Lower and upper confidence bounds.
        """
        t_crit = stats.t.ppf(1 - alpha / 2, self.df_resid)
        ci_lower = self.params - t_crit * self.bse
        ci_upper = self.params + t_crit * self.bse
        return np.column_stack([ci_lower, ci_upper])

    def summary(self) -> str:
        """Generate summary table."""
        ci = self.conf_int()
        lines = [
            "=" * 78,
            "Differentially Private Fixed Effects Results".center(78),
            "=" * 78,
            f"Observations: {self.nobs}".ljust(26) +
            f"Groups: {self.n_groups}".ljust(26) +
            f"Df Resid: {self.df_resid}",
            f"Privacy: ε={self.epsilon_used:.3f}, δ={self.delta_used:.1e}".ljust(52) +
            f"Resid Var: {self.resid_var:.4f}",
            "=" * 78,
            f"{'':>10} {'coef':>10} {'std err':>10} {'t':>10} "
            f"{'P>|t|':>10} {'[0.025':>10} {'0.975]':>10}",
            "-" * 78,
        ]
        for i in range(len(self.params)):
            name = f"x{i+1}"
            lines.append(
                f"{name:>10} {self.params[i]:>10.4f} {self.bse[i]:>10.4f} "
                f"{self.tvalues[i]:>10.3f} {self.pvalues[i]:>10.3f} "
                f"{ci[i, 0]:>10.4f} {ci[i, 1]:>10.4f}"
            )
        lines.append("=" * 78)
        lines.append("Note: Fixed effects absorbed. No intercept in output.")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"DPFixedEffectsResults(nobs={self.nobs}, n_groups={self.n_groups}, "
            f"k={len(self.params)}, ε={self.epsilon_used:.3f})"
        )


class DPFixedEffects:
    """
    Differentially Private Fixed Effects Regression.

    Uses the within transformation to eliminate fixed effects, then
    applies Noisy Sufficient Statistics to the transformed data.

    The model is:
        y_it = α_i + X_it β + ε_it

    The within transformation demeans by group:
        ỹ_it = y_it - ȳ_i
        X̃_it = X_it - X̄_i

    Then OLS on transformed data: ỹ = X̃β + ε

    Parameters
    ----------
    epsilon : float
        Privacy parameter ε.
    delta : float
        Privacy parameter δ.
    bounds_X : tuple of (min, max), optional
        Bounds on feature values. Required for proper DP.
    bounds_y : tuple of (min, max), optional
        Bounds on response variable. Required for proper DP.

    Examples
    --------
    >>> import numpy as np
    >>> from dp_econometrics.models import DPFixedEffects
    >>> # Panel data: 100 entities, 10 time periods each
    >>> n_entities, n_periods = 100, 10
    >>> n = n_entities * n_periods
    >>> groups = np.repeat(np.arange(n_entities), n_periods)
    >>> X = np.random.randn(n, 2)
    >>> # True fixed effects + coefficients
    >>> alpha = np.repeat(np.random.randn(n_entities), n_periods)
    >>> y = alpha + X @ [1.0, 2.0] + np.random.randn(n) * 0.5
    >>> model = DPFixedEffects(epsilon=5.0, delta=1e-5,
    ...                        bounds_X=(-5, 5), bounds_y=(-20, 20))
    >>> result = model.fit(y, X, groups)
    >>> print(result.params)  # Should be close to [1.0, 2.0]

    References
    ----------
    - Within transformation: standard panel data econometrics
    - Noisy sufficient statistics: Sheffet (2017)
    """

    def __init__(
        self,
        epsilon: float,
        delta: float,
        bounds_X: Optional[Tuple[float, float]] = None,
        bounds_y: Optional[Tuple[float, float]] = None,
    ):
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if delta <= 0 or delta >= 1:
            raise ValueError("delta must be in (0, 1)")

        self.epsilon = epsilon
        self.delta = delta
        self.bounds_X = bounds_X
        self.bounds_y = bounds_y

    def _within_transform(
        self,
        data: np.ndarray,
        groups: np.ndarray
    ) -> np.ndarray:
        """
        Apply within transformation (demean by group).

        Parameters
        ----------
        data : np.ndarray
            Data to transform (1D or 2D).
        groups : np.ndarray
            Group identifiers.

        Returns
        -------
        np.ndarray
            Demeaned data.
        """
        unique_groups = np.unique(groups)
        transformed = data.copy().astype(float)

        for g in unique_groups:
            mask = groups == g
            if data.ndim == 1:
                transformed[mask] = data[mask] - np.mean(data[mask])
            else:
                transformed[mask] = data[mask] - np.mean(data[mask], axis=0)

        return transformed

    def _compute_within_bounds(
        self,
        bounds: Tuple[float, float]
    ) -> Tuple[float, float]:
        """
        Compute bounds for within-transformed data.

        After demeaning, the range is at most 2x the original range.
        """
        original_range = bounds[1] - bounds[0]
        # After demeaning, max deviation from mean is at most the range
        return (-original_range, original_range)

    def fit(
        self,
        y: np.ndarray,
        X: np.ndarray,
        groups: np.ndarray,
    ) -> DPFixedEffectsResults:
        """
        Fit DP Fixed Effects model.

        Parameters
        ----------
        y : np.ndarray of shape (n,)
            Response variable.
        X : np.ndarray of shape (n, k)
            Design matrix (without constant - FE absorbs it).
        groups : np.ndarray of shape (n,)
            Group/entity identifiers.

        Returns
        -------
        DPFixedEffectsResults
            Results with coefficients and standard errors.
        """
        y = np.asarray(y).flatten()
        X = np.asarray(X)
        groups = np.asarray(groups).flatten()

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n, k = X.shape
        unique_groups = np.unique(groups)
        n_groups = len(unique_groups)

        # Degrees of freedom: n - n_groups - k
        # (lose n_groups for fixed effects, k for coefficients)
        df_resid = n - n_groups - k

        if df_resid <= 0:
            raise ValueError(
                f"Insufficient degrees of freedom: n={n}, n_groups={n_groups}, "
                f"k={k}. Need n > n_groups + k."
            )

        # Handle bounds
        bounds_X = self.bounds_X
        bounds_y = self.bounds_y

        if bounds_X is None:
            warnings.warn(
                "bounds_X not provided. Computing from data leaks privacy.",
                UserWarning
            )
            bounds_X = (X.min(), X.max())

        if bounds_y is None:
            warnings.warn(
                "bounds_y not provided. Computing from data leaks privacy.",
                UserWarning
            )
            bounds_y = (y.min(), y.max())

        # Apply within transformation
        X_within = self._within_transform(X, groups)
        y_within = self._within_transform(y, groups)

        # Compute bounds for transformed data
        bounds_X_within = self._compute_within_bounds(bounds_X)
        bounds_y_within = self._compute_within_bounds(bounds_y)

        # Clip transformed data to bounds
        X_within = np.clip(X_within, bounds_X_within[0], bounds_X_within[1])
        y_within = np.clip(y_within, bounds_y_within[0], bounds_y_within[1])

        # Split privacy budget
        eps_xtx = self.epsilon * 0.5
        eps_xty = self.epsilon * 0.5
        delta_each = self.delta / 2

        # Compute noisy sufficient statistics on transformed data
        noisy_xtx = compute_noisy_xtx(
            X_within, eps_xtx, delta_each, bounds_X_within
        )
        noisy_xty = compute_noisy_xty(
            X_within, y_within, eps_xty, delta_each,
            bounds_X_within, bounds_y_within
        )

        # Ensure positive definiteness
        min_eig = np.min(np.linalg.eigvalsh(noisy_xtx))
        if min_eig <= 0:
            warnings.warn(
                "Noisy X'X not positive definite. Adding regularization.",
                UserWarning
            )
            noisy_xtx = noisy_xtx + (abs(min_eig) + 1e-6) * np.eye(k)

        # Solve for coefficients
        try:
            params = np.linalg.solve(noisy_xtx, noisy_xty)
        except np.linalg.LinAlgError:
            warnings.warn("Singular matrix. Using pseudo-inverse.", UserWarning)
            params = np.linalg.lstsq(noisy_xtx, noisy_xty, rcond=None)[0]

        # Estimate residual variance
        # Using within-transformed data
        y_pred_within = X_within @ params
        ss_resid = np.sum((y_within - y_pred_within) ** 2)
        resid_var = ss_resid / df_resid

        # Compute variance of β̂
        try:
            xtx_inv = np.linalg.inv(noisy_xtx)
        except np.linalg.LinAlgError:
            xtx_inv = np.linalg.pinv(noisy_xtx)

        # Sampling variance
        var_sampling = resid_var * xtx_inv

        # Privacy noise variance
        sens_xtx = compute_xtx_sensitivity(bounds_X_within, k)
        sens_xty = compute_xty_sensitivity(bounds_X_within, bounds_y_within, k)

        sigma_xtx = sens_xtx * np.sqrt(2 * np.log(1.25 / delta_each)) / eps_xtx
        sigma_xty = sens_xty * np.sqrt(2 * np.log(1.25 / delta_each)) / eps_xty

        # Approximate variance from noise
        var_noise_diag = (
            sigma_xty ** 2 + params ** 2 * sigma_xtx ** 2
        ) * np.diag(xtx_inv ** 2)

        # Total variance
        var_total = np.diag(var_sampling) + var_noise_diag
        bse = np.sqrt(np.maximum(var_total, 1e-10))

        # t-statistics and p-values
        tvalues = params / bse
        pvalues = 2 * (1 - stats.t.cdf(np.abs(tvalues), df_resid))

        return DPFixedEffectsResults(
            params=params,
            bse=bse,
            tvalues=tvalues,
            pvalues=pvalues,
            nobs=n,
            n_groups=n_groups,
            df_resid=df_resid,
            epsilon_used=self.epsilon,
            delta_used=self.delta,
            resid_var=resid_var,
        )
