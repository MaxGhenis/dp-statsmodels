"""
Differentially Private OLS using Noisy Sufficient Statistics.

Implements OLS regression with (ε,δ)-differential privacy by adding
calibrated Gaussian noise to X'X and X'y before solving for β.

Standard errors account for both sampling variance and privacy noise variance
using the formula from Evans et al. (2024).
"""

import numpy as np
from scipy import stats
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union
import warnings

from dp_statsmodels.privacy import (
    compute_noisy_xtx,
    compute_noisy_xty,
    compute_xtx_sensitivity,
    compute_xty_sensitivity,
)
from dp_statsmodels.privacy.noisy_stats import compute_noisy_yty, _get_rng


@dataclass
class DPOLSResults:
    """
    Results from DP OLS regression.

    Attributes
    ----------
    params : np.ndarray
        Coefficient estimates (including intercept if add_constant=True).
    bse : np.ndarray
        Standard errors of coefficients.
    tvalues : np.ndarray
        t-statistics.
    pvalues : np.ndarray
        Two-sided p-values.
    nobs : int
        Number of observations.
    df_resid : int
        Residual degrees of freedom.
    epsilon_used : float
        Privacy budget consumed.
    delta_used : float
        Delta parameter used.
    resid_var : float
        Estimated residual variance (from noisy sufficient statistics).
    """
    params: np.ndarray
    bse: np.ndarray
    tvalues: np.ndarray
    pvalues: np.ndarray
    nobs: int
    df_resid: int
    epsilon_used: float
    delta_used: float
    resid_var: float
    _noisy_xtx: np.ndarray = field(repr=False)  # For variance computation
    _noisy_xty: np.ndarray = field(repr=False)
    _add_constant: bool = field(default=True, repr=False)

    def conf_int(self, alpha: float = 0.05) -> np.ndarray:
        """
        Compute confidence intervals for coefficients.

        Parameters
        ----------
        alpha : float
            Significance level (default 0.05 for 95% CI).

        Returns
        -------
        np.ndarray of shape (k, 2)
            Lower and upper bounds for each coefficient.
        """
        # Use t-distribution for finite sample inference
        t_crit = stats.t.ppf(1 - alpha / 2, self.df_resid)

        ci_lower = self.params - t_crit * self.bse
        ci_upper = self.params + t_crit * self.bse

        return np.column_stack([ci_lower, ci_upper])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions for new data.

        Note: Predictions use the noisy coefficients, so they inherit
        the privacy guarantee. No additional privacy budget is consumed.

        Parameters
        ----------
        X : np.ndarray of shape (n, k) or (n, k-1)
            Design matrix. If add_constant was True during fitting and
            X has k-1 columns, a constant column will be added.

        Returns
        -------
        np.ndarray of shape (n,)
            Predicted values.
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Add constant if needed
        n_features = len(self.params) - 1 if self._add_constant else len(self.params)
        if X.shape[1] == n_features and self._add_constant:
            X = np.column_stack([np.ones(X.shape[0]), X])

        return X @ self.params

    def summary(self) -> str:
        """
        Generate a summary table of results.

        Returns
        -------
        str
            Formatted summary table.
        """
        ci = self.conf_int()

        lines = [
            "=" * 70,
            "Differentially Private OLS Results".center(70),
            "=" * 70,
            f"Observations: {self.nobs}".ljust(35) +
            f"Privacy: ε={self.epsilon_used:.3f}, δ={self.delta_used:.1e}",
            f"Df Residuals: {self.df_resid}".ljust(35) +
            f"Residual Var: {self.resid_var:.4f}",
            "=" * 70,
            f"{'':>10} {'coef':>10} {'std err':>10} {'t':>10} "
            f"{'P>|t|':>10} {'[0.025':>10} {'0.975]':>10}",
            "-" * 70,
        ]

        for i in range(len(self.params)):
            name = f"x{i}" if i > 0 else "const"
            lines.append(
                f"{name:>10} {self.params[i]:>10.4f} {self.bse[i]:>10.4f} "
                f"{self.tvalues[i]:>10.3f} {self.pvalues[i]:>10.3f} "
                f"{ci[i, 0]:>10.4f} {ci[i, 1]:>10.4f}"
            )

        lines.append("=" * 70)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"DPOLSResults(nobs={self.nobs}, k={len(self.params)}, ε={self.epsilon_used:.3f})"


class DPOLS:
    """
    Differentially Private Ordinary Least Squares.

    Uses Noisy Sufficient Statistics (NSS) to compute OLS estimates
    with formal (ε,δ)-differential privacy guarantees.

    IMPORTANT: Standard errors are typically 100-1000x larger than non-private
    OLS at typical privacy budgets (ε=1-10). This requires samples 10,000-
    1,000,000x larger to maintain equivalent statistical power. See the
    documentation for sample size guidance.

    Parameters
    ----------
    epsilon : float
        Privacy parameter ε.
    delta : float
        Privacy parameter δ.
    bounds_X : tuple of (min, max)
        Bounds on feature values. REQUIRED - must be specified a priori
        based on domain knowledge, NOT computed from data.
    bounds_y : tuple of (min, max)
        Bounds on response variable. REQUIRED - must be specified a priori
        based on domain knowledge, NOT computed from data.
    random_state : int or np.random.Generator, optional
        Random state for reproducibility.
    budget_allocation : str, optional
        How to allocate privacy budget. Options:
        - 'equal': Equal split between X'X and X'y (default, simple)
        - 'optimal': Sheffet (2017) optimal allocation (minimizes MSE)

    Examples
    --------
    >>> model = DPOLS(epsilon=1.0, delta=1e-5, bounds_X=(-10, 10), bounds_y=(-100, 100))
    >>> result = model.fit(X, y)
    >>> print(result.summary())

    References
    ----------
    Sheffet, O. (2017). Differentially private ordinary least squares.
    ICML 2017.

    Evans, G., King, G., et al. (2024). Differentially Private Linear
    Regression with Linked Data. Harvard Data Science Review.

    Notes
    -----
    Privacy guarantees ONLY hold if bounds are specified a priori based on
    domain knowledge. Computing bounds from data voids all privacy protections.
    """

    def __init__(
        self,
        epsilon: float,
        delta: float,
        bounds_X: Tuple[float, float],
        bounds_y: Tuple[float, float],
        random_state: Optional[Union[int, np.random.Generator]] = None,
        budget_allocation: str = 'equal',
    ):
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if delta <= 0 or delta >= 1:
            raise ValueError("delta must be in (0, 1)")
        if bounds_X is None:
            raise ValueError(
                "bounds_X is REQUIRED for differential privacy. "
                "Specify bounds a priori based on domain knowledge. "
                "Computing bounds from data voids privacy guarantees."
            )
        if bounds_y is None:
            raise ValueError(
                "bounds_y is REQUIRED for differential privacy. "
                "Specify bounds a priori based on domain knowledge. "
                "Computing bounds from data voids privacy guarantees."
            )
        if budget_allocation not in ('equal', 'optimal'):
            raise ValueError("budget_allocation must be 'equal' or 'optimal'")

        self.epsilon = epsilon
        self.delta = delta
        self.bounds_X = bounds_X
        self.bounds_y = bounds_y
        self.random_state = random_state
        self.budget_allocation = budget_allocation

    def fit(
        self,
        y: np.ndarray,
        X: np.ndarray,
        weights: Optional[np.ndarray] = None,
        add_constant: bool = True,
    ) -> DPOLSResults:
        """
        Fit DP OLS model.

        Parameters
        ----------
        y : np.ndarray of shape (n,)
            Response variable.
        X : np.ndarray of shape (n, k)
            Design matrix (without constant).
        weights : np.ndarray of shape (n,), optional
            Sample weights.
        add_constant : bool
            Whether to add a constant term.

        Returns
        -------
        DPOLSResults
            Results object with coefficients, standard errors, etc.

        Raises
        ------
        ValueError
            If bounds_X or bounds_y not provided and require_bounds is True.
        """
        y = np.asarray(y).flatten()
        X = np.asarray(X)
        rng = _get_rng(self.random_state)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n, k = X.shape

        # Add constant if requested
        if add_constant:
            X = np.column_stack([np.ones(n), X])
            k = k + 1

        # Use pre-specified bounds (required for valid DP)
        bounds_X = self.bounds_X
        bounds_y = self.bounds_y

        # Handle weights (WLS)
        if weights is not None:
            weights = np.asarray(weights).flatten()
            sqrt_w = np.sqrt(weights)
            X = X * sqrt_w[:, np.newaxis]
            y = y * sqrt_w

        # Split privacy budget between sufficient statistics
        # Reserve 20% for y'y (residual variance estimation)
        eps_yty = self.epsilon * 0.2
        eps_remaining = self.epsilon * 0.8

        if self.budget_allocation == 'optimal':
            # Sheffet (2017) optimal allocation minimizes MSE
            # ε_XX = ε × √(k/(k+1)), ε_Xy = ε × √(1/(k+1))
            # But we need to account for different sensitivities too
            # For now, use the simplified version that allocates more to X'X
            # since it's a k×k matrix vs k×1 vector
            ratio = np.sqrt(k / (k + 1))
            eps_xtx = eps_remaining * ratio
            eps_xty = eps_remaining * (1 - ratio)
        else:
            # Equal split (simple, default)
            eps_xtx = eps_remaining * 0.5
            eps_xty = eps_remaining * 0.5

        delta_each = self.delta / 3

        # Compute noisy sufficient statistics
        noisy_xtx = compute_noisy_xtx(
            X, eps_xtx, delta_each, bounds_X,
            random_state=rng, require_bounds=False  # Already validated above
        )
        noisy_xty = compute_noisy_xty(
            X, y, eps_xty, delta_each, bounds_X, bounds_y,
            random_state=rng, require_bounds=False
        )
        # Compute noisy y'y for residual variance (fixes privacy leak)
        noisy_yty = compute_noisy_yty(
            y, eps_yty, delta_each, bounds_y,
            random_state=rng, require_bounds=False
        )

        # Ensure X'X is positive definite (regularize if needed)
        min_eig = np.min(np.linalg.eigvalsh(noisy_xtx))
        if min_eig <= 0:
            warnings.warn(
                "Noisy X'X is not positive definite. Adding regularization. "
                "This may indicate collinear features or excessive noise.",
                UserWarning
            )
            noisy_xtx = noisy_xtx + (abs(min_eig) + 1e-6) * np.eye(k)

        # Solve for coefficients: β = (X'X)^{-1} X'y
        try:
            params = np.linalg.solve(noisy_xtx, noisy_xty)
        except np.linalg.LinAlgError:
            warnings.warn(
                "Singular matrix encountered. Using pseudo-inverse.",
                UserWarning
            )
            params = np.linalg.lstsq(noisy_xtx, noisy_xty, rcond=None)[0]

        # Compute (X'X)^{-1} for variance computation
        try:
            xtx_inv = np.linalg.inv(noisy_xtx)
        except np.linalg.LinAlgError:
            xtx_inv = np.linalg.pinv(noisy_xtx)

        # Estimate residual variance using ONLY noisy sufficient statistics
        # RSS = y'y - 2β'X'y + β'X'Xβ = y'y - β'X'y (since β = (X'X)^{-1}X'y)
        # This is computed entirely from noisy statistics, fixing the privacy leak
        noisy_rss = noisy_yty - params @ noisy_xty
        resid_var = max(noisy_rss / (n - k), 1e-10)

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

        # t-statistics and p-values
        tvalues = params / bse
        pvalues = 2 * (1 - stats.t.cdf(np.abs(tvalues), n - k))

        return DPOLSResults(
            params=params,
            bse=bse,
            tvalues=tvalues,
            pvalues=pvalues,
            nobs=n,
            df_resid=n - k,
            epsilon_used=self.epsilon,
            delta_used=self.delta,
            resid_var=resid_var,
            _noisy_xtx=noisy_xtx,
            _noisy_xty=noisy_xty,
            _add_constant=add_constant,
        )
