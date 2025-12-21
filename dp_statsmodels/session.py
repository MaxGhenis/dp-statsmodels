"""
Privacy session for managing differential privacy budget across queries.

The Session is the main user-facing interface for running DP statistical
analyses while tracking cumulative privacy loss.
"""

import numpy as np
from typing import Optional, Tuple, List, Union

from dp_statsmodels.privacy import PrivacyAccountant
from dp_statsmodels.models import (
    DPOLS,
    DPOLSResults,
    DPLogit,
    DPLogitResults,
    DPFixedEffects,
    DPFixedEffectsResults,
)


class Session:
    """
    Session for running DP analyses with budget tracking.

    A Session manages a fixed privacy budget (ε, δ) and tracks
    cumulative privacy loss as queries are executed. When the budget
    is exhausted, no more queries can be run.

    IMPORTANT: Standard errors are typically 100-1000x larger than non-private
    methods at typical privacy budgets (ε=1-10). This requires samples 10,000-
    1,000,000x larger to maintain equivalent statistical power.

    Parameters
    ----------
    epsilon : float
        Total epsilon budget for the session.
    delta : float
        Total delta budget for the session.
    bounds_X : tuple of (min, max)
        Bounds for features. REQUIRED - must be specified a priori based on
        domain knowledge, NOT computed from data.
    bounds_y : tuple of (min, max)
        Bounds for response variable. REQUIRED - must be specified a priori
        based on domain knowledge, NOT computed from data.
    composition : {'basic', 'rdp'}
        Composition method for combining privacy costs.
    random_state : int or np.random.Generator, optional
        Random state for reproducibility.
    budget_allocation : str, optional
        How to allocate privacy budget. Options:
        - 'equal': Equal split between X'X and X'y (default, simple)
        - 'optimal': Sheffet (2017) optimal allocation (minimizes MSE)

    Attributes
    ----------
    epsilon_spent : float
        Cumulative epsilon consumed.
    epsilon_remaining : float
        Remaining epsilon budget.

    Examples
    --------
    >>> import dp_statsmodels.api as sm_dp
    >>> session = sm_dp.Session(epsilon=1.0, delta=1e-5,
    ...                         bounds_X=(-10, 10), bounds_y=(-100, 100))
    >>> result = session.OLS(y, X)
    >>> print(f"Budget remaining: ε = {session.epsilon_remaining:.3f}")

    Notes
    -----
    Privacy guarantees ONLY hold if bounds are specified a priori based on
    domain knowledge. Computing bounds from data voids all privacy protections.

    The session tracks privacy using composition theorems. By default,
    basic sequential composition is used, which adds epsilon values.
    """

    def __init__(
        self,
        epsilon: float,
        delta: float,
        bounds_X: Tuple[float, float],
        bounds_y: Tuple[float, float],
        composition: str = "basic",
        random_state: Optional[Union[int, np.random.Generator]] = None,
        budget_allocation: str = 'equal',
    ):
        if bounds_X is None:
            raise ValueError(
                "bounds_X is REQUIRED for differential privacy. "
                "Specify bounds a priori based on domain knowledge."
            )
        if bounds_y is None:
            raise ValueError(
                "bounds_y is REQUIRED for differential privacy. "
                "Specify bounds a priori based on domain knowledge."
            )

        self.epsilon = epsilon
        self.delta = delta
        self.bounds_X = bounds_X
        self.bounds_y = bounds_y
        self.random_state = random_state
        self.budget_allocation = budget_allocation

        self._accountant = PrivacyAccountant(
            epsilon_budget=epsilon,
            delta_budget=delta,
            composition=composition
        )

    # Context manager support
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    @property
    def epsilon_spent(self) -> float:
        """Total epsilon consumed so far."""
        return self._accountant.epsilon_spent

    @property
    def epsilon_remaining(self) -> float:
        """Remaining epsilon budget."""
        return self._accountant.epsilon_remaining

    @property
    def delta_spent(self) -> float:
        """Total delta consumed so far."""
        return self._accountant.delta_spent

    @property
    def queries(self) -> int:
        """Number of queries executed."""
        return self._accountant.queries

    def _allocate_budget(
        self,
        epsilon: Optional[float] = None,
        fraction: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Allocate privacy budget for a query.

        Parameters
        ----------
        epsilon : float, optional
            Specific epsilon to use.
        fraction : float, optional
            Fraction of remaining budget to use.

        Returns
        -------
        tuple
            (epsilon, delta) for this query.
        """
        if epsilon is not None:
            query_epsilon = epsilon
        elif fraction is not None:
            query_epsilon = self.epsilon_remaining * fraction
        else:
            # Default: use 10% of remaining budget
            query_epsilon = self.epsilon_remaining * 0.1

        # Use proportional delta
        query_delta = self.delta * (query_epsilon / self.epsilon)

        return query_epsilon, query_delta

    def OLS(
        self,
        y: np.ndarray,
        X: np.ndarray,
        weights: Optional[np.ndarray] = None,
        add_constant: bool = True,
        epsilon: Optional[float] = None,
        bounds_X: Optional[Tuple[float, float]] = None,
        bounds_y: Optional[Tuple[float, float]] = None,
    ) -> DPOLSResults:
        """
        Run differentially private OLS regression.

        Parameters
        ----------
        y : np.ndarray
            Response variable.
        X : np.ndarray
            Design matrix.
        weights : np.ndarray, optional
            Sample weights for WLS.
        add_constant : bool
            Whether to add intercept (default True).
        epsilon : float, optional
            Epsilon budget for this query. If None, uses 10% of remaining.
        bounds_X : tuple, optional
            Bounds on X. Uses session default if not provided.
        bounds_y : tuple, optional
            Bounds on y. Uses session default if not provided.

        Returns
        -------
        OLSResults
            Results with coefficients, standard errors, etc.

        Raises
        ------
        ValueError
            If privacy budget is exhausted or bounds not provided.
        """
        # Allocate budget
        query_eps, query_delta = self._allocate_budget(epsilon=epsilon)

        # Check if we can afford this query
        if not self._accountant.can_afford(query_eps, query_delta):
            raise ValueError(
                f"Privacy budget exhausted. "
                f"Requested ε={query_eps:.4f}, available ε={self.epsilon_remaining:.4f}"
            )

        # Use session defaults if bounds not provided
        if bounds_X is None:
            bounds_X = self.bounds_X
        if bounds_y is None:
            bounds_y = self.bounds_y

        # Create and fit model
        model = DPOLS(
            epsilon=query_eps,
            delta=query_delta,
            bounds_X=bounds_X,
            bounds_y=bounds_y,
            random_state=self.random_state,
            budget_allocation=self.budget_allocation,
        )

        result = model.fit(y, X, weights=weights, add_constant=add_constant)

        # Record privacy expenditure
        self._accountant.spend(
            epsilon=query_eps,
            delta=query_delta,
            query_name=f"OLS_{self.queries + 1}",
            query_type="OLS"
        )

        return result

    def Logit(
        self,
        y: np.ndarray,
        X: np.ndarray,
        add_constant: bool = True,
        epsilon: Optional[float] = None,
        bounds_X: Optional[Tuple[float, float]] = None,
        regularization: float = 0.01,
    ) -> DPLogitResults:
        """
        Run differentially private logistic regression.

        Uses objective perturbation to achieve (epsilon, delta)-DP.

        Parameters
        ----------
        y : np.ndarray
            Binary response variable (0/1).
        X : np.ndarray
            Design matrix.
        add_constant : bool
            Whether to add intercept (default True).
        epsilon : float, optional
            Epsilon budget for this query. If None, uses 10% of remaining.
        bounds_X : tuple, optional
            Bounds on X. Uses session default if not provided.
        regularization : float
            L2 regularization parameter (required for DP, default 0.01).

        Returns
        -------
        LogitResults
            Results with coefficients, standard errors, etc.

        Raises
        ------
        ValueError
            If privacy budget is exhausted.
        """
        # Allocate budget
        query_eps, query_delta = self._allocate_budget(epsilon=epsilon)

        # Check if we can afford this query
        if not self._accountant.can_afford(query_eps, query_delta):
            raise ValueError(
                f"Privacy budget exhausted. "
                f"Requested ε={query_eps:.4f}, available ε={self.epsilon_remaining:.4f}"
            )

        # Use session defaults if bounds not provided
        if bounds_X is None:
            bounds_X = self.bounds_X

        # Create and fit model
        model = DPLogit(
            epsilon=query_eps,
            delta=query_delta,
            bounds_X=bounds_X,
            regularization=regularization,
        )

        result = model.fit(y, X, add_constant=add_constant)

        # Record privacy expenditure
        self._accountant.spend(
            epsilon=query_eps,
            delta=query_delta,
            query_name=f"Logit_{self.queries + 1}",
            query_type="Logit"
        )

        return result

    def PanelOLS(
        self,
        y: np.ndarray,
        X: np.ndarray,
        groups: Optional[np.ndarray] = None,
        entity_effects: bool = True,
        epsilon: Optional[float] = None,
        bounds_X: Optional[Tuple[float, float]] = None,
        bounds_y: Optional[Tuple[float, float]] = None,
    ) -> DPFixedEffectsResults:
        """
        Run differentially private panel OLS with fixed effects.

        Uses within transformation to eliminate entity effects, then
        applies noisy sufficient statistics.

        Parameters
        ----------
        y : np.ndarray
            Response variable.
        X : np.ndarray
            Design matrix (no constant - absorbed by FE).
        groups : np.ndarray
            Group/entity identifiers.
        entity_effects : bool
            Whether to include entity fixed effects (default True).
        epsilon : float, optional
            Epsilon budget for this query. If None, uses 10% of remaining.
        bounds_X : tuple, optional
            Bounds on X. Uses session default if not provided.
        bounds_y : tuple, optional
            Bounds on y. Uses session default if not provided.

        Returns
        -------
        PanelOLSResults
            Results with coefficients, standard errors, etc.

        Raises
        ------
        ValueError
            If privacy budget is exhausted or groups not provided.
        """
        if groups is None:
            raise ValueError("groups must be provided for PanelOLS")

        if not entity_effects:
            # If no entity effects, just run regular OLS
            return self.OLS(y, X, epsilon=epsilon, bounds_X=bounds_X,
                          bounds_y=bounds_y, add_constant=True)

        # Allocate budget
        query_eps, query_delta = self._allocate_budget(epsilon=epsilon)

        # Check if we can afford this query
        if not self._accountant.can_afford(query_eps, query_delta):
            raise ValueError(
                f"Privacy budget exhausted. "
                f"Requested ε={query_eps:.4f}, available ε={self.epsilon_remaining:.4f}"
            )

        # Use session defaults if bounds not provided
        if bounds_X is None:
            bounds_X = self.bounds_X
        if bounds_y is None:
            bounds_y = self.bounds_y

        # Create and fit model
        model = DPFixedEffects(
            epsilon=query_eps,
            delta=query_delta,
            bounds_X=bounds_X,
            bounds_y=bounds_y,
        )

        result = model.fit(y, X, groups)

        # Record privacy expenditure
        self._accountant.spend(
            epsilon=query_eps,
            delta=query_delta,
            query_name=f"PanelOLS_{self.queries + 1}",
            query_type="PanelOLS"
        )

        return result

    # Backwards compatible lowercase aliases
    def ols(self, *args, **kwargs) -> DPOLSResults:
        """Alias for OLS (backwards compatibility)."""
        return self.OLS(*args, **kwargs)

    def logit(self, *args, **kwargs) -> DPLogitResults:
        """Alias for Logit (backwards compatibility)."""
        return self.Logit(*args, **kwargs)

    def fe(self, y, X, groups, **kwargs) -> DPFixedEffectsResults:
        """Alias for PanelOLS (backwards compatibility)."""
        return self.PanelOLS(y, X, groups=groups, entity_effects=True, **kwargs)

    def summary(self) -> str:
        """Get session summary including budget status."""
        return self._accountant.summary()

    def history(self) -> List[dict]:
        """Get history of all queries."""
        return self._accountant.get_history()

    # Backwards compatible alias
    def get_history(self) -> List[dict]:
        """Alias for history (backwards compatibility)."""
        return self.history()

    def __repr__(self) -> str:
        return (
            f"Session(ε={self.epsilon}, δ={self.delta}, "
            f"spent={self.epsilon_spent:.4f}, queries={self.queries})"
        )


# Backwards compatibility alias
PrivacySession = Session
