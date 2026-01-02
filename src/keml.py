# src/keml.py

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from sklearn.linear_model import LinearRegression

@dataclass
class KeMLRegressor:
    """
    Knowledge-enhanced ML regressor:

    y_exp = y_emp + δ(x, y_emp)
    δ(x, y_emp) = f_L(x, y_emp) + f_n(x, y_emp)

    - f_L: single-neuron linear model (LinearRegression).
    - f_n: nonlinear ML model (e.g., SVR, KNN, RF, XGB, ANN).

    Notes:
      * X must be the feature matrix (scaled or not; your choice).
      * y_emp is an array of empirical baseline predictions aligned with X.
    """
    base_model: Any
    fit_base_model: bool = True

    def __post_init__(self):
        self.lin_ = LinearRegression()
        self._fitted = False

    def _stack_features(self, X: np.ndarray, y_emp: np.ndarray) -> np.ndarray:
        """
        Concatenate feature matrix X with y_emp as an extra column.
        """
        y_emp_col = np.asarray(y_emp).reshape(-1, 1)
        return np.hstack([X, y_emp_col])

    def fit(self, X: np.ndarray, y: np.ndarray, y_emp: np.ndarray):
        """
        Fit the KeML model.

        Parameters
        ----------
        X : (n_samples, n_features)
            Input features (train).
        y : (n_samples,)
            Experimental ultimate strength (train).
        y_emp : (n_samples,)
            Empirical baseline predictions (train).
        """
        X = np.asarray(X)
        y = np.asarray(y).ravel()
        y_emp = np.asarray(y_emp).ravel()

        Z = self._stack_features(X, y_emp)
        delta = y - y_emp

        # 1) fit linear residual
        self.lin_.fit(Z, delta)
        delta_lin = self.lin_.predict(Z)

        # 2) fit nonlinear residual on remaining error
        delta_nl = delta - delta_lin

        if self.fit_base_model and self.base_model is not None:
            self.base_model.fit(Z, delta_nl)

        self._fitted = True
        return self

    def predict(self, X: np.ndarray, y_emp: np.ndarray) -> np.ndarray:
        """
        Predict ultimate strength using KeML.
        """
        if not self._fitted:
            raise RuntimeError("KeMLRegressor must be fitted before predicting.")

        X = np.asarray(X)
        y_emp = np.asarray(y_emp).ravel()
        Z = self._stack_features(X, y_emp)

        delta_lin = self.lin_.predict(Z)

        if self.base_model is not None:
            delta_nl = self.base_model.predict(Z)
        else:
            delta_nl = 0.0

        return y_emp + delta_lin + delta_nl
