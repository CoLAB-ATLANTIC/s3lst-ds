"""
Regression metrics
"""

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)

from ._base import Scorer


# Dummy scoring function
def dummy(*args, **kwargs) -> None:
    """
    Dummy scoring function which always return `None`.

    Parameters
    ----------

        args :
            Any positional arguments.
        kwargs :
            Any keyword arguments.

    Returns
    -------

        score : None
            Simply `None`.

    """


# Function for out-of-sample R-squared
def r2_oos_score(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    y_dummy_pred: ArrayLike,
    sample_weight: ArrayLike | None = None,
) -> float | np.ndarray:
    """
    Compute out-of-sample R-squared.

    Parameters
    ----------

        y_true : array-like
            The true target.
        y_pred : array-like
            The target predicted by the model.
        y_dummy_pred : array-like
            The target predicted by a mean dummy model (this model is simply the mean of
            the training target).
        sample_weight : array-like or None, default=None
            Weight of each sample.
    Returns
    -------

        r2_oos : float or ndarray of floats
            The out-of-sample R-squared.
    """

    mse_pred = mean_squared_error(y_true, y_pred, sample_weight=sample_weight)
    mse_dummy = mean_squared_error(y_true, y_dummy_pred, sample_weight=sample_weight)
    r2_oos = 1 - (mse_pred / mse_dummy)
    return r2_oos


# Function for residual standard error
def residual_standard_error(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    p: int,
    sample_weight: np.ndarray | pd.Series | None = None,
) -> float:
    """
    Compute residual standard error (RSE).

    Parameters
    ----------

        y_true : np.ndarray or pd.Series
            The true target.
        y_pred : np.ndarray or pd.Series
            The target predicted by the model.
        p : int
            Number of parameters in the model.
        sample_weight : np.ndarray or pd.Series or None, default=None
            Weight of each sample.
    Returns
    -------

        rse: float
            The residual standard error.
    """

    # Number of samples
    n = len(y_true)

    rse = root_mean_squared_error(
        y_true, y_pred, sample_weight=sample_weight
    ) * np.sqrt(n / (n - p))
    return rse


# Function for mean absolute error of the standardised target (using the true target
# statistics)
def mean_absolute_error_delta(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    sample_weight: np.ndarray | pd.Series | None = None,
) -> float:
    """
    Compute mean absolute error of the standardised target (using the true target
    statistics in the standardisation of both true and predicted targets).

    Parameters
    ----------

        y_true : np.ndarray or pd.Series
            The true inference target.
        y_pred : np.ndarray or pd.Series
            The inference target predicted by the model.
        sample_weight : np.ndarray or pd.Series or None, default=None
            Weight of each sample.

    Returns
    -------

        mae_delta : float
            The mean absolute error of the standardised target.
    """
    y_true_mean = y_true.mean()
    y_true_std = y_true.std()
    y_true = (y_true - y_true_mean) / y_true_std
    y_pred = (y_pred - y_true_mean) / y_true_std

    mae_delta = mean_absolute_error(y_true, y_pred, sample_weight=sample_weight)
    return mae_delta


# Function for root mean squared error of the standardised target (using the true target
# statistics)
def root_mean_squared_error_delta(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    sample_weight: np.ndarray | pd.Series | None = None,
) -> float:
    """
    Compute root mean squared error of the standardised target (using the true target
    statistics in the standardisation of both true and predicted targets).

    Parameters
    ----------

        y_true : np.ndarray or pd.Series
            The true inference target.
        y_pred : np.ndarray or pd.Series
            The inference target predicted by the model.
        sample_weight : np.ndarray or pd.Series or None, default=None
            Weight of each sample.

    Returns
    -------

        rmse_delta : float
            The root mean squared error of the standardised target.
    """
    y_true_mean = y_true.mean()
    y_true_std = y_true.std()
    y_true = (y_true - y_true_mean) / y_true_std
    y_pred = (y_pred - y_true_mean) / y_true_std

    rmse_delta = root_mean_squared_error(y_true, y_pred, sample_weight=sample_weight)
    return rmse_delta


# Function for mean bias error
def mean_bias_error(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    sample_weight: np.ndarray | pd.Series | None = None,
) -> float:
    """
    Compute mean bias error (MBE).

    Parameters
    ----------

        y_true : np.ndarray or pd.Series
            The true target.
        y_pred : np.ndarray or pd.Series
            The target predicted by the model.
        sample_weight : np.ndarray or pd.Series or None, default=None
            Weight of each sample.

    Returns
    -------

        mbe : float
            The mean bias error.
    """

    # NOTE: float() is used to convert result from np.float64 to built-in float
    mbe = float(
        (sample_weight * (y_pred - y_true)).sum() / sample_weight.sum()
        if sample_weight is not None
        else (y_pred - y_true).mean()
    )
    return mbe


# R-squared (Coefficient of determination)
r2 = Scorer(
    alias="r2",
    alias_fancy=r"$R^2$",
    func=r2_score,
    greater_is_better=True,
    dimensionless=True,
)

# Out-of-sample R-squared
r2_oos = Scorer(
    alias="r2_oos",
    alias_fancy=r"$R^2_{\mathrm{OOS}}$",
    func=r2_oos_score,
    greater_is_better=True,
    dimensionless=True,
)

# Residual standard error
rse = Scorer(
    alias="rse",
    alias_fancy=r"$\mathrm{RSE}$",
    func=residual_standard_error,
    greater_is_better=False,
    dimensionless=False,
)

# Root mean squared error
rmse = Scorer(
    alias="rmse",
    alias_fancy=r"$\mathrm{RMSE}$",
    func=root_mean_squared_error,
    greater_is_better=False,
    dimensionless=False,
)

# Root mean squared error of the standardised target (using the true target statistics)
rmse_delta = Scorer(
    alias="rmse_delta",
    alias_fancy=r"$\mathrm{RMSE}_{\delta}$",
    func=root_mean_squared_error_delta,
    greater_is_better=False,
    dimensionless=True,
)

# Mean absolute error
mae = Scorer(
    alias="mae",
    alias_fancy=r"$\mathrm{MAE}$",
    func=mean_absolute_error,
    greater_is_better=False,
    dimensionless=False,
)

# Mean absolute error of the standardised target (using the true target statistics)
mae_delta = Scorer(
    alias="mae_delta",
    alias_fancy=r"$\mathrm{MAE}_{\delta}$",
    func=mean_absolute_error_delta,
    greater_is_better=False,
    dimensionless=True,
)

# Mean bias error
mbe = Scorer(
    alias="mbe",
    alias_fancy=r"$\mathrm{MBE}$",
    func=mean_bias_error,
    greater_is_better=False,
    dimensionless=False,
)

# Akaike Information Criterion (AIC)
# NOTE: https://en.wikipedia.org/wiki/Akaike_information_criterion
aic = Scorer(
    alias="aic",
    alias_fancy=r"$\mathrm{AIC}$",
    # WARNING: missing function.
    func=dummy,
    greater_is_better=False,
    dimensionless=True,
)

# Bayesian Information Criterion (BIC)
# NOTE: https://en.wikipedia.org/wiki/Bayesian_information_criterion
bic = Scorer(
    alias="bic",
    alias_fancy=r"$\mathrm{BIC}$",
    # WARNING: missing function.
    func=dummy,
    greater_is_better=False,
    dimensionless=True,
)
