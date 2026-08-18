from typing import Self

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.dummy import DummyRegressor

from s3lst_ds.downscaling.regression_base import Regressor
from s3lst_ds.utilities.df_utils import get_i_non_nan
from s3lst_ds.utilities.metrics import (
    mae,
    mae_delta,
    mbe,
    r2,
    r2_oos,
    rmse,
    rmse_delta,
)


class DownscalerRegressor(BaseEstimator, RegressorMixin):
    """
    A non-pixel-wise regression model that also applies masks to the data before
    fitting/predicting. This regressor is to be trained with coarse data. It predicts
    either coarse or fine target with either coarse or fine predictors, respectively.

    Attributes
    ----------

    base_model : Regressor
        The general (i.e. non-pixel-wise) base model to be fitted with coarse data.

    cols_mask : list or np.ndarray, optional
        The names of the mask columns.

    N_cols_mask : int
        Number of mask columns (defined internally).

    col_y : str, default="y"
        Name for target column inferred from the training target data (defined
        internally).

    dummy_mean_model : sklearn.dummy.DummyRegressor
        An `sklearn.dummy.DummyRegressor` model that predicts the target as the
        arithmetic mean of the seen training ones, regardless of the predictor values
        (defined internally).

    """

    def __init__(
        self,
        base_model: Regressor,
        cols_mask: list | np.ndarray | None = None,
    ) -> None:

        super().__init__()
        self.base_model = base_model
        self.cols_mask = cols_mask

        # Name for target column
        self.col_y = "y"
        # Dummy mean model (a DummyRegressor model that predicts a target as the
        # arithmetic mean of the seen training ones)
        self.dummy_mean_model = None

    @property
    def cols_mask(self) -> list | np.ndarray:
        return self._cols_mask

    @cols_mask.setter
    def cols_mask(self, value: list | np.ndarray | None) -> None:
        self._cols_mask = value if value is not None else []
        # Number of mask columns
        self.N_cols_mask = len(self._cols_mask)

    def fit(
        self,
        X_and_mask_coarse: np.ndarray | pd.DataFrame,
        y_coarse: np.ndarray | pd.Series,
        sample_weight: np.ndarray | pd.Series | None = None,
    ) -> Self:
        """
        Fit the general base model to training coarse data.

        Parameters
        ----------

        X_and_mask_coarse : np.ndarray or pd.DataFrame
            The training coarse predictors and masks. If this is a nested array, the
            masks must correspond to the last columns.

        y_coarse : np.ndarray or pd.Series
            The training coarse target.

        sample_weight : np.ndarray or pd.Series or None, default=None
            Weight of each sample in the cost function.

        Returns
        -------

        self : DownscalerRegressor
            The fitted instance itself.
        """

        # If target data is a pandas Series, use its name for the respective column in
        # the concatenated data
        if isinstance(y_coarse, pd.Series):
            self.col_y = y_coarse.name

        # If X_and_mask_coarse is a DataFrame, reset its indexes. The analogous follows
        # for y_coarse and sample_weight. This is required, since indexes of
        # X_and_mask_coarse, y_coarse and sample_weight should match when combining them
        # into a single DataFrame afterwards.
        if isinstance(X_and_mask_coarse, pd.DataFrame):
            X_and_mask_coarse = X_and_mask_coarse.reset_index(drop=True)
        if isinstance(y_coarse, pd.Series):
            y_coarse = y_coarse.reset_index(drop=True)
        if isinstance(sample_weight, pd.Series):
            sample_weight = sample_weight.reset_index(drop=True)

        # If X_and_mask_coarse is an array, convert it to DataFrame and set the name of
        # its last columns as the mask ones (since the preprocessor relocates the mask
        # columns to the right-hand side)
        if isinstance(X_and_mask_coarse, np.ndarray):
            X_and_mask_coarse = pd.DataFrame(X_and_mask_coarse)
        X_and_mask_coarse.columns = (
            list(X_and_mask_coarse.columns)[: -self.N_cols_mask]
            if self.N_cols_mask != 0
            else list(X_and_mask_coarse.columns)
        ) + self.cols_mask

        # If y_coarse is an array, convert it to a Series
        if isinstance(y_coarse, np.ndarray):
            y_coarse = pd.Series(y_coarse)

        # If sample_weight is an array, convert it to a Series
        if isinstance(sample_weight, np.ndarray):
            sample_weight = pd.Series(sample_weight)

        # Combine predictors, target, masks and sample_weights together (so that all
        # records with nan in any variable may be removed later before training)
        data = X_and_mask_coarse
        data[self.col_y] = y_coarse
        if sample_weight is not None:
            data["sample_weight"] = sample_weight

        # Remove all records with nan in the data (therefore, simultaneously applying
        # the masks if there are any)
        data = data.dropna()

        # Extract masked predictors and target
        X_coarse_masked = data.drop(
            columns=list(self.cols_mask)
            + [self.col_y]
            + (["sample_weight"] if sample_weight is not None else [])
        )
        y_coarse_masked = data[self.col_y]
        sample_weight_masked = (
            data["sample_weight"] if sample_weight is not None else None
        )

        # Fit the general model with the masked training data
        self.base_model.fit(
            X_coarse_masked,
            y_coarse_masked,
            # WARNING: PLSRegression does not support sample weights in fitting.
            **(
                {"sample_weight": sample_weight_masked}
                if not isinstance(self.base_model, PLSRegression)
                else {}
            ),
        )

        # Update the dummy mean model (this model predicts a target as the arithmetic
        # mean of the training ones, regardless of the predictor values)
        self.update_dummy_mean_model(
            X_coarse_masked, y_coarse_masked, sample_weight=sample_weight_masked
        )

        # NOTE: attribute `is_fitted_` must be set to `True` to let `sklearn` know that
        # the instance is already fitted.
        self.is_fitted_ = True

        return self

    def update_dummy_mean_model(
        self,
        X_coarse_masked: pd.DataFrame,
        y_coarse_masked: pd.Series,
        sample_weight: pd.Series | None = None,
    ) -> None:
        """
        Update dummy mean model for predicting targets. This dummy mean model predicts a
        target as the arithmetic mean of the seen training ones, regardless of the
        predictor values.

        Parameters
        ----------

        X_coarse_masked : pd.DataFrame
            Masked coarse predictors.

        y_coarse_masked : pd.Series
            Masked coarse target.

        sample_weight : np.ndarray or pd.Series
            Weight of each sample in the cost function.
        """

        # Dummy mean model trained with the masked coarse predictors and target
        self.dummy_mean_model = DummyRegressor(strategy="mean").fit(
            # NOTE: the dummy regressor solely uses the training targets for its
            # training. In this current fit method, the given predictors are simply used
            # to ensure parallelism with the fit methods of all the other scikit-learn
            # models.
            X_coarse_masked,
            y_coarse_masked,
            sample_weight=sample_weight,
        )

    def predict(self, X_and_mask: np.ndarray | pd.DataFrame) -> np.ndarray:
        """
        Predict target from predictors and masks using the general base model.

        Parameters
        ----------

        X_and_mask : np.ndarray or pd.DataFrame
            Predictors and masks. If this is a nested array, the masks must correspond
            to the last columns.

        Returns
        -------

        y_pred : np.ndarray
            Predicted target.
        """

        # If X_and_mask is an array, convert it to DataFrame and set the name of its
        # last columns as the mask ones
        if isinstance(X_and_mask, np.ndarray):
            X_and_mask = pd.DataFrame(X_and_mask)
        X_and_mask.columns = (
            list(X_and_mask.columns)[: -self.N_cols_mask]
            if self.N_cols_mask != 0
            else list(X_and_mask.columns)
        ) + self.cols_mask

        data = X_and_mask

        # Get length of the data
        n = len(data)

        # Get integer indexes of the combined predictor and mask records not containing
        # any nan
        # NOTE: This will be required to later express the predicted masked target as a
        # Series with the full original length of the data (i.e. with non-nan and nan
        # values)
        i_non_nan = get_i_non_nan(data)

        # Remove all records with nan in the data (therefore, simultaneously applying
        # the masks if there are any)
        data = data.dropna()

        # Extract masked predictors
        X_masked = data.drop(columns=self.cols_mask)  # type: ignore

        # Predict the masked target
        y_masked_pred = self.base_model.predict(X_masked)

        # Express the predicted masked target as Series with the full original length
        # of the data (i.e. with non-nan and nan values)
        y_pred = np.full(n, np.nan)
        y_pred[i_non_nan] = y_masked_pred

        return y_pred

    def score(
        self,
        X_and_mask: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        sample_weight: np.ndarray | pd.Series | None = None,
    ) -> dict[str, float]:
        """
        Predict target and score prediction.

        Parameters
        ----------

        X_and_mask : np.ndarray or pd.DataFrame
            Predictors and masks.

        y : np.ndarray or pd.Series
            True target.

        sample_weight : np.ndarray or pd.Series or None, default=None
            Weight of each sample in the score.

        Returns
        -------

        score : dict[str, float]
            Prediction scores.
        """

        # If y is a Series, reset its indexes. The analogous follows for sample_weight.
        # This is required, since indexes of y, y_pred and y_dummy_pred and
        # sample_weight should match when combining them into a single DataFrame
        # afterwards.
        if isinstance(y, pd.Series):
            y = y.reset_index(drop=True)
        if isinstance(sample_weight, pd.Series):
            sample_weight = sample_weight.reset_index(drop=True)

        # Predict target from predictors using the general base model
        y_pred = self.predict(X_and_mask)

        # Predict target from predictors using the dummy mean model
        # NOTE: this is required for computing out-of-sample coefficient of
        # determination.
        # NOTE: herein, DummyRegressor's constant_ attribute is used instead of the
        # predict() method since the latter, for some reason, would make the very
        # current score method to not be used in sklearn's cross_validate and
        # GridSearchCV.
        y_dummy_pred = np.squeeze(self.dummy_mean_model.constant_)  # type: ignore

        # Combine the true and predicted targets (so that all records containing any nan
        # may be later dropped and the prediction score afterwards computed)
        data = pd.DataFrame(
            data={
                "y_true": y,
                "y_pred": y_pred,
                "y_dummy_pred": y_dummy_pred,
                **(
                    {
                        "sample_weight": sample_weight,
                    }
                    if sample_weight is not None
                    else {}
                ),
            }
        )

        # Drop nan
        data = data.dropna()

        # Compute prediction score
        score = {
            # Coefficient of determination
            "r2": r2(
                y_true=data["y_true"],
                y_pred=data["y_pred"],
                sample_weight=(
                    data["sample_weight"] if sample_weight is not None else None
                ),
            ),
            # Out-of-sample coefficient of determination
            # [NOTE: this is such that it uses a dummy mean model (simply the arithmetic
            # mean of the masked training coarse targets) as reference.]
            "r2_oos": r2_oos(
                y_true=data["y_true"],
                y_pred=data["y_pred"],
                y_dummy_pred=data["y_dummy_pred"],
                sample_weight=(
                    data["sample_weight"] if sample_weight is not None else None
                ),
            ),
            # Root mean squared error
            "rmse": rmse(
                y_true=data["y_true"],
                y_pred=data["y_pred"],
                sample_weight=(
                    data["sample_weight"] if sample_weight is not None else None
                ),
            ),
            # Root mean squared error of the standardised target (using true target
            # statistics)
            "rmse_delta": rmse_delta(
                y_true=data["y_true"],
                y_pred=data["y_pred"],
                sample_weight=(
                    data["sample_weight"] if sample_weight is not None else None
                ),
            ),
            # Mean absolute error
            "mae": mae(
                y_true=data["y_true"],
                y_pred=data["y_pred"],
                sample_weight=(
                    data["sample_weight"] if sample_weight is not None else None
                ),
            ),
            # Mean absolute error of the standardized target (using true
            # target statistics)
            "mae_delta": mae_delta(
                y_true=data["y_true"],
                y_pred=data["y_pred"],
                sample_weight=(
                    data["sample_weight"] if sample_weight is not None else None
                ),
            ),
            # Mean bias error
            "mbe": mbe(
                y_true=data["y_true"],
                y_pred=data["y_pred"],
                sample_weight=(
                    data["sample_weight"] if sample_weight is not None else None
                ),
            ),
        }

        return score
