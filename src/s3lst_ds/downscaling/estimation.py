from typing import Literal, Self

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline

from s3lst_ds.downscaling.preprocessing import DownscalerPreprocessor
from s3lst_ds.downscaling.regression import DownscalerRegressor, Regressor


class DownscalerEstimator(BaseEstimator, RegressorMixin):
    """
    A `Pipeline` with a `DownscalerPreprocessor` and a `DownscalerRegressor` combined.

    The `DownscalerPreprocessor` prepares the inputs to the `DownscalerRegressor`:
        - selects the predictors that the user wants to employ in the model;
        - scales and encodes numeric and categorical predictors, respectively.

    The `DownscalerRegressor` corresponds to a non-pixel-wise regression model that
    predicts targets using preprocessed predictors (coarse or fine). It is trained with
    coarse data.

    Attributes
    ----------

    base_model : Regressor
        The general (i.e. non-pixel-wise) base model to be fitted with coarse data.

    cols_X : list or np.ndarray
        The names of the predictor columns to regard.

    cols_mask : list or np.ndarray, optional
        The names of the mask columns to regard.

    scale : {"standardize", "min_max_normalize", None}, default="standardize"
        The scaling method to apply to numerical predictors:
            - `"standardize"`: to standardize the numerical predictors (zero mean and
            unit variance);
            - `"min_max_normalize"`: to min-max normalize the numerical predictors (to
            the range `[0, 1]`);
            - `None`: to regard the numerical predictors raw (no scaling).

    encode : {"one_hot", "dummy", None}, default="dummy"
        The encoding method to apply to the categorical predictors:
            - `"one_hot"`: to one-hot encode the categorical predictors;
            - `"dummy"`: to dummy encode the categorical predictors (one-hot encoding
            with the first component dropped);
            - `None`: to regard the categorical predictors raw (no encoding).

        Note that dummy encoding is usually considered in place of one-hot to avoid
        multicollinearity problems (one may show that a component of a one-hot encoding
        vector is fully determined by all the other components making it redundant).

    lasso_sel : bool, default=False
        Whether to use a Lasso regression for selecting the scaled-encoded `cols_X`
        predictors downstream of the preprocessor. Lasso selection is such that solely
        the input predictors associated with coefficients of the fitted Lasso regression
        model having absolute values larger than `1e-5` are selected. Note that the
        non-encoded `cols_X` predictors are regardlessly considered downstream of the
        preprocessor.

    lasso_alpha : float, default=1.0
        The regularization strength of the Lasso regression model used for selecting the
        scaled-encoded `cols_X` predictors downstream of the preprocessor. Such
        regularization strength is the multiplying constant of the weight vector L1-norm
        (sum of the absolute values of the components) in the Lasso regression objective
        function. The larger the value, the stronger the regularization. Note that this
        parameter only takes effect if `lasso_sel` is `True`.

    preprocessor : preprocessing.DownscalerPreprocessor
        The actual pre-processor.

    regressor : regresssion.DownscalerRegressor
        The actual regressor.

    pipeline : sklearn.pipeline.Pipeline
        The combination of `preprocessor` and `regressor` as an
        `sklearn.pipeline.Pipeline`.

    """

    def __init__(
        self,
        base_model: Regressor,
        cols_X: list | np.ndarray,
        cols_mask: list | np.ndarray | None = None,
        scale: Literal["standardize", "min_max_normalize"] | None = "standardize",
        encode: Literal["one_hot", "dummy"] | None = "dummy",
        lasso_sel: bool = False,
        lasso_alpha: float = 1.0,
    ) -> None:

        super().__init__()
        # NOTE: base_model is a Regressor class instance and it is an attribute of the
        # the DownscalerRegressor class instance (regressor).
        # The latter would be updated with the changes that are done on the former even
        # if outside of the latter.
        self._base_model = base_model
        self._cols_X = cols_X
        self._cols_mask = cols_mask if cols_mask is not None else []
        self._scale = scale
        self._encode = encode
        self._lasso_sel = lasso_sel
        self._lasso_alpha = lasso_alpha

        # Pre-processor
        self.preprocessor = self.get_preprocessor()
        # Regressor
        self.regressor = self.get_regressor()
        # Pipeline
        # NOTE: a sklearn Pipeline stores references to the given objects (preprocessor,
        # regressor). It does not correspond to a copy of them. Therefore, the pipeline
        # would be updated with the changes done to the objects outside of it. This
        # means that if a hyperparameter of the pipeline is to be changed, it can be
        # done through the objects themselves.
        self.pipeline = self.get_pipeline()

    @property
    def base_model(self) -> Regressor:
        return self._base_model

    @property
    def cols_X(self) -> list | np.ndarray:
        return self._cols_X

    @property
    def cols_mask(self) -> list | np.ndarray:
        return self._cols_mask

    @property
    def scale(self) -> Literal["standardize", "min_max_normalize"] | None:
        return self._scale  # type: ignore

    @property
    def encode(self) -> Literal["one_hot", "dummy"] | None:
        return self._encode  # type: ignore

    @property
    def lasso_sel(self) -> bool:
        return self._lasso_sel

    @property
    def lasso_alpha(self) -> float:
        return self._lasso_alpha

    @base_model.setter
    def base_model(self, value: Regressor) -> None:
        self._base_model = value
        self.regressor.base_model = value

    @cols_X.setter
    def cols_X(self, value: list | np.ndarray) -> None:
        self._cols_X = value
        self.preprocessor.cols_X = value

    @cols_mask.setter
    def cols_mask(self, value: list | np.ndarray | None) -> None:
        self._cols_mask = value if value is not None else []
        self.preprocessor.cols_mask = self._cols_mask
        self.regressor.cols_mask = self._cols_mask

    @scale.setter
    def scale(self, value: Literal["standardize", "min_max_normalize"] | None) -> None:
        self._scale = value
        self.preprocessor.scale = value

    @encode.setter
    def encode(self, value: Literal["one_hot", "dummy"] | None) -> None:
        self._encode = value
        self.preprocessor.encode = value

    @lasso_sel.setter
    def lasso_sel(self, value: bool) -> None:
        self._lasso_sel = value
        self.preprocessor.lasso_sel = value

    @lasso_alpha.setter
    def lasso_alpha(self, value: float) -> None:
        self._lasso_alpha = value
        self.preprocessor.lasso_alpha = value

    def get_preprocessor(self) -> DownscalerPreprocessor:
        preprocessor = DownscalerPreprocessor(
            cols_X=self.cols_X,
            cols_mask=self.cols_mask,
            scale=self.scale,
            encode=self.encode,
            lasso_sel=self.lasso_sel,
            lasso_alpha=self.lasso_alpha,
        )
        return preprocessor

    def get_regressor(self) -> DownscalerRegressor:
        regressor = DownscalerRegressor(
            base_model=self.base_model,
            cols_mask=self.cols_mask,
        )
        return regressor

    def get_pipeline(self) -> Pipeline:
        pipeline = Pipeline(
            steps=[
                ("preprocessor", self.preprocessor),
                ("regressor", self.regressor),
            ]
        )
        return pipeline

    def fit(
        self,
        X_and_mask_coarse: np.ndarray | pd.DataFrame,
        y_coarse: np.ndarray | pd.Series,
        sample_weight: np.ndarray | pd.Series | None = None,
    ) -> Self:
        """
        Fit the preprocessing transformers and the general base model to training
        coarse data.

        Parameters
        ----------

        X_and_mask_coarse : np.ndarray or pd.DataFrame
            The training coarse predictors and masks.

        y_coarse : np.ndarray or pd.Series
            The training coarse target.

        sample_weight : np.ndarray or pd.Series or None, default=None
            Weight of each sample in the cost function of the model.

        Returns
        -------

        self : DownscalerEstimator
            The fitted instance itself.
        """

        self.pipeline.fit(
            X_and_mask_coarse, y_coarse, regressor__sample_weight=sample_weight
        )

        # NOTE: attribute `is_fitted_` must be set to `True` to let `sklearn` know that
        # the instance is already fitted.
        self.is_fitted_ = True

        return self

    def predict(self, X_and_mask: np.ndarray | pd.DataFrame) -> np.ndarray:
        """
        Apply the preprocessing transformations on the issued predictors and masks, and
        predict the target with them by applying the general base model.

        Parameters
        ----------

        X_and_mask : np.ndarray or pd.DataFrame
            Predictors and masks.

        Returns
        -------

        y_pred : np.ndarray
            Predicted target.
        """

        return self.pipeline.predict(X_and_mask)  # type: ignore

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
            Prediction scores keyed by scorer alias.
        """

        return self.pipeline.score(X_and_mask, y, sample_weight=sample_weight)  # type: ignore
