from __future__ import annotations

from typing import Literal, Self

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler


class Scaler(BaseEstimator, TransformerMixin):
    """
    A transformer for scaling numerical predictors.

    The scaler is a standard scaler, a min-max normalizer or an identity transformer (if
    no scaling is wanted).

    Attributes
    ----------

    cols_mask : list or np.ndarray, optional
        The column names of the masks to regard.

    scale : {"standardize", "min_max_normalize", None}, default="standardize"
        The scaling method to apply to numerical predictors:
            - `"standardize"`: to standardize the numerical predictors (zero mean and
            unit variance);
            - `"min_max_normalize"`: to min-max normalize the numerical predictors (to
            the range `[0, 1]`);
            - `None`: to regard the numerical predictors raw (no scaling).

    pipeline : sklearn.pipeline.Pipeline
        The actual scaling pipeline (defined internally).
    """

    def __init__(
        self,
        scale: Literal["standardize", "min_max_normalize"] | None = "standardize",
        cols_mask: list | np.ndarray | None = None,
    ) -> None:
        super().__init__()

        self._scale = scale
        self._cols_mask = cols_mask if cols_mask is not None else []
        self.pipeline = self.get_pipeline()

    @property
    def cols_mask(self) -> list | np.ndarray:
        return self._cols_mask

    @property
    def scale(self) -> Literal["standardize", "min_max_normalize"] | None:
        return self._scale  # type: ignore

    @cols_mask.setter
    def cols_mask(self, value: list | np.ndarray | None) -> None:
        self._cols_mask = value if value is not None else []

    @scale.setter
    def scale(self, value: Literal["standardize", "min_max_normalize"] | None) -> None:
        self._scale = value
        self.pipeline = self.get_pipeline()

    def get_pipeline(self) -> StandardScaler | MinMaxScaler | IdentityTransformer:
        """
        Get the scaling pipeline for the numerical predictors.

        Returns
        -------

        pipeline : StandardScaler or MinMaxScaler or IdentityTransformer
            The scaling pipeline.
        """

        pipeline = (
            StandardScaler()
            if self.scale == "standardize"
            else (
                MinMaxScaler()
                if self.scale == "min_max_normalize"
                else IdentityTransformer()
            )
        ).set_output(
            transform="pandas"  # Get output as pandas DataFrame instead of np.ndarray
        )

        return pipeline  # type: ignore

    def get_X_num(self, X: pd.DataFrame):
        """
        Get the numerical predictors of the DataFrame `X`.

        Parameters
        ----------

        X : pd.DataFrame
            The DataFrame from which the numerical predictors are to be returned.

        Returns
        -------

        X_num : pd.DataFrame
            The numerical predictors of the DataFrame `X`.
        """

        cols_X_num = (
            X.select_dtypes(include=np.number)
            .columns.difference(self.cols_mask)  # type: ignore
            .tolist()
        )
        X_num = X[cols_X_num]

        return X_num

    def get_X_cat(self, X: pd.DataFrame):
        """
        Get the categorical predictors of the DataFrame `X`.

        Parameters
        ----------

        X : pd.DataFrame
            The DataFrame from which the categorical predictors are to be returned.

        Returns
        -------

        X_cat : pd.DataFrame
            The categorical predictors of the DataFrame `X`.
        """

        cols_X_cat = (
            X.select_dtypes(include=["object", "category"])
            .columns.difference(self.cols_mask)  # type: ignore
            .tolist()
        )
        X_cat = X[cols_X_cat]

        return X_cat

    def fit(
        self,
        X: pd.DataFrame,
        y=None,
        sample_weight: np.ndarray | pd.Series | None = None,
    ) -> Self:
        """
        Fit the scaler, that is, compute the relevant statistics of the numerical
        predictors of `X` so that the scaling transform may be defined and then used by
        the `transform()` method.

        Parameters
        ----------

        X : pd.DataFrame
            The training predictors and masks.
        sample_weight : np.ndarray or pd.Series or None, default=None
            Weight of each sample.

        Returns
        -------

        self : Scaler
            The fitted scaler.
        """

        # Reset the indexes of X. The analogous follows for sample_weight if is a
        # Series. This is required, since indexes of X and sample_weight should match
        # when combining them into a single DataFrame afterwards.
        X = X.reset_index(drop=True)
        if isinstance(sample_weight, pd.Series):
            sample_weight = sample_weight.reset_index(drop=True)

        # Convert sample_weight to Series if it is array
        if isinstance(sample_weight, np.ndarray):
            sample_weight = pd.Series(sample_weight)

        # Combine predictors, masks and sample_weights together (so that all records
        # with nan in any variable may be removed later before training)
        data = X
        if sample_weight is not None:
            data["sample_weight"] = sample_weight

        # Remove all records with nan in the data (therefore, simultaneously applying
        # the masks if there are any)
        data = data.dropna()

        # Extract masked numerical predictors and sample weights
        X_num_masked = self.get_X_num(
            data.drop(["sample_weight"] if sample_weight is not None else [])
        )
        sample_weight_masked = (
            data["sample_weight"] if sample_weight is not None else None
        )

        # Fit the scaler with the masked training data
        self.pipeline.fit(
            X_num_masked,
            **(
                # NOTE: scikit-learn's `MinMaxScaler` does not support sample weights as
                # these do not make sense for such scaler.
                {"sample_weight": sample_weight_masked}
                if self.scale != "min_max_normalize"
                else {}
            ),
        )

        # NOTE: attribute `is_fitted_` must be set to `True` to let `sklearn` know that
        # the instance is already fitted.
        self.is_fitted_ = True

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Scale the numerical predictors in `X` and return them combined with the
        categorical predictors and masks.

        Parameters
        ----------

        X : pd.DataFrame
            Predictors and masks.

        Returns
        -------

        X_trans : pd.DataFrame
            `X` with its numerical predictors scaled.
        """

        # Get the numerical predictors
        X_num = self.get_X_num(X)

        # Scaled the numerical predictors
        X_num_scaled = self.pipeline.transform(X_num).reset_index(drop=True)  # type: ignore

        # Get the original categorical predictors and masks
        X_cat = self.get_X_cat(X).reset_index(drop=True)
        X_mask = X[self.cols_mask].reset_index(drop=True)

        # Combine the data
        X_trans = pd.concat([X_num_scaled, X_cat, X_mask], axis="columns")

        return X_trans

    def set_output(self, transform: Literal["default", "pandas"] | None = None) -> Self:
        """
        Set output format of `transform()` and `fit_transform()` methods.

        Parameters
        ----------

        transform : {"default", "pandas", None}, default None
            Output format:
                - `"default"` or `"pandas"` - for pandas DataFrame
                - `None` - keep current output format.

        Returns
        -------

        self : Scaler
            The instance itself.

        """

        match transform:
            case "default" | "pandas":
                self.output = "pandas"
            # case "polars":
            #     self.output = "polars"

            case None:
                pass

        return self


class Encoder(BaseEstimator, TransformerMixin):
    """
    A transformer for encoding categorical predictors.

    The transformer is a one-hot or dummy encoder or an identity transformer (if no
    encoding is wanted). The encoder regards nans as a category which without further
    additional reprocessing would make the encoded nans to not be dropped in the masking
    process of the `DownscalerRegressor` (since the method `dropna()` is used). To avoid
    such issue, a decoder of the nan encodings is considered to transform them into nan
    entries and to drop the redundant column associated with the nan category.

    Attributes
    ----------

    cols_mask : list or np.ndarray, optional
        The column names of the masks to regard.

    encode : {"one_hot", "dummy", None}, default="dummy"
        The encoding method to apply to the categorical predictors:
            - `"one_hot"`: to one-hot encode the categorical predictors;
            - `"dummy"`: to dummy encode the categorical predictors (one-hot encoding
            with the first component dropped);
            - `None`: to regard the categorical predictors raw (no encoding).

        Note that dummy encoding is usually considered in place of one-hot to avoid
        multicollinearity problems (one may show that a component of a one-hot encoding
        vector is fully determined by all the other components making it redundant).

    pipeline : sklearn.pipeline.Pipeline
        The actual scaling pipeline (defined internally).
    """

    def __init__(
        self,
        encode: Literal["one_hot", "dummy"] | None = "dummy",
        cols_mask: list | np.ndarray | None = None,
    ) -> None:
        super().__init__()

        self._cols_mask = cols_mask if cols_mask is not None else []
        self._encode = encode

        # The pre-processing pipeline
        self.pipeline = self.get_pipeline()

    @property
    def cols_mask(self) -> list | np.ndarray:
        return self._cols_mask

    @property
    def encode(self) -> Literal["one_hot", "dummy"] | None:
        return self._encode  # type: ignore

    @cols_mask.setter
    def cols_mask(self, value: list | np.ndarray | None) -> None:
        self._cols_mask = value if value is not None else []

    @encode.setter
    def encode(self, value: Literal["one_hot", "dummy"] | None) -> None:
        self._encode = value
        self.pipeline = self.get_pipeline()

    def get_pipeline(self) -> Pipeline | IdentityTransformer:
        """
        Get the scaling pipeline for the numerical predictors.

        Returns
        -------

        pipeline : sklearn.pipeline.Pipeline or IdentityTransformer
            The scaling pipeline.
        """

        pipeline = (
            Pipeline(
                steps=(
                    [
                        (
                            "one_hot_encoder",
                            OneHotEncoder(sparse_output=False).set_output(
                                transform="pandas"
                            ),
                        ),
                        (
                            "nan_one_hot_decoder",
                            NanOneHotDecoder(),
                        ),
                        (
                            (
                                "first_category_dropper",
                                FirstCategoryDropper(),
                            )
                            if self.encode == "dummy"
                            else (
                                "passthrough",
                                IdentityTransformer(),
                            )
                        ),
                    ]
                )
            )
            if self.encode is not None
            else IdentityTransformer()
        )

        return pipeline  # type: ignore

    def get_X_num(self, X: pd.DataFrame):
        """
        Get the numerical predictors of the DataFrame `X`.

        Parameters
        ----------

        X : pd.DataFrame
            The DataFrame from which the numerical predictors are to be returned.

        Returns
        -------

        X_num : pd.DataFrame
            The numerical predictors of the DataFrame `X`.
        """

        cols_X_num = (
            X.select_dtypes(include=np.number)
            .columns.difference(self.cols_mask)  # type: ignore
            .tolist()
        )
        X_num = X[cols_X_num]

        return X_num

    def get_X_cat(self, X: pd.DataFrame):
        """
        Get the categorical predictors of the DataFrame `X`.

        Parameters
        ----------

        X : pd.DataFrame
            The DataFrame from which the categorical predictors are to be returned.

        Returns
        -------

        X_cat : pd.DataFrame
            The categorical predictors of the DataFrame `X`.
        """

        cols_X_cat = (
            X.select_dtypes(include=["object", "category"])
            .columns.difference(self.cols_mask)  # type: ignore
            .tolist()
        )
        X_cat = X[cols_X_cat]

        return X_cat

    def fit(
        self,
        X: pd.DataFrame,
        y=None,
        sample_weight=None,
    ) -> Self:
        """
        Fit the encoder, that is, compute the encoding maps of the categorical
        predictors of `X` so that the encoding transform may be defined and then used by
        the `transform()` method.

        Parameters
        ----------

        X : pd.DataFrame
            The training predictors and masks.

        Returns
        -------

        self : Encoder
            The fitted encoder.
        """

        # Extract categorical predictors
        X_cat = self.get_X_cat(X)

        # Fit the encoder with the training data
        self.pipeline.fit(X_cat)

        # NOTE: attribute `is_fitted_` must be set to `True` to let `sklearn` know that
        # the instance is already fitted.
        self.is_fitted_ = True

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Encode the categorical predictors in `X` and return them combined with the
        numerical predictors and masks.

        Parameters
        ----------

        X : pd.DataFrame
            Predictors and masks.

        Returns
        -------

        X_trans : pd.DataFrame
            `X` with its categorical predictors encoded.
        """

        # Get the categorical predictors
        X_cat = self.get_X_cat(X)
        # Encode the categorical predictors
        X_cat_encoded = self.pipeline.transform(X_cat).reset_index(drop=True)  # type: ignore

        # Get the original numerical predictors and masks
        X_num = self.get_X_num(X).reset_index(drop=True)
        X_mask = X[self.cols_mask].reset_index(drop=True)

        # Combine the data
        X_trans = pd.concat([X_num, X_cat_encoded, X_mask], axis="columns")

        return X_trans

    def set_output(self, transform: Literal["default", "pandas"] | None = None) -> Self:
        """
        Set output format of `transform()` and `fit_transform()` methods.

        Parameters
        ----------

        transform : {"default", "pandas", None}, default None
            Output format:
                - `"default"` or `"pandas"` - for pandas DataFrame
                - `None` - keep current output format.

        Returns
        -------

        self : Encoder
            The instance itself.

        """

        match transform:
            case "default" | "pandas":
                self.output = "pandas"
            # case "polars":
            #     self.output = "polars"

            case None:
                pass

        return self


class LassoNumPredictorSelector(BaseEstimator, TransformerMixin):
    """
    A numerical predictor selection model based on Lasso regression. With such model,
    solely the input numerical predictors associated with coefficients of the fitted
    Lasso regression model having absolute values larger than `1e-5` are selected.
    The categorical predictor and mask columns are  then appended to the selected
    numerical predictors.

    Note that this may be used to select the best scale-encoded predictors in a
    DataFrame `X`. Scaled-encoded predictors are the predictors resulting from scaling
    and encoding of the numerical and categorical predictors, becoming numerical.
    Non-encoded categorical predictors are, therefore, not considered by the Lasso
    regression model.

    Attributes
    ----------

    lasso_sel : bool, default=True
        Whether to consider Lasso-selection. If `False` no transform is applied.

    lasso_alpha : float, default=1.0
        The regularization strength of the Lasso regression model. This corresponds
        to the multiplying constant of the weight vector L1-norm (sum of the
        absolute values of the components) in the Lasso regression objective
        function. The larger the value, the stronger the regularization.

    lasso : sklearn.linear_model.Lasso
        The Lasso regression model.
    """

    def __init__(
        self,
        lasso_sel: bool = True,
        lasso_alpha: float = 1.0,
        cols_mask: list | np.ndarray | None = None,
    ) -> None:
        super().__init__()

        self._lasso_sel = lasso_sel
        self._lasso_alpha = lasso_alpha
        self._cols_mask = cols_mask if cols_mask is not None else []

        # Get the Lasso-selection pipeline
        self.pipeline = self.get_pipeline()

    @property
    def lasso_sel(self) -> bool:
        return self._lasso_sel

    @property
    def lasso_alpha(self) -> float:
        return self._lasso_alpha

    @property
    def cols_mask(self) -> list | np.ndarray:
        return self._cols_mask

    @lasso_sel.setter
    def lasso_sel(self, value: bool) -> None:
        self._lasso_sel = value
        self.pipeline = self.get_pipeline()

    @lasso_alpha.setter
    def lasso_alpha(self, value: float) -> None:
        self._lasso_alpha = value
        if self.lasso_sel is True:
            self.pipeline.set_params(estimator__alpha=self.lasso_alpha)

    @cols_mask.setter
    def cols_mask(self, value: list | np.ndarray | None) -> None:
        self._cols_mask = value if value is not None else []

    def get_pipeline(self) -> SelectFromModel | IdentityTransformer:
        """
        Get the Lasso-selection pipeline.

        Returns
        -------

        pipeline : SelectFromModel or IdentityTransformer
            The Lasso-selection pipeline.
        """

        pipeline = (
            # NOTE: https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html
            SelectFromModel(estimator=Lasso(alpha=self.lasso_alpha), threshold=0)
            if self.lasso_sel is True
            else IdentityTransformer()
        )

        return pipeline  # type: ignore

    def get_X_num(self, X: pd.DataFrame):
        """
        Get the numerical predictors of the DataFrame `X`.

        Parameters
        ----------

        X : pd.DataFrame
            The DataFrame from which the numerical predictors are to be returned.

        Returns
        -------

        X_num : pd.DataFrame
            The numerical predictors of the DataFrame `X`.
        """

        cols_X_num = (
            X.select_dtypes(include=np.number)
            .columns.difference(self.cols_mask)  # type: ignore
            .tolist()
        )
        X_num = X[cols_X_num]

        return X_num

    def get_X_cat(self, X: pd.DataFrame):
        """
        Get the categorical predictors of the DataFrame `X`.

        Parameters
        ----------

        X : pd.DataFrame
            The DataFrame from which the categorical predictors are to be returned.

        Returns
        -------

        X_cat : pd.DataFrame
            The categorical predictors of the DataFrame `X`.
        """

        cols_X_cat = (
            X.select_dtypes(include=["object", "category"])
            .columns.difference(self.cols_mask)  # type: ignore
            .tolist()
        )
        X_cat = X[cols_X_cat]

        return X_cat

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        sample_weight: np.ndarray | pd.Series | None = None,
    ) -> Self:
        """
        Fit the Lasso-based feature selector model.

        Parameters
        ----------

        X : pd.DataFrame
            The training predictors and masks.
        y : np.ndarray or pd.Series
            The training targets.
        sample_weight : np.ndarray or pd.Series or None, default=None
            Weight of each sample.

        Returns
        -------

        self : LassoNumPredictorSelector
            The transformer with the fitted Lasso-based feature selector model.
        """

        # Reset the indexes of X. The analogous follows for y and sample_weight if they
        # are Series. This is required, since indexes of X, y and sample_weight should
        # match when combining them into a single DataFrame afterwards.
        X = X.reset_index(drop=True)
        if isinstance(y, pd.Series):
            y = y.reset_index(drop=True)
        if isinstance(sample_weight, pd.Series):
            sample_weight = sample_weight.reset_index(drop=True)

        # Convert y and sample_weight to Series if they are arrays
        if isinstance(y, np.ndarray):
            y = pd.Series(y)
        if isinstance(sample_weight, np.ndarray):
            sample_weight = pd.Series(sample_weight)

        # Combine predictors, target, masks and sample_weights together (so that all
        # records with nan in any variable may be removed later before training)
        # WARNING: it is herein assumed that no predictor or mask variable in `X` is
        # named "y".
        data = X
        data["y"] = y
        if sample_weight is not None:
            data["sample_weight"] = sample_weight

        # Remove all records with nan in the data (therefore, simultaneously applying
        # the masks if there are any)
        data = data.dropna()

        # Extract masked numerical predictors, target and sample weights
        X_num_masked = self.get_X_num(
            data.drop(
                columns=["y"] + (["sample_weight"] if sample_weight is not None else [])
            )
        )
        y_masked = data["y"]
        sample_weight_masked = (
            data["sample_weight"] if sample_weight is not None else None
        )

        # Fit the selector with the masked training data
        self.pipeline.fit(X_num_masked, y_masked, sample_weight=sample_weight_masked)

        # NOTE: attribute `is_fitted_` must be set to `True` to let `sklearn` know that
        # the instance is already fitted.
        self.is_fitted_ = True

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Select best numerical predictors according to the coefficients of the fitted
        Lasso model and return them combined with the categorical predictor and mask
        data.

        Parameters
        ----------

        X : pd.DataFrame
            The DataFrame whose variables are to be selected.

        Returns
        -------

        X_trans : pd.DataFrame
            The DataFrame with the selected variables.
        """

        # Get the numerical predictors
        X_num = self.get_X_num(X)

        # Select best numerical predictors according to Lasso model
        X_num_sel = (
            X_num[self.pipeline.get_feature_names_out()]  # type: ignore
            if self.lasso_sel is True
            else X_num
        ).reset_index(drop=True)

        # Get original categorical predictor and mask variables
        X_cat = self.get_X_cat(X).reset_index(drop=True)
        X_mask = X[self.cols_mask].reset_index(drop=True)

        # Combine the data
        X_trans = pd.concat([X_num_sel, X_cat, X_mask], axis="columns")

        return X_trans

    def set_output(self, transform: Literal["default", "pandas"] | None = None) -> Self:
        """
        Set output format of `transform()` and `fit_transform()` methods.

        Parameters
        ----------

        transform : {"default", "pandas", None}, default None
            Output format:
                - `"default"` or `"pandas"` - for pandas DataFrame
                - `None` - keep current output format.

        Returns
        -------

        self :
            The instance itself.

        """

        match transform:
            case "default" | "pandas":
                self.output = "pandas"
            # case "polars":
            #     self.output = "polars"

            case None:
                pass

        return self


class VariableSelector(BaseEstimator, TransformerMixin):
    """
    A transformer that selects a subset of columns from the input DataFrame.

    Attributes
    ----------

    cols : list of str
        The columns to select.
    output : {"pandas"}, default="pandas"
        Output format of `transform()` and `fit_transform()` methods.
    """

    def __init__(self, cols):
        super().__init__()
        self.cols = cols
        self.output = "pandas"

    def fit(self, X: pd.DataFrame, y=None, sample_weight=None) -> Self:
        """
        Check if the specified columns exist in the input DataFrame.

        X : pd.DataFrame
            The DataFrame to check for the presence of the specified columns.
        """

        cols_miss = set(self.cols) - set(X.columns)
        if cols_miss:
            raise ValueError(f"Missing columns: {cols_miss}")

        # NOTE: attribute `is_fitted_` must be set to `True` to let `sklearn` know that
        # the instance is already fitted.
        self.is_fitted_ = True

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Select a subset of columns from the input DataFrame.

        Parameters
        ----------

        X : pd.DataFrame
            The DataFrame to transform.

        Returns
        -------

        X_trans : pd.DataFrame
            The transformed DataFrame.
        """

        X_trans = X[self.cols]
        return X_trans  # type: ignore

    def set_output(self, transform: Literal["default", "pandas"] | None = None) -> Self:
        """
        Set output format of `transform()` and `fit_transform()` methods.

        Parameters
        ----------

        transform : {"default", "pandas", None}, default None
            Output format:
                - `"default"` or `"pandas"` - for pandas DataFrame
                - `None` - keep current output format.

        Returns
        -------

        self : VariableSelector
            The instance itself.

        """

        match transform:
            case "default" | "pandas":
                self.output = "pandas"
            # case "polars":
            #     self.output = "polars"

            case None:
                pass

        return self


class IdentityTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that does not perform any transformation, that is, it returns the
    input as it is.

    Attributes
    ----------

    output : {"pandas"}, default="pandas"
        Output format of `transform()` and `fit_transform()` methods.
    """

    def __init__(
        self,
    ) -> None:

        super().__init__()
        self.output = "pandas"

    def fit(self, X: pd.DataFrame, y=None, sample_weight=None) -> Self:
        # NOTE: attribute `is_fitted_` must be set to `True` to let `sklearn` know that
        # the instance is already fitted.
        self.is_fitted_ = True

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Do not perform any transformation, that is, return the input as it is.

        Parameters
        ----------

        X : pd.DataFrame
            The DataFrame to transform.

        Returns
        -------

        X_trans : pd.DataFrame
            The transformed DataFrame.
        """

        return X

    def set_output(self, transform: Literal["default", "pandas"] | None = None) -> Self:
        """
        Set output format of `transform()` and `fit_transform()` methods.

        Parameters
        ----------

        transform : {"default", "pandas", None}, default None
            Output format:
                - `"default"` or `"pandas"` - for pandas DataFrame
                - `None` - keep current output format.

        Returns
        -------

        self : IdentityTransformer
            The instance itself.

        """

        match transform:
            case "default" | "pandas":
                self.output = "pandas"
            # case "polars":
            #     self.output = "polars"

            case None:
                pass

        return self


class ColumnPrefixDropper(BaseEstimator, TransformerMixin):
    """
    A transformer for dropping prefixes ending with double underscore (`"__"`) from the
    names of the columns of a given DataFrame.

    Attributes
    ----------

    output : {"pandas"}, default="pandas"
        Output format of `transform()` and `fit_transform()` methods.
    """

    def __init__(
        self,
    ) -> None:

        super().__init__()
        self.output = "pandas"

    def fit(self, X: pd.DataFrame, y=None, sample_weight=None) -> Self:
        # NOTE: attribute `is_fitted_` must be set to `True` to let `sklearn` know that
        # the instance is already fitted.
        self.is_fitted_ = True

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Drop prefixes ending with double underscore (`"__"`) from the names of the
        columns of a given DataFrame.

        Parameters
        ----------

        X : pd.DataFrame
            The DataFrame to transform.

        Returns
        -------

        X_trans : pd.DataFrame
            The transformed DataFrame.
        """

        X.columns = [col.split("__")[-1] for col in X.columns]

        return X

    def set_output(self, transform: Literal["default", "pandas"] | None = None) -> Self:
        """
        Set output format of `transform()` and `fit_transform()` methods.

        Parameters
        ----------

        transform : {"default", "pandas", None}, default None
            Output format:
                - `"default"` or `"pandas"` - for pandas DataFrame
                - `None` - keep current output format.

        Returns
        -------

        self : ColumnPrefixDropper
            The instance itself.

        """

        match transform:
            case "default" | "pandas":
                self.output = "pandas"
            # case "polars":
            #     self.output = "polars"

            case None:
                pass

        return self


class ColumnDropper(BaseEstimator, TransformerMixin):
    """
    A transformer for dropping specified columns from a given DataFrame. Such columns
    are dropped based on their prefixes with exceptions based on their suffixes.

    Attributes
    ----------

    cols_drop_prefixes : list[str]
        The prefixes of the columns to be dropped except if their suffixes are in
        `cols_exception_suffixes`. If the respective initialization parameter was not
        issued, it is set to an empty list, meaning that no column is dropped.

    cols_except_suffixes : list[str]
        The suffixes of the columns that should not be dropped even if they match the
        drop prefixes. If the respective initialization parameter was not issued, it is
        set to an empty list, meaning that no exception is considered in the dropping.

    output : {"pandas"}, default="pandas"
        Output format of `transform()` and `fit_transform()` methods.
    """

    def __init__(
        self,
        cols_drop_prefixes: list | None = None,
        cols_except_suffixes: list | None = None,
    ) -> None:

        super().__init__()
        self.cols_drop_prefixes = (
            cols_drop_prefixes if cols_drop_prefixes is not None else []
        )
        self.cols_except_suffixes = (
            cols_except_suffixes if cols_except_suffixes is not None else []
        )
        self.output = "pandas"

    def fit(self, X: pd.DataFrame, y=None, sample_weight=None) -> Self:

        # NOTE: attribute `is_fitted_` must be set to `True` to let `sklearn` know that
        # the instance is already fitted.
        self.is_fitted_ = True

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Drop columns with prefixes in `cols_drop_prefixes` from `X`,
        except if their suffixes are in `cols_except_suffixes`.

        Parameters
        ----------

        X : pd.DataFrame
            The DataFrame to transform.

        Returns
        -------

        X_trans : pd.DataFrame
            The transformed DataFrame.
        """

        X = X.drop(
            columns=[
                col
                for col in X.columns
                if any(col.startswith(prefix) for prefix in self.cols_drop_prefixes)
                and not any(
                    col.endswith(suffix) for suffix in self.cols_except_suffixes
                )
            ],
            errors="ignore",
        )

        # if self.output == "polars":
        #     X = pl.from_pandas(X)  # type: ignore

        return X

    def set_output(self, transform: Literal["default", "pandas"] | None = None) -> Self:
        """
        Set output format of `transform()` and `fit_transform()` methods.

        Parameters
        ----------

        transform : {"default", "pandas", None}, default None
            Output format:
                - `"default"` or `"pandas"` - for pandas DataFrame
                - `None` - keep current output format.

        Returns
        -------

        self : ColumnDropper
            The instance itself.

        """

        match transform:
            case "default" | "pandas":
                self.output = "pandas"
            # case "polars":
            #     self.output = "polars"

            case None:
                pass

        return self


class ColumnReorderer(BaseEstimator, TransformerMixin):
    """
    A transformer for reordering and relocating specified columns of a given DataFrame.

    Attributes
    ----------

    cols_ref : list
        The (wanted) order of the column names.

    relocate : {"left", "right"}, default="right"
        How to relocate the reordered columns: if "left" the the columns are relocated
        to the left of the DataFrame, and if 'right' to the right.

    output : {"pandas"}, default="pandas"
        Output format of `transform()` and `fit_transform()` methods.
    """

    def __init__(
        self,
        cols_ref: list,
        relocate: str = "right",
    ) -> None:

        super().__init__()
        self.cols_ref = cols_ref
        self.output = "pandas"

        # Ensure that the parameter "relocate" is either "left" or "right"
        if relocate not in ["left", "right"]:
            raise ValueError("relocate needs to be either 'left' or 'right'.")
        else:
            self.relocate = relocate

    def fit(self, X: pd.DataFrame, y=None, sample_weight=None) -> Self:

        # NOTE: attribute `is_fitted_` must be set to `True` to let `sklearn` know that
        # the instance is already fitted.
        self.is_fitted_ = True

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Reorder `X`'s columns of names `cols_ref` according to the order in that same
        list. Relocate the reordered columns to the left or right of the DataFrame as
        determined by `relocate`.

        Parameters
        ----------

        X : pd.DataFrame
            The DataFrame to transform.

        Returns
        -------

        X_trans : pd.DataFrame
            The transformed DataFrame.
        """

        non_cols_ref = [col for col in X.columns if col not in self.cols_ref]

        X = (
            X[self.cols_ref + non_cols_ref]
            if self.relocate == "left"
            else X[non_cols_ref + self.cols_ref]
        )

        # if self.output == "polars":
        #     X = pl.from_pandas(X)  # type: ignore

        return X

    def set_output(self, transform: Literal["default", "pandas"] | None = None) -> Self:
        """
        Set output format of `transform()` and `fit_transform()` methods.

        Parameters
        ----------

        transform : {"default", "pandas", None}, default None
            Output format:
                - `"default"` or `"pandas"` - for pandas DataFrame
                - `None` - keep current output format.

        Returns
        -------

        self : ColumnReorderer
            The instance itself.

        """

        match transform:
            case "default" | "pandas":
                self.output = "pandas"
            # case "polars":
            #     self.output = "polars"

            case None:
                pass

        return self


class NanOneHotDecoder(BaseEstimator, TransformerMixin):
    """
    A transformer for decoding one-hot encoded nan values in a DataFrame, that is,
    transforming them into nan entries and dropping the redundant one-hot encoding
    columns associated with the nan category.

    Note that the input of the transformer must be a pandas DataFrame. This mean that if
    one uses sklearn's OneHotEncoder to encode the data before employing the decoder in
    the pipeline, the parameter `sparse_output` of this encoder must to be set to
    `False`, and the method `set_output(transform='pandas')` must be applied.

    Furthermore, the decoder expects that the names of the input columns correspond to
    the unencoded ones suffixed with `"_CAT"` where CAT is the name of the found
    category. For instance, the name of the column for category of value `np.nan` found
    in an unencoded column of name `"a"` would correspond to `"a_nan"`. This is the
    behaviour taken by sklearn's OneHotEncoder, which by default, regards `np.nan` as a
    possible category.

    Attributes
    ----------

    output : {"pandas"}, default="pandas"
        Output format of `transform()` and `fit_transform()` methods.
    """

    def __init__(
        self,
    ) -> None:

        super().__init__()
        self.output = "pandas"

    def fit(self, X: pd.DataFrame, y=None, sample_weight=None) -> Self:

        # NOTE: attribute `is_fitted_` must be set to `True` to let `sklearn` know that
        # the instance is already fitted.
        self.is_fitted_ = True

        return self  # No fitting necessary

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform one-hot encoded nan values in the given DataFrame into nan entries and
        drop the redundant one-hot encoding columns associated with the nan category.

        Parameters
        ----------

        X : pd.DataFrame
            The DataFrame to transform.

        Returns
        -------

        X_trans : pd.DataFrame
            The transformed DataFrame.
        """
        # Get names of all one-hot encoding columns associated with the category nan
        cols_nan = [col for col in X.columns if col.endswith("_nan")]

        for col_nan in cols_nan:
            # Get the columns associated with same old (unencoded) column as the current
            # nan category one
            cols_common = [
                col for col in X.columns if col.startswith(col_nan.removesuffix("_nan"))
            ]
            # Convert the one-hot encodings of the nans associated with the current old
            # (unencoded) column to nan entries
            X.loc[X[col_nan] == 1, cols_common] = np.nan

        # Drop the redundant one-hot encoding columns associated with the category nan
        X = X.drop(columns=cols_nan)

        # if self.output == "polars":
        #     X = pl.from_pandas(X)  # type: ignore

        return X

    def set_output(self, transform: Literal["default", "pandas"] | None = None) -> Self:
        """
        Set output format of `transform()` and `fit_transform()` methods.

        Parameters
        ----------

        transform : {"default", "pandas", None}, default None
            Output format:
                - `"default"` or `"pandas"` - for pandas DataFrame
                - `None` - keep current output format.

        Returns
        -------

        self : NanOneHotDecoder
            The instance itself.

        """

        match transform:
            case "default" | "pandas":
                self.output = "pandas"
            # case "polars":
            #     self.output = "polars"

            case None:
                pass

        return self


class FirstCategoryDropper(BaseEstimator, TransformerMixin):
    """
    A transformer for dropping the first component of each one-hot encoding in a
    DataFrame, that is, dropping the respective columns.

    WARNING:
    Note that the input of the transformer must be a pandas DataFrame. This mean that if
    one uses sklearn's OneHotEncoder to encode the data before employing the dropper in
    the pipeline, the parameter `sparse_output` of this encoder must to be set to
    `False`, and the method `set_output(transform='pandas')` must be applied.

    Furthermore, the dropper expects that the names of the input columns correspond to
    the unencoded ones suffixed with `"_CAT"` where `CAT` is the name of the found
    category. For instance, the name of the column for category of value `"b"` found in
    an unencoded column of name `"a"` would correspond to `"a_b"`. This is the behaviour
    taken by sklearn's OneHotEncoder. The dropper further assumes that the category
    names do not have any underscore ("_") in it.

    Attributes
    ----------

    output : {"pandas"}, default="pandas"
        Output format of `transform()` and `fit_transform()` methods.
    """

    def __init__(
        self,
    ) -> None:

        super().__init__()
        self.output = "pandas"

    def fit(self, X: pd.DataFrame, y=None, sample_weight=None) -> Self:
        """
        Get name of the first occurring column associated with each one-hot encoding in
        the given DataFrame.

        X : pd.DataFrame
            The DataFrame from which the name of the first occurring column of each
            one-hot encoding is to be extracted.
        """
        # Get names of all categorical variables
        self.vars_ = list({col.rpartition("_")[0] for col in X.columns})

        # Get name of first occurring column associated with each variable
        # NOTE: this is safe since the order of the one-hot encoding columns associated
        # with a same categorical variable as set by `OneHotEncoder` is always the same.
        self.cols_first_ = [
            next(col for col in X.columns if col.startswith(var)) for var in self.vars_
        ]

        # NOTE: attribute `is_fitted_` must be set to `True` to let `sklearn` know that
        # the instance is already fitted.
        self.is_fitted_ = True

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Drop first component of each one-hot encoding in the given DataFrame.

        Parameters
        ----------

        X : pd.DataFrame
            The DataFrame to transform.

        Returns
        -------

        X_trans : pd.DataFrame
            The transformed DataFrame.
        """

        X = X.drop(columns=self.cols_first_)

        # if self.output == "polars":
        #     X = pl.from_pandas(X)  # type: ignore

        return X

    def set_output(self, transform: Literal["default", "pandas"] | None = None) -> Self:
        """
        Set output format of `transform()` and `fit_transform()` methods.

        Parameters
        ----------

        transform : {"default", "pandas", None}, default None
            Output format:
                - `"default"` or `"pandas"` - for pandas DataFrame
                - `None` - keep current output format.

        Returns
        -------

        self : FirstCategoryDropper
            The instance itself.

        """

        match transform:
            case "default" | "pandas":
                self.output = "pandas"
            # case "polars":
            #     self.output = "polars"

            case None:
                pass

        return self


class DownscalerPreprocessor(BaseEstimator, TransformerMixin):
    """
    A transformer for pre-processing the inputs of a `DownscalerRegressor`, which:
        - selects the predictors that the user wants to employ in the downstream model;
        - scales and encodes numeric and categorical predictors, respectively;
        - selects the best scaled-encoded predictors according to a Lasso regression
        model (while non-encoded categorical predictors pass through).

    This transformer encompasses the transformations through an sklearn `Pipeline`.

    Attributes
    ----------

    cols_X : list or np.ndarray,
        The column names of the predictors to regard.

    cols_mask : list or np.ndarray, optional
        The column names of the masks to regard.

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

    pipeline : sklearn.pipeline.Pipeline
        The actual pre-processing pipeline (defined internally).

    N_cols_X_trans : int
        Number of columns of the transformed predictors. Note that this parameter is
        only defined after fitting.

    """

    def __init__(
        self,
        cols_X: list | np.ndarray,
        cols_mask: list | np.ndarray | None = None,
        scale: Literal["standardize", "min_max_normalize"] | None = "standardize",
        encode: Literal["one_hot", "dummy"] | None = "dummy",
        lasso_sel: bool = False,
        lasso_alpha: float = 1.0,
    ) -> None:
        super().__init__()

        self._cols_X = cols_X
        self._cols_mask = cols_mask if cols_mask is not None else []
        self._scale = scale
        self._encode = encode
        self._lasso_sel = lasso_sel
        self._lasso_alpha = lasso_alpha

        # The pre-processing pipeline
        self.pipeline = self.get_pipeline()

        self.N_cols_X_trans = None

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

    @cols_X.setter
    def cols_X(self, value: list | np.ndarray) -> None:
        self._cols_X = value
        self.pipeline = self.get_pipeline()

    @cols_mask.setter
    def cols_mask(self, value: list | np.ndarray | None) -> None:
        self._cols_mask = value if value is not None else []
        self.pipeline.set_params(
            variable_selector__cols=self.cols_X + self._cols_mask,
            scaler__cols_mask=self._cols_mask,
            encoder__cols_mask=self._cols_mask,
            predictor_selector__cols_mask=self._cols_mask,
            cols_mask_reorderer__cols_ref=self._cols_mask,
        )

    @scale.setter
    def scale(self, value: Literal["standardize", "min_max_normalize"] | None) -> None:
        self._scale = value
        self.pipeline = self.get_pipeline()

    @encode.setter
    def encode(self, value: Literal["one_hot", "dummy"] | None) -> None:
        self._encode = value
        self.pipeline = self.get_pipeline()

    @lasso_sel.setter
    def lasso_sel(self, value: bool) -> None:
        self._lasso_sel = value
        self.pipeline = self.get_pipeline()

    @lasso_alpha.setter
    def lasso_alpha(self, value: float) -> None:
        self._lasso_alpha = value
        if self._lasso_sel is True:
            self.pipeline.set_params(predictor_selector__lasso_alpha=self._lasso_alpha)

    def get_cols_X_num(self, X: pd.DataFrame):
        """
        Get aliases of the numerical predictors in the input DataFrame `X`.

        Parameters
        ----------

        X : pd.DataFrame
            The DataFrame from which the aliases of numerical predictors are to be
            returned.

        Returns
        -------

        cols_X_num : list
            The aliases (column names) of the numerical predictors.
        """

        cols_X_num = X[self.cols_X].select_dtypes(include=np.number).columns.tolist()

        return cols_X_num

    def get_cols_X_cat(self, X: pd.DataFrame):
        """
        Get aliases of the categorical predictors in the input DataFrame `X`.

        Parameters
        ----------

        X : pd.DataFrame
            The DataFrame from which the aliases of categorical predictors are to be
            returned.

        Returns
        -------

        cols_X_cat : list
            The aliases (column names) of the categorical predictors.
        """

        cols_X_cat = (
            X[self.cols_X]
            .select_dtypes(include=["object", "category"])
            .columns.tolist()
        )

        return cols_X_cat

    def get_pipeline(self) -> Pipeline:
        """
        Get a pre-processing pipeline for the `DownscalerRegressor`.

        This pipeline is such that it:
        - selects the predictors that the user wants to employ in the downstream model;
        - scales and encodes numeric and categorical predictors, respectively;
        - selects the best scaled-encoded predictors according to a Lasso regression
        model (while non-encoded categorical predictors pass through).

        Returns
        -------

        pipeline : sklearn.pipeline.Pipeline
            The pre-processing pipeline.
        """

        preprocessor = Pipeline(
            steps=[
                # Select the predictors and masks that the user wants to employ in the
                # model.
                (
                    "variable_selector",
                    VariableSelector(cols=self.cols_X + self.cols_mask),
                ),
                # Scale the numerical predictors
                (
                    "scaler",
                    Scaler(
                        scale=self.scale,
                        cols_mask=self.cols_mask,
                    ),
                ),
                # Encode the categorical predictors
                (
                    "encoder",
                    Encoder(
                        encode=self.encode,
                        cols_mask=self.cols_mask,
                    ),
                ),
                # Use Lasso predictor selector model for selecting the best
                # scale-encoded predictors
                # NOTE: non-encoded categorical predictors and masks always pass
                # through.
                (
                    "predictor_selector",
                    LassoNumPredictorSelector(
                        lasso_sel=self.lasso_sel,
                        lasso_alpha=self.lasso_alpha,
                        cols_mask=self.cols_mask,
                    ),
                ),
                # Reorder the mask columns and place them at the right
                (
                    "cols_mask_reorderer",
                    ColumnReorderer(
                        cols_ref=self.cols_mask,  # type: ignore
                        relocate="right",  # type: ignore
                    ),
                ),
            ]
        ).set_output(
            transform="pandas"  # Get output as pandas DataFrame instead of np.ndarray
        )

        return preprocessor  # type: ignore

    def fit(
        self,
        X: pd.DataFrame,
        y=np.ndarray | pd.Series,
        sample_weight: np.ndarray | pd.Series | None = None,
    ) -> Self:
        """
        Fits the pre-processing pipeline. It computes statistics and encoding maps of
        the numerical and categorical predictors, respectively, in the training data and
        fits the Lasso-based feature selector model. The trained pipeline would then be
        used by the `transform()` method to perform the pre-processing of the
        predictors.

        Parameters
        ----------

        X : pd.DataFrame
            The training predictors.
        y : np.ndarray or pd.Series
            The training targets.
        sample_weight : np.ndarray or pd.Series or None, default=None
            Weight of each sample.

        Returns
        -------

        self : DownscalerPreprocessor
            The transformer with the fitted pre-processing pipeline.
        """

        X_trans = self.pipeline.fit_transform(
            X,
            y,  # type: ignore
            scaler__sample_weight=sample_weight,
            predictor_selector__sample_weight=sample_weight,
        )  # type: ignore

        # Get number of columns of the transformed predictors
        self.N_cols_X_trans = len(
            X_trans.drop(columns=self.cols_mask).columns  # type: ignore
        )

        # NOTE: attribute `is_fitted_` must be set to `True` to let `sklearn` know that
        # the instance is already fitted.
        self.is_fitted_ = True

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the pre-processing transformations on the given predictors.

        Parameters
        ----------

        X : pd.DataFrame
            The predictors.

        Returns
        -------

        X_trans : pd.DataFrame
            The transformed predictors.
        """

        X_trans = self.pipeline.transform(X)

        return X_trans  # type: ignore
