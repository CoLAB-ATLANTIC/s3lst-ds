from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Literal, Self

import joblib
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.base import BaseEstimator, RegressorMixin

from s3lst_ds.downscaling.estimation import DownscalerEstimator
from s3lst_ds.downscaling.regression import Regressor
from s3lst_ds.utilities.jobs_utils import parse_n_jobs
from s3lst_ds.utilities.logging_utils import RichLogger
from s3lst_ds.utilities.metrics import (
    mae,
    mae_delta,
    mbe,
    r2,
    r2_oos,
    rmse,
    rmse_delta,
)
from s3lst_ds.utilities.tqdm_utils import tqdm
from s3lst_ds.utilities.xr_utils import selective_reproject_match


class Downscaler(BaseEstimator, RegressorMixin):
    """
    A downscaling model that employs the a scale-invariance-based approach with residual
    correction: the fine target is estimated by a `DownscalerEstimator` from fine
    predictors and masks and corrected with the finely-resampled residual associated
    with the prediction of coarse target from coarse predictors and masks.

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

    transform : {None, "center", "standardize"}, default=None,
        The transform operation of the transformed target that is estimated by the
        `estimator`:
            - `None`: if the estimator estimates the target itself (without any
            transformation);
            - `"center"`: if the estimator estimates the centered target (that is, with
            the image-specific mean subtracted from it);
            - `"standardize"`: if the estimator estimates standardized target (that is,
            with the centered target further divided by the image-specific standard
            deviation).
        Note that this has no impact when training the `estimator`, but when inferring
        with the downscaler. To make the `estimator` estimate a transformed target, one
        must issue a transformed target in training. When inferring with the downscaler,
        the output of the estimator is transformed to its "raw" state using statistics
        of the issued coarse true raw target (if and only if `transform` is set to
        estimator's transform).

    max_workers : int, default=1
        Number of simultaneous multiple processes to consider in prediction and scoring
        with the special cases:
            - `1` or `None`: no multiprocessing is considered;
            - `-1`: all processors are used;
            - `-k`: all processors except k-1 are used.

    estimator : estimation.DownscalerEstimator,
        The preprocessing and regression pipeline.

    logger : RichLogger or None
        A rich logger for showing progress of the prediction/scoring.

    show_progress : bool, default=True
        `True` to display the downscaling progress.

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
        transform: Literal["center", "standardize"] | None = None,
        max_workers: int = 1,
        logger: RichLogger | None = None,
        show_progress: bool = True,
    ) -> None:
        """
        Initialize the downscaling model.

        Parameters
        ----------
        base_model : Regressor
            The general (i.e. non-pixel-wise) base model to be fitted with coarse data.

        cols_X : list or np.ndarray
            The names of the predictor columns to regard.

        cols_mask : list or np.ndarray, optional
            The names of the mask columns to regard.

        scale : {"standardize", "min_max_normalize", None}, default="standardize"
            The scaling method to apply to numerical predictors:
                - `"standardize"`: to standardize the numerical predictors (zero mean
                and unit variance);
                - `"min_max_normalize"`: to min-max normalize the numerical predictors
                (to the range `[0, 1]`);
                - `None`: to regard the numerical predictors raw (no scaling).

        encode : {"one_hot", "dummy", None}, default="dummy"
            The encoding method to apply to the categorical predictors:
                - `"one_hot"`: to one-hot encode the categorical predictors;
                - `"dummy"`: to dummy encode the categorical predictors (one-hot
                encoding with the first component dropped);
                - `None`: to regard the categorical predictors raw (no encoding).

            Note that dummy encoding is usually considered in place of one-hot to avoid
            multicollinearity problems (one may show that a component of a one-hot
            encoding vector is fully determined by all the other components making it
            redundant).

        lasso_sel : bool, default=False
            Whether to use a Lasso regression for selecting the scaled-encoded `cols_X`
            predictors downstream of the preprocessor. Lasso selection is such that
            solely the input predictors associated with coefficients of the fitted Lasso
            regression model having absolute values larger than `1e-5` are selected.
            Note that the non-encoded `cols_X` predictors are regardlessly considered
            downstream of the preprocessor.

        lasso_alpha : float, default=1.0
            The regularization strength of the Lasso regression model used for selecting
            the scaled-encoded `cols_X` predictors downstream of the preprocessor. Such
            regularization strength is the multiplying constant of the weight vector
            L1-norm (sum of the absolute values of the components) in the Lasso
            regression objective function. The larger the value, the stronger the
            regularization. Note that this parameter only takes effect if `lasso_sel` is
            `True`.

        transform : {None, "center", "standardize"}, default=None,
            The transform operation of the transformed target that is estimated by the
            `estimator`:
                - `None`: if the estimator estimates the target itself (without any
                transformation);
                - `"center"`: if the estimator estimates the centered target (that is,
                with the image-specific mean subtracted from it);
                - `"standardize"`: if the estimator estimates standardized target (that
                is, with the centered target further divided by the image-specific
                standard deviation).
            Note that this has no impact when training the `estimator`, but when
            inferring with the downscaler. To make the `estimator` estimate a
            transformed target, one must issue a transformed target in training. When
            inferring with the downscaler, the output of the estimator is transformed to
            its "raw" state using statistics of the issued coarse true raw target (if
            and only if `transform` is set to estimator's transform).

        max_workers : int, default=1
            Number of simultaneous multiple processes to consider in prediction and
            scoring with the special cases:
                - `1` or `None`: no multiprocessing is considered;
                - `-1`: all processors are used;
                - `-k`: all processors except k-1 are used.

        logger : RichLogger or None
            A rich logger for showing progress of the prediction/scoring.

        show_progress : bool, default=True
            `True` to display the downscaling progress.
        """
        super().__init__()
        # NOTE: attribute `is_fitted_` is set to `True` after fitting to let `sklearn`
        # know that the instance is already fitted.
        self.is_fitted_ = False
        # NOTE: base_model is a Regressor class instance and it is an attribute of the
        # an attribute of the DownscalerEstimator class instance (estimator). The latter
        # would be updated with the changes that are done on base_model even if outside
        # of the latter.
        self._base_model = base_model
        self._cols_X = cols_X
        self._cols_mask = cols_mask if cols_mask is not None else []
        self._scale = scale
        self._encode = encode
        self._lasso_sel = lasso_sel
        self._lasso_alpha = lasso_alpha
        self.transform = transform
        self.max_workers = max_workers
        self.estimator = self.get_estimator()
        self.logger = logger
        self.show_progress = show_progress

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
        """
        Get the scaling method to apply to numerical predictors:
            - `"standardize"`: to standardize the numerical predictors (zero mean and
            unit variance);
            - `"min_max_normalize"`: to min-max normalize the numerical predictors (to
            the range `[0, 1]`);
            - `None`: to regard the numerical predictors raw (no scaling).

        Returns
        -------
        scale : {"standardize", "min_max_normalize", None}
            The scaling method to apply to numerical predictors.
        """
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

    @property
    def max_workers(self) -> int:
        return self._max_workers

    @base_model.setter
    def base_model(self, value: Regressor) -> None:
        self._base_model = value
        self.estimator.base_model = value

    @cols_X.setter
    def cols_X(self, value: list | np.ndarray) -> None:
        self._cols_X = value
        self.estimator.cols_X = value

    @cols_mask.setter
    def cols_mask(self, value: list | np.ndarray | None) -> None:
        self._cols_mask = value if value is not None else []
        self.estimator.cols_mask = self._cols_mask

    @scale.setter
    def scale(self, value: Literal["standardize", "min_max_normalize"] | None) -> None:
        self._scale = value
        self.estimator.scale = value  # type: ignore

    @encode.setter
    def encode(self, value: Literal["one_hot", "dummy"] | None) -> None:
        self._encode = value
        self.estimator.encode = value

    @lasso_sel.setter
    def lasso_sel(self, value: bool) -> None:
        self._lasso_sel = value
        self.estimator.lasso_sel = value

    @lasso_alpha.setter
    def lasso_alpha(self, value: float) -> None:
        self._lasso_alpha = value
        self.estimator.lasso_alpha = value

    @max_workers.setter
    def max_workers(self, value: int) -> None:
        self._max_workers = parse_n_jobs(value)

    def get_estimator(self) -> DownscalerEstimator:
        estimator = DownscalerEstimator(
            base_model=self.base_model,
            cols_X=self.cols_X,
            cols_mask=self.cols_mask,
            scale=self.scale,
            encode=self.encode,
            lasso_sel=self.lasso_sel,
            lasso_alpha=self.lasso_alpha,
        )
        return estimator

    def fit(
        self,
        X_and_mask_coarse: np.ndarray | pd.DataFrame,
        y_coarse: np.ndarray | pd.Series,
        sample_weight: np.ndarray | pd.Series | None = None,
    ) -> Self:
        """
        Fit the estimator (preprocessing transformers and the general base model) to
        training coarse data. To make the estimator estimate a transformed target such
        as a centered or a standardized one, issue `y_coarse` with transformed true
        target values.

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

        self : Downscaler
            The fitted instance itself.

        """

        self.estimator.fit(X_and_mask_coarse, y_coarse, sample_weight=sample_weight)

        # NOTE: attribute `is_fitted_` must be set to `True` to let `sklearn` know that
        # the instance is already fitted.
        self.is_fitted_ = True

        return self

    def predict_single(
        self,
        timestamp: pd.Timestamp,
        X_and_mask_fine: np.ndarray | pd.DataFrame,
        correct: bool = True,
        X_and_mask_coarse: np.ndarray | pd.DataFrame | None = None,
        y_coarse: np.ndarray | pd.Series | None = None,
        coords_coarse: xr.Coordinates | None = None,
        coords_fine: xr.Coordinates | None = None,
        gridded: bool = True,
        dims: tuple | None = None,
        attrs: dict | None = None,
        path_out: Path | None = None,
    ) -> np.ndarray | xr.DataArray | None:
        """
        Predict fine raw target from fine predictors and masks (`X_and_mask_fine`).
        Additionally, if `correct` is set to `True`, correct prediction with
        finely-resampled residuals associated with the prediction of coarse raw target
        from coarse predictors and masks (`X_and_mask_coarse`). Note that to compute
        such residuals, the "true" coarse raw target (`y_coarse`) and the coarse and
        fine grid coordinates (`coords_coarse` and `coords_fine`) must be also issued.
        An estimator that estimates a centered or standardized target (identifiable
        through attribute `transform`) considers raw target coarse statistics in the
        respective transformations. To re-transform the target back to a "raw" state,
        the statistics of the issued `y_coarse` are herein used.

        Note that this method only predicts for a single image. To predict for multiple
        images, use `predict()`.

        Parameters
        ----------

        timestamp : pd.Timestamp
            Timestamp associated with the data.

        X_and_mask_fine : np.ndarray or pd.DataFrame
            Fine predictors and masks.

        correct : bool, default=True
            Whether to correct the predicted fine raw target (from fine predictors and
            masks, `X_and_mask_fine`) using the finely-resampled residual for the
            prediction of the coarse raw target (from coarse predictors and masks,
            `X_and_mask_coarse`).

        X_and_mask_coarse : np.ndarray or pd.DataFrame or None, default=None
            Coarse predictors and masks. It must be issued if `correct` is `True`.

        y_coarse : np.ndarray or pd.Series or None, default=None
            The "true" coarse raw target. It must be issued if `correct` is `True` or if
            `transform` is not `None`.

        coords_coarse : xarray.core.coordinates.Coordinates or None, default=None
            The coordinates of the coarse mesh. It must be issued if `correct` is
            `True`.

        coords_fine : xarray.core.coordinates.Coordinates or None, default=None
            The coordinates of the fine mesh. It must be issued if `correct` or
            `gridded` are `True`.

        gridded : bool, default=True
            Whether to return the predicted fine raw target in grid form (as an
            `xr.DataArray`) or flattened form (as a `pd.Series`).

        dims : tuple or None, default=None
            Labels for the dimensions of the predicted target if it is returned in grid
            form. If not issued, it is set to ("lat", "lon") by default.

        attrs : dict or None, default=None
            Attributes to set in the predicted target if it is returned in grid form. If
            not issued, it is set as in accordance with the CF conventions
            (https://cf-convention.github.io/Data/cf-conventions/cf-conventions-1.13/cf-conventions.pdf#temperature-units):
                {
                    "standard_name": "land_surface_temperature",
                    "long_name": "Land surface temperature",
                    "units": "K",
                }

        path_out : Path or None, default=None
            The output path of the file for the predicted image. If not issued the
            predicted target is instead returned.

        Returns
        -------

        y_fine_pred : np.ndarray or xr.DataArray or None
            Predicted fine raw target in flattened form (as a `np.ndarray`, if `gridded`
            is `False`) or grid form (as an `xr.DataArray` if `gridded` is `True`). If
            `path_out` is issued, the prediction is written to file and `None` is
            instead returned.

        """
        # Parse dims and attrs parameters
        dims = dims if dims is not None else ("lat", "lon")
        attrs = (
            attrs
            if attrs is not None
            else {
                "standard_name": "land_surface_temperature",
                "long_name": "Land surface temperature",
                "units": "K",
            }
        )

        # Raise error if residual correction is to be performed but required parameters
        # are missing
        if correct is True and any(
            elem is None
            for elem in [X_and_mask_coarse, y_coarse, coords_coarse, coords_fine]
        ):
            raise TypeError(
                "Parameters 'X_and_mask_coarse', 'y_coarse', 'coords_coarse' and"
                " 'coords_fine' must also be issued to perform residual"
                " correction."
            )
        # Raise error if transformation of predicted target into raw state is to be
        # performed but required parameters are missing
        if self.transform is not None and y_coarse is None:
            raise TypeError(
                "Parameter 'y_coarse' must also be issued to transform predicted"
                " target into raw state."
            )

        # Raise error if the predicted target is wanted in grid form (not in ravelled
        # one) but required parameters are missing
        if gridded is True and coords_fine is None:
            raise TypeError(
                "Parameter 'coords_fine' must also be issued to make predicted"
                " target gridded."
            )

        # Convert true coarse raw target to a pandas Series if it is not already and
        # residual correction is considered or the estimator estimates centered or
        # standardized target (such condition would require usage of the true coarse
        # target)
        if not isinstance(y_coarse, pd.Series) and (
            correct is True or self.transform is not None
        ):
            y_coarse = pd.Series(y_coarse)

        # Predict fine target from fine predictors and masks using the the preprocessor
        # and the base model
        y_fine_pred = pd.Series(self.estimator.predict(X_and_mask_fine))

        # Transform predicted fine target to raw state using the true target coarse
        # statistics
        if self.transform == "center":
            y_fine_pred = y_fine_pred + y_coarse.mean()  # type: ignore
        elif self.transform == "standardize":
            y_fine_pred = y_fine_pred * y_coarse.std() + y_coarse.mean()  # type: ignore

        # If gridded prediction or residual correction are wanted, transform the
        # predicted fine raw target into grid form
        # NOTE: residual correction involves reprojection of the coarse residual into
        # the fine grid. The grid of the gridded predicted fine raw target may be used
        # as target of the matching reprojection.
        if gridded is True or correct is True:
            # Get shape of the fine grid
            shape_fine = tuple(reversed(list(coords_fine.sizes.values())))  # type: ignore

            y_fine_pred = xr.DataArray(
                data=y_fine_pred.values.reshape(  # type: ignore
                    shape_fine  # type: ignore
                ),
                coords=coords_fine,
                dims=("y", "x"),
                name="LST",
            )

        # If residual correction is wanted, correct the prediction using finely-resample
        # residuals associated with the prediction of the coarse target
        if correct is True:
            # Predict coarse target from coarse predictors and masks
            y_coarse_pred = pd.Series(self.estimator.predict(X_and_mask_coarse))  # type: ignore

            # Transform predicted coarse target to raw state using the true target
            # coarse statistics
            if self.transform == "center":
                y_coarse_pred = y_coarse_pred + y_coarse.mean()  # type: ignore
            elif self.transform == "standardize":
                y_coarse_pred = (
                    y_coarse_pred * y_coarse.std() + y_coarse.mean()  # type: ignore
                )

            # Compute respective residuals
            res_coarse = y_coarse - y_coarse_pred  # type: ignore

            # Get shape of the coarse grid
            shape_coarse = tuple(reversed(list(coords_coarse.sizes.values())))  # type: ignore

            # Express the residuals in the coarse grid
            res_coarse = xr.DataArray(
                data=res_coarse.values.reshape(shape_coarse),  # type: ignore
                coords=coords_coarse,
                dims=("y", "x"),
                name="LST",
            )

            # Refine the residuals by reprojecting then to the fine grid
            res_coarse_refined = selective_reproject_match(
                data_src=res_coarse,
                data_target=y_fine_pred,  # type: ignore
            )

            # Correct the fine raw target
            y_fine_pred = y_fine_pred + res_coarse_refined

            # If ravelled (flat) predicted fine raw target is wanted, ravel it
            if gridded is False:
                y_fine_pred = y_fine_pred.values.ravel()  # type: ignore

        # In case of gridded prediction, set type, NODATA value, dimension labels and
        # attributes of the data
        if gridded is True:
            # Set data type
            y_fine_pred = y_fine_pred.astype("float32")

            # Set time coordinate
            # WARNING: it is herein assumed that the timestamp is in the UTC timezone.
            y_fine_pred = y_fine_pred.expand_dims(  # type: ignore
                dim={"time": [timestamp.tz_localize("UTC")]}
            )

            # Write NODATA value
            y_fine_pred.rio.write_nodata(  # type: ignore
                input_nodata=-999,
                encoded=True,
                inplace=True,
            )

            # Set dimension labels
            if dims is not None:
                y_fine_pred = y_fine_pred.rename({"y": dims[0], "x": dims[1]})

            # Set attributes
            if attrs is not None:
                y_fine_pred.attrs = attrs  # type: ignore
                y_fine_pred["time"].attrs = {
                    "axis": "T",
                    "standard_name": "time",
                    "long_name": "Start sensing time of the satellite acquisition",
                }

        # If writing to file, write the predicted fine raw target
        if path_out is not None:
            # Create output directory if it does not exist
            path_out.parent.mkdir(  # type: ignore
                parents=True,
                exist_ok=True,
            )
            # Write to file
            if gridded is False:
                path_out = path_out.with_suffix(".csv")
                np.savetxt(fname=path_out, X=y_fine_pred)  # type: ignore

            else:
                # NOTE: rioxarray `to_raster()` cannot handle writing to NetCDF files,
                # but `to_netcdf()` can.
                if path_out.suffix == ".nc":
                    # NetCDF cannot handle pd.Timestamp type. Time will be converted to
                    # seconds since 1972-01-01 00:00:00 UTC, as in accordance with CF
                    # conventions
                    # NOTE: see https://cf-convention.github.io/Data/cf-conventions/cf-conventions-1.13/cf-conventions.pdf#page=42
                    y_fine_pred["time"] = (
                        y_fine_pred["time"] - pd.Timestamp("1972-01-01 00:00:00Z")
                    ).dt.total_seconds()  # type: ignore
                    y_fine_pred["time"].attrs = {  # type: ignore
                        "standard_name": "time",
                        "long_name": "Time",
                        "axis": "T",
                        "units": "seconds since 1972-1-1 00:00:00Z",
                        "calendar": "proleptic_gregorian",
                    }

                    y_fine_pred.to_netcdf(path_out)  # type: ignore
                else:
                    # In the case of no suffix, `to_raster()` considers GeoTIFF.
                    if path_out.suffix in [""]:
                        path_out = path_out.with_suffix(".tif")

                    y_fine_pred.rio.to_raster(path_out)  # type: ignore

            # Set y_fine_pred to None to return None at the end of the function
            y_fine_pred = None

        return y_fine_pred  # type: ignore

    def predict(
        self,
        X_and_mask_fine: dict[pd.Timestamp, np.ndarray | pd.DataFrame],
        correct: bool = True,
        X_and_mask_coarse: dict[pd.Timestamp, np.ndarray | pd.DataFrame] | None = None,
        y_coarse: dict[pd.Timestamp, np.ndarray | pd.Series] | None = None,
        coords_coarse: dict[pd.Timestamp, xr.Coordinates] | None = None,
        coords_fine: dict[pd.Timestamp, xr.Coordinates] | None = None,
        gridded: bool = True,
        dims: tuple | None = None,
        attrs: dict | None = None,
        path_out: dict[pd.Timestamp, Path] | None = None,
        *,
        _log: bool = True,
    ) -> dict[pd.Timestamp, np.ndarray | xr.DataArray] | None:
        """
        Predict fine target for multiple images using `predict_single()` for each one.

        Parameters
        ----------

        X_and_mask_fine : dict[pd.Timestamp, np.ndarray or pd.DataFrame]
            Fine predictors and masks for each image, keyed by timestamp.

        correct : bool, default=True
            Whether to correct the predicted fine raw target for each image (from fine
            predictors and masks, `X_and_mask_fine`) using the finely-resampled residual
            for the prediction of the coarse raw target (from coarse predictors and
            masks, `X_and_mask_coarse`).

        X_and_mask_coarse : dict[pd.Timestamp, np.ndarray or pd.DataFrame] or None, default=None
            Coarse predictors and masks for each image, keyed by timestamp. It must be
            issued if `correct` is `True`.

        y_coarse : dict[pd.Timestamp, np.ndarray or pd.Series] or None, default=None
            The "true" coarse raw target, keyed by timestamp. It must be issued if
            `correct` is `True` or if `transform` is not `None`.

        coords_coarse : dict[pd.Timestamp, xarray.core.coordinates.Coordinates] or None, default=None
            The coordinates of the coarse mesh for each image, keyed by timestamp. It
            must be issued if `correct` is `True`.

        coords_fine : dict[pd.Timestamp, xarray.core.coordinates.Coordinates] or None, default=None
            The coordinates of the fine mesh for each image, keyed by timestamp. It must
            be issued if `correct` or `gridded` are `True`.

        gridded : bool, default=True
            Whether to return the predicted fine raw target for each image in grid form
            (as an `xr.DataArray`) or in flattened form (as a `pd.Series`).

        dims : tuple or None, default=None
            Labels for the dimensions of the predicted target if it is returned in grid
            form. If not issued, it is set to ("y", "x") by default.

        attrs : dict or None, default=None
            Attributes to set in the predicted target if it is returned in grid form. If
            not issued, it is set as in accordance with the CF conventions
            (https://cf-convention.github.io/Data/cf-conventions/cf-conventions-1.13/cf-conventions.pdf#temperature-units):
                {
                    "standard_name": "land_surface_temperature", "long_name": "Land
                    surface temperature", "units": "K",
                }

        path_out : dict[pd.Timestamp, Path] or None, default=None
            The output path of the file for each predicted image, keyed by timestamp. If
            not issued, the predicted target is instead returned.

        _log : bool, default=True
            Whether to log messages into terminal.

        Returns
        -------

        y_fine_pred : dict[pd.Timestamp, np.ndarray or xr.DataArray] or None
            Predicted fine raw target for each image in flattened form (as a
            `np.ndarray`, if `gridded` is `False`) or grid form (as an `xr.DataArray` if
            `gridded` is `True`). If `path_out` is issued, the predictions are written
            to files and `None` is instead returned.
        """

        if self.logger is not None and _log is True:
            self.logger.info("Predicting raw target...")

        # Transform parameters valued as None into dictionaries with None values (one
        # per image)
        X_and_mask_coarse = (
            X_and_mask_coarse
            if X_and_mask_coarse is not None
            else dict.fromkeys(X_and_mask_fine.keys(), None)  # type: ignore
        )
        y_coarse = (
            y_coarse
            if y_coarse is not None
            else dict.fromkeys(X_and_mask_fine.keys(), None)  # type: ignore
        )
        coords_coarse = (
            coords_coarse
            if coords_coarse is not None
            else dict.fromkeys(X_and_mask_fine.keys(), None)  # type: ignore
        )
        coords_fine = (
            coords_fine
            if coords_fine is not None
            else dict.fromkeys(X_and_mask_fine.keys(), None)  # type: ignore
        )
        path_out = (
            path_out
            if path_out is not None
            else dict.fromkeys(X_and_mask_fine.keys(), None)  # type: ignore
        )

        # Define progress bar
        pbar = (
            tqdm(
                # Prefix for the progressbar
                bar_format=f"{'':9}" + "{l_bar}{bar}{r_bar}",
                desc=f"{'':8}",
                total=len(X_and_mask_fine.keys()),  # type: ignore
                unit="timestamp",
                position=0,
                leave=True,  # Keep progress on the screen after completion.
                options={"console": self.logger.console},
            )
            if self.show_progress is True and self.logger is not None
            else None
        )

        # Predict fine raw targets as a dictionary
        y_fine_pred = {}
        if self.max_workers != 1:
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                # List of placeholders for the eventual result of a computation
                futures = {
                    # NOTE: Using executor.submit() can be safely used as key of
                    # dictionary since executor.submit() returns a Future object
                    # (https://docs.python.org/3/library/asyncio-future.html#future-object)
                    # and all of these objects are unique and hashable.
                    executor.submit(
                        self.predict_single,
                        timestamp=timestamp,
                        X_and_mask_fine=X_and_mask_fine[timestamp],
                        correct=correct,
                        X_and_mask_coarse=X_and_mask_coarse[timestamp],  # type: ignore
                        y_coarse=y_coarse[timestamp],  # type: ignore
                        coords_coarse=coords_coarse[timestamp],  # type: ignore
                        coords_fine=coords_fine[timestamp],  # type: ignore
                        dims=dims,
                        attrs=attrs,
                        gridded=gridded,
                        path_out=path_out[timestamp],  # type: ignore
                    ): timestamp
                    for timestamp in X_and_mask_fine  # type: ignore
                }

                for future in as_completed(futures):
                    # Add result to dictionary of results
                    timestamp = futures[future]
                    y_fine_pred[timestamp] = future.result()

                    # Update progress bar with one more count per completed process
                    if pbar is not None:
                        pbar.update()

            # Make dictionary of predictions be ordered as input X_and_mask_fine
            # NOTE: multiprocessing may output results in a different order.
            y_fine_pred = {
                timestamp: y_fine_pred[timestamp] for timestamp in X_and_mask_fine
            }

        else:
            for timestamp in X_and_mask_fine:  # noqa: PLC0206
                y_fine_pred[timestamp] = self.predict_single(
                    timestamp=timestamp,
                    X_and_mask_fine=X_and_mask_fine[timestamp],
                    correct=correct,
                    X_and_mask_coarse=X_and_mask_coarse[timestamp],  # type: ignore
                    y_coarse=y_coarse[timestamp],  # type: ignore
                    coords_coarse=coords_coarse[timestamp],  # type: ignore
                    coords_fine=coords_fine[timestamp],  # type: ignore
                    gridded=gridded,
                    dims=dims,
                    attrs=attrs,
                    path_out=path_out[timestamp],  # type: ignore
                )

                # Update progress bar with one more count per completed process
                if pbar is not None:
                    pbar.update()

        # At the end close progress bar
        if pbar is not None:
            pbar.close()

        # Convert y_fine_pred to None if it any of their values was written to files.
        if None not in path_out.values():  # type: ignore
            y_fine_pred = None

        return y_fine_pred  # type: ignore

    def predict_coarse(
        self,
        X_and_mask_coarse: dict[pd.Timestamp, np.ndarray | pd.DataFrame],
        y_coarse: dict[pd.Timestamp, np.ndarray | pd.Series] | None = None,
        coords_coarse: dict[pd.Timestamp, xr.Coordinates] | None = None,
        gridded: bool = True,
        dims: tuple | None = None,
        attrs: dict | None = None,
        path_out: dict[pd.Timestamp, Path] | None = None,
    ) -> dict[pd.Timestamp, np.ndarray | xr.DataArray] | None:
        """
        Predict coarse target for multiple images.

        Parameters
        ----------

        X_and_mask_coarse : dict[pd.Timestamp, np.ndarray or pd.DataFrame]
            Coarse predictors and masks for each image, keyed by timestamp.

        y_coarse : dict[pd.Timestamp, np.ndarray or pd.Series] or None, default=None
            The "true" coarse raw target, keyed by timestamp. It must be issued if
            `transform` is not `None`.

        coords_coarse : dict[pd.Timestamp, xarray.core.coordinates.Coordinates] or None, default=None
            The coordinates of the coarse mesh for each image, keyed by timestamp. It
            must be issued if `gridded` is `True`.

        gridded : bool, default=True
            Whether to return the predicted coarse raw target for each image in grid
            form (as an `xr.DataArray`) or in flattened form (as a `pd.Series`).

        dims : tuple or None, default=None
            Labels for the dimensions of the predicted target if it is returned in grid
            form. If not issued, it is set to ("y", "x") by default.

        attrs : dict or None, default=None
            Attributes to set in the predicted target if it is returned in grid form. If
            not issued, it is set as in accordance with the CF conventions
            (https://cf-convention.github.io/Data/cf-conventions/cf-conventions-1.13/cf-conventions.pdf#temperature-units):
                {
                    "standard_name": "land_surface_temperature", "long_name": "Land
                    surface temperature", "units": "K",
                }

        path_out : dict[pd.Timestamp, Path] or None, default=None
            The output path of the file for each predicted image, keyed by timestamp. If
            not issued, the predicted target is instead returned.

        Returns
        -------

        y_coarse_pred : dict[pd.Timestamp, np.ndarray or xr.DataArray] or None
            Predicted coarse raw target for each image in flattened form (as a
            `np.ndarray`, if `gridded` is `False`) or grid form (as an `xr.DataArray` if
            `gridded` is `True`). If `path_out` is issued, the predictions are written
            to files and `None` is instead returned.
        """

        y_coarse_pred = self.predict(
            X_and_mask_fine=X_and_mask_coarse,
            correct=False,
            y_coarse=y_coarse,
            coords_fine=coords_coarse,
            gridded=gridded,
            dims=dims,
            attrs=attrs,
            path_out=path_out,
        )

        return y_coarse_pred

    def score_single(
        self,
        timestamp: pd.Timestamp,
        X_and_mask_fine: np.ndarray | pd.DataFrame,
        y_fine: np.ndarray | pd.Series,
        correct: bool = True,
        calibrate: bool = False,
        X_and_mask_coarse: np.ndarray | pd.DataFrame | None = None,
        y_coarse: np.ndarray | pd.Series | None = None,
        coords_coarse: xr.Coordinates | None = None,
        coords_fine: xr.Coordinates | None = None,
        scorers: list[str] | None = None,
        sample_weight: np.ndarray | pd.Series | None = None,
    ) -> dict[str, float]:
        """
        Predict raw fine target and score the prediction.

        Note that this method only predicts and scores for a single image. To predict
        and score for multiple images, use `score()`.

        Parameters
        ----------
        timestamp : pd.Timestamp
            Timestamp associated with the data.

        X_and_mask_fine : np.ndarray or pd.DataFrame
            Fine predictors and masks.

        y_fine : np.ndarray or pd.Series
            The "true" raw fine target.

        correct : bool, default=True
            Whether to correct the predicted fine raw target (from fine predictors and
            masks, `X_and_mask_fine`) using the finely-resampled residual for the
            prediction of the coarse raw target (from coarse predictors and masks,
            `X_and_mask_coarse`).

        calibrate : bool, default=False
            Whether to calibrate the predicted fine target with the coarse validation
            target. This is done by offsetting and scaling the predicted fine target
            with the transform that makes the coarse true target (`y_coarse`) have the
            same mean and standard deviation as the validation coarse one (coarsened
            `y_fine`). Such transformation is an attempt to account for discrepancies
            between source and validation platforms at a common coarse grid from the
            computed scores.

        X_and_mask_coarse : np.ndarray or pd.DataFrame or None, default=None
            Coarse predictors and masks. It must be issued if `correct` is `True`.

        y_coarse : np.ndarray or pd.Series or None, default=None
            The "true" raw coarse target. It must be issued if `correct` or `calibrate`
            are `True` or `transform` is not `None`.

        coords_coarse : xarray.core.coordinates.Coordinates or None, default=None
            The coordinates of the coarse mesh. It must be issued if `correct` or
            `calibrate` are `True`.

        coords_fine : xarray.core.coordinates.Coordinates or None, default=None
            The coordinates of the fine mesh. It must be issued if `correct` or
            `calibrate` are `True`.


        scorers : list[str], default=["r2", "r2_oos", "rmse", "rmse_delta", "mae", "mae_delta", "mbe"]
            Aliases of the scorers to consider.

        sample_weight : np.ndarray or pd.Series or None, default=None
            Weight of each sample in the score.

        Returns
        -------

        score : dict[str, float]
            Prediction scores.
        """

        # Define default value for scorers argument
        if scorers is None:
            scorers = ["r2", "r2_oos", "rmse", "rmse_delta", "mae", "mae_delta", "mbe"]

        # If y_fine is a Series, reset its indexes. The analogous follows for
        # sample_weight. This is required, since indexes of y_fine, y_fine_pred and
        # y_fine_dummy_pred and sample_weight should match when combining them into a
        # single DataFrame afterwards.
        if isinstance(y_fine, pd.Series):
            y_fine = y_fine.reset_index(drop=True)
        if isinstance(sample_weight, pd.Series):
            sample_weight = sample_weight.reset_index(drop=True)

        # Predict raw fine target
        y_fine_pred = pd.Series(
            self.predict_single(
                timestamp=timestamp,
                X_and_mask_fine=X_and_mask_fine,
                correct=correct,
                X_and_mask_coarse=X_and_mask_coarse,
                y_coarse=y_coarse,
                coords_coarse=coords_coarse,
                coords_fine=coords_fine,
                gridded=False,
            )  # type: ignore
        )

        # Predict raw fine target from predictors using the dummy mean model
        # NOTE: this is required for computing out-of-sample coefficient of
        # determination
        y_fine_dummy_pred = self.estimator.pipeline.named_steps[
            "regressor"
        ].dummy_mean_model.predict(X_and_mask_fine) * (
            y_coarse.std() if self.transform == "standardize" else 1  # type: ignore
        ) + (
            y_coarse.mean() if self.transform is not None else 0  # type: ignore
        )

        # Calibrate the fine targets predicted by downscaler and dummy mean model with
        # the transform that would make the coarse true target have the same mean and
        # standard deviation as the coarsened fine validation one.
        if calibrate is True:
            # Express coarse true target in its grid
            shape_coarse = tuple(reversed(list(coords_coarse.sizes.values())))  # type: ignore
            y_coarse_grid = xr.DataArray(
                data=(
                    y_coarse.values if isinstance(y_coarse, pd.Series) else y_coarse
                ).reshape(  # type: ignore
                    shape_coarse  # type: ignore
                ),
                coords=coords_coarse,
                dims=("y", "x"),
                name="LST",
            )

            # Express fine validation target in its grid
            shape_fine = tuple(reversed(list(coords_fine.sizes.values())))  # type: ignore
            y_fine_grid = xr.DataArray(
                data=(
                    y_fine.values if isinstance(y_fine, pd.Series) else y_fine
                ).reshape(  # type: ignore
                    shape_fine  # type: ignore
                ),
                coords=coords_fine,
                dims=("y", "x"),
                name="LST",
            )

            # Reproject fine validation target to coarse grid
            y_fine_coarse = selective_reproject_match(
                data_src=y_fine_grid,
                data_target=y_coarse_grid,  # type: ignore
            )

            # Calibrate fine target predicted by downscaler
            # NOTE: https://math.stackexchange.com/a/2943606/209790
            y_fine_pred = (
                y_fine_coarse.mean().item()  # type: ignore
                + y_fine_coarse.std().item()  # type: ignore
                / y_coarse.std()  # type: ignore
                * (y_fine_pred - y_coarse.mean())  # type: ignore
            )

            # Calibrate fine target predicted by dummy mean model
            y_fine_dummy_pred = (
                y_fine_coarse.mean().item()  # type: ignore
                + y_fine_coarse.std().item()  # type: ignore
                / y_coarse.std()  # type: ignore
                * (y_fine_dummy_pred - y_coarse.mean())  # type: ignore
            )

        # Combine the true and predicted raw targets into a same DataFrame (so that all
        # records containing any nan may be later dropped and the prediction score
        # afterwards computed)
        data = pd.DataFrame(
            data={
                "y_true": y_fine,
                "y_pred": y_fine_pred,
                "y_dummy_pred": y_fine_dummy_pred,
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
            # mean of the masked inference coarse targets) as reference.]
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
            # Root mean squared error of the standardized target (using true target
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

        # Select solely scores of interest
        score = {
            scorer: score_i for scorer, score_i in score.items() if scorer in scorers
        }

        return score

    def score(
        self,
        X_and_mask_fine: dict[pd.Timestamp, np.ndarray | pd.DataFrame],
        y_fine: dict[pd.Timestamp, np.ndarray | pd.Series],
        correct: bool = True,
        calibrate: bool = False,
        X_and_mask_coarse: dict[pd.Timestamp, np.ndarray | pd.DataFrame] | None = None,
        y_coarse: dict[pd.Timestamp, np.ndarray | pd.Series] | None = None,
        coords_coarse: dict[pd.Timestamp, xr.Coordinates] | None = None,
        coords_fine: dict[pd.Timestamp, xr.Coordinates] | None = None,
        aggregate: bool = False,
        scorers: list[str] | None = None,
        sample_weight: dict[pd.Timestamp, np.ndarray | pd.Series] | None = None,
    ) -> dict[pd.Timestamp, dict[str, float]]:
        """
        Predict fine target and score for multiple images individually (if `aggregate`
        is set to `False`) or combined (if `aggregate` is set to `True`).

        Parameters
        ----------

        X_and_mask_fine : dict[pd.Timestamp, np.ndarray or pd.DataFrame]
            Fine predictors and masks, keyed by timestamp.

        y_fine : dict[pd.Timestamp, np.ndarray or pd.Series]
            The "true" fine raw target, keyed by timestamp.

        correct : bool, default=True
            Whether to correct the predicted fine raw target (from fine predictors and
            masks, `X_and_mask_fine`) using the finely-resampled residual for the
            prediction of the coarse raw target (from coarse predictors and masks,
            `X_and_mask_coarse`).

        calibrate : bool, default=False
            Whether to calibrate the predicted fine target with the coarse validation
            target for each timestamp. This is done by offsetting and scaling the
            predicted fine target with the transform that makes the coarse true target
            (`y_coarse`) have the same mean and standard deviation as the validation
            coarse one (coarsened `y_fine`) for each timestamp. Such transformation is
            an attempt to account for discrepancies between source and validation
            platforms at a common coarse grid from the computed scores.

        X_and_mask_coarse : dict[pd.Timestamp, np.ndarray or pd.DataFrame] or None, default=None
            Coarse predictors and masks, keyed by timestamp. It must be issued if
            `correct` is `True`.

        y_coarse : dict[pd.Timestamp, np.ndarray or pd.Series] or None, default=None
            The "true" coarse raw target, keyed by timestamp. It must be issued if
            `correct` or `calibrate` are `True` or if `transform` is not `None`.

        coords_coarse : dict[pd.Timestamp, xarray.core.coordinates.Coordinates] or None, default=None
            The coordinates of the coarse mesh for each image, keyed by timestamp. It
            must be issued if `correct` or `calibrate` are `True`.

        coords_fine : dict[pd.Timestamp, xarray.core.coordinates.Coordinates] or None, default=None
            The coordinates of the fine mesh for each image, keyed by timestamp. It must
            be issued if `correct` or `calibrate` are `True`.

        aggregate : bool, default=False
            Whether to compute scores for images individually (`False`) or combined
            (`True`).

        scorers : list[str], default=["r2", "r2_oos", "rmse", "rmse_delta", "mae", "mae_delta", "mbe"]
            Aliases of the scorers to consider.

        sample_weight : dict[pd.Timestamp, np.ndarray or pd.Series] or None, default=None
            Weight of each sample in the score, keyed by timestamp.

        Returns
        -------

        score : dict[pd.Timestamp, dict[str, float]]
            Prediction scores for each image (if `aggregate` is set to `False`) or all
            of them combined (if `aggregate` is set to `True`).
        """

        if self.logger is not None:
            self.logger.info("Predicting raw target and scoring...")

        # Define default value for scorers argument
        if scorers is None:
            scorers = ["r2", "r2_oos", "rmse", "rmse_delta", "mae", "mae_delta", "mbe"]

        # Transform parameters valued as None into dictionaries with None values (one
        # per image)
        X_and_mask_coarse = (
            X_and_mask_coarse
            if X_and_mask_coarse is not None
            else dict.fromkeys(X_and_mask_fine.keys(), None)  # type: ignore
        )
        y_fine = (
            y_fine
            if y_fine is not None
            else dict.fromkeys(X_and_mask_fine.keys(), None)  # type: ignore
        )
        y_coarse = (
            y_coarse
            if y_coarse is not None
            else dict.fromkeys(X_and_mask_fine.keys(), None)  # type: ignore
        )
        coords_coarse = (
            coords_coarse
            if coords_coarse is not None
            else dict.fromkeys(X_and_mask_fine.keys(), None)  # type: ignore
        )
        coords_fine = (
            coords_fine
            if coords_fine is not None
            else dict.fromkeys(X_and_mask_fine.keys(), None)  # type: ignore
        )
        sample_weight = (
            sample_weight
            if sample_weight is not None
            else dict.fromkeys(X_and_mask_fine.keys(), None)  # type: ignore
        )

        if aggregate is False:
            # Define progress bar
            pbar = (
                tqdm(
                    # Prefix for the progressbar
                    bar_format=f"{'':9}" + "{l_bar}{bar}{r_bar}",
                    desc=f"{'':8}",
                    total=len(X_and_mask_fine.keys()),  # type: ignore
                    unit="timestamp",
                    position=0,
                    leave=True,  # Keep progress on the screen after completion.
                    options={"console": self.logger.console},
                )
                if self.show_progress is True and self.logger is not None
                else None
            )

            # Predict scores as a dictionary
            score = {}
            if self.max_workers != 1:
                with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                    # List of placeholders for the eventual result of a computation
                    futures = {
                        # NOTE: Using executor.submit() can be safely used as key of
                        # dictionary since executor.submit() returns a Future object
                        # (https://docs.python.org/3/library/asyncio-future.html#future-object)
                        # and all of these objects are unique and hashable.
                        executor.submit(
                            self.score_single,
                            timestamp=timestamp,
                            X_and_mask_fine=X_and_mask_fine[timestamp],
                            y_fine=y_fine[timestamp],
                            correct=correct,
                            calibrate=calibrate,
                            X_and_mask_coarse=X_and_mask_coarse[timestamp],  # type: ignore
                            y_coarse=y_coarse[timestamp],  # type: ignore
                            coords_coarse=coords_coarse[timestamp],  # type: ignore
                            coords_fine=coords_fine[timestamp],  # type: ignore
                            scorers=scorers,
                            sample_weight=sample_weight[timestamp],  # type: ignore
                        ): timestamp
                        for timestamp in X_and_mask_fine  # type: ignore
                    }

                    for future in as_completed(futures):
                        # Add result to dictionary of results
                        timestamp = futures[future]
                        score[timestamp] = future.result()

                        # Update progress bar with one more count per completed process
                        if pbar is not None:
                            pbar.update()

                # Make dictionary of scores be ordered as input X_and_mask_fine
                # NOTE: multiprocessing may output results in a different order.
                score = {timestamp: score[timestamp] for timestamp in X_and_mask_fine}  # type: ignore

            else:
                for timestamp in X_and_mask_fine:  # noqa: PLC0206
                    score[timestamp] = self.score_single(
                        timestamp=timestamp,
                        X_and_mask_fine=X_and_mask_fine[timestamp],
                        y_fine=y_fine[timestamp],
                        correct=correct,
                        calibrate=calibrate,
                        X_and_mask_coarse=X_and_mask_coarse[timestamp],  # type: ignore
                        y_coarse=y_coarse[timestamp],  # type: ignore
                        coords_coarse=coords_coarse[timestamp],  # type: ignore
                        coords_fine=coords_fine[timestamp],  # type: ignore
                        scorers=scorers,
                        sample_weight=sample_weight[timestamp],  # type: ignore
                    )
                    # Update progress bar with one more count per completed process
                    if pbar is not None:
                        pbar.update()

            # At the end close progress bar
            if pbar is not None:
                pbar.close()

        # If parameter "aggregate" is True, score for the combined data
        else:
            # Raise error if transformation of predicted target into raw state is to
            # be performed but required parameters are missing
            if self.transform is not None and y_coarse is None:
                raise TypeError(
                    "Parameter 'y_coarse' must also be issued to transform"
                    + " predicted target into raw state."
                )

            # Convert true fine raw target for each image to pandas Series if it is
            # not already.
            y_fine = {
                timestamp: (
                    pd.Series(y_fine[timestamp])
                    if not isinstance(y_fine[timestamp], pd.Series)
                    else y_fine[timestamp]
                )
                for timestamp in X_and_mask_fine  # type: ignore
            }

            # Get statistics of true fine raw target for each image (to use them
            # later to compute RMSE of the standardized target)
            y_fine_mean = {
                timestamp: y_fine[timestamp].mean()  # type: ignore
                for timestamp in X_and_mask_fine  # type: ignore
            }
            y_fine_std = {
                timestamp: y_fine[timestamp].std()  # type: ignore
                for timestamp in X_and_mask_fine  # type: ignore
            }

            # Compute standardized true fine target for each image (using true fine
            # raw target statistics)
            y_fine_delta = {
                timestamp: (y_fine[timestamp] - y_fine_mean[timestamp])
                / y_fine_std[timestamp]
                for timestamp in X_and_mask_fine  # type: ignore
            }

            # Predict raw fine target for each image
            y_fine_pred = {
                timestamp: pd.Series(value)  # type: ignore
                for timestamp, value in self.predict(
                    X_and_mask_fine=X_and_mask_fine,
                    correct=correct,
                    X_and_mask_coarse=X_and_mask_coarse,  # type: ignore
                    y_coarse=y_coarse,
                    coords_coarse=coords_coarse,
                    coords_fine=coords_fine,
                    gridded=False,
                    _log=False,
                ).items()  # type: ignore
            }

            # Predict raw fine target for each image using the dummy mean model
            # NOTE: this is required for computing out-of-sample coefficient of
            # determination.
            y_fine_dummy_pred = {
                timestamp: pd.Series(
                    self.estimator.pipeline.named_steps[
                        "regressor"
                    ].dummy_mean_model.predict(X_and_mask_fine[timestamp])
                    * (
                        y_coarse[timestamp].std()  # type: ignore
                        if self.transform == "standardize"
                        else 1
                    )
                    + (
                        y_coarse[timestamp].mean()  # type: ignore
                        if self.transform is not None
                        else 0
                    )
                )
                for timestamp in X_and_mask_fine  # type: ignore
            }

            # Calibrate the fine targets predicted by downscaler and dummy mean model
            # with the transform that would make the coarse true target have the same
            # mean and standard deviation as the coarsened fine validation one.
            if calibrate is True:
                # Express coarse true target in its grid
                shape_coarse = {
                    timestamp: tuple(
                        reversed(list(coords_coarse[timestamp].sizes.values()))  # type: ignore
                    )  # type: ignore
                    for timestamp in X_and_mask_fine
                }
                y_coarse_grid = {
                    timestamp: xr.DataArray(
                        data=(
                            y_coarse[timestamp].values  # type: ignore
                            if isinstance(y_coarse[timestamp], pd.Series)  # type: ignore
                            else y_coarse[timestamp]  # type: ignore
                        ).reshape(  # type: ignore
                            shape_coarse[timestamp]  # type: ignore
                        ),
                        coords=coords_coarse[timestamp],  # type: ignore
                        dims=("y", "x"),
                        name="LST",
                    )
                    for timestamp in X_and_mask_fine
                }

                # Express fine validation target in its  grid
                shape_fine = {
                    timestamp: tuple(
                        reversed(list(coords_fine[timestamp].sizes.values()))  # type: ignore
                    )  # type: ignore
                    for timestamp in X_and_mask_fine
                }  # type: ignore
                y_fine_grid = {
                    timestamp: xr.DataArray(
                        data=(
                            y_fine[timestamp].values  # type: ignore
                            if isinstance(y_fine[timestamp], pd.Series)  # type: ignore
                            else y_fine[timestamp]
                        ).reshape(  # type: ignore
                            shape_fine[timestamp]  # type: ignore
                        ),
                        coords=coords_fine[timestamp],  # type: ignore
                        dims=("y", "x"),
                        name="LST",
                    )
                    for timestamp in X_and_mask_fine
                }

                # Reproject fine validation target to coarse grid
                y_fine_coarse = {
                    timestamp: selective_reproject_match(
                        data_src=y_fine_grid[timestamp],  # type: ignore
                        data_target=y_coarse_grid[timestamp],  # type: ignore
                    )
                    for timestamp in X_and_mask_fine  # type: ignore
                }

                # Calibrate fine target predicted by downscaler
                # NOTE: https://math.stackexchange.com/a/2943606/209790
                y_fine_pred = {
                    timestamp: (
                        y_fine_coarse[timestamp].mean().item()  # type: ignore
                        + y_fine_coarse[timestamp].std().item()  # type: ignore
                        / y_coarse[timestamp].std()  # type: ignore
                        * (y_fine_pred[timestamp] - y_coarse[timestamp].mean())  # type: ignore
                    )
                    for timestamp in X_and_mask_fine  # type: ignore
                }

                # Calibrate fine target predicted by dummy mean model
                y_fine_dummy_pred = {
                    timestamp: (
                        y_fine_coarse[timestamp].mean().item()  # type: ignore
                        + y_fine_coarse[timestamp].std().item()  # type: ignore
                        / y_coarse[timestamp].std()  # type: ignore
                        * (y_fine_dummy_pred[timestamp] - y_coarse[timestamp].mean())  # type: ignore
                    )
                    for timestamp in X_and_mask_fine  # type: ignore
                }

            # Compute standardized predicted fine target for each image (using true fine
            # raw target statistics)
            y_fine_pred_delta = {
                timestamp: (y_fine_pred[timestamp] - y_fine_mean[timestamp])
                / y_fine_std[timestamp]
                for timestamp in X_and_mask_fine  # type: ignore
            }

            # Combine variables of all timestamps
            y_fine = pd.concat(y_fine, ignore_index=True)  # type: ignore
            y_fine_pred = pd.concat(y_fine_pred, ignore_index=True)  # type: ignore
            y_fine_dummy_pred = pd.concat(y_fine_dummy_pred, ignore_index=True)  # type: ignore
            y_fine_delta = pd.concat(y_fine_delta, ignore_index=True)  # type: ignore
            y_fine_pred_delta = pd.concat(y_fine_pred_delta, ignore_index=True)  # type: ignore
            sample_weight = (
                pd.concat(sample_weight, ignore_index=True)  # type: ignore
                if not any(value is None for value in sample_weight.values())  # type: ignore
                else None
            )
            # Combine the true and predicted targets into a common DataFrame (so that
            # all records containing any nan may be later dropped and the prediction
            # score afterwards computed)
            data = pd.DataFrame(
                data={
                    "y_true": y_fine,
                    "y_pred": y_fine_pred,
                    "y_dummy_pred": y_fine_dummy_pred,
                    "y_true_delta": y_fine_delta,
                    "y_pred_delta": y_fine_pred_delta,
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
                # [NOTE: this is such that it uses a dummy mean model (simply the
                # arithmetic mean of the masked inference coarse targets) as
                # reference.]
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
                # Root mean squared error of the standardized target (using true
                # target statistics)
                "rmse_delta": rmse(
                    y_true=data["y_true_delta"],
                    y_pred=data["y_pred_delta"],
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
                "mae_delta": mae(
                    y_true=data["y_true_delta"],
                    y_pred=data["y_pred_delta"],
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

            # Select solely scores of interest
            score = {
                scorer: score_i
                for scorer, score_i in score.items()
                if scorer in scorers
            }

        return score  # type: ignore

    def score_coarse(
        self,
        X_and_mask_coarse: dict[pd.Timestamp, np.ndarray | pd.DataFrame],
        y_coarse: dict[pd.Timestamp, np.ndarray | pd.Series],
        aggregate: bool = False,
        scorers: list[str] | None = None,
        sample_weight: dict[pd.Timestamp, np.ndarray | pd.Series] | None = None,
    ) -> dict[pd.Timestamp, dict[str, float]]:
        """
        Predict coarse target and score for multiple images individually (if `aggregate`
        is set to `False`) or combined (if `aggregate` is set to `True`).

        Parameters
        ----------

        X_and_mask_coarse : dict[pd.Timestamp, np.ndarray or pd.DataFrame]
            Coarse predictors and masks, keyed by timestamp.

        y_coarse : dict[pd.Timestamp, np.ndarray or pd.Series]
            The "true" raw coarse target, keyed by timestamp.

        aggregate : bool, default=False
            Whether to compute scores for images individually (`False`) or combined
            (`True`).

        scorers : list[str], default=["r2", "r2_oos", "rmse", "rmse_delta", "mae", "mae_delta", "mbe"]
            Aliases of the scorers to consider.

        sample_weight : dict[pd.Timestamp, np.ndarray or pd.Series] None, default=None
            Weight of each sample in the score, keyed by timestamp.

        Returns
        -------

        score : dict[pd.Timestamp, dict[str, float]]
            Prediction scores for each image (if `aggregate` is set to `False`) or all
            of them combined (if `aggregate` is set to `True`)
        """

        score = self.score(
            X_and_mask_fine=X_and_mask_coarse,
            y_fine=y_coarse,
            correct=False,
            y_coarse=y_coarse,
            aggregate=aggregate,
            scorers=scorers,
            sample_weight=sample_weight,
        )

        return score

    def save(self, path: Path) -> None:
        """
        Write the instance to `path` with `joblib`.

        Parameters
        ----------
        path : Path
            Path to write the instance to.
        """

        joblib.dump(value=self, filename=path)
