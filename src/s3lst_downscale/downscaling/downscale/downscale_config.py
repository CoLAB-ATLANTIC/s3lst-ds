from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from sklearn.linear_model import LinearRegression

from s3lst_downscale.data_wrangling.data_wrangling import DataWrangler
from s3lst_downscale.downscaling.downscaling import Downscaler
from s3lst_downscale.downscaling.piecewise_downscaling import PiecewiseDownscaler
from s3lst_downscale.downscaling.regression import Regressor


@dataclass
class DownscaleConfig:
    """
    Configurations for wrangling the data of timestamps of interest, training a
    downscaler using the coarse data of the training timestamps, downscaling the data of
    the inference timestamps with the model as well as scoring the downscaler and
    returning or writing the results to files.

    Parameters
    ----------

    data_wrangler : DataWrangler or Path or None, default=None
        Data wrangler or a path to a Joblib file containing it. If not issued, a data
        wrangler is created from scratch using the `data_wrangler_`-prefixed parameters
        of the present `DownscaleConfig` instance. Note that `transform` parameter of
        the downscaler is in any case enforced (therefore, transforming/re-transforming
        the wrangled data) on the one of the data wrangler regardless of the previous
        value. Also, `data_wrangler_max_workers` parameter of the present
        `DownscaleConfig` instance is also in any case enforced.

    data_wrangler_path_sentinel3 : Path or None, default=None
        If `data_wrangler` is not issued: path to directory containing Sentinel-3
        product folders whose data is to be wrangled. Each of such folders must contain
        georeferenced Sentinel-3 SLSTR Level-2 LST product file
        (https://sentiwiki.copernicus.eu/web/slstr-products#S3-SLSTR-Products-L2-LST-Products)
        as well as a georeferenced Sentinel-3 Synergy Level-2 product file
        (https://sentiwiki.copernicus.eu/web/synergy-products#SYNERGYProducts-L2SYNSDRprocessingS3-Synergy-Products-L2-SYN-SDR-processing).
        Furthermore, the name of such folders must correspond to the respective start
        sensing time in the format "YYYYMMDDTHHMMSS".

    data_wrangler_path_spatial_pred : Path or None, default=None
        If `data_wrangler` is not issued: path to a NetCDF file with the spatial
        predictor data whose data is to be wrangled. If not set, no spatial predictor
        data is considered.

    data_wrangler_aoi : str or Path or None, default=None
        If `data_wrangler` is not issued: WKT string or path to AOI geometry file to
        mask out the data. The data wrangler will add the AOI to the wrangled data as
        variable `"aoi"`. If not set, no such variable is defined and no masking is
        applied.

    data_wrangler_path_landsat : Path or None, default=None
        If `data_wrangler` is not issued and `score` is `True`: Path to the directory
        containing Landsat 8/9 folders whose data is to be wrangled. Each of such
        folders must contain a `LST.TIF` file with georeferenced Landsat 8/9 Level-2
        LST data
        (https://www.usgs.gov/centers/eros/science/usgs-eros-archive-landsat-archives-landsat-8-9-olitirs-collection-2-level-2),
        having a resolution of 30 m. In the wrangling, such data and Sentinel-3's will
        be "matched" if the respective folders have the same name (it is implied here
        that the user had analysed the acquisitions obtained by the two platforms and
        set the names of the Landsat 8/9 data folders as the ones of Sentinel-3's (start
        sensing times) whose start sensing times and spatial extents are approximately
        the same). Note that Landsat data will be solely used if `score` is `True` as it
        may be used for the coarse and fine-scoring of the downscaler in the training
        and inference timestamps for which such data is available. If no Landsat data is
        issued or it is not available for the timestamps of interest, only
        coarse-scoring using Sentinel-3 LST as ground truth is performed.

    data_wrangler_vars : list[str] or None, default=None
        If `data_wrangler` is not issued: aliases of the variables to be wrangled
        besides the target (such as predictor, sample_weight and visualization
        variables). If `vars` is not issued, but `downscaler` is, it will be set to the
        aliases of the predictors (`cols_X`) considered by the latter. Otherwise, if
        `downscaler` is not issued but `downscaler_X` is, it will be set to
        `downscaler_X`, or, if not, to all aliases of the predictors (`X`) considered by
        a default `DataVars` instance (`s3lst_downscale.utilities.var_utils.DataVars`).

    data_wrangler_max_workers : int, default=1
        Number of simultaneous multiple processes to be considered by the data wrangler
        in wrangling. Note that if negative, one has the following conditions:
            - `-1`: all processors are used;
            - `-k`: all processors except k-1 are used.
        This parameter is enforced regardless of the data wrangler being issued or
        created from scratch.

    downscaler : PiecewiseDownscaler or Downscaler or Path or None, default=None
        Downscaler or a path to a Joblib file containing it. If not issued, a downscaler
        is created from scratch using the `downscaler_`-prefixed parameters of the
        present `DownscaleConfig`. Note that `downscaler_masks` and
        `downscaler_max_workers` are in any case enforced, regardless of the downscaler
        being issued or created from scratch.

    downscaler_architecture: {"single", "multi"}, default="single"
        If `downscaler` is not issued: the architecture of the downscaler to be created:
            - `"single"` - for the case of a single-timestamp one (a sub-downscaler
            per timestamp, trained with solely the coarse data of a timestamp and that
            infers solely the fine target of that same timestamp);
            - `"multi"` - for the case of a multi-timestamp one (a downscaler trained
            with the coarse data of multiple timestamps and that can infer the fine
            target of any other).

    downscaler_base_model : Regressor, default=LinearRegression()
        If `downscaler` is not issued: the regression model to be used as the base model
        of the downscaler to be created. If not issued, it is set to
        `LinearRegression()` by default.

    downscaler_X : list[str], default=["FVC", "NDWI"]
        If `downscaler` is not issued: aliases of the predictors to be considered by the
        downscaler to be created. If not issued, it is set to `["FVC", "NDWI"]`.

    downscaler_masks: list[str] or None, default=None
        Aliases of the mask variables (e.g. `["aoi"]`) to regard (wherever the variables
        have `nan` values, the respective data records are masked out). If not issued,
        it is set to `[]` and no masking is considered by the downscaler. This parameter
        is enforced regardless of the downscaler being issued or created from scratch.

    downscaler_scale : {"standardize", "min_max_normalize", None}, default="standardize"
        If `downscaler` is not issued: the scaling method to apply to numerical
        predictors:
            - `"standardize"`: to standardize the numerical predictors (zero mean and
            unit variance);
            - `"min_max_normalize"`: to min-max normalize the numerical predictors (to
            the range `[0, 1]`);
            - `None`: to regard the numerical predictors raw (no scaling).

    downscaler_encode : {"one_hot", "dummy", None}, default="dummy"
        If `downscaler` is not issued: the encoding method to apply to the categorical
        predictors:
            - `"one_hot"`: to one-hot encode the categorical predictors;
            - `"dummy"`: to dummy encode the categorical predictors (one-hot
            encoding with the first component dropped);
            - `None`: to regard the categorical predictors raw (no encoding).

        Note that dummy encoding is usually considered in place of one-hot to avoid
        multicollinearity problems (one may show that a component of a one-hot encoding
        vector is fully determined by all the other components making it redundant).

    downscaler_transform : {None, "center", "standardize"}, default=None
        If `downscaler` is not issued: the transform to apply on the coarse target and
        coarse and fine spatio-temporal predictors from a copy of the wrangled `data` in
        each `SingleDataWrangler` instance of the `data_wrangler` by using coarse data
        statistics. The transformations are set in `SingleDataWrangler's `data` with the
        same names as the original columns with the substring `"_trans"` suffixed to
        them. Note that the transformations are timestamp-specific, that is, the
        computed statistics and the applied transformations in each timestamp solely
        concern the data of that timestamp. The possible values for
        `downscaler_transform` are:
            - `None` - not transforming the data;
            - `"center"` - subtracting the mean from the data;
            - `"standardize"` - subtracting the mean from the data and dividing the
            result by the standard deviation.
        Note that such transforms are redundant for the case of the single-timestamp
        architecture. They only take effect for the multi-timestamp architecture.

    downscaler_max_workers : int, default=1
        Number of simultaneous multiple processes to be considered by the downscaler (in
        training, prediction and scoring). Note that if negative, one has the following
        conditions:
            - `-1`: all processors are used;
            - `-k`: all processors except k-1 are used.
        This parameter is enforced regardless of the downscaler being issued or created
        from scratch.

    retrain : bool, default=True
        If `downscaler` is issued, its architecture is multi-timestamp and it had been
        trained: whether to retrain the downscaler with the coarse data of the training
        timestamps. Note that training/retraining is in any case considered if the model
        had not been trained or its architecture is a single-timestamp one. A
        single-timestamp architecture is such that inference of the fine target of some
        timestamp can only be done with a downscaler trained with the coarse data of
        that same timestamp.

    timestamps_infer : list[str] or None, default=None
        Timestamps for inferring fine target with the downscaler, in any format parsable
        by `pd.Timestamp` (e.g. `"YYYY-MM-DD HH:MM:SS"`). These must correspond to the
        start sensing times of the respective acquisitions. Furthermore, they must be
        part of the timestamps of the `data_wrangler` if it was issued, or
        `data_wrangler_path_sentinel3` if it was not. If `timestamps_infer` is not
        issued, it will be set to all such timestamps.

    timestamps_fit : list[str] or None, default=None
        Timestamps for training the downscaler, in any format parsable by `pd.Timestamp`
        (e.g. `"YYYY-MM-DD HH:MM:SS"`). These must correspond to the start sensing times
        of the respective acquisitions. Furthermore, they must be part of the timestamps
        of the `data_wrangler` if it was issued, or `data_wrangler_path_sentinel3` if it
        was not. If `timestamps_fit` is not issued and the downscaler is a
        multi-timestamp one, it will be set to all such timestamps. If the architecture
        of the downscaler is single-timestamp, `timestamps_fit` is set to
        `timestamps_infer` regardless of the issued value. Note that in the case of the
        single-timestamp architecture, inference of the fine target of some timestamp
        can only be done with a downscaler trained with the coarse data of that same
        timestamp.

    sample_weight_fit: str or None, default=None
        Alias of the variable to be regarded as sample weight for training the
        downscaler. If not issued, no sample weight in training is considered.

    sample_weight_score: str or None, default=None
        Alias of the variable to be regarded as sample weight for scoring the
        downscaler. If not issued, no sample weight in scoring is considered.

    score : bool, default=True
        Whether to score the predictions in the training and inference timestamps.

    scorers : list[str], default=["r2", "rmse", "mae", "mbe"]
        Aliases of the scorers to consider in scoring.

    correct : bool, default=True
        Whether to correct the predicted fine raw target for each image (from fine
        predictors and masks, `X_and_mask_fine`) using the finely-resampled residual for
        the prediction of the coarse raw target (from coarse predictors and masks,
        `X_and_mask_coarse`).

    gridded : bool, default=True
        Whether to get the predicted fine raw target of each image in grid form (as an
        `xr.DataArray`) or in flattened form (as a `pd.Series`).

    dims : tuple or None, default=None
        If `gridded` is `True`: labels for the dimensions of the predicted gridded
        target. If not issued, it is set to `("lat", "lon")` by default.

    attrs : dict or None, default=None
        If `gridded` is `True`: attributes to set in the predicted gridded target. If
        not issued, it is set as in accordance with the CF conventions
        (https://cf-convention.github.io/Data/cf-conventions/cf-conventions-1.13/cf-conventions.pdf#temperature-units):
            ```
            {
                "standard_name": "land_surface_temperature",
                "long_name": "Land surface temperature",
                "units": "K",
            }
            ```

    path_out : Path or None, default=None
        The directory path to save the downscaled LST data, obtained scores, downscaler
        and data wrangler. If not issued, the results are instead returned.

    file_ext_grid : str, default=".nc"
        If `gridded` is `True` and `path_out` is issued: The file extension to use when
        writing the gridded predicted fine target to file (e.g. ".tif" for GeoTIFF and
        ".nc" for NetCDF). If `path_out` is issued and `gridded` is `False`, the
        predicted fine target is written in flattened form with the ".csv" extension.

    out_data_wrangler : bool, default=True
        Whether to return or write (if `path_out` is issued) the data wrangler.

    out_downscaler : bool, default=True
        Whether to return or write (if `path_out` is issued) the downscaler.

    log_mode : {None, "console", "file", "both"}, default="both"
        The logging mode for wrangling, training, inferring and scoring:
            - `None`: No logging is done;
            - `"console"`: Logging is done to console only;
            - `"file"`: Logging is done to a log file only;
            - `"both"`: Logging is done to both console and a log file.
        Note the log file would be defined as `downscale.log` at `out_dir`.
    """

    data_wrangler: DataWrangler | Path | None = None
    data_wrangler_path_sentinel3: Path | None = None
    data_wrangler_path_spatial_pred: Path | None = None
    data_wrangler_aoi: str | Path | None = None
    data_wrangler_path_landsat: Path | None = None
    data_wrangler_vars: list[str] | None = None
    data_wrangler_max_workers: int = 1
    downscaler: PiecewiseDownscaler | Downscaler | Path | None = None
    downscaler_architecture: Literal["single", "multi"] = "single"
    downscaler_base_model: Regressor = field(default_factory=lambda: LinearRegression())
    downscaler_X: list[str] = field(default_factory=lambda: ["FVC", "NDWI"])
    downscaler_masks: list[str] | None = None
    downscaler_scale: Literal["standardize", "min_max_normalize"] | None = "standardize"
    downscaler_encode: Literal["one_hot", "dummy"] | None = "dummy"
    downscaler_transform: Literal["center", "standardize"] | None = None
    downscaler_max_workers: int = 1
    timestamps_infer: list[str] | None = None
    timestamps_fit: list[str] | None = None
    sample_weight_fit: str | None = None
    sample_weight_score: str | None = None
    score: bool = True
    scorers: list[str] = field(default_factory=lambda: ["r2", "rmse", "mae", "mbe"])
    retrain: bool = True
    correct: bool = True
    gridded: bool = True
    dims: tuple | None = None
    attrs: dict | None = None
    path_out: Path | None = None
    file_ext_grid: str = ".nc"
    out_data_wrangler: bool = True
    out_downscaler: bool = True
    log_mode: Literal["console", "file", "both"] | None = "both"


# Configurations for downscaling
config = DownscaleConfig(
    # data_wrangler=Path(
    #     "/home/elio/projects/lst_downscaling/assets/results/donwscale_trial_2/data_wrangler.joblib"
    # ),
    data_wrangler_path_sentinel3=Path(
        "/home/elio/projects/lst_downscaling/assets/data/processed/sentinel3"
    ),
    data_wrangler_path_spatial_pred=Path(
        "/home/elio/projects/lst_downscaling/assets/data/processed/fixed_predictors.nc"
    ),
    # data_wrangler_aoi="POLYGON ((10 55, 11 55, 11 56, 10 56, 10 55))",
    data_wrangler_aoi="/home/elio/projects/lst_downscaling/assets/aoi/clim4cities.shp",
    data_wrangler_path_landsat=Path(
        "/home/elio/projects/lst_downscaling/assets/data/processed/landsat"
    ),
    data_wrangler_vars=[
        "FVC",
        "NDWI",
        "TCD",
        "COASTDIST",
        "IMD",
        # "season",
        # "LCZ",
        # "UD",
    ],
    # WARNING: data_wrangler_max_workers takes effect regardless of the downscaler being
    # issued or created from scratch.
    data_wrangler_max_workers=10,
    # downscaler=Path(
    #     "/home/elio/projects/lst_downscaling/assets/results/donwscale_trial_2/downscaler.joblib"
    # ),
    # downscaler=Downscaler(
    #     base_model=LinearRegression(),
    #     cols_X=["FVC", "NDWI", "COASTDIST", "IMD", "TCD"],
    #     cols_mask=["aoi"],
    #     scale="standardize",
    #     encode="dummy",
    #     max_workers=10,
    #     transform="standardize",  # type: ignore
    # ),
    downscaler_architecture="single",
    # downscaler_base_model=DummyRegressor(),
    downscaler_base_model=LinearRegression(),
    downscaler_X=["FVC", "NDWI", "COASTDIST", "IMD", "TCD"],
    # WARNING: downscaler_masks takes effect regardless of the downscaler being issued
    # or created from scratch.
    downscaler_masks=["aoi"],
    downscaler_scale="standardize",
    downscaler_encode="dummy",
    downscaler_transform="standardize",
    # WARNING: downscaler_max_workers takes effect regardless of the downscaler being
    # issued or created from scratch.
    downscaler_max_workers=10,
    # timestamps_infer=[
    #     "2020-05-30 10:17:38",
    #     "2020-06-15 10:02:39",
    #     "2022-04-19 10:15:43",
    #     "2022-10-19 10:10:10",
    #     "2023-05-08 09:59:00",
    #     "2023-06-08 09:55:13",
    #     "2023-09-04 10:13:49",
    # ],
    # timestamps_fit=[
    #     "2022-04-19 10:15:43",
    #     "2023-05-08 09:59:00",
    #     "2023-09-04 09:34:54",
    # ],
    # sample_weight_fit="UD",
    # sample_weight_score="UD",
    score=True,
    scorers=["r2", "rmse", "rmse_delta", "mae", "mae_delta", "mbe"],
    retrain=False,
    correct=True,
    gridded=True,
    dims=None,
    attrs=None,
    path_out=Path("/home/elio/projects/lst_downscaling/assets/results/downscale_trial"),
    file_ext_grid=".nc",
    out_data_wrangler=True,
    out_downscaler=True,
    log_mode="both",
)
