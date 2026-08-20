from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

from s3lst_ds.data_batching.data_batching import DataBatcher
from s3lst_ds.downscaling.downscaling import Downscaler
from s3lst_ds.downscaling.regression import Regressor


def params_tune_getter(
    trial: optuna.trial.Trial | optuna.trial.FrozenTrial,
) -> dict[str, Any]:
    """
    Get `DownscalerEstimator`'s hyperparameter values suggested by `trial` for the
    optimized tuning method `optuna.study.Study.optimize()`.

    The suggestions may be obtained using `optuna`'s [suggest
    methods](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial):
    - [`suggest_float`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial.suggest_float):
    for continuous hyperparameters;
    - [`suggest_int`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial.suggest_int):
    for integer hyperparameters;
    - [`suggest_categorical`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial.suggest_categorical):
    for categorical hyperparameters.

    WARNING: Note that there is no suggest method for iterables. Each of their
    components must be suggested separately.

    Parameters
    ----------
    trial : optuna.trial.Trial or optuna.trial.FrozenTrial

    Returns
    -------

    params_tune : dict[str, Any]
        Suggested values for `DownscalerEstimator`'s tunable hyperparameters. The keys
        correspond to the hyperparameters' full access paths with each step separated by
        double underscores. (e.g. `"base_model__formula"` in which `"base_model"` is a
        parameter of the downscaler estimator and `"formula"` is a parameter of the
        former)
    """

    # Get suggestion for number of hidden layers of the MLP so that their sizes can be
    # suggested separately.
    n_hidden_layers = trial.suggest_int(
        name="n_hidden_layers",
        low=1,
        high=3,
    )

    params_tune = {
        "scale": trial.suggest_categorical(
            name="scale",
            choices=["standardize", "min_max_normalize"],
        ),
        # Get suggestion for the sizes of the hidden layers of the MLP. Each size is
        # suggested separately.
        "base_model__hidden_layer_sizes": tuple(
            [
                trial.suggest_int(
                    name=f"base_model__hidden_layer_sizes_{i}",
                    low=16,
                    high=32,
                )
                for i in range(n_hidden_layers)
            ]
        ),
        "base_model__learning_rate_init": trial.suggest_float(
            name="base_model__learning_rate_init",
            low=1e-4,
            high=1e-2,
            log=True,
        ),
        "base_model__alpha": trial.suggest_float(
            name="base_model__alpha",
            low=1e-5,
            high=1e-1,
            log=True,
        ),
    }

    return params_tune


@dataclass
class TuneConfig:
    """
    Configurations for wrangling the data of timestamps of interest, batching it into
    cross-validation and test datasets, performing an optimized cross-validated
    hyperparameter tuning of a multi-timestamp downscaling model and subsequently
    training and testing it. Moreover, configurations for returning or writing the
    results to files are also defined.

    Batching is done with respect to the issued `timestamps`, and, therefore, the whole
    data associated with a timestamp (a scene) is fully contained within a single batch.
    The resulting batch datasets are:
        - `"test"` dataset: timestamps for which there is Landsat data.
        - `"cross_val"` dataset: random split of the remaining timestamps into
        `n_cross_val_folds` folds, stratified with respect to a categorical variable
        `var_cross_val_strat` (if issued).

    The optimized cross-validated hyperparameter tuning is done using
    [`optuna`](https://optuna.readthedocs.io/en/stable/) by trying suggested
    hyperparameter values within the search space defined in the issued
    `params_tune_getter`. The estimator of the downscaler (the downscaler before
    de-transformation and residual correction) is cross-validated for each suggested
    hyperparameter combination and the best one is selected based on the issued scoring
    metric `best_scorer`. Cross-validation is done by training the downscaler estimator
    on all cross-validation folds except one and scoring it on the latter, rotating the
    scoring fold until all are considered. The overall cross-validation score is
    computed from the arithmetic mean of the scores of each iteration.


    Training of the tuned downscaler is done using the whole cross-validation data.

    Testing is done by scoring the trained downscaler on the test dataset (for both
    Sentinel-3 and Landsat data).

    WARNING: Note that for the sake of efficiency, during tuning, solely tuner-specific
    multiprocessing (set through present `tune_n_jobs` parameter) is considered . Base
    model-specific multi-processing (set through the respective `n_jobs` parameter) is
    subsequently considered. Note that for the case of the MLPRegressor, `tune_n_jobs`
    is forcefully set to `1`, regardless of the issued value, as the MLPRegressor, by
    default, always use all available processors - and, therefore, a value of
    `tune_n_jobs` greater than `1` would impair the process.

    WARNING: Note that the units of the computed cross-validation score in the
    hyperparameter tuning are based on the ones of the transformed target (whose
    transform corresponds to the one set in the downscaler). For example, if the
    transform corresponds to `"standardize"` and `best_scorer` is set to `"rmse"`, the
    computed score corresponds to the RMSE of the standardized target, which is
    unitless.

    Attributes
    ----------

    data_batcher : DataBatcher or Path or None, default=None
        Data batcher or a path to a Joblib file containing it. If not issued, a data
        batcher is created from scratch using the `data_batcher_`-prefixed parameters of
        batcher is created from scratch using the `data_batcher_`-prefixed parameters of
        the present `TuneConfig` instance. Note that `transform` parameter of the
        downscaler is in any case enforced (therefore, transforming/re-transforming the
        wrangled data) on the one of the data batcher's data wrangler regardless of the
        previous value. Also, `data_batcher_data_wrangler_max_workers` parameter of the
        present `TuneConfig` instance is also in any case enforced.

    data_batcher_data_wrangler_path_sentinel3 : Path or None, default=None
        If `data_batcher` is not issued: path to directory containing Sentinel-3 product
        folders whose data is to be wrangled. Each of such folders must contain
        georeferenced Sentinel-3 SLSTR Level-2 LST product file
        (https://sentiwiki.copernicus.eu/web/slstr-products#S3-SLSTR-Products-L2-LST-Products)
        as well as a georeferenced Sentinel-3 Synergy Level-2 product file
        (https://sentiwiki.copernicus.eu/web/synergy-products#SYNERGYProducts-L2SYNSDRprocessingS3-Synergy-Products-L2-SYN-SDR-processing).
        Furthermore, the name of such folders must correspond to the respective start
        sensing time in the format "YYYYMMDDTHHMMSS".

    data_batcher_data_wrangler_path_spatial_pred : Path or None, default=None
        If `data_batcher` is not issued: path to a NetCDF file with the spatial
        predictor data whose data is to be wrangled. If not set, no spatial predictor
        data is considered.

    data_batcher_data_wrangler_aoi : str or Path or None, default=None
        If `data_batcher` is not issued: WKT string or path to AOI geometry file to mask
        out the data. The data wrangler will add the AOI to the wrangled data as
        variable `"aoi"`. If not set, no such variable is defined and no masking is
        applied.

    data_batcher_data_wrangler_path_landsat : Path or None, default=None
        If `data_batcher` is not issued: Path to the directory containing Landsat 8/9
        folders whose data is to be wrangled. Each of such folders must contain a
        `LST.TIF` file with georeferenced Landsat 8/9 Level-2 LST data
        (https://www.usgs.gov/centers/eros/science/usgs-eros-archive-landsat-archives-landsat-8-9-olitirs-collection-2-level-2),
        having a resolution of 30 m. In the wrangling, such data and Sentinel-3's will
        be "matched" if the respective folders have the same name (it is implied here
        that the user had analysed the acquisitions obtained by the two platforms and
        set the names of the Landsat 8/9 data folders as the ones of Sentinel-3's (start
        sensing times) whose start sensing times and spatial extents are approximately
        the same). Note that the Landsat data will be solely used for testing.

    data_batcher_data_wrangler_vars : list[str] or None, default=None
        If `data_batcher` is not issued: aliases of the variables to be wrangled besides
        the target (such as predictor, sample_weight and visualization variables). If
        `vars` is not issued, but `downscaler` is, it will be set to the aliases of the
        predictors (`cols_X`) considered by the latter. Otherwise, if `downscaler` is
        not issued but `downscaler_X` is, it will be set to `downscaler_X`, or, if not,
        to all aliases of the predictors (`X`) considered by a default `DataVars`
        instance (`s3lst_ds.utilities.var_utils.DataVars`).

    data_batcher_data_wrangler_max_workers : int, default=1
        Number of simultaneous multiple processes to be considered by the data wrangler
        in wrangling. Note that if negative, one has the following conditions:
            - `-1`: all processors are used;
            - `-k`: all processors except k-1 are used.
        This parameter is enforced regardless of the data wrangler being issued or
        created from scratch.

    data_batcher_n_cross_val_folds: int, default=5
        If `data_batcher` is not issued: number of cross-validation folds.

    data_batcher_var_cross_val_strat: str or None, default=None
        If `data_batcher` is not issued: metadata categorical variable with respect to
        which stratification in the cross-validation data splitting into folds is to be
        performed. If not defined, no stratification is considered.

    data_batcher_rnd_seed: int or np.random.RandomState or None = None
        If `data_batcher` is not issued: random seed number considered in the
        cross-validation data splitting into folds. If not defined, no such number is
        regarded.

    downscaler : Downscaler or Path or None, default=None
        Downscaler or a path to a Joblib file containing it. If not issued, a downscaler
        is created from scratch using the `downscaler_`-prefixed parameters of the
        present `TuneConfig`. Note that `downscaler_masks` and `downscaler_max_workers`
        are in any case enforced, regardless of the downscaler being issued or created
        from scratch.

    downscaler_base_model : Regressor, default=LinearRegression()
        If `downscaler` is not issued: the regression model to be used as the base model
        of the downscaler to be created. If not issued, it is set to
        `LinearRegression()` by default.

    downscaler_X : list[str] or None, default=["FVC", "NDWI"]
        If `downscaler` is not issued: aliases of the predictors to be considered by the
        downscaler to be created. If not issued, it is set to `["FVC", "NDWI"]`.

    downscaler_masks: list[str] or None, default=None
        Aliases of the mask variables (e.g. `["aoi"]`) to regard (wherever the variables
        have `nan` values, the respective data records are masked out). If not issued,
        it is set to `[]` and no masking is considered by the downscaler. This parameter
        is enforced regardless of the downscaler being issued or created from scratch.

    downscaler_scale : {"standardize", "min_max_normalize", None}, default="standardize"
        If `downscaler` is not issued: the scaling method to apply to numerical
        predictors.

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
        each `SingleDataWrangler` instance of the data batcher's `data_wrangler` by
        using coarse data statistics. The transformations are set in
        `SingleDataWrangler's `data` with the same names as the original columns with
        the substring `"_trans"` suffixed to them. Note that the transformations are
        timestamp-specific, that is, the computed statistics and the applied
        transformations in each timestamp solely concern the data of that timestamp. The
        possible values for `downscaler_transform` are:
            - `None`: not transforming the data;
            - `"center"`: subtracting the mean from the data;
            - `"standardize"`: subtracting the mean from the data and dividing the
            result by the standard deviation.
        Note that such transforms are redundant for the case of the single-timestamp
        architecture. They only take effect for the multi-timestamp architecture.

    downscaler_lasso_sel : bool, default=False
        If `downscaler` is not issued: whether to use a Lasso regression for selecting
        the scaled-encoded `downscaler_X` predictors downstream of the preprocessor.
        Lasso selection is such that solely the input predictors associated with
        coefficients of the fitted Lasso regression model having absolute values larger
        than `1e-5` are selected. Note that the non-encoded `downscaler_X` predictors
        are regardlessly considered downstream of the preprocessor.

    downscaler_lasso_alpha : float, default=1.0
        If `downscaler` is not issued: the regularization strength of the Lasso
        regression model used for selecting the scaled-encoded `downscaler_X` predictors
        downstream of the preprocessor. Such regularization strength is the multiplying
        constant of the weight vector L1-norm (sum of the absolute values of the
        components) in the Lasso regression objective function. The larger the value,
        the stronger the regularization. Note that this parameter only takes effect if
        `downscaler_lasso_sel` is `True`.

    downscaler_max_workers : int, default=1
        Number of simultaneous multiple processes to be considered by the downscaler (in
        training, prediction and scoring). Note that if negative, one has the following
        conditions:
            - `-1`: all processors are used;
            - `-k`: all processors except k-1 are used.
        This parameter is enforced regardless of the downscaler being issued or created
        from scratch.

    timestamps : list[pd.Timestamp] or list[str] or None, default=None
        If `data_batcher` is not issued: timestamps of interest either as `pd.Timestamp`
        values or in any format parsable by `pd.Timestamp` (e.g. `"YYYY-MM-DD
        HH:MM:SS"`). These must correspond to the start sensing times of the respective
        acquisitions. Furthermore, they must be part of those regarded by
        `data_batcher_data_wrangler_path_sentinel3`.  If neither `timestamps` nor
        `data_batcher` are issued, `timestamps` will be set to all of those regarded by
        such path. If `timestamps` is not issued but `data_batcher` is, `timestamps`
        will be set to all of those regarded by the batcher.

    sample_weight_fit: str or None, default=None
        Alias of the variable to be regarded as sample weight for cross-validation and
        training of the downscaler. Note that in cross-validation, `sample_weight_fit`
        is used for both training and scoring. If not issued, no sample weight in
        training and cross-validation is considered.

    sample_weight_score: str or None, default=None
        Alias of the variable to be regarded as sample weight for scoring the downscaler
        in training and testing (but not in cross-validation). If not issued, no sample
        weight in such scoring is considered.

    scorers : list[str], default=["r2", "rmse", "mae", "mbe"]
        Aliases of the scorers to consider in training and testing.

    best_scorer : str, default="rmse"
        Alias of the metric to consider in the selection of the best hyperparameter
        combination in the hyperparameter tuning of the downscaler estimator (that is,
        the downscaler before de-transformation and residual correction).

        WARNING: Note that since the estimator is the object tuned and not the
        downscaler itself, the units of the computed cross-validation metric in the
        tuning are based on the ones of the transformed target (whose transform
        corresponds to the one set in the downscaler). For example, if the transform
        corresponds to `"standardize"` and `best_scorer` is set to `"rmse"`, the
        computed metric corresponds to the RMSE of the standardized target, which is
        unitless.

    correct : bool, default=True
        Whether to correct the predicted fine raw target for each image (from fine
        predictors and masks, `X_and_mask_fine`) using the finely-resampled residual for
        the prediction of the coarse raw target (from coarse predictors and masks,
        `X_and_mask_coarse`).

    params_tune_getter: dict[str, dict[str, Any]]
        Dictionary of search spaces of the downscaler estimator hyperparameters to be
        tuned. The keys must correspond to the hyperparameters' full access paths with
        each step separated by double underscores (e.g. `"base_model__formula"` in which
        `"base_model"` is a parameter of the downscaler estimator and `"formula"` is a
        parameter of the former). The values must be dictionaries with keys:
            - `"suggest_method"`: alias of the `optuna`'s [suggest
            method](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial)
            to be used in the hyperparameter tuning of the respective hyperparameter:
                - [`"float"`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial.suggest_float):
                for continuous hyperparameters;
                - [`"int"`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial.suggest_int):
                for integer hyperparameters;
                - [`"categorical"`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial.suggest_categorical):
                for categorical hyperparameters.
            - `"suggest_kwargs"`: dictionary of keyword arguments (except `name` - as
            this one is inferred from the key) to be passed to the suggest method (e.g.,
            `{"low": 0.0, "high": 1.0}` for `"float"`).

    tune_rnd_seed: int or np.random.RandomState or None = None
        Random seed number for the sampler of hyperparameter values during the optimised
        hyperparameter tuning. If not defined, no such number is regarded.

    tune_n_trials: int, default=100
        Number of trials to perform in the optimised hyperparameter tuning.

    tune_n_jobs: int, default=1
        Number of simultaneous multiple processes to be considered in the optimised
        hyperparameter tuning. Note that if negative, one has the following conditions:
            - `-1`: all processors are used;
            - `-k`: all processors except k-1 are used.

    hparam_rnd_seed: int or np.random.RandomState or None = None
        Random seed number for the sampler of hyperparameter values during tuning. If
        not defined, no such number is regarded.

    path_out : Path or None, default=None
        The directory path to save the tuned downscaler, the obtained scores, and the
        data batcher. If not issued, the results are instead returned.

    out_data_batcher : bool, default=True
        Whether to return or write (if `path_out` is issued) the data batcher.

    log_mode : {None, "console", "file", "both"}, default="both"
        The logging mode for wrangling, batching, tuning, training, testing and writing:
            - `None`: No logging is done;
            - `"console"`: Logging is done to console only;
            - `"file"`: Logging is done to a log file only;
            - `"both"`: Logging is done to both console and a log file. Note the log
            file would be defined as `tune.log` at `out_dir`.
    """

    params_tune_getter: Callable[
        [optuna.trial.Trial | optuna.trial.FrozenTrial], dict[str, Any]
    ]
    data_batcher: DataBatcher | Path | None = None
    data_batcher_data_wrangler_path_sentinel3: Path | None = None
    data_batcher_data_wrangler_path_spatial_pred: Path | None = None
    data_batcher_data_wrangler_aoi: str | Path | None = None
    data_batcher_data_wrangler_path_landsat: Path | None = None
    data_batcher_data_wrangler_vars: list[str] | None = None
    data_batcher_data_wrangler_max_workers: int = 1
    data_batcher_n_cross_val_folds: int = 5
    data_batcher_var_cross_val_strat: str | None = None
    data_batcher_rnd_seed: int | np.random.RandomState | None = None
    downscaler: Downscaler | Path | None = None
    downscaler_base_model: Regressor = field(default_factory=lambda: LinearRegression())
    downscaler_X: list[str] = field(default_factory=lambda: ["FVC", "NDWI"])
    downscaler_masks: list[str] | None = None
    downscaler_scale: Literal["standardize", "min_max_normalize"] | None = "standardize"
    downscaler_encode: Literal["one_hot", "dummy"] | None = "dummy"
    downscaler_transform: Literal["center", "standardize"] | None = None
    downscaler_lasso_sel: bool = False
    downscaler_lasso_alpha: float = 1.0
    downscaler_max_workers: int = 1
    timestamps: list[pd.Timestamp] | list[str] | None = None
    sample_weight_fit: str | None = None
    sample_weight_score: str | None = None
    scorers: list[str] = field(default_factory=lambda: ["r2", "rmse", "mae", "mbe"])
    best_scorer: str = "rmse"
    correct: bool = True
    tune_rnd_seed: int | np.random.RandomState | None = None
    tune_n_trials: int = 100
    tune_n_jobs: int = 1
    path_out: Path | None = None
    out_data_batcher: bool = True
    log_mode: Literal["console", "file", "both"] | None = "both"


# Configurations for tuning
config = TuneConfig(
    # data_batcher=Path(
    #     Path(__file__).resolve().parents[4]
    #     / "assets/results/tune_trial/data_batcher.joblib"
    # ),
    data_batcher_data_wrangler_path_sentinel3=(
        Path(__file__).resolve().parents[4] / "assets/data/processed/sentinel3"
    ),
    data_batcher_data_wrangler_path_spatial_pred=(
        Path(__file__).resolve().parents[4]
        / "assets/data/processed/fixed_predictors.nc"
    ),
    data_batcher_data_wrangler_aoi="POLYGON ((10 55, 11 55, 11 56, 10 56, 10 55))",
    data_batcher_data_wrangler_path_landsat=(
        Path(__file__).resolve().parents[4] / "assets/data/processed/landsat"
    ),
    data_batcher_data_wrangler_vars=[
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
    data_batcher_data_wrangler_max_workers=5,
    data_batcher_n_cross_val_folds=5,
    data_batcher_var_cross_val_strat="season",
    data_batcher_rnd_seed=42,
    # downscaler=(
    #     Path(__file__).resolve().parents[4]
    #     / "assets/results/tune_trial/downscaler.joblib"
    # ),
    # downscaler=Downscaler(
    #     base_model=MLPRegressor(
    #         random_state=42,
    #         batch_size=1024,
    #         activation="relu",
    #         solver="adam",
    #         beta_1=0.9,
    #         beta_2=0.999,
    #         epsilon=1e-08,
    #         verbose=False,
    #         shuffle=True,
    #         max_iter=1000,
    #         early_stopping=True,  # NOTE: R2 is used has validation scorer
    #         validation_fraction=0.2,
    #         n_iter_no_change=10,
    #         tol=1e-3,
    #     ),
    #     cols_X=["FVC", "NDWI", "COASTDIST", "IMD", "TCD"],
    #     cols_mask=["aoi"],
    #     scale="standardize",
    #     encode="dummy",
    #     max_workers=5,
    #     transform="standardize",  # type: ignore
    # ),
    downscaler_base_model=MLPRegressor(
        random_state=42,
        batch_size=1024,
        activation="relu",
        solver="adam",
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-08,
        verbose=False,
        shuffle=True,
        max_iter=1000,
        early_stopping=True,  # NOTE: R2 is used has validation scorer
        validation_fraction=0.2,
        n_iter_no_change=10,
        tol=1e-3,
    ),
    downscaler_X=["FVC", "NDWI", "COASTDIST", "IMD", "TCD"],
    # WARNING: downscaler_masks takes effect regardless of the downscaler being issued
    # or created from scratch.
    downscaler_masks=["aoi"],
    downscaler_scale="standardize",
    downscaler_encode="dummy",
    downscaler_transform="standardize",
    downscaler_lasso_sel=False,
    downscaler_lasso_alpha=1.0,
    # WARNING: downscaler_max_workers takes effect regardless of the downscaler being
    # issued or created from scratch.
    downscaler_max_workers=5,
    # sample_weight_fit="IMD",
    # sample_weight_score="IMD",
    scorers=["r2", "rmse", "rmse_delta", "mae", "mae_delta", "mbe"],
    best_scorer="rmse",
    correct=True,
    params_tune_getter=params_tune_getter,
    tune_rnd_seed=42,
    tune_n_trials=2,
    tune_n_jobs=10,
    path_out=Path(__file__).resolve().parents[4] / "assets/results/tune_trial",
    out_data_batcher=True,
    log_mode="both",
)
