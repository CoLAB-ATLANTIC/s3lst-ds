# ---> Import packages
import logging
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from typing import Any, TypedDict

import joblib
import optuna
import pandas as pd
from rich.table import Table
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor

from s3lst_ds.data_batching.data_batching import DataBatcher
from s3lst_ds.data_wrangling.data_wrangling import DataWrangler
from s3lst_ds.downscaling.downscaling import Downscaler
from s3lst_ds.downscaling.estimation import DownscalerEstimator
from s3lst_ds.downscaling.tune.tune_config import (
    TuneConfig,
    config,
)
from s3lst_ds.utilities.exceptions_utils import (
    DataBatchingError,
    DataWranglingError,
    ReadingError,
    ScoringError,
    TrainingError,
    TuningError,
    WritingError,
)
from s3lst_ds.utilities.logging_utils import (
    RichLogger,
    get_rich_text_from_renderable,
)
from s3lst_ds.utilities.metrics import get_scorer
from s3lst_ds.utilities.tqdm_utils import tqdm
from s3lst_ds.utilities.var_utils import DataVars

# ----- Function for training downscaler with the whole cross-validation ----- #


def train(
    downscaler: Downscaler,
    data_batcher: DataBatcher,
    sample_weight: str | None = None,
    logger: RichLogger | None = None,
) -> Downscaler:
    """

    Train the `downscaler` with the respective data from `data_batcher` using
    `sample_weight` in the cost function.

    Parameters
    ----------

    downscaler : Downscaler or Path or None, default=None
        Downscaler to be trained.

    data_batcher : DataBatcher or Path or None, default=None
        Data batcher containing the training data.

    sample_weight: str or None, default=None
        Alias of the variable to be regarded as sample weight in the cost function for
        training. If not issued, no sample weight is considered.

    logger : RichLogger or None, default=None
        A rich logger for showing progress.

    Returns
    -------
    downscaler : Downscaler
        The trained downscaler.
    """

    # Train the downscaler with the training coarse data
    if logger is not None:
        logger.console.print()
        logger.info("The downscaler will now be trained.")

    try:
        with (
            logger.console.status(
                f"{'':7}Training the downscaler[yellow]...[/yellow]",
                spinner="dots",
                spinner_style="bold blue",
            )
            if logger is not None
            else nullcontext()
        ):
            # Train the downscaler
            downscaler = downscaler.fit(
                X_and_mask_coarse=data_batcher.get_data_X_and_mask(
                    batch="train",
                    grid="coarse",
                    trans=True,
                    aggregate=True,
                ),  # type: ignore
                y_coarse=data_batcher.get_data_y(
                    batch="train",
                    grid="coarse",
                    trans=True,
                    aggregate=True,
                ),  # type: ignore
                sample_weight=(
                    data_batcher.get_data(
                        batch="train",
                        grid="coarse",
                        vars=sample_weight,
                        trans=False,
                        aggregate=True,
                    )
                    if sample_weight is not None
                    else None
                ),  # type: ignore
            )

    except Exception as e:  # noqa: BLE001
        if logger is not None:
            logger.error(
                "[bold red]Error training the downscaler."
                + f"\nError message: {e}"
                + "\nRun will stop.[/bold red]",
            )
        raise TrainingError("Error training the downscaler." + f"\nError message: {e}")

    if logger is not None:
        logger.info("[bold green]Downscaler trained.[/bold green]")

    return downscaler


# --------------- Function for scoring in training and testing --------------- #


def score_train_test(
    downscaler: Downscaler,
    data_batcher: DataBatcher,
    sample_weight: str | None = None,
    scorers: list[str] | None = None,
    correct: bool = True,
    logger: RichLogger | None = None,
) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    """

    Score the `downscaler` on coarse training and coarse and fine test data from
    `data_batcher` using the issued `scorers` and `sample_weight`. In the case of
    training, solely Sentinel-3 data is used as ground-truth. In the case of coarse
    testing both Sentinel-3 and Landsat data are both used as ground-truth, while for
    the case of fine testing, solely the latter is used. Residual correction is
    considered in fine prediction if `correct` is set to `True`.

    Parameters
    ----------

    downscaler : Downscaler or Path or None, default=None
        Downscaler to be scored.

    data_batcher : DataBatcher or Path or None, default=None
        Data batcher containing the training and test data.

    sample_weight: str or None, default=None
        Alias of the variable to be regarded as sample weight for scoring the
        downscaler. If not issued, no sample weight is considered.

    scorers : list[str], default=["r2", "rmse", "mae", "mbe"]
        Aliases of the scorers to consider.

    correct : bool, default=True
        Whether to perform residual correction in fine prediction.

    logger : RichLogger or None, default=None
        A rich logger for showing progress.

    Returns
    -------
    score : dict[str, dict[str, dict[str, dict[str, float]]]]
        Training and test scores firstly keyed by batch (`"train"` or `"test"`),
        secondly by grid (`"coarse"` or `"fine"`), thirdly by ground-truth satellite
        (`"sentinel"` or `"landsat"`) and finally by scorer alias (`"r2"`, `"rmse"`,
        `"mae"`, `"mbe"`, etc.).
    """

    score = {}
    for batch in ["train", "test"]:
        for grid in ["coarse", "fine"]:
            for ground_truth in ["sentinel", "landsat"]:
                try:
                    if (
                        # Score donwscaler using Sentinel-3 data as ground truth
                        # if the grid is coarse
                        (grid == "coarse" and ground_truth == "sentinel")
                        # Score downscaler using Landsat data for both grids as
                        # ground truth in testing if such data is available
                        or (
                            batch == "test"
                            and ground_truth == "landsat"
                            and "test" in data_batcher.metadata["batch"].values
                        )
                    ):
                        if logger is not None:
                            logger.console.print()
                            logger.info(
                                "The downscaler will now be scored with respect to the"
                                f" {grid}"
                                f" {'training' if batch == 'train' else 'test'}"
                                f" data using"
                                f" {
                                    'Sentinel-3'
                                    if ground_truth == 'sentinel'
                                    else 'Landsat 8/9'
                                }"
                                " as ground truth."
                            )

                        if batch not in score:
                            score[batch] = {}
                        if grid not in score[batch]:
                            score[batch][grid] = {}

                        score[batch][grid][ground_truth] = (
                            # Coarse-score if grid is coarse
                            downscaler.score_coarse(
                                X_and_mask_coarse=data_batcher.get_data_X_and_mask(
                                    batch=batch,
                                    grid="coarse",
                                    trans=True,
                                    aggregate=False,
                                ),  # type: ignore
                                y_coarse=(
                                    # Use Sentinel-3 data as ground truth if
                                    # that is the case
                                    data_batcher.get_data_y(
                                        batch=batch,
                                        grid="coarse",
                                        trans=False,
                                        aggregate=False,
                                    )
                                    if ground_truth == "sentinel"
                                    # Use Landsat data as ground truth if that
                                    # is the case
                                    else data_batcher.get_data(
                                        batch=batch,
                                        grid="coarse",
                                        vars=data_batcher.data_wrangler.data_vars.y_val,
                                        trans=False,
                                        aggregate=False,
                                    )
                                ),  # type: ignore
                                aggregate=True,
                                scorers=scorers,
                                sample_weight=(
                                    data_batcher.get_data(
                                        batch=batch,
                                        grid="coarse",
                                        vars=sample_weight,
                                        trans=False,
                                        aggregate=False,
                                    )
                                    if sample_weight is not None
                                    else None
                                ),  # type: ignore
                            )
                            if grid == "coarse"
                            # Fine-score if grid is fine
                            else downscaler.score(
                                X_and_mask_fine=data_batcher.get_data_X_and_mask(
                                    batch=batch,
                                    grid="fine",
                                    trans=True,
                                    aggregate=False,
                                ),  # type: ignore
                                y_fine=data_batcher.get_data(
                                    batch=batch,
                                    grid="fine",
                                    vars=data_batcher.data_wrangler.data_vars.y_val,
                                    trans=False,
                                    aggregate=False,
                                ),  # type: ignore
                                correct=correct,
                                X_and_mask_coarse=data_batcher.get_data_X_and_mask(
                                    batch=batch,
                                    grid="coarse",
                                    trans=True,
                                    aggregate=False,
                                ),  # type: ignore
                                y_coarse=data_batcher.get_data_y(
                                    batch=batch,
                                    grid="coarse",
                                    trans=False,
                                    aggregate=False,
                                ),  # type: ignore
                                coords_coarse=data_batcher.get_coords(
                                    batch=batch,
                                    grid="coarse",
                                ),  # type: ignore
                                coords_fine=data_batcher.get_coords(
                                    batch=batch,
                                    grid="fine",
                                ),  # type: ignore
                                aggregate=True,
                                scorers=scorers,
                                sample_weight=(
                                    data_batcher.get_data(
                                        batch=batch,
                                        grid="fine",
                                        vars=sample_weight,
                                        trans=False,
                                        aggregate=False,
                                    )
                                    if sample_weight is not None
                                    else None
                                ),  # type: ignore
                            )
                        )
                        if logger is not None:
                            logger.info(
                                f"[bold green]Downscaler scored with respect to {grid}"
                                f" {'training' if batch == 'train' else 'test'}"
                                " data using"
                                f" {
                                    'Sentinel-3'
                                    if ground_truth == 'sentinel'
                                    else 'Landsat 8/9'
                                }"
                                " as ground truth.[/bold green]"
                            )

                except Exception as e:  # noqa: BLE001
                    if logger is not None:
                        logger.error(
                            "[bold red]Error scoring the downscaler with respect to"
                            f" {grid}"
                            f" {'training' if batch == 'train' else 'test'} data"
                            " using"
                            f" {
                                'Sentinel-3'
                                if ground_truth == 'sentinel'
                                else 'Landsat 8/9'
                            }"
                            " as ground truth."
                            f"\nError message: {e}"
                            "\nRun will stop.[/bold red]",
                        )
                    raise ScoringError(
                        f"Error scoring the downscaler with respect to {grid}"
                        f" {'training' if batch == 'train' else 'test'} data"
                        " using"
                        f" {
                            'Sentinel-3'
                            if ground_truth == 'sentinel'
                            else 'Landsat 8/9'
                        }"
                        " as ground truth."
                        f"\nError message: {e}"
                    )

    return score


# -------------- Function for getting tuning objective function -------------- #


def get_objective(
    estimator: DownscalerEstimator,
    data_batcher: DataBatcher,
    params_tune_getter: Callable[
        [optuna.trial.Trial | optuna.trial.FrozenTrial], dict[str, Any]
    ],
    sample_weight: str | None = None,
    best_scorer: str = "rmse",
) -> Callable[[optuna.trial.Trial | optuna.trial.FrozenTrial], float]:
    """
    Get objective function to use in the optimized hyperparameter tuning method
    `optuna.study.Study.optimize()` for the issued `estimator` considering the
    cross-validation data from `data_batcher` and tunable hyperparameter getter
    `params_tune_getter`. Fitting and scoring of the estimator is further performed
    considering the issued `sample_weight`. The best hyperparameter combination is
    selected based on the issued `best_scorer`.

    Parameters
    ----------

    estimator : DownscalerEstimator
        Downscaler estimator to tune.

    data_batcher : DataBatcher
        Data batcher containing the cross-validation data to use in the tuning.

    params_tune_getter: Callable[[optuna.trial.Trial or optuna.trial.FrozenTrial],
    dict[str, Any]]
        Callable that takes an `optuna.trial.Trial` or `optuna.trial.FrozenTrial` object
        and returns a dictionary of suggested (tunable) hyperparameters for the
        downscaler estimator. The keys must correspond to the hyperparameters' full
        access paths with each step separated by double underscores (e.g.
        `"base_model__formula"` in which `"base_model"` is a parameter of the downscaler
        estimator and `"formula"` is a parameter of the former).

    sample_weight_fit: str or None, default=None
        Alias of the variable to be regarded as sample weight for both training and
        scoring in the tuning. If not issued, no sample weight is considered.

    best_scorer : str, default="rmse"
        Alias of the metric to consider in the selection of the best hyperparameter
        combination.

    Returns
    -------

    objective : Callable[[optuna.trial.Trial or optuna.trial.FrozenTrial], float]
        Objective function to use in `optuna.study.Study.optimize()`.
    """

    def objective(trial: optuna.trial.Trial | optuna.trial.FrozenTrial) -> float:
        """
        Objective function to use in `optuna.study.Study.optimize()`.

        Parameters
        ----------

        trial : optuna.trial.Trial or optuna.trial.FrozenTrial

        Returns
        -------

        score : cross-validation score of the estimator for hyperparameters suggested by
        `trial`.
        """

        # Get suggested (tunable) hyperparameters
        params_tune = params_tune_getter(trial)

        # Set estimator hyperparameters
        estimator.set_params(**params_tune)

        # Cross-validate the model using the best scorer
        # NOTE: GridSearchCV is used instead of cross_validate as the latter cannot make
        # the score method to use sample weights (though it can make the fit one).
        score = (
            GridSearchCV(
                estimator=estimator,
                param_grid={},
                cv=data_batcher.get_cv(
                    data_batcher.get_data(
                        batch="cross_val",
                        grid="coarse",
                        vars="batch",
                        trans=True,
                        aggregate=True,
                    )  # type: ignore
                ),
                n_jobs=1,
                refit=False,
                return_train_score=False,
                verbose=0,
            )
            .fit(
                X=data_batcher.get_data_X_and_mask(
                    batch="cross_val",
                    grid="coarse",
                    trans=True,
                    aggregate=True,
                ),  # type: ignore
                y=data_batcher.get_data_y(
                    batch="cross_val",
                    grid="coarse",
                    trans=True,
                    aggregate=True,
                ),  # type: ignore
                sample_weight=(
                    data_batcher.get_data(
                        batch="cross_val",
                        grid="coarse",
                        vars=sample_weight,
                        aggregate=True,
                    )
                    if sample_weight is not None
                    else None
                ),
            )
            .cv_results_[f"mean_test_{best_scorer}"][0]
        )

        return score

    return objective


# ---------------------- Hyperparameter tuning function ---------------------- #


# Class for output of hyperparameter tuning
# NOTE: Since this class is inherinted from TypedDict, any of its instances would be
# regarded as a dict object at runtime.
class TuneOut(TypedDict, total=False):
    downscaler: Downscaler | Path
    params: dict[str, Any]
    score: dict[str, dict[str, dict[str, dict[str, float]]]] | Path
    data_batcher: DataBatcher | Path


def tune(
    config: TuneConfig,
) -> TuneOut:
    """
    Wrangle the data of timestamps of interest, batch it into cross-validation and test
    datasets, perform an optimized cross-validated hyperparameter tuning of a
    multi-timestamp downscaling model and subsequently train and test it. Return or
    write the results to files.

    Batching is done with respect to the issued `timestamps`, and, therefore, the whole
    data associated with a timestamp (a scene) is fully contained within a single batch.
    The resulting batch datasets are:
        - `"test"` dataset: timestamps for which there is Landsat data.
        - `"cross_val"` dataset: random split of the remaining timestamps into
        `config.n_cross_val_folds` folds, stratified with respect to a categorical
        variable `config.var_cross_val_strat` (if issued).

    The optimized cross-validated hyperparameter tuning is done using
    [`optuna`](https://optuna.readthedocs.io/en/stable/) by trying suggested
    hyperparameter values within the search space defined in the issued
    `config.params_tune_getter`. The estimator of the downscaler (the downscaler before
    de-transformation and residual correction) is cross-validated for each suggested
    hyperparameter combination and the best one is selected based on the issued scoring
    metric `config.best_scorer`. Cross-validation is done by training the downscaler
    estimator on all cross-validation folds except one and scoring it on the latter,
    rotating the scoring fold until all are considered. The overall cross-validation
    score is computed from the arithmetic mean of the scores of each iteration.

    Training of the tuned downscaler is done using the whole cross-validation data.

    Testing is done by scoring the trained downscaler on the test dataset (for both
    Sentinel-3 and Landsat data).

    WARNING: Note that for the sake of efficiency, during tuning, solely tuner-specific
    multiprocessing (set through `config.tune_n_jobs` parameter) is considered . Base
    model-specific multi-processing (set through the respective `n_jobs` parameter) is
    subsequently considered. Note that for the case of the MLPRegressor,
    `config.tune_n_jobs` is forcefully set to `1`, regardless of the issued value, as
    the MLPRegressor, by default, always use all available processors - and, therefore,
    a value of `config.tune_n_jobs` greater than `1` would impair the process.

    WARNING: Note that the units of the computed cross-validation score in the
    hyperparameter tuning are based on the ones of the transformed target (whose
    transform corresponds to the one set in the downscaler). For example, if the
    transform corresponds to `"standardize"` and `best_scorer` is set to `"rmse"`, the
    computed score corresponds to the RMSE of the standardized target, which is
    unitless.

    Parameters
    ----------

    config: TuneConfig
        Configurations for wrangling, batching, tuning, training, testing and writing.

    Returns
    -------
    dict
        Dictionary containing:
        - downscaler: Downscaler or Path
            The tuned downscaler (if `config.path_out` is not issued) or a path to the
            respective Joblib file.
        - params: dict[str, Any]
            The tuned hyperparameters of the downscaler (fixed hyperparameters are
            disregarded here).
        - score: dict[str, dict[str, dict[str, dict[str, float]]]] or Path
            Cross-validation, training and test scores either as a dictionary (if
            `config.path_out` is not issued) or as a path to the respective JSON file.
            The scores are keyed by batch ("cross_val", "train" or "test"), grid
            ("coarse" or "fine"), ground truth dataset ("sentinel" or "landsat") and
            metric ("r2", "rmse", etc.). Scores using Landsat ground truth data are
            solely computed in testing and if such data exists for the timestamps of
            interest (the Landsat data must be included in the issued
            `config.data_batcher` or in `config.data_batcher_path_landsat` if no data
            batcher is issued).
        - data_batcher: DataBatcher or Path
            The data batcher (if `config.path_out` is not issued) or a path to the
            respective Joblib file. The data batcher also contains the batched data and
            may be useful for debugging or for reusing it without need for reprocessing
            the original one. This only takes effect if parameter
            `config.out_data_batcher` is `True`.
    """

    # ---> Handle logging

    # Create logger
    logger = RichLogger(
        name="tune",
        level=logging.INFO,
        file_path=(
            Path(config.path_out) / "tune.log" if config.path_out is not None else None
        ),
        file_mode="w",
        log_mode=config.log_mode,
    )

    # Redirect optuna logs to the logger
    optuna_logger = optuna.logging.get_logger("optuna")
    optuna_logger.handlers.clear()
    for handler in logger.base_logger.handlers:
        optuna_logger.addHandler(handler)

    # Print status message
    logger.console.print()
    logger.info("[bold]Tuning multi-timestamp downscaler[/bold]")

    # ---> Create output directory
    if config.path_out is not None:
        try:
            Path(config.path_out).mkdir(parents=True, exist_ok=True)

        except Exception as e:  # noqa: BLE001
            logger.error(
                "[bold red]Error creating the output directory."
                + f"\nError message: {e}"
                + "\nRun will stop.[/bold red]",
            )
            raise WritingError(
                "Error creating the output directory." + f"\nError message: {e}"
            )

    # ---> Get downscaler if it is issued

    # Load downscaler from file if a path is issued
    if isinstance(config.downscaler, Path):
        logger.console.print()
        logger.info("The downscaler will now be loaded from file.")
        try:
            with logger.console.status(
                f"{'':7}Loading downscaler from file[yellow]...[/yellow]",
                spinner="dots",
                spinner_style="bold blue",
            ):
                downscaler = joblib.load(config.downscaler)

        except Exception as e:  # noqa: BLE001
            logger.error(
                "[bold red]Error loading the downscaler from file."
                + f"\nError message: {e}"
                + "\nRun will stop.[/bold red]",
            )
            raise ReadingError(
                "Error loading the downscaler from file." + f"\nError message: {e}"
            )

        if not isinstance(downscaler, Downscaler):
            logger.error(
                "[bold red]The loaded downscaler is not a (multi-timestamp) Downscaler"
                " object." + "\nRun will stop.[/bold red]",
            )
            raise TypeError(
                "The loaded downscaler is not a (multi-timestamp) Downscaler object."
            )
        logger.info("[bold green]Downscaler loaded from file.[/bold green]")

    # Set downscaler if provided directly
    elif isinstance(config.downscaler, Downscaler):
        downscaler = config.downscaler

    # ---> Get data batcher if it is issued
    # Load data batcher from file if a path is issued
    if isinstance(config.data_batcher, Path):
        logger.console.print()
        logger.info("The data batcher will now be loaded from file.")
        try:
            with logger.console.status(
                f"{'':7}Loading data batcher from file[yellow]...[/yellow]",
                spinner="dots",
                spinner_style="bold blue",
            ):
                data_batcher = joblib.load(config.data_batcher)

        except Exception as e:  # noqa: BLE001
            logger.error(
                "[bold red]Error loading the data batcher from file."
                + f"\nError message: {e}"
                + "\nRun will stop.[/bold red]",
            )
            raise ReadingError(
                "Error loading the data batcher from file." + f"\nError message: {e}"
            )

        logger.info("[bold green]Data batcher loaded from file.[/bold green]")

    # Set data batcher if provided directly
    elif isinstance(config.data_batcher, DataBatcher):
        data_batcher = config.data_batcher

    # ---> Parse parameters

    # Parse downscaler predictors
    downscaler_X = (
        downscaler.cols_X if config.downscaler is not None else config.downscaler_X
    )

    # Parse downscaler transform
    downscaler_transform = (
        downscaler.transform  # type: ignore
        if config.downscaler is not None
        else config.downscaler_transform
    )

    # Parse path to Landsat data
    # NOTE: Landsat data will be solely used in coarse and fine test-scoring if it was
    # issued or already contained in an issued data batcher.
    data_batcher_data_wrangler_path_landsat = (
        data_batcher.data_wrangler.path_landsat
        if config.data_batcher is not None
        else config.data_batcher_data_wrangler_path_landsat
    )

    # Parse data wrangling variables
    if config.data_batcher is not None:
        data_batcher_data_wrangler_data_vars = data_batcher.data_wrangler.data_vars
    else:
        data_batcher_data_wrangler_vars = (
            config.data_batcher_data_wrangler_vars
            if config.data_batcher_data_wrangler_vars is not None
            else (
                downscaler.cols_X
                if config.downscaler is not None
                else config.downscaler_X
                if config.downscaler_X is not None
                else None
            )
        )
        data_batcher_data_wrangler_data_vars = DataVars().subset_X(
            data_batcher_data_wrangler_vars  # type: ignore
        )

    # Parse timestamps
    timestamps = (
        [
            pd.Timestamp(timestamp)
            if not isinstance(timestamp, pd.Timestamp)
            else timestamp
            for timestamp in config.timestamps
        ]
        if config.data_batcher is None and config.timestamps is not None
        else (
            data_batcher.data_wrangler.timestamps
            if config.data_batcher is not None
            else [
                pd.Timestamp(data_batcher_data_wrangler_path_sentinel3_folder.name)
                for data_batcher_data_wrangler_path_sentinel3_folder in config.data_batcher_data_wrangler_path_sentinel3.iterdir()  # type: ignore
                if data_batcher_data_wrangler_path_sentinel3_folder.is_dir()
            ]
        )
    )

    # Parse indicators for outputting variables
    out = {
        "downscaler": True,
        "params": True,
        "score": True,
        "data_batcher": config.out_data_batcher,
    }

    # Parse output paths
    path_out = {
        object_alias: (
            (
                config.path_out
                / (
                    object_alias
                    + (
                        ".joblib"
                        if object_alias not in ["params", "score"]
                        else ".json"
                    )
                )
            )
            if config.path_out is not None and out[object_alias] is True
            else None
        )
        for object_alias in out
    }

    # ---> Update parameters of the downscaler and data wrangler if they had been issued

    # Set logger
    if config.data_batcher is not None:
        data_batcher.data_wrangler.logger = logger
    if config.downscaler is not None:
        downscaler.logger = logger

    # Update masking variables and maximum number of workers of the downscaler if it had
    # been issued
    if config.downscaler is not None:
        downscaler.cols_mask = config.downscaler_masks
        downscaler.max_workers = config.downscaler_max_workers

    # Update transform (with the one of the downscaler, if it is different from the one
    # of the data wrangler) and maximum number of workers of the data wrangler if it had
    # been issued
    if config.data_batcher is not None:
        if data_batcher.data_wrangler.transform != downscaler_transform:  # type: ignore
            data_batcher.data_wrangler.transform = downscaler_transform  # type: ignore
        data_batcher.data_wrangler.max_workers = (
            config.data_batcher_data_wrangler_max_workers
        )

    # ---> Wrangle and batch the data if a data batcher was not issued
    if config.data_batcher is None:
        logger.console.print()
        logger.info("The data will now be wrangled.")
        try:
            # Define a data wrangler
            data_wrangler = DataWrangler(
                data_vars=data_batcher_data_wrangler_data_vars,
                path_sentinel3=config.data_batcher_data_wrangler_path_sentinel3,  # type: ignore
                path_spatial_pred=config.data_batcher_data_wrangler_path_spatial_pred,  # type: ignore
                aoi=config.data_batcher_data_wrangler_aoi,
                path_landsat=data_batcher_data_wrangler_path_landsat,
                timestamps=timestamps,
                transform=downscaler_transform,  # type: ignore
                max_workers=config.data_batcher_data_wrangler_max_workers,
                logger=logger,
            )

        except Exception as e:  # noqa: BLE001
            logger.error(
                "[bold red]Error wrangling the data."
                + f"\nError message: {e}"
                + "\nRun will stop.[/bold red]",
            )
            raise DataWranglingError(
                "Error wrangling the data." + f"\nError message: {e}"
            )

        logger.info("[bold green]Data wrangled.[/bold green]")

        logger.console.print()
        logger.info("The data will now be batched.")
        try:
            with logger.console.status(
                f"{'':7}Batching the data[yellow]...[/yellow]",
                spinner="dots",
                spinner_style="bold blue",
            ):
                # Define a data batcher
                data_batcher = DataBatcher(
                    data_wrangler=data_wrangler,
                    n_cross_val_folds=config.data_batcher_n_cross_val_folds,
                    var_cross_val_strat=config.data_batcher_var_cross_val_strat,
                    rnd_seed=config.data_batcher_rnd_seed,
                )

        except Exception as e:  # noqa: BLE001
            logger.error(
                "[bold red]Error batching the data."
                + f"\nError message: {e}"
                + "\nRun will stop.[/bold red]",
            )
            raise DataBatchingError(
                "Error batching the data." + f"\nError message: {e}"
            )

        logger.info("[bold green]Data batched.[/bold green]")

    # ---> Tune the downscaler
    logger.console.print()
    logger.info("The downscaler will now be tuned.")
    try:
        # Define a downscaler if it had not been issued
        if config.downscaler is None:
            downscaler = Downscaler(
                base_model=config.downscaler_base_model,
                cols_X=downscaler_X,
                cols_mask=config.downscaler_masks,
                scale=config.downscaler_scale,
                encode=config.downscaler_encode,
                max_workers=config.downscaler_max_workers,
                transform=downscaler_transform,  # type: ignore
                logger=logger,
            )

        # Set base model processors to 1 during tuning to not impair the process (as
        # multiple processes may already be used by the tuner itself)
        if "base_model__n_jobs" in downscaler.get_params():
            base_model__n_jobs = downscaler.base_model.n_jobs  # type: ignore
            downscaler.set_params(base_model__n_jobs=1)

        # Define optimization task
        # NOTE:
        # https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.create_study.html#optuna.study.create_study
        study = optuna.create_study(
            study_name="hparam_tuning",
            # Optimization direction
            direction=(
                "minimize"
                if get_scorer(config.best_scorer).greater_is_better is False
                else "maximize"
            ),
            # Method for suggesting new hyperparameter values
            # NOTE: TPESampler (Tree-structured Parzen Estimator) uses past results
            # to guide future trials (see
            # https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html#optuna-samplers-tpesampler).
            sampler=optuna.samplers.TPESampler(seed=config.tune_rnd_seed),  # type: ignore
        )

        # Define progress bar and callback for optuna's optimizer
        logger.info("Performing tuning trials...")
        pbar = tqdm(
            # Prefix for the progressbar
            bar_format=f"{'':9}" + "{l_bar}{bar}{r_bar}",
            desc=f"{'':8}",
            total=config.tune_n_trials,  # type: ignore
            unit="trial",
            position=0,
            leave=True,  # Keep progress on the screen after completion.
            options={"console": logger.console},
        )

        def pbar_callback(
            study: optuna.study.Study, trial: optuna.trial.FrozenTrial
        ) -> None:
            """
            Callback function to update the progress bar after each trial.

            For more details about `optuna`'s callback functions, read [the
            documentation](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html#optuna.study.Study.optimize).
            """
            pbar.update()

        # Perform optimization task
        # NOTE:
        # https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html#optuna.study.Study.optimize
        study.optimize(
            func=get_objective(
                estimator=downscaler.estimator,
                data_batcher=data_batcher,
                params_tune_getter=config.params_tune_getter,
                sample_weight=config.sample_weight_fit,
                best_scorer=config.best_scorer,
            ),
            # Number of trials to perform
            n_trials=config.tune_n_trials,
            # Number of multiple processes
            # NOTE: https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/004_distributed.html#multi-thread-optimization
            # NOTE:  Note that for the case of the MLPRegressor, `tune_n_jobs` is
            # forcefully set to `1`, regardless of the issued value, as the
            # MLPRegressor, by default, always use all available processors - and,
            # therefore, a value of `tune_n_jobs` greater than `1` would impair the
            # process.
            n_jobs=(
                config.tune_n_jobs
                if not isinstance(downscaler.base_model, MLPRegressor)
                else 1
            ),
            # List of callback functions to be invoked at the end of each trial
            callbacks=[pbar_callback],
            show_progress_bar=False,
        )

        # Close progress bar
        pbar.close()

        # Get best trial number
        best_trial = study.best_trial.number

        # Get best score
        best_score = study.best_value

        # Get and set best hyperparameters
        best_params = config.params_tune_getter(study.best_trial)
        downscaler.set_params(**best_params)

        # Set base model number of processors to the issued one (since tuning is now
        # done and does not require more processors)
        if "base_model__n_jobs" in downscaler.get_params():
            downscaler.set_params(base_model__n_jobs=base_model__n_jobs)

    except Exception as e:  # noqa: BLE001
        logger.error(
            "[bold red]Error tuning the downscaler."
            + f"\nError message: {e}"
            + "\nRun will stop.[/bold red]",
        )
        raise TuningError("Error tuning the downscaler." + f"\nError message: {e}")

    logger.info(
        "[bold green]Downscaler tuned having as best hyperparameters"
        f"\n{best_params}"
        f"\nwhich were found at trial {best_trial} with cross-validation"
        f" {config.best_scorer} of {best_score:g}."
        "[/bold green]"
    )

    # ---> Retrain with the whole cross-validation data

    # Train the downscaler with the training data
    downscaler = train(
        downscaler=downscaler,
        data_batcher=data_batcher,
        sample_weight=config.sample_weight_fit,
        logger=logger,
    )

    # ---> Score the downscaler with respect to the training and test data
    score = {"cross_val": {"coarse": {"sentinel": {config.best_scorer: best_score}}}}
    score = score | score_train_test(
        downscaler=downscaler,
        data_batcher=data_batcher,
        sample_weight=config.sample_weight_fit,
        scorers=config.scorers,
        logger=logger,
    )

    # ---> Combine all the results in a dictionary

    object = {
        "downscaler": downscaler,
        "params": best_params,
        "score": score,
        "data_batcher": data_batcher,
    }

    # ---> Write downscaler, scores and data batcher to file if wanted
    for object_alias in object:  # noqa: PLC0206
        if path_out[object_alias] is not None:
            logger.console.print()
            logger.info(
                f"The {object_alias.replace('_', ' ')} will now be written to file."
            )
            try:
                with logger.console.status(
                    f"{'':7}Writing {object_alias.replace('_', ' ')} to file"
                    "[yellow]...[/yellow]",
                    spinner="dots",
                    spinner_style="bold blue",
                ):
                    if object_alias in ["params", "score"]:
                        pd.Series(object[object_alias]).to_json(
                            path_out[object_alias]  # type: ignore
                        )
                    else:
                        object[object_alias].save(path_out[object_alias])
            except Exception as e:  # noqa: BLE001
                logger.error(
                    f"[bold red]Error writing the {object_alias.replace('_', ' ')}"
                    " to file."
                    f"\nError message: {e}"
                    "\nRun will stop.[/bold red]",
                )
                raise WritingError(
                    f"Error writing the {object_alias.replace('_', ' ')} to file."
                    + f"\nError message: {e}"
                )

            logger.info(
                f"[bold green]{object_alias.replace('_', ' ').capitalize()} written"
                " to file.[/bold green]"
            )

    # ---> Show table with best hyperparameters
    table_hparams = Table(title="Best hyperparameters")
    table_hparams.add_column("Hyperparameter", justify="left")
    table_hparams.add_column("Value", justify="right")
    for param, value in best_params.items():
        table_hparams.add_row(param, f"{value}")
    logger.console.print()
    logger.info(
        "The downscaler attained the following best hyperparameters:"
        "\n\n"
        + f"{
            get_rich_text_from_renderable(
                console=logger.console,
                renderable=table_hparams,
            )
        }"
    )

    # ---> Show table with scores if they were computed
    if score is not None:
        table_score = Table(title="Metrics")
        table_score.add_column("Batch", justify="left")
        table_score.add_column("Grid", justify="left")
        table_score.add_column("Ground truth", justify="left")
        table_score.add_column("Metric", justify="left")
        table_score.add_column("Value", justify="right")
        for batch in score:
            for i, grid in enumerate(score[batch].keys()):
                for j, ground_truth in enumerate(score[batch][grid].keys()):
                    for k, scorer in enumerate(score[batch][grid][ground_truth].keys()):
                        table_score.add_row(
                            batch if i == 0 and j == 0 and k == 0 else None,
                            grid if j == 0 and k == 0 else None,
                            ground_truth if k == 0 else None,
                            scorer,
                            f"{score[batch][grid][ground_truth][scorer]}",
                        )

        logger.console.print()
        logger.info(
            "The downscaler attained the following metrics:"
            "\n\n"
            + f"{
                get_rich_text_from_renderable(
                    console=logger.console,
                    renderable=table_score,
                )
            }"
        )

    # ---> Return the results
    logger.console.print()
    return TuneOut(
        **{
            object_alias: (
                object if path_out[object_alias] is None else path_out[object_alias]
            )
            for object_alias, object in object.items()
            if out[object_alias] is True
        }  # type: ignore
    )


# --------------------------- Script's main funcion -------------------------- #
def main() -> TuneOut:  # type: ignore
    tune(config)


# If this very script is directly executed in the terminal (in that case the global
# variable __name__ corresponds to "__main__"), it runs function main()
if __name__ == "__main__":
    main()
