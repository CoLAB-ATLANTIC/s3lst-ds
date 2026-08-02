# ---> Import packages
import logging
from pathlib import Path
from typing import TypedDict

import joblib
import numpy as np
import pandas as pd
import xarray as xr
from rich.table import Table

from s3lst_downscale.data_wrangling.data_wrangling import DataWrangler
from s3lst_downscale.downscaling.downscale.downscale_config import (
    DownscaleConfig,
    config,
)
from s3lst_downscale.downscaling.downscaling import Downscaler
from s3lst_downscale.downscaling.piecewise_downscaling import PiecewiseDownscaler
from s3lst_downscale.utilities.exceptions_utils import (
    DataWranglingError,
    DownscalingError,
    ReadingError,
    ScoringError,
    TrainingError,
    WritingError,
)
from s3lst_downscale.utilities.logging_utils import (
    RichLogger,
    get_rich_text_from_renderable,
)
from s3lst_downscale.utilities.var_utils import DataVars

# --------------------------- Downscaling function --------------------------- #


# Class for output of downscaling
# NOTE: Since this class is inherinted from TypedDict, any of its instances would be
# regarded as a dict object at runtime.
class DownscaleOut(TypedDict, total=False):
    y_fine_pred: dict[pd.Timestamp, np.ndarray | xr.DataArray] | list[Path]
    score: dict[str, dict[str, dict[str, dict[str, float]]]] | Path
    downscaler: Downscaler | PiecewiseDownscaler | Path
    data_wrangler: DataWrangler | Path


def downscale(
    config: DownscaleConfig,
) -> DownscaleOut:
    """
    Wrangle the data of timestamps of interest, train a downscaler using the coarse data
    of the training timestamps, downscale the coarse data of the inference timestamps,
    score the downscaler, and return or write the results to files.

    Parameters
    ----------

    config: DownscaleConfig
        Configurations for wrangling, training, downscaling, scoring and writing.

    Returns
    -------
    dict
        Dictionary containing:
        - y_fine_pred: dict[pd.Timestamp, np.ndarray or xr.DataArray] or
        dict[pd.Timestamp, Path]
            The downscaled LST data for each inference timestamp, either as a dictionary
            of NumPy arrays (if parameter `config.path_out` is not issued and
            `config.gridded` is `False`) or Xarray DataArrays (if `config.path_out` is
            not issued and parameter `config.gridded` is `True`) or as a dictionary of
            paths to the respective files (if `config.path_out` is issued).
        - score: dict[str, dict[str, dict[str, dict[str, float]]]] or Path
            Prediction scores either as a dictionary (if `config.path_out` is not
            issued) or as a path to the respective JSON file. This only takes effect if
            `config.score` is `True`. The scores are keyed by batch ("train" or
            "infer"), grid ("coarse" or "fine"), ground truth dataset ("sentinel" or
            "landsat") and metric ("r2", "rmse", etc.). Scores using Landsat ground
            truth data are solely computed if Landsat data exists (included in the
            issued `config.data_wrangler` or in `config.data_wrangler_path_landsat` if
            no data wrangler is issued) and if such data is available for the considered
            training or inference timestamps. Training scores are solely computed if the
            downscaler is a multi-timestamp one and training is performed. In the case
            of the single-timestamp architecture, training scores do not need to be
            computed since such model is always trained with the coarse data from the
            same timestamp whose target it infers (and, thus, the training scores would
            coincide with the inference ones).
        - downscaler: Downscaler or PiecewiseDownscaler or Path
            The downscaler (if `config.path_out` is not issued) or a path to the
            respective Joblib file. This only takes effect if parameter
            `config.out_downscaler` is `True`.
        - data_wrangler: DataWrangler or Path
            The data wrangler (if `config.path_out` is not issued) or a path to the
            respective Joblib file. The data wrangler also contains the wrangled data
            and may be useful for debugging or for reusing it without need for
            reprocessing the original one. This only takes effect if parameter
            `config.out_data_wrangler` is `True`.
    """

    # ---> Handle logging

    # Create logger
    logger = RichLogger(
        level=logging.INFO,
        file_path=(
            Path(config.path_out) / "downscale.log"
            if config.path_out is not None
            else None
        ),
        file_mode="w",
        log_mode=config.log_mode,
    )

    logger.console.print()
    logger.info("[bold]Downscaling Sentinel-3 LST products[/bold]")

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
                f"{'':7}Loading downscaler from file...",
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

        if not isinstance(downscaler, (Downscaler, PiecewiseDownscaler)):
            logger.error(
                "[bold red]The loaded downscaler is neither (multi-timestamp)"
                " Downscaler nor a (single-timestamp) PiecewiseDownscaler object."
                + "\nRun will stop.[/bold red]",
            )
            raise TypeError(
                "The loaded downscaler is neither a (multi-timestamp) Downscaler nor a"
                " (single-timestamp) PiecewiseDownscaler object."
            )

        logger.info("[bold green]Downscaler loaded from file.[/bold green]")

    # Set downscaler if provided directly
    elif isinstance(config.downscaler, (Downscaler, PiecewiseDownscaler)):
        downscaler = config.downscaler

    # ---> Get data wrangler if it is issued
    # Load data wrangler from file if a path is issued
    if isinstance(config.data_wrangler, Path):
        logger.console.print()
        logger.info("The data wrangler will now be loaded from file.")
        try:
            with logger.console.status(
                f"{'':7}Loading data wrangler from file...",
                spinner="dots",
                spinner_style="bold blue",
            ):
                data_wrangler = joblib.load(config.data_wrangler)

        except Exception as e:  # noqa: BLE001
            logger.error(
                "[bold red]Error loading the data wrangler from file."
                + f"\nError message: {e}"
                + "\nRun will stop.[/bold red]",
            )
            raise ReadingError(
                "Error loading the data wrangler from file." + f"\nError message: {e}"
            )

        logger.info("[bold green]Data wrangler loaded from file.[/bold green]")

    # Set data wrangler if provided directly
    elif isinstance(config.data_wrangler, DataWrangler):
        data_wrangler = config.data_wrangler

    # ---> Parse parameters

    # Parse downscaler architecture
    downscaler_architecture = (
        type(downscaler)
        if config.downscaler is not None
        else (
            Downscaler
            if config.downscaler_architecture == "multi"
            else PiecewiseDownscaler
        )
    )

    # Parse downscaler predictors
    downscaler_X = (
        downscaler.cols_X if config.downscaler is not None else config.downscaler_X
    )

    # Parse downscaler transform
    # NOTE: in the case of the single-timestamp architecture, a timestamp-specific
    # transform of the spatio-temporal predictors is redundant and may be set to `None`.
    downscaler_transform = (
        downscaler.transform  # type: ignore
        if config.downscaler is not None and downscaler_architecture == Downscaler
        else (
            config.downscaler_transform
            if downscaler_architecture == Downscaler
            else None
        )
    )

    # Parse path to Landsat data
    # NOTE: Landsat data will be solely used in coarse and fine-scoring if `score` is
    # `True` and if it was issued or already contained in an issued data wrangler.
    data_wrangler_path_landsat = (
        (
            data_wrangler.path_landsat
            if config.data_wrangler is not None
            else config.data_wrangler_path_landsat
        )
        if config.score is True
        else None
    )

    # Parse data wrangling variables
    if config.data_wrangler is not None:
        data_wrangler_data_vars = data_wrangler.data_vars
    else:
        data_wrangler_vars = (
            config.data_wrangler_vars
            if config.data_wrangler_vars is not None
            else (
                downscaler.cols_X
                if config.downscaler is not None
                else config.downscaler_X
                if config.downscaler_X is not None
                else None
            )
        )
        data_wrangler_data_vars = DataVars().subset_X(data_wrangler_vars)  # type: ignore

    # Parse training indicator
    train = bool(
        config.downscaler is not None
        and (
            config.retrain is True
            or downscaler_architecture == PiecewiseDownscaler
            or downscaler.is_fitted_ is False
        )
        or config.downscaler is None
    )

    # Parse timestamps
    timestamps = {"sentinel": {}}
    timestamps["sentinel"]["infer"] = (
        [pd.Timestamp(timestamp) for timestamp in config.timestamps_infer]
        if config.timestamps_infer is not None
        else (
            data_wrangler.timestamps
            if config.data_wrangler is not None
            else [
                pd.Timestamp(data_wrangler_path_sentinel3_folder.name)
                for data_wrangler_path_sentinel3_folder in config.data_wrangler_path_sentinel3.iterdir()  # type: ignore
                if data_wrangler_path_sentinel3_folder.is_dir()
            ]
        )
    )
    timestamps["sentinel"]["train"] = (
        [pd.Timestamp(timestamp) for timestamp in config.timestamps_fit]
        if config.timestamps_fit is not None and downscaler_architecture == Downscaler
        else (
            timestamps["sentinel"]["infer"]
            if downscaler_architecture == PiecewiseDownscaler
            else (
                data_wrangler.timestamps
                if config.data_wrangler is not None
                else [
                    pd.Timestamp(data_wrangler_path_sentinel3_folder.name)
                    for data_wrangler_path_sentinel3_folder in config.data_wrangler_path_sentinel3.iterdir()  # type: ignore
                    if data_wrangler_path_sentinel3_folder.is_dir()
                ]
            )
        )
    )

    # Parse indicators for outputting variables
    out = {
        "y_fine_pred": True,
        "score": config.score,
        "data_wrangler": config.out_data_wrangler,
        "downscaler": config.out_downscaler,
    }

    # Parse output paths
    path_out = {
        object_alias: (
            (
                # For the case of the fine target predictions
                {
                    timestamp: config.path_out
                    / (
                        "lst_downscaled_"
                        + timestamp.strftime("%Y%m%dT%H%M%S")
                        + (config.file_ext_grid if config.gridded is True else ".csv")
                    )
                    for timestamp in timestamps["sentinel"]["infer"]
                }
                if object_alias == "y_fine_pred"
                # For the case of scores, data wrangler and downscaler
                else config.path_out
                / (object_alias + (".joblib" if object_alias != "score" else ".json"))
            )
            if config.path_out is not None and out[object_alias] is True
            else None
        )
        for object_alias in out
    }

    # ---> Update parameters of the downscaler and data wrangler if they had been issued

    # Set logger
    if config.data_wrangler is not None:
        data_wrangler.logger = logger
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
    if config.data_wrangler is not None:
        if data_wrangler.transform != downscaler_transform:  # type: ignore
            data_wrangler.transform = downscaler_transform  # type: ignore
        data_wrangler.max_workers = config.data_wrangler_max_workers

    # ---> Wrangle the data if a data wrangler was not issued
    if config.data_wrangler is None:
        logger.console.print()
        logger.info("The data will now be wrangled.")
        try:
            # Define a data wrangler
            data_wrangler = DataWrangler(
                data_vars=data_wrangler_data_vars,
                path_sentinel3=config.data_wrangler_path_sentinel3,  # type: ignore
                path_spatial_pred=config.data_wrangler_path_spatial_pred,  # type: ignore
                aoi=config.data_wrangler_aoi,
                path_landsat=data_wrangler_path_landsat,
                timestamps=list(
                    set(timestamps["sentinel"]["train"])
                    | set(timestamps["sentinel"]["infer"])
                ),
                transform=downscaler_transform,  # type: ignore
                max_workers=config.data_wrangler_max_workers,
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

    # ---> Train the downscaler if it was issued and retraining is wanted or it was
    # issued and the architecture is single-timestamp one or the downscaler is to be
    # built from scratch
    if train is True:
        logger.console.print()
        logger.info("The downscaler will now be trained.")
        try:
            # Define a downscaler if it had not been issued
            if config.downscaler is None:
                downscaler = downscaler_architecture(
                    base_model=config.downscaler_base_model,
                    cols_X=downscaler_X,
                    cols_mask=config.downscaler_masks,
                    scale=config.downscaler_scale,
                    encode=config.downscaler_encode,
                    max_workers=config.downscaler_max_workers,
                    **(
                        {"transform": downscaler_transform}
                        if downscaler_transform is not None
                        else {}
                    ),  # type: ignore
                    logger=logger,
                )

            # Train the downscaler with the training coarse data
            if downscaler_architecture == Downscaler:
                with logger.console.status(
                    f"{'':7}Training downscaler with the coarse data...",
                    spinner="dots",
                    spinner_style="bold blue",
                ):
                    downscaler.fit(
                        X_and_mask_coarse=data_wrangler.get_data_X_and_mask(
                            timestamps=timestamps["sentinel"]["train"],
                            grid="coarse",
                            trans=True,
                            aggregate=True,
                        ),  # type: ignore
                        y_coarse=data_wrangler.get_data_y(
                            timestamps=timestamps["sentinel"]["train"],
                            grid="coarse",
                            trans=True,
                            aggregate=True,
                        ),  # type: ignore
                        sample_weight=(
                            data_wrangler.get_data(
                                timestamps=timestamps["sentinel"]["train"],
                                grid="coarse",
                                vars=config.sample_weight_fit,
                                trans=False,
                                aggregate=True,
                            )
                            if config.sample_weight_fit is not None
                            else None
                        ),  # type: ignore
                    )
            else:
                downscaler.fit(
                    X_and_mask_coarse=data_wrangler.get_data_X_and_mask(
                        timestamps=timestamps["sentinel"]["train"],
                        grid="coarse",
                        trans=False,
                        aggregate=False,
                    ),  # type: ignore
                    y_coarse=data_wrangler.get_data_y(
                        timestamps=timestamps["sentinel"]["train"],
                        grid="coarse",
                        trans=False,
                        aggregate=False,
                    ),  # type: ignore
                    sample_weight=(
                        data_wrangler.get_data(
                            timestamps=timestamps["sentinel"]["train"],
                            grid="coarse",
                            vars=config.sample_weight_fit,
                            trans=False,
                            aggregate=False,
                        )
                        if config.sample_weight_fit is not None
                        else None
                    ),  # type: ignore
                )

        except Exception as e:  # noqa: BLE001
            logger.error(
                "[bold red]Error training the downscaler."
                + f"\nError message: {e}"
                + "\nRun will stop.[/bold red]",
            )
            raise TrainingError(
                "Error training the downscaler." + f"\nError message: {e}"
            )

        logger.info("[bold green]Downscaler trained.[/bold green]")

    # ---> Downscale
    logger.console.print()
    logger.info("LST will now be downscaled.")
    try:
        # Downscale for each inference timestamp
        y_fine_pred = downscaler.predict(
            X_and_mask_fine=data_wrangler.get_data_X_and_mask(
                timestamps=timestamps["sentinel"]["infer"],
                grid="fine",
                trans=True,
                aggregate=False,
            ),  # type: ignore
            correct=config.correct,
            X_and_mask_coarse=data_wrangler.get_data_X_and_mask(
                timestamps=timestamps["sentinel"]["infer"],
                grid="coarse",
                trans=True,
                aggregate=False,
            ),  # type: ignore
            y_coarse=data_wrangler.get_data_y(
                timestamps=timestamps["sentinel"]["infer"],
                grid="coarse",
                trans=False,
                aggregate=False,
            ),  # type: ignore
            coords_coarse=data_wrangler.get_coords(
                timestamps=timestamps["sentinel"]["infer"], grid="coarse"
            ),  # type: ignore
            coords_fine=data_wrangler.get_coords(
                timestamps=timestamps["sentinel"]["infer"], grid="fine"
            ),  # type: ignore
            gridded=config.gridded,
            dims=config.dims,
            attrs=config.attrs,
            path_out=path_out["y_fine_pred"],  # type: ignore
        )

    except Exception as e:  # noqa: BLE001
        logger.error(
            "[bold red]Error downscaling."
            + f"\nError message: {e}"
            + "\nRun will stop.[/bold red]",
        )
        raise DownscalingError("Error downscaling." + f"\nError message: {e}")

    logger.info("[bold green]LST downscaled.[/bold green]")

    # ---> Score if wanted
    if config.score is True:
        # Initialize score dictionary
        score = {}

        # Get timestamps for which there is Landsat data available
        timestamps["landsat"] = {
            batch: (
                list(
                    set(timestamps["sentinel"][batch])
                    & set(data_wrangler.timestamps_landsat)
                )
                if data_wrangler_path_landsat is not None
                else None
            )
            for batch in ["train", "infer"]
        }
        for batch in ["train", "infer"]:
            for grid in ["coarse", "fine"]:
                for ground_truth in ["sentinel", "landsat"]:
                    try:
                        if (
                            # Score downscaler on the training data if it is a
                            # multi-timestamp one and was trained/retrained
                            (
                                batch == "train"
                                and train is True
                                and downscaler_architecture == Downscaler
                            )
                            # Score downscaler on the inference data
                            or batch == "infer"
                        ) and (
                            # Score donwscaler using Sentinel-3 data as ground truth
                            # if the grid is coarse
                            (grid == "coarse" and ground_truth == "sentinel")
                            # Score downscaler using Landsat data for both grids as
                            # ground truth only if such data is available
                            or (
                                ground_truth == "landsat"
                                and timestamps["landsat"][batch] is not None
                            )
                        ):
                            logger.console.print()
                            logger.info(
                                "The downscaler will now be scored with respect to the"
                                f" {grid}"
                                f" {'training' if batch == 'train' else 'inference'}"
                                f" data using"
                                f" {'Sentinel-3' if ground_truth == 'sentinel' else 'Landsat 8/9'}"
                                " as ground truth."
                            )

                            if batch not in score:
                                score[batch] = {}
                            if grid not in score[batch]:
                                score[batch][grid] = {}

                            score[batch][grid][ground_truth] = (
                                # Coarse-score if grid is coarse
                                downscaler.score_coarse(
                                    X_and_mask_coarse=data_wrangler.get_data_X_and_mask(
                                        timestamps=timestamps[ground_truth][batch],
                                        grid="coarse",
                                        trans=True,
                                        aggregate=False,
                                    ),  # type: ignore
                                    y_coarse=(
                                        # Use Sentinel-3 data as ground truth if
                                        # that is the case
                                        data_wrangler.get_data_y(
                                            timestamps=timestamps[ground_truth][batch],
                                            grid="coarse",
                                            trans=False,
                                            aggregate=False,
                                        )
                                        if ground_truth == "sentinel"
                                        # Use Landsat data as ground truth if that
                                        # is the case
                                        else data_wrangler.get_data(
                                            timestamps=timestamps[ground_truth][batch],
                                            grid="coarse",
                                            vars=data_wrangler_data_vars.y_val,
                                            trans=False,
                                            aggregate=False,
                                        )
                                    ),  # type: ignore
                                    aggregate=True,
                                    scorers=config.scorers,
                                    sample_weight=(
                                        data_wrangler.get_data(
                                            timestamps=timestamps[ground_truth][batch],
                                            grid="coarse",
                                            vars=config.sample_weight_score,
                                            trans=False,
                                            aggregate=False,
                                        )
                                        if config.sample_weight_score is not None
                                        else None
                                    ),  # type: ignore
                                )
                                if grid == "coarse"
                                # Fine-score if grid is fine
                                else downscaler.score(
                                    X_and_mask_fine=data_wrangler.get_data_X_and_mask(
                                        timestamps=timestamps["landsat"][batch],
                                        grid="fine",
                                        trans=True,
                                        aggregate=False,
                                    ),  # type: ignore
                                    y_fine=data_wrangler.get_data(
                                        timestamps=timestamps["landsat"][batch],
                                        grid="fine",
                                        vars=data_wrangler_data_vars.y_val,
                                        trans=False,
                                        aggregate=False,
                                    ),  # type: ignore
                                    correct=config.correct,
                                    X_and_mask_coarse=data_wrangler.get_data_X_and_mask(
                                        timestamps=timestamps["landsat"][batch],
                                        grid="coarse",
                                        trans=True,
                                        aggregate=False,
                                    ),  # type: ignore
                                    y_coarse=data_wrangler.get_data_y(
                                        timestamps=timestamps["landsat"][batch],
                                        grid="coarse",
                                        trans=False,
                                        aggregate=False,
                                    ),  # type: ignore
                                    coords_coarse=data_wrangler.get_coords(
                                        timestamps=timestamps["landsat"][batch],
                                        grid="coarse",
                                    ),  # type: ignore
                                    coords_fine=data_wrangler.get_coords(
                                        timestamps=timestamps["landsat"][batch],
                                        grid="fine",
                                    ),  # type: ignore
                                    aggregate=True,
                                    scorers=config.scorers,
                                    sample_weight=(
                                        data_wrangler.get_data(
                                            timestamps=timestamps["landsat"][batch],
                                            grid="fine",
                                            vars=config.sample_weight_score,
                                            trans=False,
                                            aggregate=False,
                                        )
                                        if config.sample_weight_score is not None
                                        else None
                                    ),  # type: ignore
                                )
                            )

                            logger.info(
                                f"[bold green]Downscaler scored with respect to {grid}"
                                f" {'training' if batch == 'train' else 'inference'}"
                                " data using"
                                f" {'Sentinel-3' if ground_truth == 'sentinel' else 'Landsat 8/9'}"
                                " as ground truth.[/bold green]"
                            )

                    except Exception as e:  # noqa: BLE001
                        logger.error(
                            "[bold red]Error scoring the downscaler with respect to"
                            f" {grid}"
                            f" {'training' if batch == 'train' else 'inference'} data"
                            " using"
                            f" {'Sentinel-3' if ground_truth == 'sentinel' else 'Landsat 8/9'}"
                            " as ground truth."
                            f"\nError message: {e}"
                            "\nRun will stop.[/bold red]",
                        )
                        raise ScoringError(
                            f"Error scoring the downscaler with respect to {grid}"
                            f" {'training' if batch == 'train' else 'inference'} data"
                            " using"
                            f" {'Sentinel-3' if ground_truth == 'sentinel' else 'Landsat 8/9'}"
                            " as ground truth."
                            f"\nError message: {e}"
                        )

    else:
        score = None

    # ---> Combine all the results in a dictionary
    object = {
        "y_fine_pred": y_fine_pred,
        "score": score,
        "downscaler": downscaler,
        "data_wrangler": data_wrangler,
    }

    # ---> Write scores, downscaler and data wrangler to file if wanted
    for object_alias in object:  # noqa: PLC0206
        if object_alias != "y_fine_pred" and path_out[object_alias] is not None:
            logger.console.print()
            logger.info(
                f"The {object_alias.replace('_', ' ')} will now be written to file."
            )
            try:
                with logger.console.status(
                    f"{'':7}Writing {object_alias.replace('_', ' ')} to file...",
                    spinner="dots",
                    spinner_style="bold blue",
                ):
                    if object_alias == "score":
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
    return DownscaleOut(
        **{
            object_alias: (
                object if path_out[object_alias] is None else path_out[object_alias]
            )
            for object_alias, object in object.items()
            if out[object_alias] is True
        }  # type: ignore
    )


# --------------------------- Script's main funcion -------------------------- #
def main() -> DownscaleOut:  # type: ignore
    downscale(config)


# If this very script is directly executed in the terminal (in that case the global
# variable __name__ corresponds to "__main__"), it runs function main()
if __name__ == "__main__":
    main()
