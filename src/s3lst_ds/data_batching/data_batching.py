from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.model_selection import BaseCrossValidator, KFold, StratifiedKFold

from s3lst_ds.data_wrangling.data_wrangling import DataWrangler


class DataBatcher:
    """
    A class for batching data into cross-validation folds and a test set.

    All splits are timestamp-specific, that is, the data of each timestamp is fully
    contained by its associated batch. The timestamps considered for testing are the
    ones whose Landsat data exists. Note that none of the Sentinel-3 data of such
    timestamps will be used for training/cross-validation. The cross-validation
    timestamps are randomly split into `n_cross_val_folds` folds with stratification (if
    `var_cross_val_strat` is not `None`) and considering `rnd_seed` as random seed
    number.

    Attributes
    ----------

    data_wrangler : DataWrangler
        Wrangler for Sentinel-3, spatial predictor, AOI and Landsat data of multiple
        timestamps.

    n_cross_val_folds : int, default=5
        Number of cross-validation folds.

    var_cross_val_strat: str or None, default=None
        Metadata categorical variable with respect to which stratification in the
        cross-validation data splitting into folds is to be performed. If not defined,
        no stratification is considered.

    rnd_seed : int or RandomState instance or None, default=None
        Random seed number considered in the cross-validation data splitting into
        folds. If not defined, no such number is regarded.

    metadata_cross_val_splitter : BaseCrossValidator
        Get cross-validation splitter for the metadata of the wrangled data. The
        splitter is a `StratifiedKFold` instance if `var_cross_val_strat` is issued or
        `KFold` otherwise.

    metadata : pd.DataFrame
        Metadata DataFrame associated with the wrangled data having columns:
            - `"timestamp"`
            - `"season"`: season associated with timestamp;
            - `"landsat_exists"`: indicator of existence of Landsat data;
            - `"batch"`: batch alias assigned to timestamp.

    batches : list[str]
        Aliases of the data batches:
            - `"test"`: test set;
            - `"cross_val_1"`, ..., `"cross_val_n_cross_val_folds"`: cross-validation
            folds.
    batch_fancy : dict[str, str]
        Mapper between batch aliases and their fancy counterparts.
    """

    # ---> Instance methods
    def __init__(
        self,
        data_wrangler: DataWrangler,
        n_cross_val_folds: int = 5,
        var_cross_val_strat: str | None = None,
        rnd_seed: int | np.random.RandomState | None = None,
    ) -> None:
        """
        Initialize instance by performing batching of the wrangled data into
        cross-validation folds and test set by assigning a batch alias to the respective
        timestamps. Stratification on variable `var_cross_val_strat` is performed if
        issued.

        Parameters
        ----------
        data_wrangler : DataWrangler
            Wrangler for Sentinel-3, spatial predictor, AOI and Landsat data of multiple
            timestamps.

        n_cross_val_folds : int, default=5
            Number of cross-validation folds.

        var_cross_val_strat: str or None, default=None
            Categorical variable with respect to which stratification in the
            cross-validation data splitting into folds is to be performed. If not
            defined, no stratification is considered.

        rnd_seed : int or RandomState instance or None, default=None
            Random seed number considered in the cross-validation data splitting into
            folds. If not defined, no such number is regarded.
        """
        self.data_wrangler = data_wrangler
        self.n_cross_val_folds = n_cross_val_folds
        self.var_cross_val_strat = var_cross_val_strat
        self.rnd_seed = rnd_seed

        # Get cross-validation splitter for the metadata of the wrangled data
        self.metadata_cross_val_splitter = self.get_metadata_cross_val_splitter()

        # Get aliases of the data batches
        self.batches = self.get_batches()

        # Get mapper between aliases of the data batches and their fancy counterparts
        self.batch_fancy = self.get_batch_fancy()

        # Perform batching of the metadata into cross-validation folds and test set
        self.batch_metadata()

        # Perform batching of the wrangled data
        self.batch_data()

    @property
    def metadata(self):
        return self.data_wrangler.metadata

    def get_metadata_cross_val_splitter(self) -> BaseCrossValidator:
        """
        Get cross-validation splitter for the metadata of the wrangled data. The
        splitter is a `StratifiedKFold` instance if `var_cross_val_strat` is issued or
        `KFold` otherwise.

        Returns
        -------
        metadata_cross_val_splitter : BaseCrossValidator
            Cross-validation splitter for the metadata of the wrangled data.
        """

        splitter_cls = (
            StratifiedKFold if self.var_cross_val_strat is not None else KFold
        )

        metadata_cross_val_splitter = splitter_cls(
            n_splits=self.n_cross_val_folds,
            random_state=self.rnd_seed,
            shuffle=True,
        )

        return metadata_cross_val_splitter

    def batch_metadata(self) -> None:
        """
        Batch the metadata into cross-validation folds and a test set. This is done by
        defining column `"batch"` in the `metadata` DataFrame with values:
            - `"test"`: for timestamps whose Landsat data exists;
            - `"cross_val_1"`, ..., `"cross_val_n_cross_val_folds"`: for timestamps
            whose Landsat data does not exist, randomly batched into `n_cross_val_folds`
            cross-validation folds with random state `rnd_seed` and stratification (if
            `var_cross_val_strat` is issued).
        """

        # Batch timestamps into a test set if they have Landsat data associated with
        # them or into a cross-validation set, if they have not
        self.metadata["batch"] = self.metadata["landsat_exists"].apply(
            lambda landsat_exists: "test" if landsat_exists is True else "cross_val"
        )

        # Get position indexes of each cross-validation timestamp fold in the
        # cross-validation metadata
        metadata_cross_val = self.metadata[self.metadata["batch"] == "cross_val"]

        i_cross_val_folds = [
            i_train__i_val__iteration[1]
            for i_train__i_val__iteration in self.metadata_cross_val_splitter.split(
                X=metadata_cross_val,
                y=(
                    metadata_cross_val[self.var_cross_val_strat]
                    if self.var_cross_val_strat is not None
                    else None
                ),
            )
        ]

        # Batch the cross-validation timestamps into folds in the cross-validation
        # metadata
        for n_cross_val_fold, i_cross_val_fold in enumerate(i_cross_val_folds, start=1):
            metadata_cross_val.loc[
                metadata_cross_val.index[i_cross_val_fold], "batch"
            ] = f"cross_val_{n_cross_val_fold}"

        # Batch the cross-validation timestamps into folds in the whole metadata
        self.metadata.loc[self.metadata["batch"] == "cross_val", "batch"] = (
            self.metadata[self.metadata["batch"] == "cross_val"]["timestamp"].map(
                metadata_cross_val.set_index("timestamp")["batch"]
            )
        )

    def batch_data(self) -> None:
        """
        Create `"batch"` variable in the wrangled data with the aliases of the
        respective batches.
        """

        # Create timestamp variable in the wrangled data if not existing already
        # NOTE: this variable will be required to create the batch label one.
        for timestamp in self.data_wrangler.timestamps:
            for grid in self.data_wrangler.grids:
                if (
                    "timestamp"
                    not in self.data_wrangler.single_data_wrangler[
                        timestamp
                    ].mapper_var_to_var_trans[grid]
                ):
                    # Set data and update variable mapper
                    self.data_wrangler.set_data(
                        values=timestamp.strftime("%Y-%m-%d %H:%M:%S"),  # type: ignore
                        vars="timestamp",
                        timestamps=timestamp,
                        grid=grid,  # type: ignore
                        trans=False,
                    )

                    # Set data type of timestamp variable to category
                    # NOTE: this dramatically increases the speed of creating batch
                    # variable from timestamp.
                    self.data_wrangler.single_data_wrangler[timestamp].data[grid][
                        "timestamp"
                    ] = (
                        self.data_wrangler.single_data_wrangler[timestamp]
                        .data[grid]["timestamp"]
                        .astype("category")
                    )

        # Create batch variable in the wrangled data with the aliases of the respective
        # batches
        self.data_wrangler.set_data(
            values=self.data_wrangler.apply(
                vars="timestamp",
                func=lambda timestamp: (
                    self.metadata.set_index("timestamp")["batch"].loc[timestamp]
                    # NOTE: after masking (in data wrangling), the timestamp variable
                    # associated with pixels outside of the AOI become nan. One herein
                    # defines the batch variable to be nan when the timestamp variable
                    # also is.
                    if not pd.isna(timestamp)
                    else np.nan
                ),
            ),
            vars="batch",
        )

    def get_batches(self) -> list[str]:
        """
        Get aliases of the data batches:
            - `"test"` - test set;
            - `"cross_val_1"`, ..., `"cross_val_n_cross_val_folds"` - cross-validation
            folds.

        Returns
        -------
        batches : list[str]
            Aliases of the data batches.
        """
        batches = ["test"] + [
            f"cross_val_{i}" for i in range(1, 1 + self.n_cross_val_folds)
        ]

        return batches

    def get_batch_fancy(self) -> dict[str, str]:
        """
        Get mapper between aliases of the data batches and their fancy counterparts.

        Returns
        -------
        batch_fancy : dict[str, str]
            Mapper between aliases of the data batches and their fancy counterparts.
        """
        batch_fancy = {
            "train": "training",
            "cross_val": "cross-validation",
            "test": "test",
            **{
                f"cross_val_{k}": f"cross-validation fold {k}"
                for k in range(1, 1 + self.n_cross_val_folds)
            },
        }
        return batch_fancy

    def get_cv(
        self, data: pd.DataFrame | pd.Series
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Get training and validation position-indexes of issued `data` for each
        cross-validation iteration. Note that `data` must be a pandas DataFrame with
        column `"batch"` containing the batch aliases of the data records, or simply a
        pandas Series corresponding to this very `"batch"` column. The `data` may be
        obtained using method `get_data(batch="cross_val", grid="coarse",
        aggregate=True)` of an instance of the current class.

        Parameters
        ----------

        data : pd.DataFrame or pd.Series
            Data whose position-indexes are to be extracted for the training and
            validation sets of each cross-validation iteration. Must be a pandas
            DataFrame with column `"batch"` containing the batch aliases of the data
            record, or simply a pandas Series corresponding to this very `"batch"`
            column.

        Returns
        -------
        cv : list[tuple[np.ndarray, np.ndarray]]
            A list of tuples of training and validation position-indexes of `data`: a
            tuple for each cross-validation iteration. Each tuple is comprised by two
            entries: the first containing the training position-indexes and the second
            the validation ones.
        """

        # Extract batch alias variable from data
        batch = data["batch"] if isinstance(data, pd.DataFrame) else data

        # Get training and validation position-indexes of the data for each
        # cross-validation iteration
        cv = [
            (
                # Training position-indexes for current iteration
                batch.index.get_indexer(
                    batch[
                        batch.isin(
                            [
                                f"cross_val_{j}"
                                for j in range(1, 1 + self.n_cross_val_folds)
                                if j != i
                            ]
                        )
                    ].index
                ),
                # Validation position-indexes for current iteration
                batch.index.get_indexer(batch[batch == f"cross_val_{i}"].index),
            )
            for i in range(1, 1 + self.n_cross_val_folds)
        ]

        return cv

    def get_coords(
        self,
        batch: str | None = None,
        grid: Literal["coarse", "fine"] | None = None,
    ) -> (
        dict[pd.Timestamp, xr.core.coordinates.DatasetCoordinates]  # type: ignore
        | dict[str, dict[pd.Timestamp, xr.core.coordinates.DatasetCoordinates]]  # type: ignore
        | dict[
            pd.Timestamp,
            dict[Literal["coarse", "fine"], xr.core.coordinates.DatasetCoordinates],  # type: ignore
        ]
        | dict[
            str,
            dict[
                pd.Timestamp,
                dict[Literal["coarse", "fine"], xr.core.coordinates.DatasetCoordinates],  # type: ignore
            ],
        ]
    ):
        """
        Get Sentinel-3's coordinates associated with issued `batch` and `grid` aliases.

        Note that if `batch` or `grid` alias are not issued, the returned value
        corresponds to coordinates of all batches or grids, respectively, keyed by batch
        or grid aliases. The coordinates are also keyed by timestamp.

        Parameters
        ----------

        batch : str or None, default=None
            Batch alias associated with the coordinates. If not issued, the coordinates
            of all batches are considered. If set to `"cross_val"` or `"train"`, the
            coordinates of all cross-validation folds are considered.

        grid : {"coarse", "fine", None}, default=None
            Alias of the grid associated with the coordinates. If not issued, the
            coordinates of both grids are returned.


        Returns
        -------
        coords : dict[pd.Timestamp, xr.core.coordinates.DatasetCoordinates] or dict[str,
        dict[pd.Timestamp, xr.core.coordinates.DatasetCoordinates]] or dict[
            pd.Timestamp, dict[{coarse", "fine"},
            xr.core.coordinates.DatasetCoordinates],
        ] or dict[
            str, dict[
                pd.Timestamp, dict[{"coarse", "fine"},
                xr.core.coordinates.DatasetCoordinates],
            ],
        ]
            Coordinates associated with Sentinel-3's issued `batch` and `grid` aliases.
            Note that if `batch` or `grid` alias are not issued, the returned value
            corresponds to coordinates of all batches or grids, respectively, keyed by
            batch or grid aliases. The coordinates are also keyed by timestamp.
        """

        coords = {
            batch_: {
                timestamp: {
                    grid_: self.data_wrangler.single_data_wrangler[timestamp].coords[
                        grid_
                    ]
                    for grid_ in (
                        [grid] if grid is not None else self.data_wrangler.grids
                    )
                }
                for timestamp in self.metadata[
                    (
                        self.metadata["batch"] == batch_
                        if batch_ not in ["cross_val", "train"]
                        else self.metadata["batch"].str.startswith("cross_val")
                    )
                ]["timestamp"]
            }
            for batch_ in ([batch] if batch is not None else self.batches)
        }

        # Squeeze
        if grid is not None:
            for batch_ in [batch] if batch is not None else self.batches:
                for timestamp in self.metadata[
                    (
                        self.metadata["batch"] == batch_
                        if batch_ not in ["cross_val", "train"]
                        else self.metadata["batch"].str.startswith("cross_val")
                    )
                ]["timestamp"]:
                    coords[batch_][timestamp] = coords[batch_][timestamp][grid]
        if batch is not None:
            coords = coords[batch]

        return coords

    def get_metadata(
        self,
        batch: str | None = None,
        vars: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame | dict[str, pd.Series | pd.DataFrame]:
        """
        Get values of metadata `vars` associated with batched and wrangled data for
        issued `batch`.

        Note that if `vars` is not issued, all metadata variables are returned. Also, if
        `batch` is not issued, the returned value corresponds to metadata of all batches
        keyed by batch. If `batch` is set to `"cross_val"` or `"train"` the metadata of
        all cross-validation folds is considered.

        Parameters
        ----------

        batch : str or None, default=None
            Batch alias associated with the metadata. If not issued, the metadata of all
            batches is considered. If set to `"cross_val"` or `"train"`, the metadata of
            all cross-validation folds is considered.

        vars : str or list[str] or None, default=None
            Variables of the metadata to return. If not issued, all metadata variables
            are returned.

        Returns
        -------

        metadata : pd.Series or pd.DataFrame or dict[str, pd.Series or pd.DataFrame]
            Values of metadata `vars` associated with batched and wrangled data for
            issued `batch`. Note that if `vars` is not issued, all metadata variables
            are returned. Also, if `batch` is not issued, the returned value corresponds
            to metadata of all batches keyed by batch. If `batch` is set to
            `"cross_val"` or `"train"` the metadata of all cross-validation folds is
            considered.
        """

        metadata = {
            batch_: self.metadata[
                (
                    self.metadata["batch"] == batch_
                    if batch_ not in ["cross_val", "train"]
                    else self.metadata["batch"].str.startswith("cross_val")
                )
            ][vars if vars is not None else self.metadata.columns]
            for batch_ in ([batch] if batch is not None else self.batches)
        }

        # Squeeze
        if batch is not None:
            metadata = metadata[batch]

        return metadata  # type: ignore

    def get_data(
        self,
        batch: str | None = None,
        grid: Literal["coarse", "fine"] | None = None,
        vars: str | list[str] | None = None,
        trans: bool = False,
        aggregate: bool = False,
    ) -> (
        pd.Series
        | pd.DataFrame
        | dict[pd.Timestamp, pd.Series | pd.DataFrame]
        | dict[Literal["coarse", "fine"], pd.Series | pd.DataFrame]
        | dict[str, pd.Series | pd.DataFrame]
        | dict[pd.Timestamp, dict[Literal["coarse", "fine"], pd.Series | pd.DataFrame]]
        | dict[str, dict[Literal["coarse", "fine"], pd.Series | pd.DataFrame]]
        | dict[str, dict[pd.Timestamp, pd.Series | pd.DataFrame]]
        | dict[
            str,
            dict[
                pd.Timestamp,
                dict[Literal["coarse", "fine"], pd.Series | pd.DataFrame],
            ],
        ]
    ):
        """
        Get batched, wrangled, and, if `trans` is `True`, further transformed data
        `vars` for issued `batch` and `grid` aliases.

        Note that if `vars` is not issued, the data of all variables is returned. Also,
        if `batch` or `grid` are not issued, the returned value corresponds to data of
        all batches or grids, respectively, keyed by batch or grid aliases. If `batch`
        is set to `"cross_val"` or `"train"` the data of all cross-validation folds is
        considered. If `aggregate` is `True`, the data instead of also being keyed by
        timestamp is aggregated with respect to it. If the instance has no
        transformation (attribute `transform` is `None`), the untransformed data is the
        one considered regardless of the value of `trans`.

        Parameters
        ----------

        batch : str or None, default=None
            Batch alias associated with the data. If not issued, the data of all batches
            is considered. If set to `"cross_val"` or `"train"`, the data of all
            cross-validation folds is considered.

        grid : {"coarse", "fine", None}, default=None
            Alias of the grid associated with the data. If not issued, the data of both
            grids is returned.

        vars : str or list[str] or None, default=None
            Variables of the data to return. If not issued, the data of all variables is
            returned.

        trans : bool, default=False
            Whether to get transformed data.

        aggregate: bool, default=False
            Whether to aggregate the data with respect to timestamps.

        Returns
        -------

        data : pd.Series or pd.DataFrame or dict[pd.Timestamp, pd.Series or
        pd.DataFrame] or dict[{"coarse", "fine"}, pd.Series or pd.DataFrame] or
        dict[str, pd.Series or pd.DataFrame] or dict[pd.Timestamp, dict[{"coarse",
        "fine"}, pd.Series or pd.DataFrame]] or dict[str, dict[{"coarse", "fine"},
        pd.Series or pd.DataFrame]] or dict[str, dict[pd.Timestamp, pd.Series or
        pd.DataFrame]] or dict[
            str, dict[
                pd.Timestamp, dict[{"coarse", "fine"}, pd.Series or pd.DataFrame],
            ],
        ]
            Batched, wrangled and, if `trans` is `True`, further transformed data `vars`
            for issued `batch` and `grid` aliases. Note that if `vars` is not issued,
            the data of all variables is returned. If `batch` or `grid` are not issued,
            the returned value corresponds to data of all batches or grids,
            respectively, keyed by batch or grid aliases. If `batch` is set to
            `"cross_val"` or `"train"` the data of all cross-validation folds is
            considered. If `aggregate` is `True`, the data instead of also being keyed
            by timestamp is aggregated with respect to it. If the `DataWrangler`
            instance has no transformation (attribute `transform` is `None`), the
            untransformed data is the one considered regardless of the value of `trans`.
        """

        data = {
            batch_: {
                timestamp: {
                    grid_: self.data_wrangler.single_data_wrangler[timestamp].get_data(
                        grid=grid_,
                        vars=vars,
                        trans=trans,
                    )
                    for grid_ in (
                        [grid] if grid is not None else self.data_wrangler.grids
                    )
                }
                for timestamp in self.metadata[
                    (
                        self.metadata["batch"] == batch_
                        if batch_ not in ["cross_val", "train"]
                        else self.metadata["batch"].str.startswith("cross_val")
                    )
                ]["timestamp"]
            }
            for batch_ in ([batch] if batch is not None else self.batches)
        }

        # If wanted, aggregate (concatenate) the data with respect to timestamps
        if aggregate is True:
            data = {
                batch_: {
                    grid_: pd.concat(
                        [
                            data[batch_][timestamp][grid_]
                            for timestamp in self.metadata[
                                (
                                    self.metadata["batch"] == batch_
                                    if batch_ not in ["cross_val", "train"]
                                    else self.metadata["batch"].str.startswith(
                                        "cross_val"
                                    )
                                )
                            ]["timestamp"]
                        ],  # type: ignore
                        ignore_index=True,
                    )
                    for grid_ in (
                        [grid] if grid is not None else self.data_wrangler.grids
                    )
                }
                for batch_ in ([batch] if batch is not None else self.batches)
            }

            # NOTE: when concatenating the data, categorical columns may cease to be
            # categorical, hence the necessity of re-setting their type after
            # concatenation.
            for batch_ in [batch] if batch is not None else self.batches:
                for grid_ in [grid] if grid is not None else self.data_wrangler.grids:
                    if not isinstance(vars, str):
                        X_cat = [
                            var
                            for var in self.data_wrangler.data_vars.X_cat
                            if var in data[batch_][grid_].columns
                        ]
                        data[batch_][grid_][X_cat] = data[batch_][grid_][X_cat].astype(
                            "category"
                        )
                    else:
                        if vars in self.data_wrangler.data_vars.X_cat:
                            data[batch_][grid_] = data[batch_][grid_].astype("category")

        # Squeeze
        if grid is not None:
            for batch_ in [batch] if batch is not None else self.batches:
                if aggregate is True:
                    data[batch_] = data[batch_][grid]  # type: ignore
                else:
                    for timestamp in self.metadata[
                        (
                            self.metadata["batch"] == batch_
                            if batch_ not in ["cross_val", "train"]
                            else self.metadata["batch"].str.startswith("cross_val")
                        )
                    ]["timestamp"]:
                        data[batch_][timestamp] = data[batch_][timestamp][grid]  # type: ignore
        if batch is not None:
            data = data[batch]

        return data  # type: ignore

    def get_data_X_and_mask(
        self,
        batch: str | None = None,
        grid: Literal["coarse", "fine"] | None = None,
        trans: bool = False,
        aggregate: bool = False,
    ) -> (
        pd.DataFrame
        | dict[pd.Timestamp, pd.DataFrame]
        | dict[Literal["coarse", "fine"], pd.DataFrame]
        | dict[str, pd.DataFrame]
        | dict[pd.Timestamp, dict[Literal["coarse", "fine"], pd.DataFrame]]
        | dict[str, dict[Literal["coarse", "fine"], pd.DataFrame]]
        | dict[str, dict[pd.Timestamp, pd.DataFrame]]
        | dict[
            str,
            dict[
                pd.Timestamp,
                dict[Literal["coarse", "fine"], pd.DataFrame],
            ],
        ]
    ):
        """
        Get batched, wrangled and, if `trans` is `True`, further transformed predictor
        and AOI mask data for issued `timestamp` and `grid` alias.

        Note that if `batch` or `grid` are not issued, the returned value corresponds to
        data of all batches or grids, respectively, keyed by batch or grid aliases. If
        `batch` is set to `"cross_val"` or `"train"` the data of all cross-validation
        folds is considered. If `aggregate` is `True`, the data instead of also being
        keyed by timestamp is aggregated with respect to it. If the `DataWrangler`
        instance has no transformation (attribute `transform` is `None`), the
        untransformed data is the one considered regardless of the value of `trans`.

        Parameters
        ----------

        batch : str or None, default=None
            Batch alias associated with the data. If not issued, the data of all batches
            is considered. If set to `"cross_val"` or `"train"`, the data of all
            cross-validation folds is considered.

        grid : {"coarse", "fine", None}, default=None
            Alias of the grid associated with the data. If not issued, the data of both
            grids is returned.

        trans : bool, default=False
            Whether to get transformed data.


        aggregate: bool, default=False
            Whether to aggregate the data with respect to timestamps.

        Returns
        -------
        data_X_and_mask : pd.DataFrame or dict[pd.Timestamp, pd.DataFrame] or
        dict[{"coarse", "fine"}, pd.DataFrame] or dict[str, pd.DataFrame] or
        dict[pd.Timestamp, dict[{"coarse", "fine"}, pd.DataFrame]] or dict[str,
        dict[{"coarse", "fine"}, pd.DataFrame]] or dict[str, dict[pd.Timestamp,
        pd.DataFrame]] or dict[
            str, dict[
                pd.Timestamp, dict[{"coarse", "fine"}, pd.DataFrame],
            ],
        ]
            Batched, wrangled and, if `trans` is `True`, further transformed predictor
            and AOI mask data for issued `timestamp` and `grid` alias. Note that if
            `batch` or `grid` are not issued, the returned value corresponds to data of
            all batches or grids, respectively, keyed by batch or grid aliases. If
            `batch` is set to `"cross_val"` or `"train"` the data of all
            cross-validation folds is considered. If `aggregate` is `True`, the data
            instead of also being keyed by timestamp is aggregated with respect to it.
            If the `DataWrangler` instance has no transformation (attribute `transform`
            is `None`), the untransformed data is the one considered regardless of the
            value of `trans`.
        """

        return self.get_data(
            batch=batch,
            grid=grid,
            vars=self.data_wrangler.data_vars.X
            + (["aoi"] if self.data_wrangler.aoi is not None else []),  # type: ignore
            trans=trans,
            aggregate=aggregate,
        )  # type: ignore

    def get_data_y(
        self,
        batch: str | None = None,
        grid: Literal["coarse", "fine"] | None = None,
        trans: bool = False,
        aggregate: bool = False,
    ) -> (
        pd.Series
        | dict[pd.Timestamp, pd.Series]
        | dict[Literal["coarse", "fine"], pd.Series]
        | dict[str, pd.Series]
        | dict[pd.Timestamp, dict[Literal["coarse", "fine"], pd.Series]]
        | dict[str, dict[Literal["coarse", "fine"], pd.Series]]
        | dict[str, dict[pd.Timestamp, pd.Series]]
        | dict[
            str,
            dict[
                pd.Timestamp,
                dict[Literal["coarse", "fine"], pd.Series],
            ],
        ]
    ):
        """
        Get batched, wrangled and, if `trans` is `True`, further transformed target data
        for issued issued `timestamp` and `grid` alias.

        Note that if `batch` or `grid` are not issued, the returned value corresponds to
        data of all batches or grids, respectively, keyed by batch or grid aliases. If
        `batch` is set to `"cross_val"` or `"train"` the data of all cross-validation
        folds is considered. If `aggregate` is `True`, the data instead of also being
        keyed by timestamp is aggregated with respect to it. If the `DataWrangler`
        instance has no transformation (attribute `transform` is `None`), the
        untransformed data is the one considered regardless of the value of `trans`.

        Parameters
        ----------

        batch : str or None, default=None
            Batch alias associated with the data. If not issued, the data of all batches
            is considered. If set to `"cross_val"` or `"train"`, the data of all
            cross-validation folds is considered.

        grid : {"coarse", "fine", None}, default="coarse"
            Alias of the grid associated with the data. If not issued, the data of both
            grids is returned.

        trans : bool, default=False
            Whether to get transformed data.

        aggregate: bool, default=False
            Whether to aggregate the data with respect to timestamps.

        Returns
        -------
        data_y : pd.Series or dict[pd.Timestamp, pd.Series] or dict[{"coarse", "fine"},
        pd.Series] or dict[str, pd.Series] or dict[pd.Timestamp, dict[{"coarse",
        "fine"}, pd.Series]] or dict[str, dict[{"coarse", "fine"}, pd.Series]] or
        dict[str, dict[pd.Timestamp, pd.Series]] or dict[str, dict[pd.Timestamp,
        dict[{"coarse", "fine"}, pd.Series]]]
            Batched, wrangled and, if `trans` is `True`, further transformed target data
            for issued `timestamp` and `grid` alias. Note that if `timestamp` or `grid`
            are not issued, the returned value corresponds to data of all timestamps or
            grids, respectively, keyed by timestamp or grid alias. If `aggregate` is
            `True`, the data instead of being keyed by timestamp is aggregated with
            respect to it. If the `DataWrangler` instance has no transformation
            (attribute `transform` is `None`), the untransformed data is the one
            considered regardless of the value of `trans`.
        """

        return self.get_data(
            batch=batch,
            grid=grid,
            vars=self.data_wrangler.data_vars.y,
            trans=trans,
            aggregate=aggregate,
        )  # type: ignore

    def set_data(
        self,
        values: (
            dict[pd.Timestamp, pd.Series | pd.DataFrame]
            | dict[str, dict[pd.Timestamp, pd.Series | pd.DataFrame]]
            | dict[
                pd.Timestamp, dict[Literal["coarse", "fine"], pd.Series | pd.DataFrame]
            ]
            | dict[
                str,
                dict[
                    pd.Timestamp,
                    dict[Literal["coarse", "fine"], pd.Series | pd.DataFrame],
                ],
            ]
        ),
        vars: str | list[str] | None = None,
        batch: str | None = None,
        grid: Literal["coarse", "fine"] | None = None,
        trans: bool = False,
    ) -> None:
        """
        Set batched, wrangled and, if `trans` is `True`, further transformed data `vars`
        of issued `batch` and `grid` aliases to `values`.

        Note that `vars` may correspond to new variables. If not defined, `vars` is set
        to all variables of the data. If `batch` or `grid` is not issued, the data of
        all batches or grids, respectively, is set. If there is no transform in the
        instance (attribute `transform` is `None`), the untransformed data is the one
        considered regardless of the value of `trans`.

        Parameters
        ----------

        values: dict[pd.Timestamp, pd.Series or pd.DataFrame] or dict[str, dict[pd.Timestamp, pd.Series or pd.DataFrame]] or dict[pd.Timestamp, dict[{"coarse", "fine"}, pd.Series or pd.DataFrame]] or dict[str, dict[pd.Timestamp, dict[{"coarse", "fine"}, pd.Series or pd.DataFrame]]]]
            Values to set.

        vars : str or list[str] or None, default=None
            Variables of `single_data_wrangler`s data to set. Note that `vars` may
            correspond to new variables. If not defined, `vars` is set to all variables
            of the data.

        batch : str or None, default=None
            Alias of the batch associated with the data. If not issued, the data of all
            batches is set. If set to `"cross_val"` or `"train"`, the data of all
            cross-validation folds is set.

        grid : {"coarse", "fine", None}, default=None
            Alias of the grid associated with the data. If not issued, the data of both
            grids is set.

        trans : bool, default=False
            Whether to set transformed data.
        """

        for batch_ in [batch] if batch is not None else self.batches:
            for timestamp in self.metadata[
                (
                    self.metadata["batch"] == batch_
                    if batch_ not in ["cross_val", "train"]
                    else self.metadata["batch"].str.startswith("cross_val")
                )
            ]["timestamp"]:
                self.data_wrangler.single_data_wrangler[timestamp].set_data(
                    values=(
                        values[timestamp]
                        if batch is not None
                        else values[batch_][timestamp]  # type: ignore
                    ),
                    vars=vars,
                    grid=grid,
                    trans=trans,
                )

    def apply(
        self,
        vars: str | list[str] | None = None,
        batch: str | None = None,
        grid: Literal["coarse", "fine"] | None = None,
        trans: bool = False,
        aggregate: bool = False,
        **pandas_kwargs: Any,
    ) -> (
        dict[pd.Timestamp, pd.Series | pd.DataFrame]
        | dict[str, dict[pd.Timestamp, pd.Series | pd.DataFrame]]
        | dict[pd.Timestamp, dict[Literal["coarse", "fine"], pd.Series | pd.DataFrame]]
        | dict[
            str,
            dict[
                pd.Timestamp, dict[Literal["coarse", "fine"], pd.Series | pd.DataFrame]
            ],
        ]
    ):
        """
        Use `pandas`' `apply` method (of arguments `pandas_kwargs`) on batched, wrangled
        and, if `trans` is `True`, further transformed data `vars` of issued `batch`
        and `grid` aliases.

        Note that if not defined, `vars` is set to all variables of the data. If `batch`
        or `grid` are not issued, the data of all batches or grids are used and the
        returned value is keyed by batch or grid aliases, respectively. If `aggregate`
        is `True`, the data instead also of being keyed by timestamp is aggregated with
        respect to it. If the instance has no transformation (attribute `transform` is
        `None`), the untransformed data is the one considered regardless of the value of
        `trans`.

        Parameters
        ----------

        vars : str or list[str] or None, default=None
            Variables of `single_data_wrangler`s data to use in `apply`. If not issued,
            the whole data is used.

        batch : str or None, default=None
            Alias of the batch associated with the data. If not issued, the data of all
            batches is used. If set to `"cross_val"` or `"train"`, the data of all
            cross-validation folds is used.

        grid : {"coarse", "fine", None}, default=None
            Alias of the grid associated with the data. If not issued, the data of both
            grids is used.

        trans : bool, default=False
            Whether to consider transformed data.

        aggregate: bool, default=False
            Whether to aggregate the result with respect to timestamps.

        pandas_kwargs :
            Keyword arguments of `pandas`' `apply` method.

        Returns
        -------

        dict[pd.Timestamp, pd.Series or pd.DataFrame] or dict[str, dict[pd.Timestamp,
        pd.Series or pd.DataFrame]] or dict[pd.Timestamp, dict[{"coarse", "fine"},
        pd.Series or pd.DataFrame]] or dict[
            str, dict[
                pd.Timestamp, dict[{"coarse", "fine"}, pd.Series or pd.DataFrame]
            ],
        ]
            Result of `pandas`' `apply` method (of arguments `pandas_kwargs`) on
            batched, wrangled and, if `trans` is `True`, further transformed data `vars`
            of issued `timestamp` and `grid` alias. Note that if not defined, `vars` is
            set to all variables of the data. If `batch` or `grid` are not issued, the
            data of all batches or grids are used and the returned value is keyed by
            batch or grid aliases, respectively. If `aggregate` is `True`, the data
            instead also of being keyed by timestamp is aggregated with respect to it.
            If the instance has no transformation (attribute `transform` is `None`), the
            untransformed data is the one considered regardless of the value of `trans`.
        """

        result = {
            batch_: {
                timestamp: {
                    grid_: self.data_wrangler.get_data(
                        timestamps=timestamp,
                        grid=grid_,  # type: ignore
                        vars=vars,
                        trans=trans,
                        aggregate=False,
                    ).apply(  # type: ignore
                        **pandas_kwargs
                    )
                    for grid_ in (
                        [grid] if grid is not None else self.data_wrangler.grids
                    )
                }
                for timestamp in self.metadata[
                    (
                        self.metadata["batch"] == batch_
                        if batch_ not in ["cross_val", "train"]
                        else self.metadata["batch"].str.startswith("cross_val")
                    )
                ]["timestamp"]
            }
            for batch_ in ([batch] if batch is not None else self.batches)
        }

        # If wanted, aggregate (concatenate) the result with respect to timestamps
        if aggregate is True:
            result = {
                batch_: {
                    grid_: pd.concat(
                        [
                            result[batch_][timestamp][grid_]
                            for timestamp in self.metadata[
                                (
                                    self.metadata["batch"] == batch_
                                    if batch_ not in ["cross_val", "train"]
                                    else self.metadata["batch"].str.startswith(
                                        "cross_val"
                                    )
                                )
                            ]["timestamp"]
                        ],
                        ignore_index=True,
                    )
                    for grid_ in (
                        [grid] if grid is not None else self.data_wrangler.grids
                    )
                }
                for batch_ in ([batch] if batch is not None else self.batches)
            }

        # Squeeze
        if grid is not None:
            for batch_ in [batch] if batch is not None else self.batches:
                if aggregate is True:
                    result[batch_] = result[batch_][grid]  # type: ignore
                else:
                    for timestamp in self.metadata[
                        (
                            self.metadata["batch"] == batch_
                            if batch_ not in ["cross_val", "train"]
                            else self.metadata["batch"].str.startswith("cross_val")
                        )
                    ]["timestamp"]:
                        result[batch_][timestamp] = result[batch_][timestamp][grid]  # type: ignore
        if batch is not None:
            result = result[batch]

        return result  # type: ignore

    def apply_set_data(
        self,
        vars_apply: str | list[str] | None = None,
        vars_set: str | list[str] | None = None,
        batch: str | None = None,
        grid: Literal["coarse", "fine"] | None = None,
        trans_apply: bool = False,
        trans_set: bool | None = None,
        **pandas_kwargs: Any,
    ) -> None:
        """
        Use `pandas`' `apply` method (of arguments `pandas_kwargs`) on wrangled and, if
        `trans_apply` is `True`, further transformed data `vars` of issued `batch` and
        `grid` aliases and set the result to `vars_set` as transformed data if
        `trans_set` is `True` or as untransformed data if otherwise.

        If `vars_apply` is not defined, it is set to all variables of the data. If
        `vars_set` or `trans_set` are not defined, they are set to `vars_apply` or
        `trans_apply`, respectively. If `batch` or `grid` are not issued, the data of
        all batches or grids (coarse and fine), respectively, is used and set. If there
        is no transform in the instance (attribute `transform` is `None`), the
        untransformed data is the one considered regardless of the value of `trans`.

        Parameters
        ----------

        vars_apply : str or list[str] or None, default=None
            Variables of `single_data_wrangler`s data to use in `apply`. If not issued,
            the whole data is used.

        vars_set : str or list[str] or None, default=None
            Variables of `single_data_wrangler`s data to use in `set_data`. If not
            issued, it is set to `vars_apply`.

        batch : str or None, default=None
            Alias of the batch associated with the data. If not issued, the data of all
            batches is used. If set to `"cross_val"` or `"train"`, the data of all
            cross-validation folds is used.

        grid : {"coarse", "fine", None}, default=None
            Alias of the grid associated with the data. If not issued, the data of both
            grids is used.

        trans_apply : bool, default=False
            Whether to consider transformed data in `apply`.

        trans_set : bool or None, default=None
            Whether to consider transformed data in `set_data`. If not issued, it is set
            to `trans_apply`.

        pandas_kwargs :
            Keyword arguments of `pandas`' `apply` method.
        """

        if vars_set is None:
            vars_set = vars_apply

        if trans_set is None:
            trans_set = trans_apply

        self.set_data(
            values=self.apply(
                vars=vars_apply,
                batch=batch,
                grid=grid,
                trans=trans_apply,
                aggregate=False,
                **pandas_kwargs,
            ),  # type: ignore
            vars=vars_set,
            batch=batch,
            grid=grid,
            trans=trans_set,
        )

    def dropna(
        self,
        batch: str | None = None,
        grid: Literal["coarse", "fine"] | None = None,
        **pandas_kwargs: Any,
    ) -> None:
        """
        Use `pandas`' `dropna` method (of arguments `pandas_kwargs` with `inplace=True`)
        on wrangled untransformed and transformed data associated with the issued
        `batch` and `grid` aliases.

        Note that if `batch` or `grid` are not issued, the data of all batches or
        grids is considered, respectively.

        Parameters
        ----------

        batch : str or None, default=None
            Alias of the batch associated with the data. If not issued, the data of all
            batches is considered. If set to `"cross_val"` or `"train"`, the data of all
            cross-validation folds is considered.

        grid : {"coarse", "fine", None}, default=None
            Alias of the grid associated with the data. If not issued, the data of both
            grids is considered.

        pandas_kwargs :
            Keyword arguments of `pandas`' `dropna` method.
        """

        # Remove pandas `dropna` argument `inplace` if it exists since `inplace=True`
        # will be enforced.
        pandas_kwargs.pop("inplace", None)
        for batch_ in [batch] if batch is not None else self.batches:
            for timestamp in self.metadata[
                (
                    self.metadata["batch"] == batch_
                    if batch_ not in ["cross_val", "train"]
                    else self.metadata["batch"].str.startswith("cross_val")
                )
            ]["timestamp"]:
                for grid_ in [grid] if grid is not None else self.data_wrangler.grids:
                    self.data_wrangler.single_data_wrangler[timestamp].data[
                        grid_
                    ].dropna(
                        **pandas_kwargs,
                        inplace=True,
                    )

    def save(self, path: Path) -> None:
        """
        Write the instance to `path` with `joblib`.

        Parameters
        ----------
        path : Path
            Path to write the instance to.
        """

        joblib.dump(value=self, filename=path)
