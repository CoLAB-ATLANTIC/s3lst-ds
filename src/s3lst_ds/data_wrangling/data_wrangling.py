from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, ClassVar, Literal

import joblib
import pandas as pd
import rioxarray as rxr  # noqa: F401
import xarray as xr

from s3lst_ds.data_wrangling.single_data_wrangling import SingleDataWrangler
from s3lst_ds.utilities.jobs_utils import parse_n_jobs
from s3lst_ds.utilities.logging_utils import RichLogger
from s3lst_ds.utilities.time_utils import get_season
from s3lst_ds.utilities.tqdm_utils import tqdm
from s3lst_ds.utilities.var_utils import DataVars


def get_single_data_wrangler(
    timestamp,
    data_vars,
    path_sentinel3,
    path_spatial_pred,
    aoi,
    path_landsat,
    transform,
):
    """
    Get a `SingleDataWrangler` instance for data of  timestamp `timestamp` by reading
    Sentinel-3, spatial predictor, AOI and Landsat data from issued paths
    `path_sentinel3`, `path_spatial_pred`, `aoi`, `path_landsat`, reprojecting it to
    Sentinel-3 coarse and fine grids, combining it (including the temporal predictor
    data) and masking it for each grid and further transforming it with the issued
    `transform`.

    The `SingleDataWrangler` instance is returned as value of a dictionary which also
    includes `timestamp`. The values are keyed by labels `"single_data_wrangler"` and
    `"timestamp"`, respectively.

    Parameters
    ----------

    timestamp : pd.Timestamp
        Timestamp of interest. Only used for keying the `SingleDataWrangler` instance.

    data_vars : DataVars
        Aliases for predictors and target and their kinds.

    path_sentinel3 : Path
        Path to a Sentinel-3 data folder. This folder must contain a georeferenced
        Sentinel-3 SLSTR Level-2 LST product file
        (https://sentiwiki.copernicus.eu/web/slstr-products#S3-SLSTR-Products-L2-LST-Products)
        and a georeferenced Sentinel-3 Synergy Level-2 product file
        (https://sentiwiki.copernicus.eu/web/synergy-products#SYNERGYProducts-L2SYNSDRprocessingS3-Synergy-Products-L2-SYN-SDR-processing).

    path_spatial_pred : Path or None, default=None
        Path to the NetCDF file with the spatial predictor data. If not set, no spatial
        predictor data is considered.

    aoi : Path or str or None, default=None
        WKT string representing the AOI, or path to its shapefile. If not set, no AOI is
        considered and no masking is applied.

    path_landsat : Path or None, default=None
        Path to a Landsat 8/9 data folder for validation. This folder must contain a
        `LST.TIF` file for georeferenced Level-2 LST, having a resolution of 30 m
        (https://www.usgs.gov/centers/eros/science/usgs-eros-archive-landsat-archives-landsat-8-9-olitirs-collection-2-level-2).
        If not set, no Landsat data is considered.

    transform : {None, "center", "standardize"}, default=None
        The transform to apply on the coarse target (as well as validation one) and
        coarse and fine spatio-temporal predictors from a copy of the wrangled `data` by
        using coarse data statistics. The transformations are set in `data` with the
        same
        names as the original columns with the substring `"_trans"` suffixed to
        them. The possible values for `transform` are:
            - `None` - not transforming the data;
            - `"center"` - subtracting the mean from the data;
            - `"standardize"` - subtracting the mean from the data and diving the
            result by the standard deviation.

    Returns
    -------
    timestamp__single_data_wrangler : dict[str, pd.Timestamp | SingleDataWrangler]
        The `timestamp` and respective `SingleDataWrangler` instance keyed by labels
        `"timestamp"` and `"single_data_wrangler"`, respectively.
    """

    single_data_wrangler = SingleDataWrangler(
        data_vars=data_vars,
        path_sentinel3=path_sentinel3,
        path_spatial_pred=path_spatial_pred,
        aoi=aoi,
        path_landsat=path_landsat,
        transform=transform,
    )
    return {"timestamp": timestamp, "single_data_wrangler": single_data_wrangler}


class DataWrangler:
    """
    A class for wrangling Sentinel-3, spatial predictor, AOI and and possibly validation
    Landsat data associated with multiple timestamps.

    Attributes
    ----------

    data_vars : DataVars
        Aliases for predictors and target and their kinds.

    path_sentinel3 : Path
        Path to the directory containing Sentinel-3 data folders. Each folder contains a
        georeferenced Sentinel-3 SLSTR Level-2 LST product file
        (https://sentiwiki.copernicus.eu/web/slstr-products#S3-SLSTR-Products-L2-LST-Products)
        and a georeferenced Sentinel-3 Synergy Level-2 product file
        (https://sentiwiki.copernicus.eu/web/synergy-products#SYNERGYProducts-L2SYNSDRprocessingS3-Synergy-Products-L2-SYN-SDR-processing).

    path_spatial_pred : Path or None, default=None
        Path to the NetCDF file with the spatial predictor data. If not set, no spatial
        predictor data is considered.

    aoi : Path or str or None, default=None
        WKT string representing the AOI, or path to its shapefile. If not set, no AOI is
        considered and no masking is applied.

    path_landsat : Path or None, default=None
        Path to the directory containing Landsat 8/9 data folders. Each folder contains
        a `LST.TIF` file for georeferenced Level-2 LST, having a resolution of 30 m
        (https://www.usgs.gov/centers/eros/science/usgs-eros-archive-landsat-archives-landsat-8-9-olitirs-collection-2-level-2).
        In the wrangling, such data and Sentinel-3's will be "matched" if the respective
        folders have the same name (it is implied here that the user had analysed the
        acquisitions obtained by the two platforms and set the names of the Landsat 8/9
        data folders as the ones of Sentinel-3's (start sensing times) whose start
        sensing times and spatial extents are approximately the same).

    transform : {None, "center", "standardize"}, default=None
        The transform to apply on the coarse target (as well as validation one) and
        coarse and fine spatio-temporal predictors from a copy of the wrangled `data` in
        each `SingleDataWrangler` instance by using coarse data statistics. The
        transformations are set in
        `SingleDataWrangler's `data` with the same names as the original columns with
        the substring `"_trans"` suffixed to them. Note that the transformations are
        timestamp-specific, that is, the computed statistics and the applied
        transformations in each timestamp solely concern the data of that timestamp. The
        possible values for `transform` are:
            - `None` - not transforming the data;
            - `"center"` - subtracting the mean from the data;
            - `"standardize"` - subtracting the mean from the data and diving the result
            by the standard deviation.

    max_workers : int, default=1
        Number of simultaneous multiple processes to consider in wrangling with the
        special cases:
            - `1` or `None`: no multiprocessing is considered;
            - `-1`: all processors are used;
            - `-k`: all processors except k-1 are used.

    timestamps : list[pd.Timestamp]
        Start sensing times associated with each Sentinel-3 data folder of interest.

    timestamps_landsat : list[pd.Timestamp]
        Start sensing times associated with each Sentinel-3 data folder of interest for
        which there is Landsat data available.

    single_data_wrangler: dict[pd.Timestamp, SingleDataWrangler]
        `SingleDataWrangler` instances keyed by respective timestamp. These correspond
        timestamp-specific data wranglers.

    metadata : pd.DataFrame
        Metadata DataFrame associated with the wrangled data having columns:
            - `"timestamp"`
            - `"season"` - season associated with timestamp;
            - `"landsat_exists"` - indicator of existence of Landsat data;

    grids: ("coarse", "fine")
        Aliases of the wrangled coarse and fine Sentinel-3 grids:
            - `"coarse"`, with resolution of approximately 1000 m, obtained from the
            original Sentinel-3 LST data after it being clipped to the AOI bounds;
            - `"fine"`, with resolution of approximately 300 m, obtained from the
            original Sentinel-3 SYN data after it being clipped to the AOI bounds.

    logger : RichLogger or None
        A rich logger for showing progress of the wrangling.

    show_progress : bool, default=True
        `True` to display the wrangling progress.


    """

    # ---> Class attributes
    grids: ClassVar[tuple[str, ...]] = ("coarse", "fine")

    # ---> Instance methods
    def __init__(
        self,
        data_vars: DataVars,
        path_sentinel3: Path,
        aoi: Path | str | None = None,
        path_spatial_pred: Path | None = None,
        path_landsat: Path | None = None,
        timestamps: list[pd.Timestamp] | None = None,
        transform: Literal["center", "standardize"] | None = None,  # type: ignore
        max_workers: int = 1,
        logger: RichLogger | None = None,
        show_progress: bool = True,
    ) -> None:
        """
        Initialize DataWrangler instance by reading Sentinel-3, spatial predictor, AOI
        and Landsat data from issued paths `path_sentinel3`, `path_spatial_pred`,
        `aoi`, `path_landsat`, reprojecting it to Sentinel-3 coarse and fine grids,
        combining and masking it for each grid and further transforming it with the
        issued `transform`.

        Parameters
        ----------
        data_vars : DataVars
            Aliases for predictors and target and their kinds.

        path_sentinel3 : Path
            Path to the directory containing Sentinel-3 data folders. Each folder
            contains a georeferenced Sentinel-3 SLSTR Level-2 LST product file
            (https://sentiwiki.copernicus.eu/web/slstr-products#S3-SLSTR-Products-L2-LST-Products)
            and a georeferenced Sentinel-3 Synergy Level-2 product file
            (https://sentiwiki.copernicus.eu/web/synergy-products#SYNERGYProducts-L2SYNSDRprocessingS3-Synergy-Products-L2-SYN-SDR-processing).

        path_spatial_pred : Path or None, default=None
            Path to the NetCDF file with the spatial predictor data. If not set, no
            spatial predictor data is considered.

        aoi : Path or str or None, default=None
            WKT string representing the AOI, or path to its shapefile. If not set, no
            AOI is considered and no masking is applied.

        path_landsat : Path or None, default=None
            Path to the directory containing Landsat 8/9 data folders. Each folder
            contains a `LST.TIF` file for georeferenced Level-2 LST, having a resolution
            of 30 m
            (https://www.usgs.gov/centers/eros/science/usgs-eros-archive-landsat-archives-landsat-8-9-olitirs-collection-2-level-2).
            In the wrangling, such data and Sentinel-3's will be "matched" if the
            respective folders have the same name (it is implied here that the user had
            analysed the acquisitions obtained by the two platforms and set the names of
            the Landsat 8/9 data folders as the ones of Sentinel-3's (start sensing
            times) whose start sensing times and spatial extents are approximately the
            same).

        timestamps : list[pd.Timestamp] | None = None
            Start sensing times associated with each Sentinel-3 data folder of interest.
            If not issued, all Sentinel-3 data folders are considered.

        transform : {None, "center", "standardize"}, default=None
            The transform to apply on the coarse target and coarse and fine
            spatio-temporal predictors from a copy of the wrangled `data` in each
            `SingleDataWrangler` instance by using coarse data statistics. The
            transformations are set in `SingleDataWrangler's `data` with the same names
            as the original columns with the substring `"_trans"` suffixed to them. Note
            that the transformations are timestamp-specific, that is, the computed
            statistics and the applied transformations in each timestamp solely concern
            the data of that timestamp. The possible values for `transform` are:
                - `None` - not transforming the data;
                - `"center"` - subtracting the mean from the data;
                - `"standardize"` - subtracting the mean from the data and diving the
                result by the standard deviation.

        max_workers : int, default=1
            Number of simultaneous multiple processes to consider in wrangling with the
            special cases:
                - `1` or `None`: no multiprocessing is considered;
                - `-1`: all processors are used;
                - `-k`: all processors except k-1 are used.

        logger : RichLogger or None
            A rich logger for showing progress of the wrangling.

        show_progress : bool, default=True
            `True` to display the wrangling progress.
        """

        self.data_vars = data_vars
        self.path_sentinel3 = path_sentinel3
        self.path_spatial_pred = path_spatial_pred
        self.aoi = aoi
        self.path_landsat = path_landsat
        self._transform = transform
        self.max_workers = parse_n_jobs(max_workers)
        self.logger = logger
        self.show_progress = show_progress

        # Get names of Sentinel-3 data folders of interest
        foldernames_sentinel3 = (
            [
                path_sentinel3_folder.name
                for path_sentinel3_folder in path_sentinel3.iterdir()
                if path_sentinel3_folder.is_dir()
            ]
            if timestamps is None
            else [timestamp.strftime("%Y%m%dT%H%M%S") for timestamp in timestamps]
        )
        # Check if Sentinel-3 data folders exist for all issued timestamps of interest
        if timestamps is not None:
            missing_foldernames_sentinel3 = [
                foldername_sentinel3
                for foldername_sentinel3 in foldernames_sentinel3
                if not (path_sentinel3 / foldername_sentinel3).is_dir()
            ]
            if missing_foldernames_sentinel3:
                raise FileNotFoundError(
                    "The following Sentinel-3 data folders do not exist in"
                    f" {path_sentinel3}:"
                    "\n"
                    + "\n".join(
                        [
                            str(f"{missing_foldername_sentinel3!r}")
                            for missing_foldername_sentinel3 in missing_foldernames_sentinel3
                        ]
                    )
                )

        # Get start sensing times associated with the Sentinel-3 data folders of
        # interest
        self.timestamps = (
            [
                pd.Timestamp(foldername_sentinel3)
                for foldername_sentinel3 in foldernames_sentinel3
            ]
            if timestamps is None
            else timestamps
        )

        # Sort timestamps and names of the Sentinel-3 data folders in ascending order of
        # the timestamps
        self.timestamps, foldernames_sentinel3 = [
            list(timestamp__foldername_sentinel3)
            for timestamp__foldername_sentinel3 in zip(
                *sorted(zip(self.timestamps, foldernames_sentinel3))
            )
        ]

        # Get a SingleDataWrangler instance for each timestamp and perform wrangling of
        # the respective data
        if self.logger is not None:
            self.logger.info(  # type: ignore
                "Getting a SingleDataWrangler instance for each timestamp and"
                " performing wrangling of the respective data..."
            )

        pbar = (
            tqdm(
                # Prefix for the progressbar
                bar_format=f"{'':9}" + "{l_bar}{bar}{r_bar}",
                desc=f"{'':8}",
                total=len(self.timestamps),
                unit="timestamp",
                position=0,
                leave=True,  # Keep progress on the screen after completion.
                options={"console": self.logger.console},
            )
            if self.show_progress is True and self.logger is not None
            else None
        )
        self.single_data_wrangler = {}
        if self.max_workers != 1:
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                # List of placeholders for the eventual result of a computation
                futures = [
                    # NOTE: Using executor.submit() can be safely used as key of
                    # dictionary since executor.submit() returns a Future object
                    # (https://docs.python.org/3/library/asyncio-future.html#future-object)
                    # and all of these objects are unique and hashable.
                    executor.submit(
                        get_single_data_wrangler,
                        timestamp=timestamp,
                        data_vars=data_vars,
                        path_sentinel3=path_sentinel3 / foldername_sentinel3,
                        path_spatial_pred=path_spatial_pred,
                        aoi=aoi,
                        path_landsat=(
                            (path_landsat / foldername_sentinel3)
                            if (
                                path_landsat is not None
                                and (path_landsat / foldername_sentinel3).exists()
                            )
                            else None
                        ),
                        transform=transform,
                    )
                    for timestamp, foldername_sentinel3 in zip(
                        self.timestamps, foldernames_sentinel3
                    )
                ]

                for future in as_completed(futures):
                    # Get result of the completed future (the timestamps and
                    # the single data wranglers)
                    result = future.result()
                    timestamp = result["timestamp"]
                    single_data_wrangle = result["single_data_wrangler"]
                    # Add result to dictionary of results
                    self.single_data_wrangler[timestamp] = single_data_wrangle

                    # Update progress bar with one more count per completed process
                    if pbar is not None:
                        pbar.update()
        else:
            for timestamp, foldername_sentinel3 in zip(
                self.timestamps, foldernames_sentinel3
            ):
                # Get current single data wrangler
                timestap__single_data_wrangler = get_single_data_wrangler(
                    timestamp=timestamp,
                    data_vars=data_vars,
                    path_sentinel3=path_sentinel3 / foldername_sentinel3,
                    path_spatial_pred=path_spatial_pred,
                    aoi=aoi,
                    path_landsat=(
                        (path_landsat / foldername_sentinel3)
                        if (
                            path_landsat is not None
                            and (path_landsat / foldername_sentinel3).exists()
                        )
                        else None
                    ),
                    transform=transform,
                )
                # Set current single data wrangler
                self.single_data_wrangler[
                    timestap__single_data_wrangler["timestamp"]
                ] = timestap__single_data_wrangler["single_data_wrangler"]

                # Update progress bar with one more count per completed process
                if pbar is not None:
                    pbar.update()

        # At the end close progress bar
        if pbar is not None:
            pbar.close()

        # Infer the metadata of the wrangled data (timestamps, season and existence of
        # Landsat data)
        self.metadata = self.extract_metadata()

        # Get timestamps for which there is Landsat data available
        self.timestamps_landsat = self.metadata[self.metadata["landsat_exists"]][
            "timestamp"
        ].tolist()

    @property
    def transform(self) -> Literal["center", "standardize"] | None:
        return self._transform  # type: ignore

    @transform.setter
    def transform(self, value: Literal["center", "standardize"] | None) -> None:
        self._transform = value
        for timestamp in self.timestamps:
            self.single_data_wrangler[timestamp].transform = value

    def extract_metadata(self) -> pd.DataFrame:
        """
        Extract the metadata of the wrangled data (timestamps, season and existence of
        Landsat data).

        Returns
        -------
        metadata : pd.DataFrame
            Metadata DataFrame associated with the wrangled data having columns:
                - `"timestamp"`
                - `"season"` - season associated with timestamp;
                - `"landsat_exists"` - indicator of existence of Landsat data.
        """

        metadata = pd.DataFrame({"timestamp": self.timestamps})
        metadata["season"] = metadata["timestamp"].apply(get_season)
        metadata["landsat_exists"] = metadata["timestamp"].apply(
            lambda timestamp: (
                self.single_data_wrangler[timestamp].path_landsat is not None
            )
        )

        return metadata

    def get_metadata(
        self,
        timestamps: pd.Timestamp | list[pd.Timestamp] | None = None,
        vars: str | list[str] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Get values of metadata `vars` associated with the timestamps `timestamps` of the
        wrangled data.

        Note that if `timestamps` or `vars` are not issued, the returned value
        corresponds to the metadata of of all timestamps or metadata variables,
        respectively.

        Parameters
        ----------
        timestamps : pd.Timestamp or list[pd.Timestamp] or None, default=None
            Timestamps associated with the metadata. If not issued, the metadata of all
            timestamps is considered.
        vars : str or list[str] or None, default=None
            Variables of the metadata to return. If not issued, all metadata variables
            are returned.

        Returns
        -------

        metadata : pd.Series or pd.DataFrame
        Get values of metadata `vars` associated with the timestamps `timestamps` of the
        wrangled data. Note that if `timestamps` or `vars` are not issued, the returned
        value corresponds to the metadata of of all timestamps or metadata variables,
        respectively.
        """

        metadata = (
            self.metadata
            if timestamps is None
            else self.metadata[
                self.metadata["timestamp"].isin(
                    [timestamps] if isinstance(timestamps, pd.Timestamp) else timestamps
                )
            ]
        )[vars if vars is not None else self.metadata.columns]

        return metadata

    def get_coords(
        self,
        timestamps: pd.Timestamp | list[pd.Timestamp] | None = None,
        grid: Literal["coarse", "fine"] | None = None,
    ) -> (
        xr.core.coordinates.DatasetCoordinates  # type: ignore
        | dict[pd.Timestamp, xr.core.coordinates.DatasetCoordinates]  # type: ignore
        | dict[
            pd.Timestamp,
            dict[Literal["coarse", "fine"], xr.core.coordinates.DatasetCoordinates],  # type: ignore
        ]
    ):
        """
        Get Sentinel-3's coordinates associated with issued `timestamps` and `grid`
        alias.

        Note that if `timestamps` or `grid` alias are not issued, the returned value
        corresponds to coordinates of all timestamps or grids, respectively, keyed by
        timestamp or grid alias.

        Parameters
        ----------

        timestamps : pd.Timestamp or list[pd.Timestamp] or None, default=None
            Timestamps associated with the coordinates. If not issued, the coordinates
            of all timestamps is considered.

        grid : {"coarse", "fine", None}, default=None
            Alias of the grid associated with the coordinates. If not issued, the
            coordinates of both grids are returned.


        Returns
        -------
        coords : xr.core.coordinates.DatasetCoordinates or dict[pd.Timestamp,
        xr.core.coordinates.DatasetCoordinates] or dict[pd.Timestamp, dict[{"coarse",
        "fine"}, xr.core.coordinates.DatasetCoordinates]]
            Coordinates associated with Sentinel-3's issued `timestamps` and `grid`
            alias. Note that if `timestamps` or `grid` are not issued, the returned
            value corresponds to coordinates of all timestamps or grids, respectively,
            keyed by timestamp or grid alias.
        """

        coords = {
            timestamp_: {
                grid_: self.single_data_wrangler[timestamp_].coords[grid_]
                for grid_ in ([grid] if grid is not None else self.grids)
            }
            for timestamp_ in (
                [timestamps]
                if isinstance(timestamps, pd.Timestamp)
                else timestamps
                if isinstance(timestamps, list)
                else self.timestamps
            )
        }

        # Squeeze
        if grid is not None:
            for timestamp_ in (
                [timestamps]
                if isinstance(timestamps, pd.Timestamp)
                else timestamps
                if isinstance(timestamps, list)
                else self.timestamps
            ):
                coords[timestamp_] = coords[timestamp_][grid]  # type: ignore
        if isinstance(timestamps, pd.Timestamp):
            coords = coords[timestamps]

        return coords

    def get_shape(
        self,
        timestamps: pd.Timestamp | list[pd.Timestamp] | None = None,
        grid: Literal["coarse", "fine"] | None = None,
    ) -> (
        tuple[int, int]
        | dict[pd.Timestamp, tuple[int, int]]
        | dict[pd.Timestamp, dict[Literal["coarse", "fine"], tuple[int, int]]]
    ):
        """
        Get Sentinel-3's grid shape associated with issued `timestamps` and `grid`
        alias.

        Note that if `timestamps` or `grid` are not issued, the returned value
        corresponds to shapes of all timestamps or grids, respectively, keyed by
        timestamp or grid alias.

        Parameters
        ----------

        timestamps : pd.Timestamp or list[pd.Timestamp] or None, default=None
            Timestamps associated with the shapes. If not issued, the shapes of all
            timestamps is considered.

        grid : {"coarse", "fine", None}, default=None
            Alias of the grid associated with the shapes. If not issued, the shapes of
            both grids are returned.

        Returns
        -------
        shape : tuple[int, int] or dict[pd.Timestamp, tuple[int, int]] or
        dict[pd.Timestamp, dict[{"coarse", "fine"}, tuple[int, int]]]
            Grid shape associated with Sentinel-3's issued `timestamps` and `grid`
            alias. Note that if `timestamps` or `grid` are not issued, the returned
            value corresponds to shapes of all timestamps or grids, respectively, keyed
            by timestamp or grid alias.
        """

        shape = {
            timestamp_: {
                grid_: self.single_data_wrangler[timestamp_].shape[grid_]
                for grid_ in ([grid] if grid is not None else self.grids)
            }
            for timestamp_ in (
                [timestamps]
                if isinstance(timestamps, pd.Timestamp)
                else timestamps
                if isinstance(timestamps, list)
                else self.timestamps
            )
        }

        # Squeeze
        if grid is not None:
            for timestamp_ in (
                [timestamps]
                if isinstance(timestamps, pd.Timestamp)
                else timestamps
                if isinstance(timestamps, list)
                else self.timestamps
            ):
                shape[timestamp_] = shape[timestamp_][grid]  # type: ignore
        if isinstance(timestamps, pd.Timestamp):
            shape = shape[timestamps]

        return shape  # type: ignore

    def get_data(
        self,
        timestamps: pd.Timestamp | list[pd.Timestamp] | None = None,
        grid: Literal["coarse", "fine"] | None = None,
        vars: str | list[str] | None = None,
        trans: bool = False,
        aggregate: bool = False,
    ) -> (
        pd.Series
        | pd.DataFrame
        | dict[pd.Timestamp, pd.Series | pd.DataFrame]
        | dict[Literal["coarse", "fine"], pd.Series | pd.DataFrame]
        | dict[pd.Timestamp, dict[Literal["coarse", "fine"], pd.Series | pd.DataFrame]]
    ):
        """
        Get wrangled and, if `trans` is `True`, further transformed data `vars` for
        issued `timestamps` and `grid` alias.

        Note that if `vars` is not issued, the data of all variables is returned. Also,
        if `timestamps` or `grid` are not issued, the returned value corresponds to data
        of all timestamps or grids, respectively, keyed by timestamp or grid alias. If
        `aggregate` is `True`, the data instead of being keyed by timestamp is
        aggregated with respect to it. If the instance has no transformation (attribute
        `transform` is `None`), the untransformed data is the one considered regardless
        of the value of `trans`.

        Parameters
        ----------

        timestamps : pd.Timestamp or list[pd.Timestamp] or None, default=None
            Timestamps associated with the data. If not issued, the data of all
            timestamps is considered.

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
        dict[pd.Timestamp, dict[{"coarse", "fine"}, pd.Series or pd.DataFrame]]
            Wrangled and, if `trans` is `True`, further transformed data `vars` for
            issued `timestamps` and `grid` alias. Note that if `vars` is not issued, the
            data of all variables is returned. Also, if `timestamps` or `grid` are not
            issued, the returned value corresponds to data of all timestamps or grids,
            respectively, keyed by timestamp or grid alias. If `aggregate` is `True`,
            the data instead of being keyed by timestamp is aggregated with respect to
            it. If the instance has no transformation (attribute `transform` is `None`),
            the untransformed data is the one considered regardless of the value of
            `trans`.
        """

        data = {
            timestamp_: {
                grid_: self.single_data_wrangler[timestamp_].get_data(
                    grid=grid_,  # type: ignore
                    vars=vars,
                    trans=trans,
                )
                for grid_ in ([grid] if grid is not None else self.grids)
            }
            for timestamp_ in (
                [timestamps]
                if isinstance(timestamps, pd.Timestamp)
                else timestamps
                if isinstance(timestamps, list)
                else self.timestamps
            )
        }

        # If wanted, aggregate (concatenate) the data with respect to timestamps
        if not isinstance(timestamps, pd.Timestamp) and aggregate is True:
            data = {
                grid_: pd.concat(
                    [
                        data[timestamp_][grid_]
                        for timestamp_ in (
                            timestamps
                            if isinstance(timestamps, list)
                            else self.timestamps
                        )
                    ],  # type: ignore
                    ignore_index=True,
                )
                for grid_ in ([grid] if grid is not None else self.grids)
            }

            # NOTE: when concatenating the data, categorical columns may cease to be
            # categorical, hence the necessity of re-setting their type after
            # concatenation.
            for grid_ in [grid] if grid is not None else self.grids:
                if not isinstance(vars, str):
                    X_cat = [
                        var
                        for var in self.data_vars.X_cat
                        if var in data[grid_].columns
                    ]
                    data[grid_][X_cat] = data[grid_][X_cat].astype("category")
                else:
                    if vars in self.data_vars.X_cat:
                        data[grid_] = data[grid_].astype("category")

        # Squeeze
        if grid is not None:
            if not isinstance(timestamps, pd.Timestamp) and aggregate is True:
                data = data[grid]
            else:
                for timestamp_ in (
                    [timestamps]
                    if isinstance(timestamps, pd.Timestamp)
                    else timestamps
                    if isinstance(timestamps, list)
                    else self.timestamps
                ):
                    data[timestamp_] = data[timestamp_][grid]  # type: ignore
        if isinstance(timestamps, pd.Timestamp):
            data = data[timestamps]  # type: ignore

        return data  # type: ignore

    def get_data_X_and_mask(
        self,
        timestamps: pd.Timestamp | list[pd.Timestamp] | None = None,
        grid: Literal["coarse", "fine"] | None = None,
        trans: bool = False,
        aggregate: bool = False,
    ) -> (
        pd.DataFrame
        | dict[pd.Timestamp, pd.DataFrame]
        | dict[Literal["coarse", "fine"], pd.DataFrame]
        | dict[pd.Timestamp, dict[Literal["coarse", "fine"], pd.DataFrame]]
    ):
        """
        Get wrangled and, if `trans` is `True`, further transformed predictor and AOI
        mask data for issued `timestamps` and `grid` alias.

        Note that if `timestamps` or `grid` are not issued, the returned value
        corresponds to data of all timestamps or grids, respectively, keyed by timestamp
        or grid alias. If `aggregate` is `True`, the data instead of being keyed by
        timestamp is aggregated with respect to it. If the instance has no
        transformation (attribute `transform` is `None`), the untransformed data is the
        one considered regardless of the value of `trans`.

        Parameters
        ----------

        timestamps : pd.Timestamp or list[pd.Timestamp] or None, default=None
            Timestamps associated with the data. If not issued, the data of all
            timestamps is considered.

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
        dict[{"coarse", "fine"}, pd.DataFrame] or dict[pd.Timestamp, dict[{"coarse",
        "fine"}, pd.DataFrame]]
            Wrangled and, if `trans` is `True`, further transformed predictor and AOI
            mask data for issued `timestamps` and `grid` alias. Note that if
            `timestamps` or `grid` are not issued, the returned value corresponds to
            data of all timestamps or grids, respectively, keyed by timestamp or grid
            alias. If `aggregate` is `True`, the data instead of being keyed by
            timestamp is aggregated with respect to it. If the instance has no
            transformation (attribute `transform` is `None`), the untransformed data is
            the one considered regardless of the value of `trans`.
        """

        return self.get_data(
            timestamps=timestamps,
            grid=grid,
            vars=self.data_vars.X + (["aoi"] if self.aoi is not None else []),  # type: ignore
            trans=trans,
            aggregate=aggregate,
        )  # type: ignore

    def get_data_y(
        self,
        timestamps: pd.Timestamp | list[pd.Timestamp] | None = None,
        grid: Literal["coarse", "fine"] | None = None,
        trans: bool = False,
        aggregate: bool = False,
    ) -> (
        pd.Series
        | dict[pd.Timestamp, pd.Series]
        | dict[Literal["coarse", "fine"], pd.Series]
        | dict[pd.Timestamp, dict[Literal["coarse", "fine"], pd.Series]]
    ):
        """
        Get wrangled and, if `trans` is `True`, further transformed target data for
        issued issued `timestamps` and `grid` alias.

        Note that if `timestamps` or `grid` are not issued, the returned value
        corresponds to data of all timestamps or grids, respectively, keyed by timestamp
        or grid alias. If `aggregate` is `True`, the data instead of being keyed by
        timestamp is aggregated with respect to it. If the instance has no
        transformation (attribute `transform` is `None`), the untransformed data is the
        one considered regardless of the value of `trans`.

        Parameters
        ----------

        timestamps : pd.Timestamp or list[pd.Timestamp] or None, default=None
            Timestamps associated with the data. If not issued, the data of all
            timestamps is considered.

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
        pd.Series] or dict[pd.Timestamp, dict[{"coarse", "fine"}, pd.Series]]
            Wrangled and, if `trans` is `True`, further transformed target data for
            issued `timestamps` and `grid` alias. Note that if `timestamps` or `grid`
            are not issued, the returned value corresponds to data of all timestamps or
            grids, respectively, keyed by timestamp or grid alias. If `aggregate` is
            `True`, the data instead of being keyed by timestamp is aggregated with
            respect to it. If the instance has no transformation (attribute `transform`
            is `None`), the untransformed data is the one considered regardless of the
            value of `trans`.
        """

        return self.get_data(
            timestamps=timestamps,
            grid=grid,
            vars=self.data_vars.y,
            trans=trans,
            aggregate=aggregate,
        )  # type: ignore

    def set_data(
        self,
        values: (
            pd.Series
            | pd.DataFrame
            | dict[Literal["coarse", "fine"], pd.Series | pd.DataFrame]
            | dict[pd.Timestamp, pd.Series | pd.DataFrame]
            | dict[
                pd.Timestamp, dict[Literal["coarse", "fine"], pd.Series | pd.DataFrame]
            ]
        ),
        vars: str | list[str] | None = None,
        timestamps: pd.Timestamp | list[pd.Timestamp] | None = None,
        grid: Literal["coarse", "fine"] | None = None,
        trans: bool = False,
    ) -> None:
        """
        Set wrangled and, if `trans` is `True`, further transformed data `vars` of
        issued `timestamps` and `grid` alias to `values`.

        Note that `vars` may correspond to new variables. If not defined, `vars` is set
        to all variables of the data. If `timestamps` or `grid` is not issued, the data
        of all timestamps or grids, respectively, is set. If there is no transform in
        the instance (attribute `transform` is `None`), the untransformed data is the
        one considered regardless of the value of `trans`.

        Parameters
        ----------

        values : dict[pd.Timestamp, pd.Series or pd.DataFrame] or dict[pd.Timestamp, dict[{"coarse", "fine"}, pd.Series or pd.DataFrame]]
            Values to set.

        vars : str or list[str] or None, default=None
            Variables of `single_data_wrangler`s data to set. Note that `vars` may
            correspond to new variables. If not defined, `vars` is set as all variables
            of the data.

        timestamps : pd.Timestamp or list[pd.Timestamp] or None, default=None
            Timestamps associated with the data. If not issued, the data of all
            timestamps is set.

        grid : {"coarse", "fine", None}, default=None
            Alias of the grid associated with the data. If not issued, the data of both
            grids is set.

        trans : bool, default=False
            Whether to set transformed data.
        """

        for timestamp_ in (
            [timestamps]
            if isinstance(timestamps, pd.Timestamp)
            else timestamps
            if isinstance(timestamps, list)
            else self.timestamps
        ):
            self.single_data_wrangler[timestamp_].set_data(
                values=(
                    values
                    if isinstance(timestamps, pd.Timestamp)
                    else values[timestamp_]  # type: ignore
                ),
                vars=vars,
                grid=grid,
                trans=trans,
            )

    def apply(
        self,
        vars: str | list[str] | None = None,
        timestamps: pd.Timestamp | list[pd.Timestamp] | None = None,
        grid: Literal["coarse", "fine"] | None = None,
        trans: bool = False,
        aggregate: bool = False,
        **pandas_kwargs: Any,
    ) -> (
        pd.Series
        | pd.DataFrame
        | dict[pd.Timestamp, pd.Series | pd.DataFrame]
        | dict[Literal["coarse", "fine"], pd.Series | pd.DataFrame]
        | dict[
            pd.Timestamp,
            dict[Literal["coarse", "fine"], pd.Series | pd.DataFrame],
        ]
    ):
        """
        Use `pandas`' `apply` method (of arguments `pandas_kwargs`) on wrangled and, if
        `trans` is `True`, further transformed data `vars` of issued `timestamps` and
        `grid` alias.

        Note that if not defined, `vars` is set to all variables of the data. If
        `timestamps` or `grid` are not issued, the data of all timestamps or grids are
        used and the returned value is keyed by timestamp or grid alias, respectively.
        If `aggregate` is `True`, the data instead of being keyed by timestamp is
        aggregated with respect to it. If the instance has no transformation (attribute
        `transform` is `None`), the untransformed data is the one considered regardless
        of the value of `trans`.

        Parameters
        ----------

        vars : str or list[str] or None, default=None
            Variables of `single_data_wrangler`s data to use in `apply`. If not issued,
            the whole data is used.

        timestamps : pd.Timestamp or list[pd.Timestamp] or None, default=None
            Timestamps associated with the data. If not issued, the data of all
            timestamps is used.

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

        pd.Series or pd.DataFrame or dict[pd.Timestamp, pd.Series or pd.DataFrame] or
        dict[{"coarse", "fine"}, pd.Series or pd.DataFrame] or dict[pd.Timestamp,
        dict[{"coarse", "fine"}, pd.Series or pd.DataFrame]]
            Result of `pandas`' `apply` method (of arguments `pandas_kwargs`) on
            wrangled and, if `trans` is `True`, further transformed data `vars` of
            issued `timestamps` and `grid` alias. Note that if not defined, `vars` is
            set to all variables of the data. If `timestamps` or `grid` are not issued,
            the data of all timestamps or grids are used and the returned value is keyed
            by timestamp or grid alias, respectively. If `aggregate` is `True`, the data
            instead of being keyed by timestamp is aggregated with respect to it. If the
            instance has no transformation (attribute `transform` is `None`), the
            untransformed data is the one considered regardless of the value of `trans`.
        """

        result = {
            timestamp_: {
                grid_: self.get_data(
                    timestamps=timestamp_,
                    grid=grid_,  # type: ignore
                    vars=vars,
                    trans=trans,
                    aggregate=False,
                ).apply(  # type: ignore
                    **pandas_kwargs
                )
                for grid_ in ([grid] if grid is not None else self.grids)
            }
            for timestamp_ in (
                [timestamps]
                if isinstance(timestamps, pd.Timestamp)
                else timestamps
                if isinstance(timestamps, list)
                else self.timestamps
            )
        }

        # If wanted, aggregate (concatenate) the result with respect to timestamps
        if not isinstance(timestamps, pd.Timestamp) and aggregate is True:
            result = {
                grid_: pd.concat(
                    [
                        result[timestamp_][grid_]
                        for timestamp_ in (
                            timestamps
                            if isinstance(timestamps, list)
                            else self.timestamps
                        )
                    ],
                    ignore_index=True,
                )
                for grid_ in ([grid] if grid is not None else self.grids)
            }

        # Squeeze
        if grid is not None:
            if not isinstance(timestamps, pd.Timestamp) and aggregate is True:
                result = result[grid]
            else:
                for timestamp_ in (
                    [timestamps]
                    if isinstance(timestamps, pd.Timestamp)
                    else timestamps
                    if isinstance(timestamps, list)
                    else self.timestamps
                ):
                    result[timestamp_] = result[timestamp_][grid]  # type: ignore
        if isinstance(timestamps, pd.Timestamp):
            result = result[timestamps]  # type: ignore

        return result  # type: ignore

    def apply_set_data(
        self,
        vars_apply: str | list[str] | None = None,
        vars_set: str | list[str] | None = None,
        timestamps: pd.Timestamp | list[pd.Timestamp] | None = None,
        grid: Literal["coarse", "fine"] | None = None,
        trans_apply: bool = False,
        trans_set: bool | None = None,
        **pandas_kwargs: Any,
    ) -> None:
        """
        Use `pandas`' `apply` method (of arguments `pandas_kwargs`) on wrangled and, if
        `trans_apply` is `True`, further transformed data `vars_apply` of issued
        `timestamps` and `grid` alias and set the result to `vars_set` as transformed
        data if `trans_set` is `True` or as untransformed data if otherwise.

        If `vars_apply` is not defined, it is set to all variables of the data. If
        `vars_set` or `trans_set` are not defined, they are set to `vars_apply` or
        `trans_apply`, respectively. If `timestamps` or `grid` are not issued, the data
        of all timestamps or grids (coarse and fine), respectively, is used and set. If
        there is no transform in the instance (attribute `transform` is `None`), the
        untransformed data is the one considered regardless of the values of
        `trans_apply` and `trans_set`.

        Parameters
        ----------

        vars_apply : str or list[str] or None, default=None
            Variables of `single_data_wrangler`s data to use in `apply`. If not issued,
            the whole data is used.

        vars_set : str or list[str] or None, default=None
            Variables of `single_data_wrangler`s data to use in `set_data`. If not
            issued, it is set to `vars_apply`.

        timestamps : pd.Timestamp or list[pd.Timestamp] or None, default=None
            Timestamps associated with the data. If not issued, the data of all
            timestamps is used.

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
                timestamps=timestamps,
                grid=grid,
                trans=trans_apply,
                aggregate=False,
                **pandas_kwargs,
            ),  # type: ignore
            vars=vars_set,
            timestamps=timestamps,
            grid=grid,
            trans=trans_set,
        )

    def dropna(
        self,
        timestamps: pd.Timestamp | list[pd.Timestamp] | None = None,
        grid: Literal["coarse", "fine"] | None = None,
        **pandas_kwargs: Any,
    ) -> None:
        """
        Use `pandas`' `dropna` method (of arguments `pandas_kwargs`) on wrangled
        untransformed and transformed data associated with the issued `timestamps` and
        `grid` alias.

        Note that if `timestamps` or `grid` are not issued, the data of all timestamps
        or grids is considered, respectively.

        Parameters
        ----------

        timestamps : pd.Timestamp or list[pd.Timestamp] or None, default=None
            Timestamps associated with the data. If not issued, the data of all
            timestamps is considered.

        grid : {"coarse", "fine", None}, default=None
            Alias of the grid associated with the data. If not issued, the data of both
            grids is considered.

        pandas_kwargs :
            Keyword arguments of `pandas`' `dropna` method.
        """

        # Remove pandas `dropna` argument `inplace` if it exists since a set after drop
        # will be enforced (which is equivalent to `inplace=True`).
        pandas_kwargs.pop("inplace", None)
        for timestamp_ in (
            [timestamps]
            if isinstance(timestamps, pd.Timestamp)
            else timestamps
            if isinstance(timestamps, list)
            else self.timestamps
        ):
            for grid_ in [grid] if grid is not None else self.grids:
                self.single_data_wrangler[timestamp_].data[grid_] = (
                    self.single_data_wrangler[timestamp_]
                    .data[grid_]
                    .dropna(
                        **pandas_kwargs,
                    )
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
