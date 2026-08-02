from pathlib import Path
from typing import ClassVar, Literal

import joblib
import numpy as np
import pandas as pd
import rioxarray as rxr  # noqa: F401
import xarray as xr

from s3lst_downscale.data_reading.s3_reader import Sen3Loader
from s3lst_downscale.utilities.geometry_utils import load_aoi_to_gdf
from s3lst_downscale.utilities.time_utils import get_season
from s3lst_downscale.utilities.var_utils import DataVars
from s3lst_downscale.utilities.xr_utils import (
    selective_reproject_match,
    shape_to_raster_mask,
)


class SingleDataWrangler:
    """
    A class for wrangling Sentinel-3, spatial predictor, AOI and and possibly validation
    Landsat data associated with a single sensing timestamp.

    Parameters
    ----------

    data_vars : DataVars
        Aliases for predictors and target and their kinds.

    path_sentinel3 : Path
        Path to a Sentinel-3 data folder. This folder must contain a georeferenced
        Sentinel-3 SLSTR Level-2 LST product file
        (https://sentiwiki.copernicus.eu/web/slstr-products#S3-SLSTR-Products-L2-LST-Products)
        and a georeferenced Sentinel-3 Synergy Level-2 product file
        (https://sentiwiki.copernicus.eu/web/synergy-products#SYNERGYProducts-L2SYNSDRprocessingS3-Synergy-Products-L2-SYN-SDR-processing).

    path_spatial_pred : Path or None, default=None
        Path to the NetCDF file with the pure spatial predictor data. If not set, no
        spatial predictor data is considered.

    aoi : Path or str or None, default=None
        WKT string representing the AOI, or path to its shapefile. If not set, no AOI is
        considered and no masking is applied.

    path_landsat : Path or None, default=None
        Path to a Landsat 8/9 data folder for validation. This folder must contain a
        `LST.TIF` for georeferenced Level-2 LST, having a resolution of 30 m
        (https://www.usgs.gov/centers/eros/science/usgs-eros-archive-landsat-archives-landsat-8-9-olitirs-collection-2-level-2).
        If not set, no Landsat data is considered.

    transform : {None, "center", "standardize"}, default=None
        The transform to apply on the coarse target (as well as validation one) and
        coarse and fine spatio-temporal predictors from a copy of the wrangled `data` by
        using coarse data statistics. The transformations are set in `data` with the
        same names as the original columns with the substring `"_trans"` suffixed to
        them. Note that the transformations are timestamp-specific, that is, the
        computed statistics and the applied transformations in each timestamp solely
        concern the data of that timestamp. The possible values for `transform` are:
            - `None` - not transforming the data;
            - `"center"` - subtracting the mean from the data;
            - `"standardize"` - subtracting the mean from the data and diving the result
            by the standard deviation.

    timestamp : pd.Timestamp
        Start sensing time associated with Sentinel-3 SLSTR Level-2 LST product file.

    data : dict[{"coarse", "fine"}, pd.DataFrame]
        A dictionary of wrangled (untransformed and transformed) coarse and fine data
        keyed by grid alias. This data corresponds to the combination of Sentinel-3,
        spatial and temporal predictor, AOI and possibly validation Landsat data.

    mapper_var_to_var_trans : dict[{"coarse", "fine"}, dict[str, str]]
        A dictionary mapping the original variable names to the the new ones after
        transformation, keyed by grid. If `transform` is `None`, the new names coincide
        with the original ones. Otherwise, the new names of the transformed variables
        correspond to the original names with the substring `"_trans"` suffixed to them
        while the remainder coincide with the original ones.

    mapper_var_trans_to_var : dict[{"coarse", "fine"}, dict[str, str]]
        A dictionary mapping the variable names after transformation to the original
        ones, keyed by grid. If `transform` is `None`, the new names coincide with the
        original ones. Otherwise, the new names of the transformed variables correspond
        to the original names with the substring `"_trans"` suffixed to them while the
        remainder coincide with the original ones.

    grids: ("coarse", "fine")
        Aliases of the wrangled coarse and fine Sentinel-3 grids:
            - `"coarse"`, with resolution of approximately 1000 m, obtained from the
            original Sentinel-3 LST data after it being clipped to the AOI bounds;
            - `"fine"`, with resolution of approximately 300 m, obtained from the
            original Sentinel-3 SYN data after it being clipped to the AOI bounds.

    coords: dict[{"coarse", "fine"}, xr.core.coordinates.DatasetCoordinates]
        Coordinates of the wrangled coarse and fine Sentinel-3 grids keyed by grid
        alias.

    shape: dict[{"coarse", "fine"}, tuple[int, int]]
        Shapes of the wrangled coarse and fine Sentinel-3 grids keyed by grid alias.
        Each shape corresponds to a tuple of integers `(n_y, n_x)` where `n_y` and `n_x`
        are the lengths of the grid matrices along the `y` and `x` directions,
        respectively.
    """

    # ---> Class attributes
    grids: ClassVar[tuple[str, ...]] = ("coarse", "fine")

    # ---> Instance methods
    def __init__(
        self,
        data_vars: DataVars,
        path_sentinel3: Path,
        aoi: str | Path | None = None,
        path_spatial_pred: Path | None = None,
        path_landsat: Path | None = None,
        transform: Literal["center", "standardize"] | None = None,  # type: ignore
    ) -> None:
        """
        Initialize instance by reading Sentinel-3, spatial predictor, AOI and Landsat
        data from issued paths `path_sentinel3`, `path_spatial_pred`, `aoi`,
        `path_landsat`, reprojecting it to Sentinel-3 coarse and fine grids, combining
        it (including the temporal predictor data) and masking it for each grid and
        further transforming it with the issued `transform`.

        Parameters
        ----------
        data_vars : DataVars
            Aliases for predictors and target and their kinds.

        path_sentinel3 : Path
            Path to a Sentinel-3 data folder. This folder must contain a georeferenced
            Sentinel-3 SLSTR Level-2 LST product file
            (https://sentiwiki.copernicus.eu/web/slstr-products#S3-SLSTR-Products-L2-LST-Products)
            and a georeferenced Sentinel-3 Synergy Level-2 product file
            (https://sentiwiki.copernicus.eu/web/synergy-products#SYNERGYProducts-L2SYNSDRprocessingS3-Synergy-Products-L2-SYN-SDR-processing).

        path_spatial_pred : Path or None, default=None
            Path to the NetCDF file with the spatial predictor data. If not set, no
            spatial predictor data is considered.

        aoi : Path or str or None, default=None
            WKT string representing the AOI, or path to AOI shapefile. If not set, no
            AOI is considered and no masking is applied.

        path_landsat : Path | None, default=None
            Path to a Landsat 8/9 data folder for validation. This folder must contain
            `LST.TIF` file for georeferenced Level-2 LST, having a resolution of 30 m
            (https://www.usgs.gov/centers/eros/science/usgs-eros-archive-landsat-archives-landsat-8-9-olitirs-collection-2-level-2).
            If not set, no Landsat data is considered.

        transform : {None, "center", "standardize"}, default=None
            The transform to apply on the coarse target (as well as validation one) and
            coarse and fine spatio-temporal predictors from a copy of the wrangled
            `data` by using coarse data statistics. The transformations are set in
            `data` with the same names as the original columns with the substring
            `"_trans"` suffixed to them. The possible values for `transform` are:
                - `None` - not transforming the data;
                - `"center"` - subtracting the mean from the data;
                - `"standardize"` - subtracting the mean from the data and diving the
                result by the standard deviation.


        """
        self.data_vars = data_vars
        self.path_sentinel3 = path_sentinel3
        self.path_spatial_pred = path_spatial_pred
        self.aoi = aoi
        self.path_landsat = path_landsat
        self._transform = transform

        # Read the data
        # NOTE: this creates instance attributes `sentinel_3_data`, `spatial_pred_data`,
        # `aoi`, `landsat_data` and `timestamp` (Sentinel-3 LST's start sensing time).
        self.read_data()

        # Reproject the original data (`sentinel_3_data`, `spatial_pred_data`, `aoi` and
        # `landsat_data`) to coarse and fine grids.
        self.reproject_data()

        # Get Sentinel-3 grids' coordinates and shapes
        # NOTE: this creates instance attributes `coords` and `shape`
        self.get_coords()
        self.get_shape()

        # Combine all data for each grid, including temporal predictors, convert
        # Datasets to DataFrames and mask them with the AOI
        # NOTE: this creates instance attribute `data` and deletes `sentinel_3_data`,
        # `spatial_pred_data`, `aoi` and `landsat_data`.
        self.combine_convert_mask_data()

        # Get transformed coarse target and coarse and fine spatio-temporal predictors
        # as according to the issued `transform`, by using coarse data statistics. The
        # transform may correspond to none, centring or standardization. The
        # transformations are defined in `data` with the same column names as the
        # original data with substring`"_trans"` suffixed to them.
        self.transform_data()

    @property
    def transform(self) -> Literal["center", "standardize"] | None:
        return self._transform  # type: ignore

    @transform.setter
    def transform(self, value: Literal["center", "standardize"] | None) -> None:
        self._transform = value
        self.transform_data()

    def read_data(self) -> None:
        """
        Get Sentinel-3, spatial predictor, AOI and possibly Landsat data and set them as
        class instance attributes `sentinel3_data`, `spatial_pred_data`, `aoi` and
        `landsat_data`.
        """
        self.sentinel3_data, self.timestamp = self.read_sentinel3_data()
        if self.path_spatial_pred is not None:
            self.spatial_pred_data = self.read_spatial_pred_data()
        if self.aoi is not None:
            self.aoi = load_aoi_to_gdf(self.aoi)  # type: ignore
        if self.path_landsat is not None:
            self.landsat_data = self.read_landsat_data()

    def reproject_data(self) -> None:
        """
        Reproject the original data to coarse and fine grids. These grids correspond to
        the Sentinel-3 original ones after them being clipped to the AOI bounds.
        """
        self.clip_reproject_sentinel3_data()
        if self.path_spatial_pred is not None:
            self.reproject_spatial_pred_data()
        if self.aoi is not None:
            self.rasterize_aoi()
        if self.path_landsat is not None:
            self.reproject_landsat_data()

    def get_coords(self) -> None:
        """
        Extract coordinates of the wrangled coarse and fine Sentinel-3 grids and store
        them in a dictionary keyed by grid alias. Set the dictionary as class instance
        attribute `coords`.
        """
        self.coords = {grid: self.sentinel3_data[grid].coords for grid in self.grids}  # type: ignore

    def get_shape(self) -> None:
        """
        Extract shapes of the wrangled coarse and fine Sentinel-3 grids and store them
        in a dictionary keyed by grid alias. Set the dictionary as class instance
        attribute `shape`. Each shape would correspond to a tuple of integers `(n_y,
        n_x)` where `n_y` and `n_x` are the lengths of the grid matrices along the `y`
        and `x` directions, respectively.
        """
        self.shape = {
            grid: tuple(reversed(list(self.coords[grid].sizes.values())))
            for grid in self.grids
        }

    def combine_convert_mask_data(self) -> None:
        """
        Combine all data (including pure temporal variables) for each grid resulting in
        a dictionary of Datasets keyed by grid alias. These Datasets are then converted
        into DataFrames, all variables' data not within the AOI (DataFrame column of
        name `"aoi"`) is masked out as nan, and the data types of the predictor columns
        are set in accordance with to the attributes of the issued `data_vars`:

            - `data_vars.X_num` (list of numeric predictors);
            - `data_vars.X_cat` (list of categorical predictors).

        The resultant data is set as instance attribute `data` and the original ones
        (`sentinel3_data`, `spatial_pred_data`, `aoi` and `landsat_data`) are deleted.

        Note that masking is required to later make target and spatio-temporal predictor
        statistics (mean and standard deviation) be computed from solely the values
        within the AOI. Note also that setting data types of the predictor columns is
        required to later make the pre-processing pipeline properly discriminate numeric
        and categorical predictors (since discrimination is done based on the data
        types).
        """

        # Combine data for each grid
        self.data = {
            grid: xr.combine_by_coords(
                [
                    data_i[grid]  # type: ignore
                    for data_i in [self.sentinel3_data]
                    + ([self.aoi] if self.aoi is not None else [])
                    + (
                        [self.spatial_pred_data]
                        if self.path_spatial_pred is not None
                        else []
                    )
                    + ([self.landsat_data] if self.path_landsat is not None else [])
                    if grid in data_i  # type: ignore
                ],  # type: ignore
                compat="no_conflicts",
            )
            for grid in self.grids
        }

        # Convert Datasets into DataFrames
        self.data = {
            grid: self.data[grid]
            .to_dataframe(
                # NOTE: dim_order is set in this way to make the resultant column data
                # to have the same order as the one of a two-dimensional numpy array
                # subject to ravel().
                dim_order=tuple(reversed(list(self.data[grid].dims)))
            )
            .reset_index()  # type: ignore
            .drop(columns=self.data[grid]._coord_names)
            for grid in self.grids
        }

        # Combine DataFrames with pure temporal variables
        # NOTE: timestamp is used as string instead of pd.Timestamp - since
        # one-hot-encoder does not properly handle NaT values of pd.Timestamp.
        if "timestamp" in self.data_vars.X_t:
            for grid in self.grids:
                self.data[grid]["timestamp"] = self.timestamp.strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
        if "season" in self.data_vars.X_t:
            for grid in self.grids:
                self.data[grid]["season"] = get_season(self.timestamp)

        # Mask out as nan all variables' records not within the AOI
        if self.aoi is not None:
            for grid in self.grids:
                self.data[grid].loc[self.data[grid]["aoi"].isna(), :] = None

        # Set data types of the predictor columns
        for grid in self.grids:
            self.data[grid][self.data_vars.X_num] = self.data[grid][
                self.data_vars.X_num
            ].apply(pd.to_numeric)

            self.data[grid][self.data_vars.X_cat] = self.data[grid][
                self.data_vars.X_cat
            ].astype("category")

        # Delete original data attributes
        del self.sentinel3_data
        if self.path_spatial_pred is not None:
            del self.spatial_pred_data
        if self.aoi is not None:
            del self.aoi
        if self.path_landsat is not None:
            del self.landsat_data

    def transform_data(self) -> None:
        """
        Apply `transform` on coarse target (as well as validation one) and coarse and
        fine spatio-temporal predictors from a copy of the wrangled `data` by using
        coarse data statistics and set the result as columns with the same names as the
        original ones with substring `"_trans"` suffixed to them. Possible values for
        `transform` are:
            - `None` - not transforming the data;
            - `"center"` - subtracting the mean from the data;
            - `"standardize"` - subtracting the mean from the data and diving the result
            by the standard deviation.

        Further define attributes `mapper_var_to_var_trans` and
        `mapper_var_trans_to_var` corresponding to dictionaries mapping the original
        variable names in each grid to the the new ones after transformation and the
        inverse, respectively. If `transform` is `None`, the new names coincide with the
        original ones. Otherwise, the new names of the transformed variables correspond
        to the original names with the substring `"_trans"` suffixed to them while the
        remainder coincide with the original ones.
        """

        # Drop any variable suffixed with `"_trans"` (so that in the case of `transform`
        # being later set to `None` no transformed variable is kept)
        self.data = {
            grid: self.data[grid].drop(
                columns=[
                    var for var in self.data[grid].columns if var.endswith("_trans")
                ]
            )
            for grid in self.grids
        }

        # Transform the data if there is a transform
        if self.transform == "center":  # type: ignore
            self.center_data()
        if self.transform == "standardize":  # type: ignore
            self.standardize_data()

        # Get original variable names in each grid
        vars = {
            grid: [var for var in self.data[grid].columns if not var.endswith("_trans")]
            for grid in self.grids
        }

        # Get mapper between original variables names and the ones after transformation
        # in each grid
        self.mapper_var_to_var_trans = {
            grid: {
                var: (
                    f"{var}_trans" if f"{var}_trans" in self.data[grid].columns else var
                )
                for var in vars[grid]
            }
            for grid in self.grids
        }

        # Get the inverse mapper
        self.mapper_var_trans_to_var = {
            grid: {
                var_trans: var
                for var, var_trans in self.mapper_var_to_var_trans[grid].items()
            }
            for grid in self.grids
        }

    def read_sentinel3_data(
        self,
    ) -> tuple[dict[Literal["coarse", "fine"], xr.Dataset], pd.Timestamp]:
        """
        Get Sentinel-3's LST and SYN-derived variables (of aliases `data_vars.X_xt`)
        from data folder at `path_sentinel3` as well as LST's start sensing time.

        Returns
        -------
        (data, timestamp) : tuple[dict[{"coarse", "fine"}, xr.Dataset]], pd.Timestamp]
            Tuple of two entries: a dictionary with the Sentinel-3 LST and SYN-derived
            variables (as xarray Datasets) keyed by a grid resolution alias; and LST's
            start sensing time. The grid resolution aliases correspond to:
                - `"coarse"` (resolution of 1000 m) for the case of the LST data;
                - `"fine"` (resolution of 300 m) for the case of the SYN-derived
                variables.
        """
        # Define Sen3Loader object
        sen3loader = Sen3Loader(path=self.path_sentinel3)

        # Get Sentinel-3 data using the Sen3Loader
        data = {
            # Coarse data (original resolution of 1000 m): LST data
            "coarse": sen3loader.get_var_by_name(self.data_vars.y),
            # Fine data (original resolution of 300 m): SYN-derived data
            "fine": xr.Dataset(
                {
                    X_xt_i: sen3loader.get_var_by_name(X_xt_i)
                    for X_xt_i in self.data_vars.X_xt
                }
            ),
        }

        # Get start sensing time associated with Sentinel-3 LST product
        timestamp = sen3loader.timestamp

        return (data, timestamp)  # type: ignore

    def read_spatial_pred_data(self) -> xr.Dataset:
        """
        Get data of spatial predictors (of aliases `data_vars.X_x`) from NetCDF file at
        the issued `path_spatial_pred`. These have a resolution of 0.002 degrees.

        Returns
        -------
        data : xr.Dataset
            The spatial predictor data.
        """

        data = xr.open_dataset(
            self.path_spatial_pred,  # type: ignore
            # Mask out NODATA values and scale
            mask_and_scale=True,
            # Properly decode coordinates and CRS
            decode_coords="all",
        )[self.data_vars.X_x]

        return data

    def read_landsat_data(self) -> xr.Dataset:
        """
        Get Landsat 8/9 LST variable from data folder at `path_landsat`. This has a
        resolution of 30 m.

        Returns
        -------
        data : xr.Dataset
            Landsat LST variable (as xarray Dataset).
        """

        # Get Landsat LST data
        data = (
            xr.open_dataarray(
                self.path_landsat / f"{self.data_vars.y}.TIF",  # type: ignore
                # Mask out NODATA values and scale
                mask_and_scale=True,
                # Properly decode coordinates and CRS
                decode_coords="all",
            )
            # Name DataArray
            .rename(self.data_vars.y_val)
            # Remove singleton "band" dimension
            .squeeze("band", drop=True)
            # Convert DataArray to Dataset
            .to_dataset()
        )

        return data

    def clip_reproject_sentinel3_data(self) -> None:
        """
        Clip Sentinel-3 data to AOI bounds and compute coarse SYN-derived variables from
        fine ones. Clipping of the Sentinel-3 data is done to solely consider the region
        of interest and simultaneously reduce the size of the data.
        """

        # Clip the Sentinel-3 grid and data to AOI bounds
        if self.aoi is not None:
            self.sentinel3_data = {
                grid: self.sentinel3_data[grid].rio.clip_box(  # type: ignore
                    *self.aoi.total_bounds  # type: ignore
                )
                for grid in self.grids
            }

        # Compute Sentinel-3 coarse variables from fine ones
        for X_xt_i in self.data_vars.X_xt:
            self.sentinel3_data["coarse"][X_xt_i] = selective_reproject_match(
                data_src=self.sentinel3_data["fine"][X_xt_i],
                data_target=self.sentinel3_data["coarse"],
            )

    def reproject_spatial_pred_data(self) -> None:
        """
        Compute coarse and fine spatial predictor data from original one and
        Sentinel-3's coarse and fine grids.
        """

        self.spatial_pred_data = {
            grid: selective_reproject_match(
                data_src=self.spatial_pred_data,  # type: ignore
                data_target=self.sentinel3_data[grid],  # type: ignore
            )
            for grid in self.grids
        }

    def reproject_landsat_data(self) -> None:
        """
        Compute coarse and fine Landsat data from original one and Sentinel-3's fine
        grid.
        """

        self.landsat_data = {
            grid: selective_reproject_match(
                data_src=self.landsat_data,  # type: ignore
                data_target=self.sentinel3_data[grid],  # type: ignore
            )
            for grid in self.grids
        }

    def rasterize_aoi(self) -> None:
        """
        Rasterize AOI shape into Sentinel-3's coarse and fine grids.
        """

        # Rasterize AOI shape into coarse and fine grids of Sentinel-3
        # NOTE: Each nested object corresponds to a two-dimensional numpy array.
        self.aoi = {
            grid: shape_to_raster_mask(
                shape=self.aoi,  # type: ignore
                data_target=self.sentinel3_data[grid],  # type: ignore
                fill=np.nan,
            )
            for grid in self.grids
        }

        # Convert nested AOI numpy arrays into xarray DataArrays
        self.aoi = {
            grid: xr.DataArray(
                data=self.aoi[grid],
                coords=self.sentinel3_data[grid].coords,  # type: ignore
                dims=tuple(reversed(list(self.sentinel3_data[grid].dims))),  # type: ignore
                name="aoi",
            )
            for grid in self.grids
        }

    def center_data(self) -> None:
        """
        Defined centered coarse target (as well as validation one) and coarse and fine
        spatio-temporal predictors from a copy of the wrangled `data` by using the
        coarse data statistics. Centering corresponds to subtracting the mean from the
        data. The transformed variables are defined in `data` with the same names as the
        original ones but with the substring `"_trans"` suffixed to them.
        """

        # Get means of the coarse target and spatio-temporal predictors
        vars_coarse_mean = self.data["coarse"][
            self.data_vars.X_xt
            + [self.data_vars.y]
            + ([self.data_vars.y_val] if self.path_landsat is not None else [])
        ].mean()

        # Define centered coarse target and coarse and fine spatio-temporal predictors
        # of the data using the coarse means
        for grid in self.grids:
            # Get names of the target and spatio-temporal predictors of the current grid
            # except the one of the fine target
            vars = self.data_vars.X_xt + (
                (
                    [self.data_vars.y]
                    + ([self.data_vars.y_val] if self.path_landsat is not None else [])
                )
                if grid == "coarse"
                else []
            )
            # Transform
            self.data[grid][[f"{var}_trans" for var in vars]] = (  # type: ignore
                self.data[grid][vars] - vars_coarse_mean[vars]
            )

    def standardize_data(self) -> None:
        """
        Define standardized coarse target (as well as validation one) and coarse and fine
        spatio-temporal predictors from a copy of the wrangled `data` by using coarse
        data statistics. Standardizing corresponds to subtracting te mean from the data
        and dividing the result by the standard deviation. The transformed variables are
        defined in `data` with the same names as the original ones but with the
        substring `"_trans"` suffixed to them.
        """

        # Get means and standard deviations of the coarse target and spatio-temporal
        # predictors
        vars_coarse_mean = self.data["coarse"][
            self.data_vars.X_xt
            + [self.data_vars.y]
            + ([self.data_vars.y_val] if self.path_landsat is not None else [])
        ].mean()
        vars_coarse_std = self.data["coarse"][
            self.data_vars.X_xt
            + [self.data_vars.y]
            + ([self.data_vars.y_val] if self.path_landsat is not None else [])
        ].std()
        # Define standardized coarse target and coarse and fine spatio-temporal
        # predictors of the data using the coarse means
        for grid in self.grids:
            # Get names of the target and spatio-temporal predictors of the current grid
            # except the one of the fine target
            vars = self.data_vars.X_xt + (
                (
                    [self.data_vars.y]
                    + ([self.data_vars.y_val] if self.path_landsat is not None else [])
                )
                if grid == "coarse"
                else []
            )
            self.data[grid][[f"{var}_trans" for var in vars]] = (  # type: ignore
                self.data[grid][vars] - vars_coarse_mean[vars]
            ) / vars_coarse_std[vars]

    def parse_vars(
        self,
        vars: str | list[str] | None = None,
        grid: Literal["coarse", "fine"] | None = None,
        trans: bool = False,
    ) -> dict[Literal["coarse", "fine"], str | list[str]]:
        """
        Parse `vars` argument in `get_data()` and `set_data()` methods.

        Parameters
        ----------

        vars : str or list[str] or None, default=None
            Variables of the data to regard. If not issued, all variables are
            considered.

        grid : {"coarse", "fine", None}, default=None
            Alias of the grid associated with the data. If not issued, the data of both
            grids is considered.

        trans : bool, default=False
            Whether to consider transformed data.

        Returns
        -------
        vars : dict[{"coarse", "fine"}, str | list[str]]
            Data's column names of interest associated with `vars` and constrained to
            `grid` and `trans`. Note that the returned value is keyed by grid alias.
        """

        vars = {
            grid_: (
                # If untransformed data is wanted or there is not transform
                (
                    # If a variable or a particular set of variables is wanted
                    vars
                    if vars is not None
                    # If all variables are wanted
                    else list(self.mapper_var_to_var_trans[grid_].keys())
                )
                if trans is False or self.transform is None  # type: ignore
                # If transformed data is wanted and there is a transform
                # If a variable is wanted
                else (
                    self.mapper_var_to_var_trans[grid_][vars]
                    if isinstance(vars, str)
                    # If a particular set of variables is wanted
                    else (
                        [self.mapper_var_to_var_trans[grid_][var] for var in vars]
                        if isinstance(vars, list)
                        # If all variables are wanted
                        else list(self.mapper_var_to_var_trans[grid_].values())
                    )
                )
            )
            for grid_ in ([grid] if grid is not None else self.grids)
        }  # type: ignore

        return vars  # type: ignore

    def get_data(
        self,
        grid: Literal["coarse", "fine"] | None = None,
        vars: str | list[str] | None = None,
        trans: bool = False,
    ) -> (
        pd.Series
        | pd.DataFrame
        | dict[Literal["coarse", "fine"], pd.Series | pd.DataFrame]
    ):
        """
        Get wrangled and, if `trans` is `True`, further transformed data `vars` for
        issued `grid` alias.

        Note that if `vars` is not issued, the data of all variables is returned. Also,
        if `grid` is not issued, the returned value corresponds to a dictionary of data
        of both grids (coarse and fine), keyed by grid alias. If the instance has no
        transformation (attribute `transform` is `None`), the untransformed data is the
        one considered regardless of the value of `trans`.

        Parameters
        ----------

        grid : {"coarse", "fine", None}, default=None
            Alias of the grid associated with the data. If not issued, the data of both
            grids is returned.

        vars : str or list[str] or None, default=None
            Variables of the data to return. If not issued, the data of all variables is
            returned.

        trans : bool, default=False
            Whether to get transformed data.

        Returns
        -------
        data : pd.Series or pd.DataFrame or dict[{"coarse", "fine"}, pd.Series or
        pd.DataFrame]
            Wrangled and, if `trans` is `True`, further transformed data `vars` for
            issued `grid` alias. Note that if `vars` is not issued, the data of all
            variables is returned. Also, if `grid` is not issued, the returned value
            corresponds to a dictionary of data of both grids (coarse and fine) keyed by
            grid alias. If the instance has no transformation (attribute `transform` is
            `None`), the untransformed data is the one considered regardless of the
            value of `trans`.
        """

        vars = self.parse_vars(vars=vars, grid=grid, trans=trans)  # type: ignore

        data = {
            grid_: self.data[grid_][vars[grid_]]  # type: ignore
            for grid_ in ([grid] if grid is not None else self.grids)
        }

        # If transformed data is wanted and there is a transform, convert names of the
        # transformed variables to the original ones (that is, without suffix "_trans")
        # so that they may be recognized later (e.g. in DataWrangler and Downscaler)
        if trans is True and self.transform is not None:  # type: ignore
            for grid_ in [grid] if grid is not None else self.grids:
                for var in (
                    [vars[grid_]]  # type: ignore
                    if isinstance(vars[grid_], str)  # type: ignore
                    else vars[grid_]  # type: ignore
                ):
                    if isinstance(data[grid_], pd.Series):
                        data[grid_] = data[grid_].rename(
                            self.mapper_var_trans_to_var[grid_][var]
                        )
                    else:
                        data[grid_] = data[grid_].rename(
                            columns=self.mapper_var_trans_to_var[grid_],  # type: ignore
                        )
        # Squeeze
        if grid is not None:
            data = data[grid]

        return data  # type: ignore

    def get_data_X_and_mask(
        self,
        grid: Literal["coarse", "fine"] | None = None,
        trans: bool = False,
    ) -> pd.DataFrame | dict[Literal["coarse", "fine"], pd.DataFrame]:
        """
        Get wrangled and, if `trans` is `True`, further transformed predictor and AOI
        mask data for issued `grid` alias.

        Note that if `grid` is not issued, the returned value corresponds to a
        dictionary of data of both grids (coarse and fine), keyed by grid alias. If
        there is no transform in the instance (attribute `transform` is `None`), the
        untransformed data is the one considered regardless of the value of `trans`.

        Parameters
        ----------

        grid : {"coarse", "fine", None}, default=None
            Alias of the grid associated with the data. If not issued, the data of both
            grids is returned.

        trans : bool, default=False
            Whether to get transformed data.

        Returns
        -------
        data_X_and_mask : pd.DataFrame or dict[{"coarse", "fine"}, pd.DataFrame]
            Wrangled and, if `trans` is `True`, further transformed predictor and AOI
            mask data for issued `grid` alias. If `grid` is not issued, the returned
            value corresponds to a dictionary of data of both grids (coarse and fine)
            keyed by grid alias. If there is no transform in the instance (attribute
            `transform` is `None`), the untransformed data is the one considered
            regardless of the value of `trans`.
        """

        return self.get_data(
            grid=grid,
            vars=self.data_vars.X + (["aoi"] if self.aoi is not None else []),  # type: ignore
            trans=trans,
        )  # type: ignore

    def get_data_y(
        self,
        grid: Literal["coarse", "fine"] | None = None,
        trans: bool = False,
    ) -> pd.Series | dict[Literal["coarse", "fine"], pd.Series]:
        """
        Get wrangled and, if `trans` is `True`, further transformed target data for
        issued `grid` alias. If `grid` is not issued, the returned value corresponds to
        a dictionary of data of both grids (coarse and fine), keyed by grid alias. If
        there is no transform in the instance (attribute `transform` is `None`), the
        untransformed data is the one considered regardless of the value of `trans`.
        There is no transformed fine target in any case.

        Parameters
        ----------

        grid : {"coarse", "fine", None}, default="coarse"
            Alias of the grid associated with the data. If not issued, the data of both
            grids is returned.

        trans : bool, default=False
            Whether to get transformed data.

        Returns
        -------
        data_y : pd.Series or dict[{"coarse", "fine"}, pd.Series]
            Wrangled and, if `trans` is `True`, further transformed target data for
            issued `grid` alias. If `grid` is not issued, the returned value corresponds
            to a dictionary of data of both grids (coarse and fine) keyed by grid alias.
            If there is no transform in the instance (attribute `transform` is `None`),
            the untransformed data is the one considered regardless of the value of
            `trans`. There is no transformed fine target in any case.
        """

        return self.get_data(grid=grid, vars=self.data_vars.y, trans=trans)  # type: ignore

    def set_data(
        self,
        values: (
            pd.Series
            | pd.DataFrame
            | dict[Literal["coarse", "fine"], pd.Series | pd.DataFrame]
        ),
        vars: str | list[str] | None = None,
        grid: Literal["coarse", "fine"] | None = None,
        trans: bool = False,
    ) -> None:
        """
        Set wrangled and, if `trans` is `True`, further transformed data `vars` of
        issued `grid` alias to `values`.

        Note that `vars` may correspond to new variables. If not defined, `vars` is set
        to all variables of the data. If `grid` is not issued, the data of both grids
        (coarse and fine) is set. If there is no transform in the instance (attribute
        `transform` is `None`), the untransformed data is the one considered regardless
        of the value of `trans`.

        Parameters
        ----------

        values: pd.Series or pd.DataFrame or dict[{"coarse", "fine"}, pd.Series or
        pd.DataFrame]
            Values to set.

        vars : str or list[str] or None, default=None
            Variables of the data to set. Note that `vars` may correspond to new
            variables. If not defined, `vars` is set as all variables of the data.

        grid : {"coarse", "fine", None}, default=None
            Alias of the grid associated with the data. If not issued, the data of both
            grids is set.

        trans : bool, default=False
            Whether to set transformed data.
        """

        vars = self.parse_vars(vars=vars, grid=grid, trans=trans)  # type: ignore

        for grid_ in [grid] if grid is not None else self.grids:
            self.data[grid_][vars[grid_]] = (  # type: ignore
                values if grid is not None else values[grid_]  # type: ignore
            )

        # If a variable with a completely new name is set, add it to the mappers of
        # names `mapper_var_to_var_trans` and `mapper_var_trans_to_var`
        for grid_ in [grid] if grid is not None else self.grids:
            for var in [vars[grid_]] if isinstance(vars[grid_], str) else vars[grid_]:  # type: ignore
                if var not in (
                    list(self.mapper_var_to_var_trans[grid_].keys())
                    + list(self.mapper_var_to_var_trans[grid_].values())
                ):
                    self.mapper_var_to_var_trans[grid_][var] = var
                    self.mapper_var_trans_to_var[grid_][var] = var

    def save(self, path: Path) -> None:
        """
        Write the instance to `path` with `joblib`.

        Parameters
        ----------
        path : Path
            Path to write the instance to.
        """

        joblib.dump(value=self, filename=path)
