# ---> Import packages
from __future__ import annotations

import gc
import logging
import os
import shutil
import zipfile
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray as rxr
import xarray as xr
from rasterio.control import GroundControlPoint
from rasterio.warp import Resampling

# Import configurations for querying, downloading and filtering Sentinel-3 LST and SYN
# products
from s3lst_ds.data_download.download_sentinel3_config import (
    Sentinel3Config,
    config,
)

# Import querying and downloading utilities associated with CDSE's OData API
from s3lst_ds.utilities.cdse_utils import (
    CDSEAuthState,
    download_cdse,
    query_cdse,
)
from s3lst_ds.utilities.exceptions_utils import (
    CleanupError,
    CloudCoverComputationError,
    CloudCoverLimitError,
    ConfigError,
    InvalidProductError,
    MaskCloudsError,
    MultipleMatchingProdutsError,
    NoMatchingProductError,
    NoProductFoundError,
    ReprojectionError,
    UnsupportedProductError,
    UnzipError,
)

# Import geometry utilities
from s3lst_ds.utilities.geometry_utils import (
    footprint_aoi_overlap,
    load_aoi_to_gdf,
    to_convex_hull_wkt,
)
from s3lst_ds.utilities.jobs_utils import parse_n_jobs
from s3lst_ds.utilities.logging_utils import RichLogger
from s3lst_ds.utilities.snappy_utils import import_esa_snappy
from s3lst_ds.utilities.warnings_utils import suppress_warnings

# ---> Configure environment

# Suppress general warnings
suppress_warnings()

# ------ Reproject Sentinel-3 product, subset to bounding box and bands ------ #


class Sen3Processor:
    """
    A class for processing Sentinel-3 products, namely:
    - reproject to EPSG:4326;
    - subset to bands of interest;
    - subset to the bounding box of a geometry;
    - write to NetCDF file;
    - delete original file.

    Attributes
    ----------

    engine : {"rioxarray", "snappy"}, default="rioxarray"
        The alias of the backend Python package to use in the processing of the
        Sentinel-3 products. This is either
        - [`"rioxarray"`](https://corteva.github.io/rioxarray/stable/);
        - [`"snappy"`](https://github.com/senbox-org/esa-snappy) (that is, the
        aliases for the respective backend Python packages).

    esa_snappy : ModuleType or None
        ESA's `snappy` imported module if the selected engine corresponds to `"snappy"`,
        otherwise `None`.

    logger : RichLogger or None, default=None
        A rich logger for showing the progress of the processing.

    """

    def __init__(
        self,
        engine: Literal["rioxarray", "snappy"] = "rioxarray",
        logger: RichLogger | None = None,
    ) -> None:
        self._engine = engine
        self.logger = logger
        self.esa_snappy = self.get_esa_snappy()

    @property
    def engine(self) -> Literal["rioxarray", "snappy"]:
        return self._engine  # type: ignore

    @engine.setter
    def engine(self, value: Literal["rioxarray", "snappy"]) -> None:
        self._engine = value
        self.esa_snappy = self.get_esa_snappy()

    def get_esa_snappy(self) -> ModuleType | None:
        """
        Get the imported `esa_snappy` module if the selected engine corresponds to
        `"snappy"`.

        Returns
        -------
        ModuleType or None
            The imported `esa_snappy` module if the selected engine corresponds to
            `"snappy"`, otherwise `None`.
        """
        return (
            import_esa_snappy(logger=self.logger) if self.engine == "snappy" else None
        )

    def process(
        self,
        prod_path: Path,
        geometry: str | None = None,
    ) -> None:
        """
        Reproject the Sentinel-3 product at `prod_path` to EPSG:4326, subset it to
        bands of interest and to the bounding box of `geometry` using issued
        `engine`, write result to NetCDF file and delete original file.

        The `engine` may be either correspond to
        [`"rioxarray"`](https://corteva.github.io/rioxarray/stable/) or
        [`"snappy"`](https://github.com/senbox-org/esa-snappy) (that is, the aliases
        for the respective backend Python packages).

        The bands of interest correspond to:

        - `LST` and `bayes_in` if the product is of LST kind;
        - `SDR_Oa06`, `SDR_Oa08` and `SDR_Oa17` if the product is of SYN kind.

        Parameters
        ----------
        prod_path : Path
            Path to the Sentinel-3 product to be processed.
        geometry : str or None, default=None
            A WKT string representing the area of interest (AOI) to which the product
            should be clipped. If `None`, no clipping is performed.
        """

        if self.engine == "snappy":
            self.process_snappy(prod_path=prod_path, geometry=geometry)
        else:
            # NOTE: rioxarray may raise warnings during reprojection, namely,
            # "CPLE_NotSupported in warp options does not support option
            # SRC_METHOD". These warnings actually do not make sense since the
            # option does take an effect. The warnings may then be safely ignored.
            if self.logger is not None:
                level = self.logger.level
                self.logger.level = logging.ERROR
                self.process_rioxarray(prod_path=prod_path, geometry=geometry)
                self.logger.level = level

    def process_rioxarray(
        self,
        prod_path: Path,
        geometry: str | None = None,
    ) -> None:
        """
        Reproject with rioxarray the product at `prod_path` to EPSG:4326, subset it to
        bands of interest and to the bounding box of `geometry` write result to NetCDF
        file and delete original file.

        The bands of interest correspond to:

        - `LST` and `bayes_in` if the product is of LST kind;
        - `SDR_Oa06`, `SDR_Oa08` and `SDR_Oa17` if the product is of SYN kind.

        Georeferencing is such that Ground Control Points (GCPs) that associate image
        row and column coordinates to spatial coordinates are defined from the source
        data. The respective transform is obtained using Thin Plate Spline (TPS) applied
        on these GCPs. The coordinates of the source dataset are then converted to
        spatial ones using it. A regular grid is then fitted to the resultant
        coordinates and the variable values on each pixel are estimated from the source
        ones and source and target grids' spatial coordinates using nearest neighbour
        resampling.

        For more details on how georeferencing can be done with rioxarray, see [this
        Medium article](https://archive.ph/Ltqa2) and this [GitHub
        discussion](https://github.com/corteva/rioxarray/discussions/329#discussioncomment-1401571).

        Parameters
        ----------
        prod_path : Path
            Path to the Sentinel-3 product to be processed.
        geometry : str or None, default=None
            A WKT string representing the area of interest (AOI) to which the product
            should be clipped. If `None`, no clipping is performed.
        """

        # ---> Infer kind of product (LST or SYN), define respective bands of interest and
        # the files associated with them as well as a parameters for the definition of
        # ground control points
        if "SL_2_LST" in prod_path.name:
            # Variables of interest
            vars = ["LST", "bayes_in"]

            # Resampling method for each variable in the georeferencing
            # NOTE: nearest neighbour resampling is used for `bayes_in` since it is a
            # categorical variable.
            # NOTE: from all tried numerical resampling methods (bilinear, cubic, cubic
            # spline, average and laczos), average was found to be the one producing the
            # best test fine perfomance.
            # NOTE: from all tried categorical resampling methods (nearest neighbour and
            # mode), nearest neighbour was found to be the one producing the best test fine
            # perfomance. Note, however, that the mode method produced almost coincident
            # results - the differences may in fact be regarded as negligible.
            resampling = {
                "LST": Resampling.average,
                "bayes_in": Resampling.nearest,
            }

            # Mapper between axis aliases and coordinate variables
            coord = {"y": "latitude_in", "x": "longitude_in", "z": "elevation_in"}

            # Step on image row and column coordinates to consider for the definition of
            # ground control points (GCPs), to be used in the georeferencing of the data
            # WARNING: if too small, errors may arise due to combined near-coincidence of
            # the GCPs and floating point truncation.
            gcp_step = 25

        elif "SY_2_SYN" in prod_path.name:
            # Variables of interest
            vars = ["SDR_Oa06", "SDR_Oa08", "SDR_Oa17"]

            # Resampling method for each variable in the georeferencing
            resampling = {
                "SDR_Oa06": Resampling.average,
                "SDR_Oa08": Resampling.average,
                "SDR_Oa17": Resampling.average,
            }

            # Mapper between axis aliases and coordinate variables
            coord = {"y": "lat", "x": "lon", "z": "altitude"}

            # Step on image row and column coordinates to consider for the definition of
            # ground control points (GCPs), to be used in the georeferencing of the data
            # WARNING: if too small, errors may arise due to combined near-coincidence of
            # the GCPs and floating point truncation.
            gcp_step = 75
        else:
            raise UnsupportedProductError(
                "Product"
                f"\n{prod_path.name!r}"
                "\nis not supported."
                "\nIt must be of LST or SYN type."
            )
        axes = coord.keys()

        # Mapper between variable aliases and the names of the files where they exist
        file = {
            "latitude_in": "geodetic_in.nc",
            "longitude_in": "geodetic_in.nc",
            "elevation_in": "geodetic_in.nc",
            "LST": "LST_in.nc",
            "bayes_in": "flags_in.nc",
            "lat": "geolocation.nc",
            "lon": "geolocation.nc",
            "altitude": "geolocation.nc",
            "SDR_Oa06": "Syn_Oa06_reflectance.nc",
            "SDR_Oa08": "Syn_Oa08_reflectance.nc",
            "SDR_Oa17": "Syn_Oa17_reflectance.nc",
        }

        # ---> Read, reproject and subset product
        # Read data of each variable of interest
        data_vars = {
            var: rxr.open_rasterio(
                filename=f"netcdf:{Path(prod_path)}/{file[var]}",
                variable=var,
                # Mask out NODATA values and scale
                mask_and_scale=True,
            )
            .load()  # type: ignore
            .squeeze(drop=True)[var]
            for var in vars
        }

        # Read data of each coordinate variable
        data_coords = {
            axis: rxr.open_rasterio(
                filename=f"netcdf:{Path(prod_path)}/{file[coord[axis]]}",
                variable=coord[axis],
                # Mask out NODATA values and scale
                mask_and_scale=True,
            )
            .load()  # type: ignore
            .squeeze(drop=True)[coord[axis]]
            for axis in axes
        }

        # Convert AOI geometry to a GeoDataFrame
        geometry = load_aoi_to_gdf(geometry)  # type: ignore

        # Merge coordinates into single dataset
        data_coords = xr.merge(data_coords.values(), compat="no_conflicts")

        # Set the NODATA value of `bayes_in` since it is missing it
        if "bayes_in" in vars:
            data_vars["bayes_in"].rio.write_nodata(255, encoded=True, inplace=True)

        # Get list of ground control points (points mapping image row and column coordinates
        # to spatial x, y, z coordinates)
        # NOTE: row and column coordinates are 0 at upper left corner fo the upper left
        # pixel. And the values of the xarray datasets are associated with the pixel
        # centres.
        # NOTE: A step on image row and column coordinates is considered in the definition
        # of ground control points (GCPs).
        gcps = []
        for row in range(0, data_coords.sizes["y"], gcp_step):
            for col in range(0, data_coords.sizes["x"], gcp_step):
                gcps.append(
                    GroundControlPoint(
                        row=row + 0.5,
                        col=col + 0.5,
                        x=data_coords[coord["x"]].values[row][col],
                        y=data_coords[coord["y"]].values[row][col],
                        z=data_coords[coord["z"]].values[row][col],
                    )
                )

        # Set CRS (Coordinate Reference System) of the variables of interest
        # WARNING: rio.reproject method requires the source CRS to be set even though it is
        # not used when GCPs are considered in the reprojection
        data_vars = {
            var: data_vars[var].rio.write_crs(input_crs="EPSG:4326", inplace=True)
            for var in vars
        }

        # Georeference the data
        # NOTE: a transform from image row and column coordinates to spatial coordinates is
        # inferred from the GCPs. The coordinates of the source dataset are then converted
        # to spatial ones using it. A regular grid and its resolution are then fitted to the
        # resultant coordinates and the variable values on each pixel are estimated from the
        # source ones and the source and regular grid spatial coordinates.
        # WARNING: it seems that reprojection with GCPs is agnostic to image row and column
        # coordinate values of the source product since no previous change to them had
        # affected the outcome. The method seems to repopulate them with values from 0.5 to
        # N - 0.5 where N is the number of rows and columns, respectively.
        data_vars = {
            var: data_vars[var].rio.reproject(
                # CRS of the target grid
                dst_crs="EPSG:4326",
                # Use GCPs for the reprojection
                # NOTE: https://rasterio.readthedocs.io/en/stable/api/rasterio.warp.html#rasterio.warp.reproject
                gcps=gcps,
                # Resampling method
                # https://rasterio.readthedocs.io/en/stable/api/rasterio.enums.html#rasterio.enums.Resampling
                resampling=resampling[var],
                # Use thin plate splines (TPS) for inferring the transform from GCPs
                # NOTE: `src_method` is an option of the function
                # `GDALCreateGenImgProjTransformer2`
                # (https://gdal.org/en/stable/api/gdal_alg.html#_CPPv432GDALCreateGenImgProjTransformer212GDALDatasetH12GDALDatasetH12CSLConstList)
                # used by the base method `rasterio.warp.reproject`
                # (https://rasterio.readthedocs.io/en/stable/api/rasterio.warp.html#rasterio.warp.reproject)
                # considered by `rio.reproject`.
                src_method="GCP_TPS",
                # Use all except 1 available CPU cores for the reprojection
                # NOTE: `num_threads` is an option of the function
                # `GDALWarpOptions`
                # (https://gdal.org/en/stable/api/gdalwarp_cpp.html#_CPPv415GDALWarpOptions)
                # used by the base method `rasterio.warp.reproject`
                # (https://rasterio.readthedocs.io/en/stable/api/rasterio.warp.html#rasterio.warp.reproject)
                # considered by `rio.reproject`.
                num_threads=parse_n_jobs(-2),
            )
            for var in vars
        }

        # Merge the data
        data_vars = xr.merge(data_vars.values(), compat="no_conflicts")

        # Clip the data to the AOI geometry
        if geometry is not None:
            # Transform AOI geometry to the same CRS as the data
            geometry = geometry.to_crs(crs=data_vars.rio.crs)  # type: ignore

            # Get AOI bounds
            minx, miny, maxx, maxy = geometry.total_bounds  # type: ignore

            # Clip the data to the AOI bounding box
            data_vars = data_vars.rio.clip_box(minx, miny, maxx, maxy)

        # Write the data to NetCDF
        data_vars.to_netcdf(f"{prod_path}.nc")

        # Close files
        data_vars.close()
        data_coords.close()

        # Collect garbage
        gc.collect()

        # Delete original product file
        shutil.rmtree(prod_path)

    def process_snappy(
        self,
        prod_path: Path,
        geometry: str | None = None,
    ) -> None:
        """
        Reproject with snappy the product at `prod_path` to EPSG:4326, subset it to
        bands of interest and to the bounding box of `geometry` write result to NetCDF
        file and delete original file.

        The bands of interest correspond to:

        - `LST` and `bayes_in` if the product is of LST kind;
        - `SDR_Oa06`, `SDR_Oa08` and `SDR_Oa17` if the product is of SYN kind.

        Note that snappy considers band `bayes_in` of the LST product files to have 0 as
        NODATA value when 0 is also a possible non-NODATA value. Fortunately, this
        NODATA value is not imposed when reading the file through snappy (as one may
        check that through p.getBand("bayes_in").isNoDataValueUsed()). Since the data
        type of the band is uint8, the allowed values for NODATA are integers between 0
        and 255. And since the maximum meaningful value is 128 (associated with the
        seventh bit) (one may confirm this by using rioxarray and checking the variable
        attributes), it was decided to set NODATA to 255. Note that in contrast with
        snappy, rioxarray does not consider `bayes_in` to have its NODATA value set.

        Parameters
        ----------
        prod_path : Path
            Path to the Sentinel-3 product to be processed.
        geometry : str or None, default=None
            A WKT string representing the area of interest (AOI) to which the product
            should be clipped. If `None`, no clipping is performed.
        """

        # ---> Infer kind of product (LST or SYN) and define respective bands of interest
        if "SL_2_LST" in prod_path.name:
            prod_kind = "LST"
            band_names = ["LST", "bayes_in"]
        elif "SY_2_SYN" in prod_path.name:
            prod_kind = "SYN"
            band_names = ["SDR_Oa06", "SDR_Oa08", "SDR_Oa17"]
        else:
            raise UnsupportedProductError(
                "Product"
                f"\n{prod_path.name!r}"
                "\nis not supported."
                "\nIt must be of LST or SYN type."
            )

        # ---> Read, reproject and subset product with snappy

        # Read Sentinel-3 product
        # NOTE: It returns an object of type org.esa.snap.core.datamodel.Product
        # (http://step.esa.int/docs/v5.0/apidoc/engine/org/esa/snap/core/datamodel/Product.html))
        p = self.esa_snappy.ProductIO.readProduct(str(prod_path / "xfdumanifest.xml"))  # type: ignore

        # In the case of the LST product, set an appropriate NODATA value (255) to the
        # "bayes_in" band since snappy considers it to value 0 (which is also a possible
        # meaningful value).
        if prod_kind == "LST":
            p.getBand("bayes_in").setNoDataValue(255)
            p.getBand("bayes_in").setNoDataValueUsed(True)

        # Get identifier of the geographic CRS
        geo_crs_identifier = (
            self.esa_snappy.ArrayList(  # type: ignore
                p.getSceneGeoCoding().getGeoCRS().getIdentifiers()
            )
            .get(0)
            .toString()
        )

        # Reproject product
        # NOTE: Read the documentation for Reproject operator:
        # http://step.esa.int/docs/v5.0/apidoc/engine/org/esa/snap/core/gpf/common/reproject/ReprojectionOp.html
        params = self.esa_snappy.HashMap()  # type: ignore
        params.put("crs", geo_crs_identifier)  # Target CRS
        # Resampling method (e.g. "Nearest", "Bilinear", "Bicubic")
        # NOTE: It seems that there is a bug in Snappy: for some product types,
        # "Nearest" resampling method is picked regardless of the picked value for the
        # "resamplingName" parameter (see https://senbox.atlassian.net/browse/SNAP-1365)
        params.put("resamplingName", "Bilinear")
        p_reproj = self.esa_snappy.GPF.createProduct(  # type: ignore
            # Name of the operator to be applied on the source product
            "Reproject",
            # HashMap of parameters to be used by the operator
            params,
            # Source product
            p,
        )

        # Create a subset product constrained to the bounding box of the issued
        # geometry, having solely the wanted bands and disregarding any tie-point grid
        # (the tie-point grids are herein not required)
        # NOTE: Read the documentation for Subset operator:
        # http://step.esa.int/docs/v5.0/apidoc/engine/org/esa/snap/core/gpf/common/SubsetOp.html
        params = self.esa_snappy.HashMap()  # type: ignore
        params.put("bandNames", self.esa_snappy.String(", ".join(band_names)))  # type: ignore
        if geometry is not None:
            params.put("geoRegion", self.esa_snappy.String(geometry))  # type: ignore
        params.put("tiePointGridNames", self.esa_snappy.String(","))  # type: ignore
        p_subset = self.esa_snappy.GPF.createProduct(  # type: ignore
            # Name of the operator to be applied on the source product
            "Subset",
            # HashMap of parameters to be used by the operator
            params,
            # Source product
            p_reproj,
        )

        # ---> Get reprojected product coordinates (longitudes and latitudes)
        # Get reprojected product dimensions
        width = p_subset.getSceneRasterWidth()
        height = p_subset.getSceneRasterHeight()

        # Get the reprojected product coordinates
        geo_coding = p_subset.getSceneGeoCoding()
        # NOTE: Pixel coordinates (that is, positions along the pixel matrix in units of
        # pixel width and height, respectively) in the PixelPos() method correspond to the
        # pixel indices with an offset of 0.5 to get the spatial coordinates (that is, the
        # longitudes and latitudes) at the centre of the pixels and not at their upper left
        # corner.
        lon = [
            geo_coding.getGeoPos(
                self.esa_snappy.PixelPos(i_x + 0.5, 0 + 0.5),  # type: ignore
                None,  # type: ignore
            ).getLon()
            for i_x in range(width)
        ]

        lat = [
            geo_coding.getGeoPos(
                self.esa_snappy.PixelPos(0 + 0.5, i_y + 0.5),  # type: ignore
                None,  # type: ignore
            ).getLat()
            for i_y in range(height)
        ]

        # ---> Read bands' values and attributes, and create an Xarray Dataset from them
        bands = []
        for band_name in band_names:
            # Get band from product
            band = p_subset.getBand(band_name)

            # Get the band values
            data = np.zeros((height, width), np.float32)
            # NOTE: readPixels() already applies scaling and offsetting, and, therefore,
            # the returned values are already the physical ones.
            band.readPixels(0, 0, width, height, data)

            # Get band attributes
            nodata = band.getNoDataValue()
            attrs = {
                "long_name": (
                    band.getDescription() if band.getDescription() is not None else ""
                ),
                "units": (band.getUnit() if band.getUnit() is not None else ""),
                "valid_max": data[data != band.getNoDataValue()].max(),
                "valid_min": data[data != band.getNoDataValue()].min(),
            }

            # Create DataArray associated with the band
            # NOTE: as mentioned in rioxarray documentation
            # (https://corteva.github.io/rioxarray/html/rioxarray.html#rioxarray.open_rasterio)
            # coords should correspond to the coordinates of the pixels' centres, hence
            # why before special care was taken to ensure that these were then ones
            # extracted.
            band = xr.DataArray(
                data=data,
                name=band_name,
                dims=("y", "x"),
                coords={
                    "x": lon,
                    "y": lat,
                },
                attrs=attrs,
            )

            # Define the DataArray CRS
            band.rio.write_crs(geo_crs_identifier, inplace=True)

            # Define the DataArray NODATA value
            band.rio.write_nodata(nodata, inplace=True)

            # Append DataArray to list
            bands.append(band)

        # Create Dataset from DataArrays
        bands = xr.merge(bands, compat="no_conflicts")

        # Remove Dataset attributes since xr.merge() defined them as the ones of the first
        # DataArray.
        bands.attrs = {}

        # Set Dataset spatial dimension attributes
        bands["x"].attrs = {
            "axis": "X",
            "long_name": "longitude",
            "standard_name": "longitude",
            "valid_max": lon[-1],
            "valid_min": lon[0],
            "units": "degrees_east",
        }
        bands["y"].attrs = {
            "axis": "Y",
            "long_name": "latitude",
            "standard_name": "latitude",
            "valid_max": lat[0],
            "valid_min": lat[-1],
            "units": "degrees_north",
        }

        # Write the Dataset to NetCDF
        bands.to_netcdf(f"{prod_path}.nc")

        # Close products
        # NOTE: in Windows, if the produces are not closed, the program may not be able to
        # delete the product source files later.
        p.dispose()
        p_reproj.dispose()
        p_subset.dispose()

        # Delete original product file
        shutil.rmtree(prod_path)


# ------------ Data class to hold product info from query results ------------ #
@dataclass
class ProdInfo:
    """
    Simple data class to hold particular product info obtained from query results.

    Attributes
    ----------

    id : str
        Product Id (to be used for performing download through the API).

    name : str
        Product file name.

    platform: str
        Sentinel-3 satellite serial identifier: "A" (for Sentinel-3A) or "B" (for
        Sentinel-3B).

    generating_centre:
        Identifier of the generating centre for the data product. The parameter may be
        inferred from the product file name. Note that this seems to be different from
        attribute "processingCenter".

        For more details check respective entry in the Sentinel 3 File Naming Convention
        (https://sentinels.copernicus.eu/documents/247904/1964331/Sentinel-3_PDGS_File_Naming_Convention#page=13)
        as well as the news on Sentinel Online for updates on the possible values
        (https://sentinels.copernicus.eu/web/sentinel/-/copernicus-sentinel-3a-data-user-news)

    cycle: int
        Cycle number at the start sensing time. A cycle corresponds to the required time
        for a same Sentinel-3 satellite pass through the same location (385 full
        orbits). The cycle number is the number identifier for that cycle, which
        increments by 1 every time a cycle is completed. The parameter may be inferred
        from the product file name or from the respective attribute ("cycleNumber") in
        the query result.

        For more details check respective entry in the Sentinel 3 File Naming Convention
        (https://sentinels.copernicus.eu/documents/247904/1964331/Sentinel-3_PDGS_File_Naming_Convention#page=13)
        as well as the Sentinel-3 Mission Wiki page
        (https://sentiwiki.copernicus.eu/web/s3-mission#S3Mission-OrbitS3-Mission-Orbittrue).

    relative_orbit: int
        Relative orbit number within the cycle at the start sensing time. This is a
        number identifier of the orbit in the cycle. It starts at 1 and ends at 385,
        incrementing by 1 every time an orbit is completed and resetting to 1after 385.
        The parameter may be inferred from the product file name or from the respective
        attribute ("relativeOrbitNumber") in the query result.

        For more details check respective entry in the Sentinel 3 File Naming Convention
        (https://sentinels.copernicus.eu/documents/247904/1964331/Sentinel-3_PDGS_File_Naming_Convention#page=13)
        as well as the documentation for the Sentinel-3 KML Orbit Files
        (https://sentiwiki.copernicus.eu/__attachments/1672112/GMV-GMESPOD-SRN-0012%20-%20Sentinel%203%20KML%20orbit%20files%202016%20-%201.0.pdf?inst-v=e49045fb-24ed-4f58-8b10-4339088f6644#page=11).

    orbit: int
        Absolute orbit number. This is a number identifier for the orbit since a
        reference time. It increments by 1 every time an orbit is completed, but without
        resetting to 1 after 385 being completed as in the case of the relative orbit
        number. Due to a tandem operational phase (for calibration) of the Sentinel-3
        satellites, the computation of the absolute orbit number from the cycle and
        relative orbit number is not straightforward. Indeed, the actual formulas are

        - for the case of Sentinel-3A: `orbit = cycle * 385 + relative_orbit - 412`
        - for the case of Sentinel-3B: `orbit = cycle * 385 + relative_orbit - 4348`

        For more details, check this post in ESA's step forum
        (https://forum.step.esa.int/t/how-to-calculate-sentinel-3-absolute-orbit-number-from-filename/27507/2)

    start_sensing_time: str
        Start sensing datetime. The parameter may be obtained from the respective
        attribute (more specifically, value of key "Start" of attribute "ContentDate")
        or inferred from the product file name (though with different string formats).

        For more details, check respective entry in the Sentinel 3 File Naming
        Convention
        (https://sentinels.copernicus.eu/documents/247904/1964331/Sentinel-3_PDGS_File_Naming_Convention#page=12)

    end_sensing_time: str
        End sensing datetime. The parameter may be obtained from the respective
        attribute (more specifically, value of key "End" of attribute "ContentDate")
        or inferred from the product file name (though with different string formats).

        For more details, check respective entry in the Sentinel 3 File Naming
        Convention
        (https://sentinels.copernicus.eu/documents/247904/1964331/Sentinel-3_PDGS_File_Naming_Convention#page=12)


    cloud_cover: float
        Cloud cover fraction (in percentage) in the whole sensed area.

    """

    id: str
    name: str
    platform: str
    generating_centre: str
    cycle: int | None
    relative_orbit: int
    orbit: int
    start_sensing_time: str
    end_sensing_time: str
    cloud_cover: float | None

    @staticmethod
    def load_from_df(products: pd.DataFrame) -> list[ProdInfo]:
        """
        Get a list of ProdInfo instances from a DataFrame of query items.
        """
        prod_infos = [
            ProdInfo(
                id=prod["Id"],  # type: ignore
                name=prod["Name"],  # type: ignore
                platform=prod["platformSerialIdentifier"],  # type: ignore
                generating_centre=prod["Name"][82:85],  # type: ignore
                # NOTE: "cycleNumber" might not be a field of the product for some
                # datetimes
                cycle=(prod["cycleNumber"] if "cycleNumber" in prod.index else None),
                relative_orbit=prod["relativeOrbitNumber"],  # type: ignore
                orbit=prod["orbitNumber"],  # type: ignore
                start_sensing_time=prod["Name"][16:31],  # type: ignore
                end_sensing_time=prod["Name"][32:47],  # type: ignore
                # NOTE: "cloudCover" might not be a field of the product for some
                # datetimes
                cloud_cover=(
                    prod["cloudCover"] if "cloudCover" in prod.index else None
                ),
            )
            for i, prod in products.iterrows()
        ]

        # Sort ProdInfo instances by start sensing time
        prod_infos = sorted(prod_infos, key=lambda x: x.start_sensing_time)

        return prod_infos


# ------- Expand field "Attributes" of queried items as separate fields ------ #
def expand_attributes(df: pd.DataFrame) -> pd.DataFrame:
    """Expand field "Attributes" of queried items as separate fields."""
    # NOTE: the field "Attributes" of the queried items contains a list of
    # dictionaries for each item. Each dictionary describes a particular attribute.
    # The attribute name and value are keyed by "Name" and "Value", respectively.

    attributes = df["Attributes"].apply(
        lambda x: pd.Series({attr["Name"]: attr["Value"] for attr in x})
    )
    # Combine the original fields of the queried items and the ones of the expanded
    # attributes fields
    return pd.concat([df.drop(columns="Attributes"), attributes], axis="columns")


# --------- Compute cloud cover fraction in an AOI for Sentinel-3 LST -------- #
def compute_cloud_cover_aoi_lst(prod_path: Path, aoi: gpd.GeoDataFrame) -> float:
    """
    Compute cloud cover fraction (in percentage) in given `aoi` for Sentinel-3 LST
    product at `prod_path`'s.

    According to Sentinel-3 SLSTR Land User Handbook
    (https://sentiwiki.copernicus.eu/__attachments/1672112/OMPC.ACR.HBK.002%20-%20Sentinel%203%20SLSTR%20Land%20Handbook%202024%20-%201.4.pdf#page=30),
    clouded pixels have `bayes_in` variable of `flags_in` file with bit 1 activated
    (bits indexes start from 0). For more details check also discussion in the STEP
    Forum
    (https://forum.step.esa.int/t/sentinel-3-slstr-level-2-lst-problem-with-clouds-and-temperature-amplitude/22551/4)
    (https://forum.step.esa.int/t/sentinel-3-slstr-level-2-lst-problem-with-clouds-and-temperature-amplitude/22551/6).
    To check the map between `bayes_in` bits and the descriptions of the flagged
    conditions, have a look at the band attributes in the original product file.
    """

    # Read data from product NetCDF file
    data = xr.open_dataset(
        prod_path,
        # False, to not replace NODATA instances by nan
        # NOTE: NODATA instances should not be transformed into nan since the bitwise
        # operations below can only be applied to integers.
        mask_and_scale=False,
        engine="netcdf4",
        # To appropriately decode spatial_ref as coordinate
        decode_coords="all",
    )

    if "cloud_cover_aoi" not in data.attrs or (
        "cloud_cover_aoi" in data.attrs and data.attrs["cloud_cover_aoi"] is not None
    ):
        # Get cloud flag data
        cloud_flag = data["bayes_in"].copy()

        # Transform AOI geometry to the same CRS as the cloud flag DataArray
        aoi = aoi.to_crs(crs=cloud_flag.rio.crs)  # type: ignore

        # Clip the cloud flag DataArray to the AOI geometry
        cloud_flag_aoi = cloud_flag.rio.clip(geometries=aoi.geometry)

        # Convert DataArray to int so that following bit-wise operations are possible
        cloud_flag_aoi = cloud_flag_aoi.astype(np.int32)

        # Get the cloud flag data where pixels are actually clouded
        # NOTE: clouded pixels have bit 1 activated
        # NOTE: operation 1 << 1 corresponds to a left shift by 1 bits of the binary
        # number associated the decimal 1 (0b1 -> 0b10), which then returns the decimal
        # number associated with the result (integer 2). The bit-wise AND (&) operation
        # between cloud_flag_aoi & (1 << 1) returns the decimal number resulting from
        # the bit-wise AND operation between the bits of these operands (e.g. 11 & 7 = 3
        # since these have only bits 0 and 1 which are in common and simultaneously
        # activated, and the activation of these bits corresponds to the binary number
        # 0b11 and decimal number 3). Since (1 << 1) corresponds to the activation of
        # bit 1, the operation cloud_flag_aoi & (1 <<
        # 1) returns (1 << 1) if the bit 1 cloud_flag_aoi is also activated.
        cloud_flag_aoi_true = cloud_flag_aoi.where(
            (cloud_flag_aoi != cloud_flag_aoi.rio.nodata)
            & ((cloud_flag_aoi & (1 << 1)) == (1 << 1))
        )

        # Get cloud flag data where pixels are without NODATA
        cloud_flag_aoi = cloud_flag_aoi.where(
            cloud_flag_aoi != cloud_flag_aoi.rio.nodata
        )

        # Get number of pixels without NODATA
        N_pixels = cloud_flag_aoi.count().item()

        # Get number of clouded pixels
        N_clouded_pixels = cloud_flag_aoi_true.count().item()

        # Compute cloud cover fraction in percentage
        cloud_cover_aoi = N_clouded_pixels / N_pixels * 100

        # Set cloud cover fraction as attribute
        data.attrs["cloud_cover_aoi"] = cloud_cover_aoi

        # Rewrite dataset
        Path(prod_path).unlink()  # Delete original file
        data.to_netcdf(prod_path)

    else:
        cloud_cover_aoi = data.attrs["cloud_cover_aoi"]

    return cloud_cover_aoi  # type: ignore


# ----------------- Mask out clouded pixels in the LST product --------------- #


def mask_clouds(prod_path: Path, logger: RichLogger | None = None) -> None:
    """
    Mask out clouded pixels in the LST product at `prod_path` and rewrite it.
    """
    try:
        # Read data from product NetCDF file
        data = xr.open_dataset(
            prod_path,
            # False, to not replace NODATA instances by nan
            # NOTE: NODATA instances should not be transformed into nan since the
            # bitwise operations below can only be applied to integers.
            mask_and_scale=False,
            engine="netcdf4",
            # To appropriately decode spatial_ref as coordinate
            decode_coords="all",
        )

        if "cloud_masked" not in data.attrs or (
            "cloud_masked" in data.attrs and data.attrs["cloud_masked"] == 0
        ):
            if logger is not None:
                logger.info("Product will now be masked out with respect to clouds.")

            with (
                logger.console.status(
                    f"{'':7}Masking out clouded pixels in the LST product"
                    "[yellow]...[/yellow]",
                    spinner="dots",
                    spinner_style="bold blue",
                )
                if logger is not None
                else nullcontext()
            ):
                # Get cloud flag data as int so that the following bit-wise operations
                # are possible
                data["cloud"] = data["bayes_in"].copy().astype(np.int32)

                # Mask out as NODATA the cloud flag data whose pixels are not clouded
                # NOTE: clouded pixels have bit 1 activated
                # NOTE: operation 1 << 1 corresponds to a left shift by 1 bits of the
                # binary number associated the decimal 1 (0b1 -> 0b10), which then
                # returns the decimal number associated with the result (integer 2). The
                # bit-wise AND (&) operation between cloud & (1 << 1) returns the
                # decimal number resulting from the bit-wise AND operation between the
                # bits of these operands (e.g. 11 & 7 = 3 since these have only bits 0
                # and 1 which are in common and simultaneously activated, and the
                # activation of these bits corresponds to the binary number 0b11 and
                # decimal number 3). Since (1 << 1) corresponds to the activation of bit
                # 1, the operation cloud & (1 << 1) returns (1 << 1) if the bit 1 in
                # cloud is also activated.
                data["cloud"] = data["cloud"].where(
                    (data["cloud"] != data["cloud"].rio.nodata)
                    & ((data["cloud"] & (1 << 1)) == (1 << 1)),
                    other=data["cloud"].rio.nodata,
                )

                # Mask out as NODATA the clouded pixels in LST data
                data["LST"] = data["LST"].where(
                    data["cloud"] == data["cloud"].rio.nodata,
                    other=data["LST"].rio.nodata,
                )

                # Set cloud masking indicator
                data.attrs["cloud_masked"] = 1

                # Rewrite dataset
                Path(prod_path).unlink()  # Delete original file
                data.to_netcdf(prod_path)

            if logger is not None:
                logger.info(
                    "[bold green]Product masked out with respect to clouds."
                    "[/bold green]"
                )

        # If already masked out, no masking will be done
        else:
            if logger is not None:
                logger.info(
                    "[bold green]Product is already masked out with respect to clouds."
                    "[/bold green]"
                )

    except Exception as e:  # noqa: BLE001
        if logger is not None:
            logger.error(
                "[bold red]Error masking out clouds in product"
                + f"\n{prod_path.name!r}."
                + f"\nError message: {e}"
                + "\nRun will stop.[/bold red]",
            )
        raise MaskCloudsError(
            "Error masking out clouds in product"
            + f"\n{prod_path.name!r}."
            + f"\nError message: {e}"
        )


# -------------- Download and unzip a product through OData API -------------- #
def download_product(
    prod_info: ProdInfo,
    path_out_dir: Path,
    access_token: str,
    logger: RichLogger | None = None,
) -> None:
    """
    Download product of info `prod_info` (if it had not already been downloaded) through
    the OData API (with respective `access_token`) to folder `path_out_dir`. Further
    unzip it if it is zipped and log the processes with `logger`.
    """

    # Define path to file
    path_out_file = path_out_dir / prod_info.name

    # If the file does not exist, download it
    # NOTE: prod_info.name used in the definition of path_out_file came from query
    # output and would not contain suffix ".zip" if the respective file is a zipped one.
    # If it is zipped, the name of the downloaded file would also come without the
    # ".zip" suffix. The current function is such that suffix ".zip" is added to the
    # file name after it being downloaded, the file is unzipped and the zipped file is
    # deleted. Also, after unzipping, the product is processed and further transformed
    # into a NetCDF file (outside of this function). To properly check if the file
    # exists for any circumstance (even in one in which an error or a keyboard
    # interruption had occurred), one should use a pattern that accepts the existence or
    # not of the ".zip" or ".nc" suffixes in the file name.
    if not any(path_out_file.parent.glob(pattern=f"{path_out_file.name}*")):
        # Create accommodating directory if it does not exist
        path_out_file.parent.mkdir(parents=True, exist_ok=True)

        download_cdse(
            products_info=pd.DataFrame(
                {"Id": [prod_info.id], "Name": [prod_info.name]}
            ),
            path_out_dir=path_out_file.parent,
            access_token=access_token,
            max_workers=1,
            logger=logger,
        )

        if logger is not None:
            logger.info("[bold green]Product downloaded.[/bold green]")

    # If the file exists state that no download is required
    else:
        if logger is not None:
            logger.info("[bold green]Product was already downloaded." + "[/bold green]")

    # Get exact path to the product file (that is, with the appropriate suffix)
    path_out_file = next(
        iter(path_out_file.parent.glob(pattern=f"{path_out_file.name}*"))
    )

    # Check if file is zipped
    is_zipped = zipfile.is_zipfile(path_out_file)

    if logger is not None:
        logger.info(
            "[bold green]Product is already unzipped.[/bold green]"
            if not is_zipped
            else "Product will now be unzipped."
        )

    # Unzip product file if it is zipped and delete the zip file
    if is_zipped:
        try:
            # Add "zip" suffix to file name if it does not have it already.
            # NOTE: this is required to avoid coincidence of zip and unzipped product
            # names when unzipping.
            path_out_file = (
                path_out_file.rename(f"{path_out_file}.zip")
                if path_out_file.suffix != ".zip"
                else path_out_file
            )

            with zipfile.ZipFile(path_out_file, "r") as zip_ref:
                # Unzip
                zip_ref.extractall(path_out_file.parent)
            # Delete the zip file
            path_out_file.unlink()
            if logger is not None:
                logger.info("[bold green]Product unzipped.[/bold green]")
        except Exception as e:  # noqa: BLE001
            if logger is not None:
                logger.error(
                    "[bold red]Error unzipping product"
                    + f"\n{path_out_file.name!r}."
                    + f"\nError message: {e}"
                    + "\nRun will stop.[/bold red]",
                )
            raise UnzipError(
                "Error unzipping product"
                + f"\n{path_out_file.name!r}."
                + f"\nError message: {e}"
            )


def delete_last_product(folder: Path, logger: RichLogger | None = None) -> None:
    # Delete last product folder
    shutil.rmtree(folder, ignore_errors=True)
    if logger is not None:
        logger.warning(
            f"[bold yellow]Deleted incomplete product folder\n{folder}[/bold yellow]"
        )


# -------- Query, download and filter Sentinel-3 LST and SYN products -------- #
def download_products(
    # Configuration parameters for querying, downloading and filtering Sentinel-3 LST
    # and SYN products.
    config: Sentinel3Config,
) -> None:

    # ---> Handle logging

    # Create logger
    logger = RichLogger(
        name="download",
        level=logging.INFO,
        file_path=(Path(config.out_dir) / "download_sentinel3.log"),
        file_mode="w",
        log_mode=config.log_mode,
    )

    # ---> Handle configurations

    # Create instance of Sentinel-3 product processor if processing is to be done
    if config.process is True:
        sen3_processor = Sen3Processor(engine=config.process_engine, logger=logger)

    # Validate input oldest start and newest end sensing dates
    if (
        isinstance(config.start_sensing_dates, list)
        and not isinstance(config.end_sensing_dates, list)
    ) or (
        not isinstance(config.start_sensing_dates, list)
        and isinstance(config.end_sensing_dates, list)
    ):
        logger.error(
            "[bold red]Oldest start sensing date is a list but newest end sensing"
            " date is not. If one is a list the other must also be, with each of"
            " their elements being associated with the respective ones of the"
            " other."
            "\nThe run will be stopped."
            "[/bold red]",
        )

        raise ConfigError(
            "Oldest start sensing date is a list but newest end sensing date"
            " is not. If one is a list the other must also be, with each of their"
            " elements being associated with the respective ones of the other."
            "\nThe run will be stopped."
        )

    # Convert oldest start and newest end sensing dates to lists if they are not already
    config.start_sensing_dates = (
        [config.start_sensing_dates]
        if not isinstance(config.start_sensing_dates, list)
        else config.start_sensing_dates
    )  # type: ignore
    config.end_sensing_dates = (
        [config.end_sensing_dates]
        if not isinstance(config.end_sensing_dates, list)
        else config.end_sensing_dates
    )  # type: ignore

    # Set the sensing date range for the query if any of the limits are not provided in
    # the configuration file
    # NOTE:
    # - If the newest end date (`end_sensing_dates`) is missing, set it as the current
    #   date
    #
    # - If the oldest start date (`start_sensing_dates`) is missing, set it as 31 days
    # before newest end date (`end_sensing_dates`)
    #
    # Note that the revisit times of Sentinel-3 OLCI and SLSTR are less than 2 days and
    # less than 1 day, respectively
    # (https://sentiwiki.copernicus.eu/web/s3-mission#S3Mission-OrbitS3-Mission-Orbittrue),
    # and the uploading time after acquisition for NR (Near Real Time), and NT (Non-Time
    # Critical) products less than 3 hours, and after 48 hours (but possibly up to 1
    # month), respectively, for OLCI
    # (https://sentiwiki.copernicus.eu/__attachments/1672112/OMPC.ACR.HBK.001%20-%20Sentinel%203%20OLCI%20Land%20Handbook%20-%201.3.pdf#page=24)
    # and within 3 hours, and within 24-48 hours, respectively, for SLSTR
    # (https://sentiwiki.copernicus.eu/__attachments/1672112/OMPC.ACR.HBK.002%20-%20Sentinel%203%20SLSTR%20Land%20Handbook%202024%20-%201.4.pdf#page=26).
    for i in range(len(config.end_sensing_dates)):  # type: ignore
        if config.end_sensing_dates is None:
            config.end_sensing_dates[i] = (  # type: ignore
                pd.Timestamp.now().strftime("%Y-%m-%d")
            )
    for i in range(len(config.start_sensing_dates)):  # type: ignore
        if config.start_sensing_dates is None:
            config.start_sensing_dates[i] = (  # type: ignore
                pd.Timestamp(config.end_sensing_dates[i])  # type: ignore
                - pd.Timedelta(days=31)
            ).strftime("%Y-%m-%d")

    # Convert oldest start and newest end sensing dates to de facto dates (to also
    # handle cases in which other formats such as timestamps are issued)
    start_sensing_dates = [
        pd.Timestamp(date).strftime("%Y-%m-%d")
        for date in config.start_sensing_dates  # type: ignore
    ]
    end_sensing_dates = [
        pd.Timestamp(date).strftime("%Y-%m-%d")
        for date in config.end_sensing_dates  # type: ignore
    ]

    # Set start sensing time as millisecond 0 of the start sensing date (in UTC)
    start_sensing_times = [
        pd.Timestamp(date).strftime("%Y-%m-%dT%H:%M:%S.%f"[:-3] + "Z")
        for date in start_sensing_dates
    ]

    # Set end sensing time as last millisecond of the end sensing date (in UTC)
    end_sensing_times = [
        (
            pd.Timestamp(date) + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)
        ).strftime("%Y-%m-%dT%H:%M:%S.%f"[:-3] + "Z")
        for date in end_sensing_dates
    ]

    logger.console.print()  # type: ignore
    logger.info(
        "Downloading Sentinel-3 products for dates"
        + ("\n- " if len(start_sensing_dates) > 1 else " ")
        + ",\n- ".join(
            [
                f"{start_sensing_date} to {end_sensing_date}"
                for start_sensing_date, end_sensing_date in zip(
                    start_sensing_dates, end_sensing_dates
                )
            ]
        )
        + "."
    )

    # Read the AOI geometry (if issued) as a GeoPandas DataFrame for which filtering of
    # the downloaded LST data is to be done with respect to cloud cover fraction
    aoi = load_aoi_to_gdf(config.geometry_aoi)

    # Read the querying geometry WKT (if issued) as a string. If not issued, but
    # `geometry_aoi` is, the `geometry_query` will be set as the convex hull of
    # `geometry_aoi`.
    # NOTE: Queried data would be considered of interest if its geometry has at least
    # one point in common with this one.
    if config.geometry_query is not None:
        if isinstance(config.geometry_query, Path):
            with open(config.geometry_query, "r") as f:
                config.geometry_query = f.read().strip()

    elif aoi is not None:
        config.geometry_query = to_convex_hull_wkt(aoi)

    # ---> Query Sentinel-3 LST products (and filter returned queried items)
    logger.console.print()  # type: ignore

    with logger.console.status(
        f"{'':7}Querying Sentinel-3 LST products[yellow]...[/yellow]",
        spinner="dots",
        spinner_style="bold blue",
    ):
        # Get list of all timestamps at the start of all months between defined start
        # and end sensing times including these start and end sensing times
        times = [
            [start_sensing_time]
            + pd.date_range(
                # Left bound
                start=pd.Timestamp(start_sensing_time),  # type: ignore
                # Right bound
                end=pd.Timestamp(end_sensing_time),  # type: ignore
                # "neither", to not include bounds in the date range
                inclusive="neither",
                # Use month start frequency
                freq="MS",
            )
            .strftime("%Y-%m-%dT%H:%M:%S.%f"[:-3] + "Z")
            .tolist()
            + [end_sensing_time]
            for start_sensing_time, end_sensing_time in zip(
                start_sensing_times, end_sensing_times
            )
        ]

        # Query
        # NOTE: several queries are performed. These are done month by month so that the
        # number of query items returned in each query is not greater than the API
        # limit,
        # 1000. The items from all queries are then concatenated into a single
        # DataFrame.
        lst_query = pd.concat(
            [
                pd.DataFrame(
                    query_cdse(
                        collection="SENTINEL-3",
                        includes=["SL_2_LST", f"_{config.timeliness}_"],
                        start_sensing_time=times[i][j],  # type: ignore
                        end_sensing_time=times[i][j + 1],  # type: ignore
                        geometry=config.geometry_query,
                        orderby_start_sensing_time="asc",
                        # WARNING: max_query_items may be ajusted if needed, but never
                        # surpassing 1000.
                        max_query_items=1000,
                        logger=logger,
                    )["value"],
                )
                for i in range(len(times))
                for j in range(len(times[i]) - 1)
            ],
            ignore_index=True,
        )

        # If not empty, filter queried items according to minimum and maximum cloud
        # cover fraction, relative orbit number and orbit direction
        if not lst_query.empty:
            # Remove duplicated items (which may happen if overlapping occurs between
            # each issued sensing daterange limits)
            lst_query = lst_query.drop_duplicates(subset="Name")

            # Express field "Attributes" of the queried items as expanded fields, each
            # one associated a single attribute
            lst_query = expand_attributes(lst_query)

            # Filter queried items according to minimum and maximum cloud cover
            # fraction, relative orbit number and orbit direction and disregard half and
            # full orbit products.
            # NOTE: half and full orbit products contain 4 underscores ("____") in the
            # frame along track coordinate of the <instance_id> substring in the product
            # name. For more details see
            # https://user.eumetsat.int/resources/user-guides/sentinel-safe-format-guide/.
            # NOTE: field "cloudCover" might not be in the data for some datetimes.
            lst_query = lst_query[~(lst_query["Name"].str[77:81] == "____")]

            if (
                "cloudCover" in lst_query.columns
                and config.cloud_cover_lims is not None
            ):
                lst_query = lst_query[
                    lst_query["cloudCover"].isna()
                    | (
                        (lst_query["cloudCover"] >= config.cloud_cover_lims[0])
                        & (lst_query["cloudCover"] <= config.cloud_cover_lims[1])
                    )
                ]
            if config.relative_orbit is not None:
                lst_query = lst_query[
                    lst_query["relativeOrbitNumber"].isin(config.relative_orbit)
                ]
            if config.orbit_dir is not None:
                lst_query = lst_query[
                    lst_query["orbitDirection"].isin(config.orbit_dir)
                ]

    # Check if query results are empty and raise error in that case
    if lst_query.empty:
        logger.warning(
            "[bold yellow]No Sentinel-3 LST products were found for the requested"
            " period, AOI, cloud cover fraction limits, relative orbit number or"
            " orbit direction."
            "\nThe run will stop.[/bold yellow]"
        )
        raise NoProductFoundError(
            "No Sentinel-3 LST products were found for the requested period, AOI,"
            " cloud cover fraction limits, relative orbit number or orbit"
            " direction.\nTry using different filtering criteria."
        )

    else:
        logger.info(
            f"[bold green]{len(lst_query)} Sentinel-3 LST products have been found"
            + " in the query:[/bold green][green]\n"
            + "\n".join(lst_query["Name"].tolist())
            + "[/green]"
        )

    # WARNING: For Open-Cosmos project only: consider solely the LST product that has
    # the maximum footprint overlap with the AOI. If there are multiple products with
    # maximum footprint overlap, select the one with the smallest cloud cover fraction.
    if config.filter_max_footprint_aoi_overlap:
        if len(lst_query) > 1:
            logger.info(
                "Solely the LST product that has the maximum footprint overlap with"
                " the AOI will be considered."
            )
            # Get footprint-AOI overlap (in percentage) for each queried item.
            # NOTE: if no AOI is defined, the overlap is set to 0 %.
            lst_query["FootprintAoiOverlap"] = lst_query["Footprint"].apply(
                lambda x: footprint_aoi_overlap(
                    footprint_wkt=x,
                    aoi_gdf=aoi,
                )
            )

            # Get maximum overlap value
            max_overlap = lst_query["FootprintAoiOverlap"].max()
            # Keep only products with maximum AOI overlap
            lst_query = lst_query[
                lst_query["FootprintAoiOverlap"] == max_overlap
            ].copy()

            # If multiple products have the maximum AOI overlap, select the one with the
            # lowest cloud cover fraction by firstly sorting the queried items by cloud
            # cover fraction in ascending order.
            if len(lst_query) > 1:
                if "cloudCover" in lst_query.columns:
                    lst_query = lst_query.sort_values(
                        by="cloudCover",
                        ascending=True,
                        na_position="last",
                    )
                    logger.warning(
                        "[bold yellow]Multiple products with maximum footprint/AOI"
                        " overlap detected. Product with lowest cloud cover"
                        " fraction in the footprint was selected.[/bold yellow]"
                    )
                else:
                    logger.warning(
                        "[bold yellow]Multiple products with maximum AOI overlap"
                        " detected, but discrimination with respect to cloud cover"
                        " fraction cannot be performed since 'cloudCover' attribute"
                        " is missing. First product was selected.[/bold yellow]"
                    )
        # Select the first product
        lst_query = lst_query.iloc[[0]].reset_index(drop=True)

    # Get LST products notable info from query results
    lst_infos = ProdInfo.load_from_df(lst_query)  # type: ignore

    # Create a CDSE authentication state object for handling the fetching of a valid
    # CDSE access token
    # NOTE: whenever auth.get_access_token() is called, the access token is
    # generated, refreshed or regenerated as required to make it valid.
    auth = CDSEAuthState(
        username=os.environ["CDSE_USER"],
        password=os.environ["CDSE_PASS"],
    )

    for lst_info in lst_infos:
        # ---> Query Sentinel-3 SYN products that accompany current LST product
        logger.console.print()  # type: ignore

        with logger.console.status(
            f"{'':7}Querying Sentinel-3 SYN product[yellow]...[/yellow]",
            spinner="dots",
            spinner_style="bold blue",
        ):
            # Query API for Sentinel-3 SYN products
            syn_query = pd.DataFrame(
                query_cdse(
                    collection="SENTINEL-3",
                    includes=["SY_2_SYN", f"_{config.timeliness}_"],
                    start_sensing_time=(
                        (
                            pd.Timestamp(lst_info.start_sensing_time)
                            - pd.Timedelta(seconds=1)
                        ).strftime("%Y-%m-%dT%H:%M:%S.%f"[:-3] + "Z")
                    ),
                    end_sensing_time=(
                        (
                            pd.Timestamp(lst_info.end_sensing_time)
                            + pd.Timedelta(seconds=1)
                        ).strftime("%Y-%m-%dT%H:%M:%S.%f"[:-3] + "Z")
                    ),
                    geometry=config.geometry_query,
                    orderby_start_sensing_time="asc",
                    max_query_items=1000,
                    logger=logger,
                )["value"]
            )

            if not syn_query.empty:
                # Express field "Attributes" of the queried items as expanded fields,
                # each one associated an attribute
                # NOTE: the field "Attributes" of the queried items contains a list of
                # dictionaries for each item. Each dictionary describes a particular
                # attribute. The attribute name and value is keyed by "Name" and
                # "Value", respectively.
                syn_query = expand_attributes(syn_query)

                # Filter SYN queried items so that that they "accompany" the current LST
                # product (for that, they must be associated with the same satellite
                # platform, orbit number and start and end sensing times).
                syn_query = syn_query[
                    (syn_query["platformSerialIdentifier"] == lst_info.platform)
                    & (syn_query["orbitNumber"] == lst_info.orbit)
                    # & (syn_query["Name"].str[82:85] == lst_info.generating_centre)
                    & (syn_query["Name"].str[16:31] == lst_info.start_sensing_time)
                    & (syn_query["Name"].str[32:47] == lst_info.end_sensing_time)
                ]

        # Check if the query results are not empty neither multiple. If they are, do not
        # download current LST product neither its companion SYN products and advance to
        # the next LST product. If the query results are singular, get the SYN product
        # info.
        if syn_query.empty:
            logger.warning(
                "[bold yellow]No companion Sentinel-3 SYN product was found in the"
                + " SYN query for LST's"
                + f"\n{lst_info.name!r}."
                + (
                    (
                        "\nThe LST product will not be downloaded and the process"
                        " will advance to the next one."
                    )
                    if config.filter_max_footprint_aoi_overlap is False
                    else (
                        "\nThe LST product will not be downloaded and the process"
                        " will terminate."
                    )
                )
                + "[/bold yellow]"
            )
            if config.filter_max_footprint_aoi_overlap is False:
                continue
            else:
                raise NoMatchingProductError(
                    "No matching Sentinel-3 SYN product was found for the LST product. "
                    "Try using different filtering criteria."
                )
        elif len(syn_query) > 1:
            logger.warning(
                "[bold yellow]Multiple companion Sentinel-3 SYN products were found"
                + " in the SYN query for LST's"
                + f"\n{lst_info.name!r}."
                + "\nTry using stricter querying filters to get a unique companion"
                + " product instead."
                + f"\nFound companion products:\n{
                    ',\n'.join(f'{name!r}' for name in syn_query['Name'].to_list())
                }."
                + (
                    (
                        "\nThe LST product will not be downloaded and the process"
                        " will advance to the next one."
                    )
                    if config.filter_max_footprint_aoi_overlap is False
                    else (
                        "\nThe LST product will not be downloaded and the process"
                        " will terminate."
                    )
                )
                + "[/bold yellow]"
            )
            if config.filter_max_footprint_aoi_overlap is False:
                continue
            else:
                raise MultipleMatchingProdutsError(
                    "Multiple companion Sentinel-3 SYN products were found in the SYN"
                    " query for LST's."
                    "\nTry using stricter querying filters to get a unique companion."
                )
        else:
            # Get SYN product info
            syn_info = ProdInfo.load_from_df(syn_query.iloc[[0]])[0]
            logger.info(
                "[bold green]Singular companion Sentinel-3 SYN product"
                + f"\n{syn_info.name!r}"
                + "\nwas found in the SYN query for LST's"
                + f"\n{lst_info.name!r}.[/bold green]"
                + "\nLST product will now be downloaded."
            )

        # ---> Download and unzip the current LST product

        # Download and unzip the current LST product
        download_product(
            prod_info=lst_info,
            path_out_dir=config.out_dir / lst_info.start_sensing_time,
            access_token=auth.get_access_token(),
            logger=logger,
        )

        # Get path to the unzipped LST product
        lst_path = config.out_dir / lst_info.start_sensing_time / lst_info.name

        if config.process is True:
            # ---> Reproject LST product to EPSG:4326, subset to bands of interest and
            # the bounding box of a geometry, write result to NetCDF file and delete
            # original file.
            # NOTE: it is important to subset the product to a bounding box since some
            # files are too huge to be handled directly (these files probably include
            # large regions not corresponding to the satellite overpassing one,
            # containing NODATA instances).

            # If a reprojected product does not exist, reproject the product if wanted
            if not lst_path.with_name(f"{lst_path.name}.nc").exists():
                logger.info("Product will now be reprojected and subsetted.")

                with logger.console.status(
                    f"{'':7}Reprojecting and subsetting the LST product[yellow]..."
                    + "[/yellow]",
                    spinner="dots",
                    spinner_style="bold blue",
                ):
                    try:
                        sen3_processor.process(
                            prod_path=lst_path,
                            geometry=config.geometry_query,
                        )
                        error = None
                    except Exception as e:  # noqa: BLE001
                        # Capture exception to log it after the console.status() scope
                        # or the message will be badly printed
                        error = e

                # If error occurred while checking, stop the run
                if error is not None:
                    if (
                        "zero-size array to reduction operation maximum which has no"
                        + " identity"
                        in str(error)
                    ):
                        logger.warning(
                            "[bold yellow]Empty array error when reprojecting or"
                            " subsetting LST product"
                            + f"\n{lst_info.name!r}."
                            + f"\nError message: {error}"
                            + (
                                "\nProduct was already invalid from source and"
                                " could not be used. Deleting product folder and"
                                " advancing to the next one."
                                if config.filter_max_footprint_aoi_overlap is False
                                else (
                                    "\nProduct was already invalid from source and"
                                    " cannot be used. Product folder will be"
                                    " deleted and the run will stop. Try using"
                                    " different filtering criteria."
                                )
                            )
                            + "[/bold yellow]"
                        )
                        # Delete the last product folder
                        delete_last_product(
                            folder=config.out_dir / lst_info.start_sensing_time,
                            logger=logger,
                        )
                        if config.filter_max_footprint_aoi_overlap is False:
                            continue
                        else:
                            raise InvalidProductError(
                                "Empty array error when reprojecting or subsetting LST"
                                + " product"
                                + f"\n{lst_info.name!r}."
                                + f"\nError message: {error}"
                                + "\nProduct was already invalid from source and could"
                                " not be used."
                                "\nProduct folder was, therefore, deleted."
                                " Try using different filtering criteria."
                            )
                    else:
                        logger.error(
                            "[bold red]Failed reprojecting or subsetting LST"
                            + f" product\n{lst_info.name!r}."
                            + f"\nError message: {error}"
                            + "\nFile is most likely corrupted (possibly due to a"
                            " stopped download or unzip process) and cannot be"
                            " used."
                            "\nProduct folder will be deleted and the run will"
                            " stop.[/bold red]",
                        )
                        # Delete the last product folder
                        delete_last_product(
                            folder=config.out_dir / lst_info.start_sensing_time,
                            logger=logger,
                        )
                        raise ReprojectionError(
                            "Failed reprojecting or subsetting LST product"
                            + f"\n{lst_info.name!r}."
                            + f"\nError message: {error}"
                            + "\nFile was most likely corrupted (possibly due to a"
                            " stopped download or unzip process) and could not be used."
                            + "\nProduct folder was, therefore, deleted."
                        )

                else:
                    logger.info(
                        "[bold green]Product reprojected and subsetted.[/bold green]"
                    )

            # If it already exists, no reprojection or subsection will be done
            else:
                logger.info(
                    "[bold green]Product is already reprojected and subsetted."
                    "[/bold green]"
                )

            # ---> Check cloud cover fraction in the AOI
            # Get path to the reprojected LST product
            lst_path = lst_path.with_name(f"{lst_path.name}.nc")

            if aoi is not None:
                with logger.console.status(
                    f"{'':7}Checking if AOI's cloud cover fraction in the LST product"
                    + " is smaller than the configured[cyan]"
                    + f" {config.cloud_cover_max_aoi}[/cyan] %[yellow]...[/yellow]",
                    spinner="dots",
                    spinner_style="bold blue",
                ):
                    try:
                        cloud_cover_aoi = compute_cloud_cover_aoi_lst(
                            prod_path=lst_path, aoi=aoi
                        )
                        error = None
                    except Exception as e:  # noqa: BLE001
                        # Capture exception to log it after the console.status() scope
                        # or the message will be badly printed
                        error = e

                # If error occurred while checking stop the run
                if error is not None:
                    logger.error(
                        "[bold red]Failed computing cloud cover fraction over the"
                        " AOI for LST product"
                        + f"\n{lst_info.name!r}."
                        + f"\nError message: {error}"
                        + "\nProduct folder will be deleted and the run stopped."
                        "[/bold red]",
                    )
                    # Delete the last product folder
                    delete_last_product(
                        folder=config.out_dir / lst_info.start_sensing_time,
                        logger=logger,
                    )
                    raise CloudCoverComputationError(
                        "Failed computing cloud cover fraction over the AOI for LST"
                        " product"
                        + f"\n{lst_info.name!r}."
                        + f"\nError message: {error}"
                        + "\nProduct folder was, therefore, deleted."
                    )

            # ---> Mask out clouded pixels in the LST data if wanted and the cloud cover
            # fraction in the issued AOI does not exceed the issued limit.
            if config.mask_clouds is True and (
                aoi is None
                or (aoi is not None and cloud_cover_aoi <= config.cloud_cover_max_aoi)
            ):
                mask_clouds(prod_path=lst_path, logger=logger)

        # ---> Download, unzip, reproject and subset companion SYN product if for the
        # LST product the cloud cover fraction in the issued AOI does not exceed the
        # issued limit. If the cloud cover fraction exceeds the limit, delete the LST
        # product, respective folder, and continue to the processing of the next LST
        # product.
        if (
            config.process is False
            or aoi is None
            or (aoi is not None and cloud_cover_aoi <= config.cloud_cover_max_aoi)
        ):
            if config.process is True and aoi is not None:
                logger.info(
                    "[bold green]Cloud cover fraction in the AOI for LST product is"
                    + f" {cloud_cover_aoi:.2f} %, not exceeding issued maximum"
                    + f" limit, {config.cloud_cover_max_aoi} %.[/bold green]"
                    + "\nThe companion SYN product will now be downloaded."
                )
            else:
                logger.info("The companion SYN product will now be downloaded.")

            # Download and unzip the companion SYN product
            download_product(
                prod_info=syn_info,
                path_out_dir=config.out_dir / lst_info.start_sensing_time,
                access_token=auth.get_access_token(),
                logger=logger,
            )

            # Get path to the unzipped SYN product
            syn_path = config.out_dir / lst_info.start_sensing_time / syn_info.name

            # Reproject SYN product to EPSG:4326, subset to bands of interest and the
            # bounding box of a geometry, write result to NetCDF file and delete
            # original file
            if config.process is True:
                if not syn_path.with_name(f"{syn_path.name}.nc").exists():
                    logger.info("Product will now be reprojected and subsetted.")

                    with logger.console.status(
                        f"{'':7}Reprojecting and subsetting the SYN product"
                        + "[yellow]...[/yellow]",
                        spinner="dots",
                        spinner_style="bold blue",
                    ):
                        try:
                            sen3_processor.process(
                                prod_path=syn_path,
                                geometry=config.geometry_query,
                            )

                            error = None
                        except Exception as e:  # noqa: BLE001
                            # Capture exception to log it after the console.status()
                            # scope or the message will be badly printed
                            error = e

                    # If error occurred while checking, stop the run
                    if error is not None:
                        if (
                            "zero-size array to reduction operation maximum which has"
                            + " no identity"
                            in str(error)
                        ):
                            logger.warning(
                                "[bold yellow]Empty array error when reprojecting"
                                + " or subsetting SYN product"
                                + f"\n{syn_info.name!r}."
                                + f"\nError message: {error}"
                                + (
                                    "\nProduct was already invalid from source and"
                                    " could not be used. Deleting product folder"
                                    " and advancing to the next one."
                                    if config.filter_max_footprint_aoi_overlap is False
                                    else (
                                        "\nProduct was already invalid from source"
                                        " and cannot be used. Product folder will"
                                        " be deleted and the run will stop. Try"
                                        " using different filtering criteria."
                                    )
                                )
                                + "[/bold yellow]"
                            )
                            # Delete the last product folder
                            delete_last_product(
                                folder=config.out_dir / syn_info.start_sensing_time,
                                logger=logger,
                            )

                            if config.filter_max_footprint_aoi_overlap is False:
                                continue
                            else:
                                raise InvalidProductError(
                                    "Empty array error when reprojecting or subsetting"
                                    + " SYN product"
                                    + f"\n{syn_info.name!r}."
                                    + f"\nError message: {error}"
                                    + "\nProduct was already invalid from source and"
                                    + " could not be used."
                                    + "\nProduct folder was, therefore, deleted."
                                    + " Try using different filtering criteria."
                                )

                        else:
                            logger.error(
                                "[bold red]Failed reprojecting or subsetting SYN"
                                " product"
                                + f"\n{syn_info.name!r}."
                                + f"\nError message: {error}"
                                + "\nFile is most likely corrupted (possibly due to"
                                " a stopped download or unzip process) and cannot"
                                " be used."
                                "\nProduct folder will be deleted and the run will"
                                " stop.[/bold red]"
                            )
                            # Delete the last product folder
                            delete_last_product(
                                folder=config.out_dir / syn_info.start_sensing_time,
                                logger=logger,
                            )
                            raise ReprojectionError(
                                "Failed reprojecting or subsetting SYN product."
                                + f"\n{syn_info.name!r}."
                                + f"\nError message: {error}"
                                + "\nFile was most likely corrupted (possibly due to a"
                                " stopped download or unzip process) and could not be"
                                " used."
                                "\nProduct folder was, therefore, deleted and"
                                " the run stopped."
                                "\nTry running again."
                            )
                    else:
                        logger.info(
                            "[bold green]Product reprojected and subsetted."
                            "[/bold green]"
                        )

                # If it already exists, no reprojection or subsection will be done
                else:
                    logger.info(
                        "[bold green]Product is already reprojected and subsetted."
                        + "[/bold green]"
                    )

        else:
            logger.warning(
                "[bold yellow]Cloud cover fraction in the AOI for LST product"
                + f"\n{lst_info.name!r}"
                + f"\nis {cloud_cover_aoi:.2f} %, which exceeds the issued limit,"
                + f" {config.cloud_cover_max_aoi} %."
                + "\nThis case will not be considered. The LST product will be"
                + " deleted as well as its subfolder if it only contained that"
                + " product, and the companion SYN product will not be downloaded. "
                + (
                    "The run will advance to the processing of the next LST product."
                    if config.filter_max_footprint_aoi_overlap is False
                    else "Run will stop. Try using different filtering criteria."
                )
                + "[/bold yellow]"
            )

            # Delete LST product
            try:
                lst_path.unlink()
                logger.warning(
                    f"[bold yellow]LST product {lst_path.name!r} was deleted."
                    + "[/bold yellow]"
                )
            except Exception as e:  # noqa: BLE001
                logger.error(
                    f"[bold red]Error removing LST product {lst_path.name!r}."
                    + f"\nError message: {e}"
                    + "\nRun will stop.[/bold red]",
                )
                raise CleanupError(
                    f"Error removing LST product {lst_path.name!r}."
                    + f"\nError message: {e}"
                )

            # If the subfolder of the deleted LST product is empty, delete it
            if not any(lst_path.parent.iterdir()):
                try:
                    shutil.rmtree(lst_path.parent)
                    logger.warning(
                        f"[bold yellow]Subfolder {lst_path.parent.name!r} was"
                        + " deleted.[/bold yellow]"
                    )
                except Exception as e:  # noqa: BLE001
                    logger.error(
                        "[bold red]Error removing subfolder"
                        + f" {lst_path.parent.name!r}\nError message: {e}"
                        + "\nRun will stop.[/bold red]",
                    )
                    raise CleanupError(
                        f"Error removing subfolder {lst_path.parent.name!r}"
                        + f"\nError message: {e}"
                    )
            if config.filter_max_footprint_aoi_overlap is False:
                continue
            else:
                raise CloudCoverLimitError(
                    "Cloud cover fraction in the AOI for the LST product"
                    + f"\n{lst_info.name!r}"
                    + f"\nis {cloud_cover_aoi:.2f} %, which exceeds the issued limit,"
                    + f" {config.cloud_cover_max_aoi} %."
                    + "\nThis case will not be considered."
                    + " Run will stop. Try using different filtering criteria."
                )


# --------------------------- Script's main funcion -------------------------- #
# For querying, downloading and filtering of Sentinel-3 LST and SYN products
def main() -> None:
    download_products(config)


# If this very script is directly executed in the terminal (in that case the global
# variable __name__ corresponds to "__main__"), it runs function main()
if __name__ == "__main__":
    main()
