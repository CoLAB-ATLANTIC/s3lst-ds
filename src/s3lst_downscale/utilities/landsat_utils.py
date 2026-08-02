import logging
from pathlib import Path
from typing import cast

import rioxarray as rioxr
import xarray as xr
from rioxarray.merge import merge_arrays


def merge_spatially(directories: list[Path], out_dir: Path) -> None:
    """
    Merge band tiles associated with the same variable and save the result as a TIF file
    to the directory `out_dir`. These band tiles come from TIF files in the issued
    `directories`.
    """

    # Create directory to accomodate merged data files if it does not already exist
    out_dir.mkdir(parents=True, exist_ok=True)

    # Get paths to the files to be merged (all TIF files in the row folders)
    tile_files = [
        file
        for directory in directories
        for file in directory.glob("*")
        if file.suffix.lower() in {".tif", ".tiff"}
    ]

    # Get unique band identifiers for the files (the last part of the name with an
    # underscore in between, e.g. 'QA_PIXEL', 'SR_B4', 'SR_B5', 'ST_B10') - this is
    # the label for the variable that the band is associated with
    var_ids = list(
        {"_".join(tile_file.stem.split("_")[-2:]) for tile_file in tile_files}
    )

    # Get geospatially merged files, one per variable
    for var_id in var_ids:
        # Get list of all files in the row folders having the current band identifier
        tile_files_match = [
            tile_file
            for directory in directories
            for tile_file in directory.glob("*")
            if "_".join(tile_file.stem.split("_")[-2:]) == var_id
        ]

        # If there is a file with same identifier as the current one per each row
        # folder, proceed with the merge of these files
        if len(tile_files_match) == len(directories):
            # List of DataArrays, one per matching file
            tiles = [
                rioxr.open_rasterio(f, mask_and_scale=False) for f in tile_files_match
            ]
            # Geospatially merge the DataArrays
            merged = merge_arrays(tiles)  # type: ignore
            merged.name = var_id
            # Write merged DataArray as raster file to the merge folder
            merged.rio.to_raster((out_dir / f"{var_id}.TIF").resolve())

        # If there is a file missing, state that, and do not proceed with the merge of
        # these files
        else:
            logging.warning(  # noqa: LOG015
                f"[bold yellow]Skipping {var_id}: not all matching files found."
                + "[/bold yellow]"
            )


def scale_landsat_band(tif_path: Path, gain: float, offset: float) -> xr.DataArray:
    """
    Read the file at path `tif_path`, scale the data by `gain` and displace it by
    `offset` to obtain physical values. Further reproject it to EPSG:4326 and replace
    negative values with nan (as these are not physically possible, resulting from
    sensor anomalities). Return the result as DataArray.
    """
    # Read the file as DataArray
    darray = cast(xr.DataArray, rioxr.open_rasterio(tif_path, mask_and_scale=True))
    # Scale and offset the data
    # NOTE: the nodata value is re-written after the operation since the latter resets
    # the former to the default value
    darray = darray * gain + offset
    darray.rio.write_nodata(-32768, encoded=True, inplace=True)
    # Reproject to CRS EPSG:4326
    darray_reproj = darray.rio.reproject("EPSG:4326")
    darray_reproj.rio.write_nodata(-32768, encoded=True, inplace=True)

    # Replace negative values by nan
    darray_reproj = darray_reproj.where(darray_reproj > 0)
    darray_reproj.rio.write_nodata(-32768, encoded=True, inplace=True)

    return darray_reproj


def get_ndvi(prod_dir: Path) -> xr.DataArray:
    """
    Return NDVI by reading the data from the raster files associated with Band 4 SR and
    Band 5 SR, found at `prod_dir`, scaling and offsetting them (to obtain physical
    values), further reproject to EPSG:4326, removing the negative values (as these are
    not physically possible, resulting from sensor anomalities) and applying the NDVI
    formula.

    NOTE: info on the Landsat 8-9 bands associated with NDVI (Band 4 SR and Band 5 SR)
    as well as the values of the scaling factor and offset are issued at table 6-1 of
    "Landsat 8-9 C2 L2SP Guide"
    (https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/media/files/LSDS-1619_Landsat8-9-Collection2-Level2-Science-Product-Guide-v6.pdf#page=18)
    and USGS FAQ
    (https://www.usgs.gov/faqs/what-are-band-designations-landsat-satellites)
    (https://www.usgs.gov/faqs/how-do-i-use-a-scale-factor-landsat-level-2-science-products).
    """
    sr_b4 = scale_landsat_band(next(prod_dir.glob("*SR_B4.*")), 0.0000275, -0.2)
    sr_b5 = scale_landsat_band(next(prod_dir.glob("*SR_B5.*")), 0.0000275, -0.2)
    ndvi = (sr_b5 - sr_b4) / (sr_b5 + sr_b4)
    ndvi.rio.write_nodata(-32768, encoded=True, inplace=True)
    return ndvi


def get_lst(prod_dir: Path) -> xr.DataArray:
    """
    Return LST by reading the data from raster file associated with Band 10 ST, found at
    `prod_dir`, scaling and offsetting it (to obtain physical values), further reproject
    to EPSG:4326 and removing the negative values (as these are not physically possible,
    resulting from sensor anomalities).

    NOTE: info on the Landsat 8-9 band associated with LST (Band 10 ST) as well as the
    values of the scaling factor and offset are issued at row for  of table 6-1 of
    "Landsat 8-9 C2 L2SP Guide"
    (https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/media/files/LSDS-1619_Landsat8-9-Collection2-Level2-Science-Product-Guide-v6.pdf#page=18)
    and USGS FAQ
    (https://www.usgs.gov/faqs/what-are-band-designations-landsat-satellites)
    (https://www.usgs.gov/faqs/how-do-i-use-a-scale-factor-landsat-level-2-science-products).
    """
    lst_file = next(prod_dir.glob("*ST_B10.*"))
    return scale_landsat_band(lst_file, 0.00341802, 149.0)
