from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import rioxarray as rioxr  # noqa: F401
import xarray as xr

from s3lst_downscale.utilities.exceptions_utils import FailedProdReadingError


@dataclass
class Sen3Loader:
    """Data class for Sentinel-3 data"""

    path: Path
    lst: xr.Dataset = field(init=False)
    syn: xr.Dataset = field(init=False)
    timestamp: pd.Timestamp = field(init=False)

    def __post_init__(self):
        """Load Sentinel-3 data and extract LST's starting sensing time"""
        try:
            self.lst = self._load_file(next(self.path.glob("*LST*.SEN3.nc")))
            self.syn = self._load_file(next(self.path.glob("*SYN*.SEN3.nc")))
            self.timestamp = self._get_timestamp()

        # NOTE: There is no error message in the case of an StopIteration exception
        # (which occurs when in the next() function there is no element to iterate over
        # (which means that the file is missing)). This is then herein handled in a
        # way that is different from general exception's.
        except StopIteration:
            raise FailedProdReadingError("Sentinel-3 LST or SYN files not found.")

        except Exception as e:  # noqa: BLE001
            raise FailedProdReadingError(
                "Error reading Sentinel-3 LST or SYN files." + f"\nError message: {e}"
            )

    def _load_file(self, field_path) -> xr.Dataset:
        """Load Sentinel-3 LST or SYN netCDF files from directory"""

        data = xr.open_dataset(
            field_path,  # type: ignore
            # Mask out NODATA values and scale
            mask_and_scale=True,
            # Properly decode coordinates and CRS
            decode_coords="all",
        )

        if "lon" in data.coords:
            data = data.rename({"lon": "x"})

        if "lat" in data.coords:
            data = data.rename({"lat": "y"})

        return data

    def _get_timestamp(self) -> pd.Timestamp:
        """
        Extract the start sensing timestamp from the Sentinel-3 LST product name.

        Returns
        -------

        timestamp : pd.Timestamp
            Start sensing timestamp associated with Sentinel-3 LST product.

        """
        timestamp = pd.Timestamp(next(self.path.glob("*LST*.SEN3.nc")).name[16:31])

        return timestamp

    def calc_ndvi(self) -> xr.DataArray:
        """
        Calculate NDVI from Sentinel-3 SYN surface directional reflectances associated
        with the OLCI channels [08, 17].
        """
        # Compute NDVI
        # NOTE: the result is an unamed DataArray whose data corresponds to the
        # result of the operation on the data of the raw DataArrays, having the same
        # attributes "dims" and "coords". But the remainder of the attributes would
        # be undefined. For instance, "attrs", "rio.nodata" and "rio.encoded_nodata"
        # would correspond to the respective default values ({}, None and None,
        # respectively.)
        ndvi = (self.syn.SDR_Oa17 - self.syn.SDR_Oa08) / (
            self.syn.SDR_Oa17 + self.syn.SDR_Oa08
        )

        ndvi = ndvi.rename("NDVI")
        ndvi.attrs = self.syn.SDR_Oa08.attrs
        ndvi.attrs["long_name"] = "Normalized Difference Vegetation Index"
        ndvi.attrs["standard_name"] = "normalized_difference_vegetation_index"
        ndvi.attrs["title"] = (
            "SYN L2, NDVI from surface directional reflectance associated with OLCI"
            + " channels [08, 17]"
        )
        ndvi.attrs["units"] = None
        ndvi.attrs["valid_max"] = np.nanmax(ndvi.data)
        ndvi.attrs["valid_min"] = np.nanmin(ndvi.data)
        for key in SYN_ATTRS_TO_REMOVE:
            ndvi.attrs.pop(key, None)

        # Set the NODATA value
        # NOTE: the data of ndvi is already encoded (that is, the NODATA instances
        # correspond to nan) since the respective computation bands (a08 and a17) were
        # also encoded. One may set the encoding NODATA value (the one used in the
        # exported files) through method rio.write_nodata() with encoded=True.
        ndvi.rio.write_nodata(-32768, encoded=True, inplace=True)

        return ndvi

    def calc_ndwi(self) -> xr.DataArray:
        """
        Calculate the Normalized Difference Water Index from Sentinel-3 OLCI SYN surface
        directional reflectance product [channels: 06, 17].
        """

        ndwi = (self.syn.SDR_Oa06 - self.syn.SDR_Oa17) / (
            self.syn.SDR_Oa06 + self.syn.SDR_Oa17
        )

        ndwi = ndwi.rename("NDWI")
        ndwi.attrs = self.syn.SDR_Oa06.attrs
        ndwi.attrs["long_name"] = "Normalized Difference Water Index"
        ndwi.attrs["standard_name"] = "normalized_difference_water_index"
        ndwi.attrs["title"] = (
            "SYN L2, NDWI from surface directional reflectance associated with OLCI"
            + " channels [06, 17]"
        )
        ndwi.attrs["units"] = None
        ndwi.attrs["valid_max"] = np.nanmax(ndwi.data)
        ndwi.attrs["valid_min"] = np.nanmin(ndwi.data)
        for key in SYN_ATTRS_TO_REMOVE:
            ndwi.attrs.pop(key, None)

        ndwi.rio.write_nodata(-32768, encoded=True, inplace=True)

        return ndwi

    def calc_fvc(self) -> xr.DataArray:
        """
        Calculate the Fractional Vegetation Cover (FVC) from Sentinel-3 OLCI SYN
        product. The FVC is calculated using the NDVI and the NDVI thresholds for bare
        soil and full vegetation cover.
        """
        # Compute FVC from NDVI
        ndvi = self.calc_ndvi()
        ndvi_min = np.nanmin(ndvi.values)
        ndvi_max = np.nanmax(ndvi.values)

        # NOTE: see equation 7 of wang2022 (doi: 10.3390/rs14225752)
        fvc = 1 - ((ndvi_max - ndvi) / (ndvi_max - ndvi_min)) ** 0.625

        fvc = fvc.rename("FVC")
        fvc.attrs = ndvi.attrs
        fvc.attrs["long_name"] = "Fractional Vegetation Cover"
        fvc.attrs["standard_name"] = "fractional_vegetation_cover"
        fvc.attrs["title"] = "SYN L2, FVC from NDVI"

        fvc.attrs["units"] = None
        fvc.attrs["valid_max"] = np.nanmax(fvc.data)
        fvc.attrs["valid_min"] = np.nanmin(fvc.data)
        for key in SYN_ATTRS_TO_REMOVE:
            fvc.attrs.pop(key, None)

        fvc.rio.write_nodata(-32768, encoded=True, inplace=True)

        return fvc

    def get_var_by_name(self, var: str) -> xr.DataArray:
        """
        Fetches and returns variables based on their name.
        """
        match var.lower():
            case "lst":
                return self.lst.LST.to_dataset()
            case "ndvi":
                return self.calc_ndvi()
            case "ndwi":
                return self.calc_ndwi()
            case "fvc":
                return self.calc_fvc()
            case _:
                raise ValueError(
                    f"Unknown variable name: {var}. "
                    "Available name: 'LST','NDVI', 'NDWI', 'FVC'."
                )


SYN_ATTRS_TO_REMOVE = [
    "wavelength",
    "STATISTICS_APPROXIMATE",
    "STATISTICS_MAXIMUM",
    "STATISTICS_MEAN",
    "STATISTICS_MINIMUM",
    "STATISTICS_STDDEV",
    "STATISTICS_VALID_PERCENT",
    "ancillary_variables",
    "bandwidth",
]
