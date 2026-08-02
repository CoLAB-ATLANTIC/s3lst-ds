import os
import warnings

from rasterio import logging


def suppress_warnings():
    """
    Suppress general, user, rasterio and GDAL warnings. In the case of rasterio and
    GDAL, this is done by setting their minimum logging level to `"ERROR"`.

    Returns
    -------
    None
    """

    # Ignore general warnings
    warnings.filterwarnings("ignore")

    # Ignore user warnings (this is particularly useful for suppressing sklearn's
    # convergence warnings)
    os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"

    # Supress rasterio and GDAL CPL (Common Portability Library) warnings by setting
    # minimum logging level to ERROR instead of INFO
    logging.getLogger().setLevel(logging.ERROR)
    os.environ["CPL_LOG"] = "ERROR"
