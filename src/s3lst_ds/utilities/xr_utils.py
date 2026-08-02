from typing import ClassVar, Literal, cast

import geopandas as gpd
import numpy as np
import xarray as xr
from rasterio.features import rasterize
from rasterio.warp import Resampling

from s3lst_ds.utilities.var_utils import DataVars


class ReprojectDefaultParams:
    """
    A class that defines default values for the parameters of the reprojection
    functions.

    Class Parameters
    ----------------

    resampling : dict[str, dict[str, Resampling]], default=1.0
        Resampling methods for coarsening and refining numerical and categorical
        variables. Defaults to:
        ```
        {
            "num": {
                "coarse": Resampling.average,
                "fine": Resampling.bilinear,
            },
            "cat": {
                "coarse": Resampling.mode,
                "fine": Resampling.nearest,
            },
        }
        ```

    fill_na_with_bilinear : bool, default=True
        Whether to fill missing values using bilinear interpolation in the case of
        reprojection with cubic interpolation, cubic spline interpolation or Lanczos
        interpolation for resampling.
    """

    # Resampling methods
    # NOTE: https://rasterio.readthedocs.io/en/stable/api/rasterio.enums.html#rasterio.enums.Resampling
    resampling: ClassVar[dict[str, dict[str, Resampling]]] = {
        "num": {
            "coarse": Resampling.average,
            "fine": Resampling.cubic_spline,
        },
        "cat": {
            "coarse": Resampling.mode,
            "fine": Resampling.nearest,
        },
    }

    # Whether to fill missing values using bilinear interpolation in the case of cubic
    # interpolation, cubic spline interpolation or Lanczos interpolation.
    fill_na_with_bilinear: ClassVar[bool] = True

    @classmethod
    def set_resampling(
        cls,
        value: Resampling | dict[str, Resampling] | dict[str, dict[str, Resampling]],
        var_type: Literal["num", "cat"] | None = None,
        res_change_type: Literal["coarse", "fine"] | None = None,
    ) -> None:
        """
        Set resampling method `value` to resolution change type `res_change_type` of
        variable type `var_type` of `resampling` (default resampling methods for
        numerical and categorical variables).

        Parameters
        ----------
        value : dict[str, dict[str, Resampling]]
            The resampling methods to set. If `var_type` is not specified, `value`
            should be a dictionary firstly keyed by variable type (`"num"` or `"cat"`)
            and secondly by resolution change type (`"coarse"` or `"fine"`). The
            analogous follows for the cases of `var_type` being specified and
            `res_change_type` not, and both being specified.

        var_type : Literal["num", "cat"] or None, default=None
            The variable type for which to set the resampling method.

        res_change_type : Literal["coarse", "fine"] or None, default=None
            The resolution change type for which to set the resampling method.
        """
        if var_type is not None and res_change_type is not None:
            cls.resampling[var_type][res_change_type] = value  # type: ignore
        elif var_type is not None and res_change_type is None:
            cls.resampling[var_type] = value  # type: ignore
        elif var_type is None:
            cls.resampling = value  # type: ignore

    @classmethod
    def get_resampling(
        cls,
        var_type: Literal["num", "cat"] | None = None,
        res_change_type: Literal["coarse", "fine"] | None = None,
    ) -> Resampling | dict[str, Resampling] | dict[str, dict[str, Resampling]]:
        """
        Get resampling method associated resolution change  type `res_change_type` of
        variable type `var_type` from `resampling` (default resampling methods for
        numerical and categorical variables).

        Parameters
        ----------
        var_type : Literal["num", "cat"] or None, default=None

        res_change_type : Literal["coarse", "fine"] or None, default=None
            The resolution change type of the wanted resampling method.

        Returns
        -------
        dict[str, dict[str, Resampling]]
            The set default resampling methods. If `var_type` is not specified, the
            returned value corresponds to a dictionary firstly keyed by variable type
            (`"num"` or `"cat"`) and secondly by resolution change type (`"coarse"` or
            `"fine"`).  The analogous follows for the cases of `var_type` being
            specified and `res_change_type` not, and both being specified.
        """

        if var_type is not None and res_change_type is not None:
            value = cls.resampling[var_type][res_change_type]
        elif var_type is not None and res_change_type is None:
            value = cls.resampling[var_type]
        elif var_type is None:
            value = cls.resampling

        return value

    @classmethod
    def set_fill_na_with_bilinear(cls, value: bool) -> None:
        """
        Set `fill_na_with_bilinear` (whether to fill missing values using bilinear
        interpolation in the case of reprojection with cubic interpolation, cubic spline
        interpolation or Lanczos interpolation for resampling.) to `value`.

        Parameters
        ----------
        value : bool
            Whether to fill missing values using bilinear interpolation in the case of
            reprojection with cubic interpolation, cubic spline interpolation or Lanczos
            interpolation for resampling.
        """
        cls.fill_na_with_bilinear = value

    @classmethod
    def get_fill_na_with_bilinear(cls) -> bool:
        """
        Get `fill_na_with_bilinear` (whether to fill missing values using bilinear
        interpolation in the case of reprojection with cubic interpolation, cubic spline
        interpolation or Lanczos interpolation for resampling.).

        Returns
        -------
        bool
            Whether to fill missing values using bilinear interpolation in the case of
            reprojection with cubic interpolation, cubic spline interpolation or Lanczos
            interpolation for resampling.
        """
        return cls.fill_na_with_bilinear


def selective_reproject_match_data_array(
    data_src: xr.DataArray,
    data_target: xr.DataArray | xr.Dataset,
) -> xr.DataArray:
    """
    Reproject DataArray `data_src` to match match the resolution, projection, and region
    of DataArray or Dataset `data_target` using an appropriate resampling method
    (according to the kinds of the supported variables `data_vars` and change of scale).
    If `ReprojectDefaultParams.fill_na_with_bilinear` is `True` and `resampling`
    is `Resampling.cubic`, `Resampling.cubic_spline` or `Resampling.lanczos`, the
    missing values resulting from the reprojection are filled with the values resulting
    from the reprojection of a clone using bilinear interpolation.
    """

    # Get aliases of the supported variables
    data_vars = DataVars()

    # Check if variable of the DataArray is supported
    if data_src.name not in data_vars.vars:
        raise ValueError(f"Variable {data_src.name!r} is not supported.")

    # Check if source and target data have the same CRS
    if data_src.rio.crs != data_target.rio.crs:
        raise ValueError(
            "Source and target data must have the same CRS. Resampling "
            " method depends on source and target resolutions, which in turn, depend on"
            " the respective CRS. Change of CRS to a common one also requires a"
            "  resampling method, which leads to a circular problem."
        )

    # Compute source and target effective resolutions
    res_x_src, res_y_src = map(abs, data_src.rio.resolution())
    res_x_target, res_y_target = map(abs, data_target.rio.resolution())
    res = {
        "src": np.sqrt(abs(res_x_src) * abs(res_y_src)),
        "target": np.sqrt(abs(res_x_target) * abs(res_y_target)),
    }

    # Get appropriate resampling method
    resampling = ReprojectDefaultParams.get_resampling(
        var_type="num" if data_src.name in data_vars.vars_num else "cat",
        res_change_type="coarse" if res["src"] < res["target"] else "fine",
    )

    # Reproject DataArray
    reprojected_data = reproject_match(
        data_src=data_src,
        data_target=data_target,
        resampling=resampling,  # type: ignore
    )

    # Fill the missing values with the values obtained from the reprojection of a clone
    # using bilinear interpolation.
    # NOTE: If the resampling method requires too many points (such as cubic
    # interpolation, cubic spline interpolation and Lanczos interpolation), nan areas
    # may be extended from the reprojection. These may be filled with the values
    # obtained from the reprojection of a clone using bilinear interpolation.
    if ReprojectDefaultParams.get_fill_na_with_bilinear() is True and resampling in [
        Resampling.cubic,
        Resampling.cubic_spline,
        Resampling.lanczos,
    ]:
        reprojected_data = reprojected_data.fillna(
            reproject_match(
                data_src=data_src,
                data_target=data_target,
                resampling=Resampling.bilinear,
            )
        )

    return cast(xr.DataArray, reprojected_data)


def selective_reproject_match(
    data_src: xr.DataArray | xr.Dataset,
    data_target: xr.DataArray | xr.Dataset,
) -> xr.DataArray | xr.Dataset:
    """
    Reproject DataArray or Dataset `data_src` to match match the resolution, projection,
    and region of DataArray or Dataset `data_target` using an appropriate resampling
    method for each variable (according to the kinds of the supported variables
    `data_vars`).
    """

    # Get aliases of the supported variables
    data_vars = DataVars()
    # For the case of the source data corresponding to a DataArray
    if isinstance(data_src, xr.DataArray):
        # Check if variable of the DataArray is supported
        if data_src.name not in data_vars.vars:
            raise ValueError(f"Variable {data_src.name!r} is not supported.")

        # Reproject DataArray
        reprojected_data = selective_reproject_match_data_array(data_src, data_target)

    # For the case of the source data corresponding to a Dataset
    elif isinstance(data_src, xr.Dataset):
        reprojected_data = xr.Dataset(
            {
                var: selective_reproject_match_data_array(data_src[var], data_target)
                for var in data_src.data_vars
            }
        )

    return reprojected_data


def shape_to_raster_mask(
    shape: gpd.GeoDataFrame,
    data_target: xr.DataArray | xr.Dataset,
    fill: float = np.nan,
    all_touched: bool = True,
) -> np.ndarray:
    """
    Reproject geometry `shape` to match the resolution and region of `data_target` and
    convert the result to a NumPy array with `1` within the AOI and `fill` out of it.
    This function assumes that `shape` has the same CRS as `data_target`.

    Parameters
    ----------
    shape : gpd.GeoDataFrame
        The input geometry.
    data_target : xr.DataArray or xr.Dataset
        Object with target grid.
    fill : float, default=np.nan
        Used as fill value for all areas not covered by input geometry.
    all_touched : boolean, default=True
        If `True`, all pixels touched by the geometry will be burned in. If false, only
        pixels whose center is within the polygon or that are selected by Bresenham's
        line algorithm will be burned in.

    Returns
    -------
    raster : np.ndarray
        A 2D NumPy array with `1` within the AOI and `fill` out of it, having resolution
        and region of `data_target`.
    """

    transform = data_target.rio.transform()
    width = data_target.sizes["x"]
    height = data_target.sizes["y"]

    return rasterize(
        shapes=shape.dissolve().geometry,
        fill=fill,  # type: ignore
        out_shape=(height, width),
        transform=transform,
        all_touched=all_touched,
        dtype="float32",
    )


def reproject_match(
    data_src: xr.DataArray | xr.Dataset,
    data_target: xr.DataArray | xr.Dataset,
    resampling: Resampling,
) -> xr.DataArray | xr.Dataset:
    """
    Reproject DataArray or Dataset `data_src` to match match the resolution, projection,
    and region of DataArray or Dataset `data_target` using the issued `resampling`
    method. If `ReprojectDefaultParams.fill_na_with_bilinear` is `True` and `resampling`
    is `Resampling.cubic`, `Resampling.cubic_spline` or `Resampling.lanczos`, the
    missing values resulting from the reprojection are filled with the values resulting
    from the reprojection of a clone using bilinear interpolation.
    """

    # NOTE: rio.reproject_match() reprojects a DataArray or Dataset to match the
    # resolution, projection and region of another. The resultant object would have the
    # same attributes "rio.nodata", "rio.encoded_nodata" and "attrs" as the source
    # object. And it would have the same attribute "coords" as the target object. The
    # resultant attribute "data" would correspond to a reprojection of the original data
    # to the target coordinate reference system, with the computation of the values at
    # the target coordinates through interpolation. The area of the target object not
    # covered by the source one would be filled with the value of the parameter "nodata"
    # of rio.reproject_match(). If the parameter "nodata" is not set, the NODATA value
    # of the source object, if this exists, is used. Note that setting or not the
    # parameter "nodata", would not change the behaviour of the method for defining the
    # attributes "rio.nodata", "rio.encoded_nodata" of the resultant object. These
    # would, anyway, correspond to the source ones.
    # NOTE: for more details see
    # https://corteva.github.io/rioxarray/stable/examples/reproject_match.html.
    reprojected_data = data_src.rio.reproject_match(
        data_target,
        resampling=resampling,
    )

    # Fill the missing values with the values obtained from the reprojection of a clone
    # using bilinear interpolation.
    # NOTE: If the resampling method requires too many points (such as cubic
    # interpolation, cubic spline interpolation and Lanczos interpolation), nan areas
    # may be extended from the reprojection. These may be filled with the values
    # obtained from the reprojection of a clone using bilinear interpolation.
    if ReprojectDefaultParams.get_fill_na_with_bilinear() is True and resampling in [
        Resampling.cubic,
        Resampling.cubic_spline,
        Resampling.lanczos,
    ]:
        reprojected_data = reprojected_data.fillna(
            data_src.rio.reproject_match(
                data_target,
                resampling=Resampling.bilinear,  # type: ignore
            )
        )

    # NOTE: according to the documentation, after performing reproject_match(), one
    # should replace the resulting coordinates by the target ones since they may differ
    # due to floating precision. For more details see
    # https://corteva.github.io/rioxarray/stable/examples/reproject_match.html#Raster-Calculations.
    reprojected_data = reprojected_data.assign_coords(
        {
            "x": data_target.x,
            "y": data_target.y,
        }
    )

    return reprojected_data
