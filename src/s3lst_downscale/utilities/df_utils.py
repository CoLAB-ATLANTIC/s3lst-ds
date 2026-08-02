import numpy as np
import pandas as pd


def get_lims_with_tol(
    data: pd.DataFrame,
    tol: float = 0.05,
) -> tuple[float, float]:
    """
    Get absolute minimum and maximum (that is, the limits of the limits of all columns)
    of the issued `data` considering a tolerance of factor `tol` in both sides. The
    tolerance is defined as the difference between the limits scaled by `tol`.
    Therefore, the returned values are a tuple whose entries correspond to

        - `min = min - tol * (max - min)`
        - `max = max + tol * (max - min)`

    Parameters
    ----------
    data : pd.DataFrame

    tol : float, default = 0.05,
        Tolerance factor to use in the transformation of the limits.

    Returns
    -------
    (min, max) : tuple[float, float]
        Tuple with the absolute minimum and maximum of the issued data after considering
        a tolerance in both sides.

    """
    # Get absolute minima and maxima in the issued DataFrame
    lims = (data.min().min(), data.max().max())

    # Add tolerance to the extrema
    lims = (
        lims[0] - tol * (lims[1] - lims[0]),
        lims[1] + tol * (lims[1] - lims[0]),
    )

    return lims


def get_i_non_nan(
    data: pd.DataFrame,
) -> np.ndarray:
    """
    Get the integer indexes of the rows of a DataFrame that do not contain any nan.

    Parameters
    ----------
    data : pd.DataFrame


    Returns
    -------
    i_non_nan : np.ndarray
        The integer indexes of the rows of the DataFrame `data` that do not contain any
        nan.

    """
    i_non_nan = data.index.get_indexer(data.dropna().index)

    return i_non_nan  # type: ignore
