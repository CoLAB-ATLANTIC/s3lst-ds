"""
Common code for all metrics.
"""

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike


class Scorer:
    """
    A scoring class to be used in grid-searching.

    The `Scorer` class allows for the convenient wrapping of a scoring function,
    determining the appropriate methods for extracting the optimum value and its index
    from an array-like of scores (`max`/`min` and `argmax`/`argmin`).

    Attributes
    ----------

    alias : str
        An alias for the scorer.

    alias_fancy : str
        A fancy alias for the scorer. This alias could be used in plotting. LaTeX syntax
        is allowed.

    func : Callable
        The scoring function.

    greater_is_better : bool, default=True
        Whether `func` is a score function (default), meaning high is good, or a loss
        function, meaning low is good.

    dimensionless : bool, default=True
        Whether the score is physically non-dimensional (i.e. without physical units).
    """

    def __init__(
        self,
        alias: str,
        alias_fancy: str,
        func: Callable,
        greater_is_better: bool = True,
        dimensionless: bool = True,
    ):
        self.alias = alias
        self.alias_fancy = alias_fancy
        self.func = func
        self.greater_is_better = greater_is_better
        self.dimensionless = dimensionless

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def get_opt(
        self,
        scores: ArrayLike,
        *args,
        **kwargs,
    ):
        """
        Get optimum value in an array-like of scores.

        Parameters
        ----------

            scores : array-like
                The scores.
            *args:
                Other positional arguments passed to the numpy optimizer function
                (`max`/ `min`).
            **kwargs : dict
                Other keyword parameters passed to the numpy optimizer function
                (`max`/ `min`).

        Returns
        -------

            opt :
                The optimum score value.
        """
        opt = (
            np.max(scores, *args, **kwargs)  # type: ignore
            if self.greater_is_better is True
            else np.min(scores, *args, **kwargs)  # type: ignore
        )

        return opt

    def get_argopt(
        self,
        scores: ArrayLike,
        *args,
        **kwargs,
    ):
        """
        Get index of optimum value in an array-like of scores.

        Parameters
        ----------

            scores : array-like
                The scores.
            *args:
                Other positional arguments passed to the numpy optimizer function
                (`argmax`/ `argmin`).
            **kwargs : dict
                Other keyword parameters passed to the numpy optimizer function
                (`argmax`/ `argmin`).

        Returns
        -------

            i_opt :
                The index of the optimum score value.
        """
        i_opt = (
            np.argmax(scores, *args, **kwargs)
            if self.greater_is_better is True
            else np.argmin(scores, *args, **kwargs)
        )

        return i_opt
