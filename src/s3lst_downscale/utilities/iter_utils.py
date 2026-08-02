from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np


def difference(
    minuend: list | np.ndarray | set, subtrahend: list | np.ndarray | set
) -> list | np.ndarray | set:
    """
    Perform set difference operation, that is, drop elements of `subtrahend` from
    `minuend`, preserving the type and order of the latter.

    Parameters
    ----------
    minuend : list | np.ndarray | set
        Iterable whose elements are to be filtered.
    subtrahend : list | np.ndarray | set
        Elements to drop from the `minuend`.

    Returns
    -------
    difference : list | np.ndarray | set
        Iterable of elements in `minuend` that are not in `subtrahend`, having the type
        and order of the former.
    """

    difference = [item for item in minuend if item not in subtrahend]

    difference = (
        np.array(difference)
        if isinstance(minuend, np.ndarray)
        else set(difference)
        if isinstance(minuend, set)
        else difference
    )

    return difference


def intersection(
    iter_1: list | np.ndarray | set, iter_2: list | np.ndarray | set
) -> list | np.ndarray | set:
    """
    Perform set intersection operation, that is, keep elements of `iter_1` that are
    also in `iter_2`, preserving the type and order of the former.

    Parameters
    ----------
    iter_1 : list | np.ndarray | set
        Iterable whose elements are to be filtered.
    iter_2 : list | np.ndarray | set
        Elements to keep from `iter_1`.

    Returns
    -------
    intersection : list | np.ndarray | set
        Iterable of elements in `iter_1` that are also in `iter_2`, having the type and
        order of the former.
    """

    intersection = [item for item in iter_1 if item in iter_2]

    intersection = (
        np.array(intersection)
        if isinstance(iter_1, np.ndarray)
        else set(intersection)
        if isinstance(iter_1, set)
        else intersection
    )

    return intersection


def union(
    iter_1: list | np.ndarray | set, iter_2: list | np.ndarray | set
) -> list | np.ndarray | set:
    """
    Perform set union operation, that is, append to `iter_1` the elements of `iter_2`
    that are not already in `iter_1`, preserving the type of the former and the order of
    both.

    Parameters
    ----------
    iter_1 : list | np.ndarray | set
        Iterable to which the elements of `iter_2` that are not already included are to
        be appended.
    iter_2 : list | np.ndarray | set
        Iterable whose elements that are not already included in `iter_1` are to be
        appended.

    Returns
    -------
    union : list | np.ndarray | set
        Iterable `iter_1` with the elements of `iter_2` that are not already included
        appended to it, having the type of the former and the order of both.
    """

    union = list(iter_1) + [item for item in iter_2 if item not in iter_1]

    union = (
        np.array(union)
        if isinstance(iter_1, np.ndarray)
        else set(union)
        if isinstance(iter_1, set)
        else union
    )

    return union


def flatten(iter: Iterable) -> Iterator[Any]:
    """
    Recursively flatten a nested iterable into a flat generator.

    This function yields elements from arbitrarily nested iterables (e.g., lists,
    tuples, sets), while treating strings and bytes as non-iterable variables.

    Parameters
    ----------
    iter : Iterable
        An iterable variable.

    Yields
    ------
    Any
        The individual non-iterable elements from the nested structure.

    """
    for item in iter:
        if isinstance(item, Iterable) and not isinstance(item, (str, bytes)):
            yield from flatten(item)
        else:
            yield item
