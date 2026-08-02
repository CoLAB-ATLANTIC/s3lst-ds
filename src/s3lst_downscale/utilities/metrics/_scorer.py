"""
Functions for handling scorers
"""

import copy
from typing import Any

from . import (
    mae,
    mae_delta,
    mbe,
    r2,
    r2_oos,
    rmse,
    rmse_delta,
    rse,
)

# Dictionary of available scorers
_SCORERS = {
    "r2": r2,
    "r2_oos": r2_oos,
    "rse": rse,
    "rmse": rmse,
    "rmse_delta": rmse_delta,
    "mae": mae,
    "mae_delta": mae_delta,
    "mbe": mbe,
}


def get_scorer_names():
    """Get the names of all available scorers.

    These names can be passed to `metrics.get_scorer()` to retrieve the scorer object.

    Returns
    -------
    list of str
        Names of all available scorers.

    """
    return sorted(_SCORERS.keys())


def get_scorer(alias: str | None):
    """
    Get a scorer from its alias.

    Function `metrics.get_scorer_names()` can be used to retrieve the names of all
    available scorers.

    Parameters
    ----------
    alias : str or None
        Scoring method as string alias.

    Returns
    -------
    scorer : callable
        The scorer.

    Notes
    -----
    This function always returns a copy of the scorer object. Calling `get_scorer` twice
    for the same scorer results in two separate scorer objects.
    """

    if isinstance(alias, str):
        try:
            scorer = copy.deepcopy(_SCORERS[alias])
        except KeyError:
            raise ValueError(
                f"Alias {alias!r} is not associated with any scorer. "
                "Use metrics.get_scorer_names() to get valid options."
            )

    else:
        raise TypeError(f"Expected a string, got {type(alias).__name__} instead.")

    return scorer


def fancify(scores: dict[str, Any]) -> dict[str, Any]:
    """
    Replace scorer aliases used as keys in the issued dictionary `scores` by their fancy
    aliases.

    Parameters
    ----------
    scores : dict[str, Any]
        A dictionary of scores keyed by scorer aliases.

    Returns
    -------
    scores : dict[str, Any]
        The issued dictionary with its keys replaced by scorer fancy aliases.

    """

    scores = {get_scorer(alias).alias_fancy: value for alias, value in scores.items()}

    return scores
