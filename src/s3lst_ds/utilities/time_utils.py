from typing import Literal

import pandas as pd


def get_season(time: pd.Timestamp) -> Literal["Winter", "Spring", "Summer", "Autumn"]:
    """
    Get season of the year from timestamp `time`. The season would correspond to
        - `"Winter"` if the month of `time` is either December, January or February;
        - `"Spring"` if the month of `time` is either March, April or May;
        - `"Summer"` if the month of `time` is either June, July or August;
        - `"Autumn"` if the month of `time` is either September, October or November.

    Parameters
    ----------
    time : pd.Timestamp
        Timestamp from which the year season is to be inferred.

    Returns
    -------
    {"Winter", "Spring", "Summer", "Autumn"}
        The year season associated with the issued `time`.

    """
    match time.month:
        case 12 | 1 | 2:
            season = "Winter"
        case 3 | 4 | 5:
            season = "Spring"
        case 6 | 7 | 8:
            season = "Summer"
        case 9 | 10 | 11:
            season = "Autumn"

    return season
