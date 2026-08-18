from pathlib import Path
from typing import ClassVar

import pandas as pd


class LCZBase:
    """
    A base class for handling Local Climate Zone (LCZ) numerical codes, labels,
    descriptions, colors and indexex, and their inter-conversion. These attributes
    derive from the file at `path_mapper`.

    The map between codes, labels and descriptions is based on Ana Oliveira et al.'s
    2020 work (doi: 10.1016/j.mex.2020.101150). The map between codes and colors is
    based on the QGIS standard.

    Attributes
    ----------

    path_mapper : Path
        Path to CSV mapper file with the LCZ attributes.

    vars_cat : tuple[str, ...]
        Aliases of the categorical variables in the mapper.

    mapper : pd.DataFrame
        The mapper derived from the CSV files at `path_mapper`. The resultant mapping
        variables are:

            - `"code"`;
            - `"label"`;
            - `"description"`;
            - `"color"`;
            - `"index"`;

    values : dict[str, list]
        Unique values of the `mapper` variables.
    """

    # ---> Class attributes
    # Path to mapper file
    path_mapper = Path(__file__).resolve().parent / "_mapper.csv"

    # Categorical variables
    vars_cat: ClassVar[tuple[str, ...]] = (
        "code",
        "label",
        "description",
        "color",
        "index",
    )

    # ---> Subclass initialization method
    # Method for initializing subclass cls (that is, LCZ) of this very class LCZBase
    def __init_subclass__(cls):
        cls.mapper = cls.get_mapper()
        cls.values = cls.get_unique_mapper_values()

    # ---> Class methods
    @classmethod
    def get_mapper(cls) -> pd.DataFrame:
        """
        Get DataFrame mapper between LCZ codes, labels, descriptions, colors and
        indexes.

        Parameters
        ----------
        cls : type
            An `LCZ` class.

        Returns
        -------
        mapper : pd.DataFrame
            The mapper.
        """

        # Get mapper associated with LCZ attributes
        mapper = pd.read_csv(
            cls.path_mapper,
            dtype={
                "code": str,
                "label": str,
                "description": str,
                "color": str,
                "index": int,
            },
        )

        return mapper

    @classmethod
    def get_unique_mapper_values(cls) -> dict[str, list]:
        """
        Get unique values of LCZ's DataFrame `mapper` variables.

        Parameters
        ----------
        cls : type
            An `LCZ` class.

        Returns
        -------
        values : dict[str, list]
            A dictionary of lists of unique values of LCZ's DataFrame `mapper`
            variables, keyed by variable.
        """

        # Get unique values of LCZ's DataFrame mapper variables
        values = {var: cls.mapper[var].unique().tolist() for var in cls.mapper.columns}

        return values  # type: ignore

    @classmethod
    def convert(
        cls,
        values: pd.Series,
        source: str,
        target: str,
    ) -> pd.Series:
        """
        Convert `values` of a `source` variable into a `target` one. The supported
        `source` and `target` variables correspond to the `mapper` variables:

            - `"code"`;
            - `"label"`;
            - `"description"`;
            - `"color"`;
            - `"index"`.

        Parameters
        ----------
        cls : type
            A `LCZ` class.
        values : pd.Series
            The values to convert.
        source : str
            The variable associated with the issued `values`.
        target : str
            The variable to convert the `values` into.

        Returns
        -------
        converted_values : pd.Series
            The respective converted values.
        """

        # If the source variable is an LCZ code, convert the values to string if they
        # are not already (as the mapper assumes that they have this type)
        if source in [
            "code",
        ] and not pd.api.types.is_object_dtype(values):
            # NOTE: the values are converted to integers before being converted to
            # strings to avoid decimal points appearing in the strings in the case of
            # the data being originally floats. Also, a value-by-value conversion is
            # required for the case of the values corresponding to floats and containing
            # nan.
            values = values.apply(
                lambda value: str(int(value)) if pd.notna(value) else None
            )

        # Get target values from source ones
        converted_values = values.map(cls.mapper.set_index(source)[target])

        # Convert target values to a categorical Series if the target variable is a
        # categorical one
        if target in cls.vars_cat:
            converted_values = pd.Series(
                data=pd.Categorical(
                    values=converted_values,
                    categories=cls.values[target],
                ),
                index=converted_values.index,
            )

        return converted_values


class LCZ(LCZBase):
    """
    Local Climate Zone (LCZ) class (a subclass of `LCZBase`) for handling LCZ codes,
    labels, descriptions, colors and indexes and their inter-conversion.
    """
