from typing import ClassVar, Literal

from s3lst_ds.utilities.iter_utils import difference, intersection, union


class DataVars:
    """
    Class with aliases and units for predictors and target and their kinds.

    Attributes
    ----------
    alias_fancy : dict[str, str]
        Class attribute: mapper between variable aliases and their fancy counterparts.

    units_fancy : dict[str, str]
        Class attribute: mapper between variable aliases and their fancy units.

    X_xt : list[str]
        Base attribute: aliases of the spatio-temporal predictors. Default corresponds
        to a list containing:
            - `"NDVI"`
            - `"FVC"`
            - `"NDWI"`

    X_x : list[str]
        Base attribute: aliases of the pure spatial predictors. Default corresponds to a
        list containing:
            - `"TCD"`
            - `"DEM"`
            - `"IMD"`
            - `"TOPEX"`
            - `"COASTDIST"`
            - `"LCZ"`
            - `"UD"`

    X_t : list[str]
        Base attribute: aliases of the pure temporal predictors. Default corresponds to
        a list containing:
            - `"timestamp"`
            - `"season"`

    X_cat : list[str]
        Base attribute: aliases of the categorical predictors. Default corresponds to a
        list containing:
            - `"LCZ"`,
            - `"timestamp"`,
            - `"season"`.

    y : str
        Base attribute: alias of the target variable. Default corresponds to `"LST"`.

    y_val : str
        Base attribute: alias of the validating target variable. Default corresponds to
        `"LST_landsat"`.

    X_num : list[str]
        Derived attribute: aliases of the numerical predictors.

    X : list[str]
        Derived attribute: aliases of all predictors (numerical and categorical).

    vars_xt : list[str]
        Derived attribute: aliases of the spatio-temporal variables (spatio-temporal
        predictors and target including the validating target).

    vars_x : list[str]
        Derived attribute: aliases of the pure spatial variables (spatial predictors).

    vars_t : list[str]
        Derived attribute: aliases of the pure temporal variables (temporal predictors).

    vars_num : list[str]
        Derived attribute: aliases of the numerical variables (numerical predictors and
        target, including the validating target).

    vars_cat : list[str]
        Derived attribute: aliases of the categorical variables (categorical
        predictors).

    vars : list[str]
        Derived attribute: aliases of all variables (predictors and target).
    """

    alias_fancy: ClassVar[dict[str, str]] = {
        "NDVI": r"$\mathrm{NDVI}$",
        "FVC": r"$\mathrm{FVC}$",
        "NDWI": r"$\mathrm{NDWI}$",
        "TCD": r"$\mathrm{TCD}$",
        "DEM": r"$\mathrm{DEM}$",
        "IMD": r"$\mathrm{IMD}$",
        "TOPEX": r"$\mathrm{TOPEX}$",
        "COASTDIST": r"$\mathrm{DCOAST}$",
        "LCZ": r"$\mathrm{LCZ}$",
        "UD": r"$\mathrm{UD}$",
        "timestamp": r"$\mathrm{Timestamp}$",
        "season": r"$\mathrm{Season}$",
        "LST": r"$\mathrm{LST}$",
        "LST_landsat": r"$\mathrm{LST}$",
    }

    units_fancy: ClassVar[dict[str, str]] = {
        "NDVI": r" $[-]$",
        "FVC": r" $[-]$",
        "NDWI": r" $[-]$",
        "LST": r" $[\mathrm{K}]$",
        "LST_landsat": r" $[\mathrm{K}]$",
    }

    def __init__(
        self,
        X_xt: list[str] | None = None,
        X_x: list[str] | None = None,
        X_t: list[str] | None = None,
        X_cat: list[str] | None = None,
        y: str | None = None,
        y_val: str | None = None,
    ) -> None:

        self.X_xt = (
            X_xt
            if X_xt is not None
            else [
                "NDVI",
                "FVC",
                "NDWI",
            ]
        )
        self.X_x = (
            X_x
            if X_x is not None
            else [
                "TCD",
                "DEM",
                "IMD",
                "TOPEX",
                "COASTDIST",
                "LCZ",
                "UD",
            ]
        )
        self.X_t = (
            X_t
            if X_t is not None
            else [
                "timestamp",
                "season",
            ]
        )
        self.X_cat = (
            X_cat
            if X_cat is not None
            else [
                "LCZ",
                "timestamp",
                "season",
            ]
        )
        self.y = y if y is not None else "LST"
        self.y_val = y_val if y_val is not None else "LST_landsat"

        # Define derived attributes
        self._update()

    def _update(self) -> None:
        """
        Update derived attributes. The derived attributes are:
        - `X_num` (`list[str]`): aliases of the numerical predictors.
        - `X` (`list[str]`): aliases of all predictors (numerical and categorical).
        - `vars_xt` (`list[str]`): aliases of the spatio-temporal variables
        (spatio-temporal predictors and target).
        - `vars_x` (`list[str]`): aliases of the pure spatial variables (spatial
        predictors).
        - `vars_t` (`list[str]`): aliases of the pure temporal variables (temporal
        predictors).
        - `vars_num` (`list[str]`): aliases of the numerical variables (numerical
        predictors and target).
        - `vars_cat` (`list[str]`): aliases of the categorical variables
        (categorical predictors).
        - `vars` (`list[str]`): aliases of all variables (predictors and target).
        """
        self.X_num = difference(self.X_xt + self.X_x + self.X_t, self.X_cat)

        self.X = self.X_num + self.X_cat  # type: ignore
        self.vars_xt = self.X_xt + [self.y, self.y_val]
        self.vars_x = self.X_x
        self.vars_t = self.X_t
        self.vars_num = self.X_num + [self.y, self.y_val]  # type: ignore
        self.vars_cat = self.X_cat
        self.vars = self.vars_num + self.vars_cat

    def set_y(self, y: str) -> "DataVars":
        """
        Set `y` as target alias.

        Parameters
        ----------
        y : str
            Target alias to set.

        Returns
        -------

        data_vars : DataVars
            A new `DataVars` instance with the issued target alias `y` and derived
            parameters updated.
        """

        # Create a copy of the current instance with the issued `y`
        data_vars = DataVars(
            X_xt=self.X_xt,
            X_x=self.X_x,
            X_t=self.X_t,
            X_cat=self.X_cat,
            y=y,
            y_val=self.y_val,
        )
        return data_vars

    def set_y_val(self, y_val: str) -> "DataVars":
        """
        Set `y_val` as validating target alias.

        Parameters
        ----------
        y_val : str
            Validating target alias to set.

        Returns
        -------

        data_vars : DataVars
            A new `DataVars` instance with the issued validating target alias `y_val`
            and derived parameters updated.
        """

        # Create a copy of the current instance with the issued `y_val`
        data_vars = DataVars(
            X_xt=self.X_xt,
            X_x=self.X_x,
            X_t=self.X_t,
            X_cat=self.X_cat,
            y=self.y,
            y_val=y_val,
        )
        return data_vars

    def subset_X(self, X: list[str]) -> "DataVars":
        """
        Subset predictor aliases to `X` and update the derived variable accordingly.

        Parameters
        ----------
        X : list[str]
            Predictor aliases to keep.

        Returns
        -------
        data_vars : DataVars
            A new `DataVars` instance with the issued predictor aliases `X` and derived
            parameters updated.
        """

        # Create a copy of the current instance with the predictor aliases subsetted to
        # the ones of interest
        data_vars = DataVars(
            X_xt=intersection(X, self.X_xt),  # type: ignore
            X_x=intersection(X, self.X_x),  # type: ignore
            X_t=intersection(X, self.X_t),  # type: ignore
            X_cat=intersection(X, self.X_cat),  # type: ignore
            y=self.y,
            y_val=self.y_val,
        )
        return data_vars

    def remove_X(self, X: str | list[str]) -> "DataVars":
        """
        Remove predictor aliases `X` and update the derived variable accordingly.

        Parameters
        ----------
        X : str or list[str]
            Predictor aliases to remove.

        Returns
        -------
        data_vars : DataVars
            A new `DataVars` instance with the issued predictor aliases `X` removed and
            derived parameters updated.
        """

        X = [X] if isinstance(X, str) else X

        # Create a copy of the current instance with the issued predictor aliases
        # removed
        data_vars = DataVars(
            X_xt=difference(self.X_xt, X),  # type: ignore
            X_x=difference(self.X_x, X),  # type: ignore
            X_t=difference(self.X_t, X),  # type: ignore
            X_cat=difference(self.X_cat, X),  # type: ignore
            y=self.y,
            y_val=self.y_val,
        )
        return data_vars

    def add_X(self, X: str, kind_xt: Literal["xt", "x", "t"], cat: bool) -> "DataVars":
        """
        Add alias `X` of a predictor of spatio-temporal kind `kind_xt` and categorical
        predictor indication `cat`, and update the derived variable accordingly.

        Parameters
        ----------
        X : str
            Predictor alias to add.

        kind_xt : {"xt", "x", "t"}
            Spatio-temporal kind of the predictor to add. It can be:
            - `"xt"`: spatio-temporal predictor.
            - `"x"`: pure spatial predictor.
            - `"t"`: pure temporal predictor.

        cat : bool
            Whether the predictor to add is categorical or not.

        Returns
        -------
        data_vars : DataVars
            A new `DataVars` instance with the issued predictor alias `X` added and
            derived parameters updated.
        """

        # Create a copy of the current instance with the issued predictor aliases added
        data_vars = DataVars(
            X_xt=union(self.X_xt, X) if kind_xt == "xt" else self.X_xt,  # type: ignore
            X_x=union(self.X_x, X) if kind_xt == "x" else self.X_x,  # type: ignore
            X_t=union(self.X_t, X) if kind_xt == "t" else self.X_t,  # type: ignore
            X_cat=union(self.X_cat, X) if cat else self.X_cat,  # type: ignore
            y=self.y,
            y_val=self.y_val,
        )
        return data_vars
