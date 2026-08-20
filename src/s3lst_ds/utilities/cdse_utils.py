import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import requests
from shapely import wkt
from shapely.errors import ShapelyError

from s3lst_ds.utilities.exceptions_utils import (
    AccessTokenGenerationError,
    AccessTokenRefreshError,
    GeometryError,
    JSONDecodeError,
    WritingError,
)
from s3lst_ds.utilities.logging_utils import RichLogger
from s3lst_ds.utilities.tqdm_utils import tqdm


# ----------------------- Getter of CDSE's access token ---------------------- #
@dataclass
class CDSEAuthState:
    """
    A class for holding CDSE's authentication state (credentials, refresh token, and
    time of last access token generation) and obtaining a valid access token (through
    generation, refresh or regeneration).

    Attributes
    ----------
    username : str
        User mail for accessing CDSE (only required for access token generation).

    password : str
        User password for accessing CDSE (only required for access token generation).

    refresh_token : str or None, default=None
        The refresh token that had been previously returned in the former access token
        generation request (only required for access token refresh).

    time_last_access_token_generation : datetime.time or None, default=None
        Time of the last access token generation. If `None`, the function would assume
        that no access token had been previously generated (it would then request the
        generation of a new one).
    """

    username: str
    password: str

    refresh_token: str | None = None
    time_last_access_token_generation: pd.Timestamp | None = None

    def get_access_token(self) -> str:
        """
        Get a valid CDSE access token (generated, refreshed or regenerated).

        According to CDSE's Quotas and Limitations page
        (https://documentation.dataspace.copernicus.eu/Quotas.html#copernicus-general-users),
        in the case of the OData API, the token remains active only for 10 minutes.
        However, it may be refreshed as many times as wanted within the first 60 minutes
        through a refresh request using the refresh token previously issued as response
        to the former access token generation request
        (https://documentation.dataspace.copernicus.eu/APIs/OData.html#product-download).
        After those 60 min, a new access token would need to be generated.

        The access token generation request requires login credentials (`username` and
        `password`) while the access token refresh solely requires the `refresh_token`.
        Note that refreshing solely re-activates the access token. The request returns
        the same access and refresh tokens.

        The number of active access tokens is limited to 100. The current value may be
        known from CDSE's Device Activity page
        (https://identity.dataspace.copernicus.eu/auth/realms/CDSE/account/account-security/device-activity).


        Returns
        -------
        access_token : str
            A valid CDSE access token.
        """

        tokens = get_access_and_refresh_tokens(
            username=self.username,
            password=self.password,
            refresh_token=self.refresh_token,
            time_last_access_token_generation=self.time_last_access_token_generation,
        )

        self.refresh_token = tokens["refresh_token"]
        self.time_last_access_token_generation = tokens[
            "time_last_access_token_generation"
        ]

        return tokens["access_token"]


def get_access_and_refresh_tokens(
    username: str | None = None,
    password: str | None = None,
    refresh_token: str | None = None,
    time_last_access_token_generation: pd.Timestamp | None = None,
    logger: RichLogger | None = None,
) -> dict[str, Any]:
    """
    Get CDSE's access and refresh tokens for downloading data.

    According to CDSE's Quotas and Limitations page
    (https://documentation.dataspace.copernicus.eu/Quotas.html#copernicus-general-users),
    in the case of the OData API, the token remains active only for 10 minutes. However,
    it may be refreshed as many times as wanted within the first 60 minutes through a
    refresh request using the refresh token previously issued as response to the former
    access token generation request
    (https://documentation.dataspace.copernicus.eu/APIs/OData.html#product-download).
    After those 60 min, a new access token would need to be generated.

    The access token generation request requires login credentials (`username` and
    `password`) while the access token refresh solely requires the `refresh_token`. Note
    that refreshing solely re-activates the access token. The request returns the same
    access and refresh tokens.

    Parameters
    ----------
    username : str
        User mail for accessing CDSE (only required for access token generation).

    password : str
        User password for accessing CDSE (only required for access token generation).

    refresh_token : str
        The refresh token that had been previously returned in the former access token
        generation request (only required for access token refresh).

    time_last_access_token_generation : datetime.time or None, default=None
        Time of the last access token generation. If `None`, the function would assume
        that no access token had been previously generated (it would then request the
        generation of a new one).

    logger : RichLogger or None, default=None
        A rich logger for stating progress.

    Returns
    -------
    tokens_and_time_last_gen: dict[str, str]
        Dictionary with access and refresh tokens as well as the time of the last access
        token generation. These are keyed by "access_token", "refresh_token" and
        "time_last_access_token_generation". If the request is such that no new access
        token was generated, the value for the latter is set as the one of the argument
        `time_last_access_token_generation`.

    """

    # Define a maximum time past last access token generation after which a new one
    # would need to be done, in seconds (s)
    time_threshold_generation = 45 * 60

    # Get difference between current time and the one of the last access token
    # generation
    time_diff = (
        (pd.Timestamp.now() - time_last_access_token_generation).total_seconds()
        if time_last_access_token_generation is not None
        else None
    )

    # If no access token had been previously generated (in that case
    # time_last_access_token_generation None), or it had been but the time past it is
    # longer than the threshold, generate a new token.
    if time_diff is None or time_diff > time_threshold_generation:
        # Check for issued username and password
        if username is None:
            if logger is not None:
                logger.error(
                    "[bold red]Access token generation failed. No username was issued."
                    + "\nRun will stop.[/bold red]"
                )
            raise AccessTokenGenerationError(
                "Access token generation failed. No username was issued."
            )

        if password is None:
            if logger is not None:
                logger.error(
                    "[bold red]Access token generation failed. No password was issued."
                    + "\nRun will stop.[/bold red]"
                )
            raise AccessTokenGenerationError(
                "Access token generation failed. No password was issued."
            )

        try:
            # Send POST request to generate the access token
            r = requests.post(
                "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
                data={
                    "client_id": "cdse-public",
                    "grant_type": "password",
                    "username": username,
                    "password": password,
                },
            )
            # Raise an exception in case of request error
            r.raise_for_status()

            # Update the time for the access token generation
            time_last_access_token_generation = pd.Timestamp.now()

            # Return new access and refresh tokens as well as the time of the access
            # token generation if no error occurred
            r_json = r.json()
            return {
                "access_token": r_json["access_token"],
                "refresh_token": r_json["refresh_token"],
                "time_last_access_token_generation": time_last_access_token_generation,
            }

        except Exception as e:  # noqa: BLE001
            if logger is not None:
                logger.error(
                    "[bold red]Access token generation failed."
                    + f"\nError message: {e}"
                    + "\nRun will stop.[/bold red]",
                )
            raise AccessTokenGenerationError(
                "Access token generation failed." + f"\nError message: {e}"
            )

    # If an access token had been previously generated and the time past it is no longer
    # than the threshold, refresh the access token.
    else:
        # Check for issued refresh token
        if refresh_token is None:
            if logger is not None:
                logger.error(
                    "[bold red]Access token refresh failed. No refresh token was"
                    + " issued.\nRun will stop.[/bold red]",
                )
            raise AccessTokenRefreshError(
                "Access token refresh failed. No refresh token was issued."
            )

        try:
            # Send POST request to refresh the access token
            r = requests.post(
                "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
                data={
                    "client_id": "cdse-public",
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                },
            )
            # Raise an exception in case of request error
            r.raise_for_status()

            # Return access and refresh tokens as well as the time of the access token
            # generation if no error occurred
            r_json = r.json()
            return {
                "access_token": r_json["access_token"],
                "refresh_token": r_json["refresh_token"],
                "time_last_access_token_generation": time_last_access_token_generation,
            }

        except Exception as e:  # noqa: BLE001
            if logger is not None:
                logger.error(
                    "[bold red]Access token refresh failed."
                    + f"\nError message: {e}"
                    + "\nRun will stop.[/bold red]",
                )
            raise AccessTokenRefreshError(
                "Access token refresh failed." + f"\nError message: {e}"
            )


# ----------------------------- CDSE data querier ---------------------------- #
def query_cdse(
    collection: str,
    includes: str | list[str] | tuple[str] | None = None,
    excludes: str | list[str] | tuple[str] | None = None,
    start_sensing_time: str | None = None,
    end_sensing_time: str | None = None,
    geometry: str | None = None,
    orderby_start_sensing_time: Literal["desc", "asc"] | None = "asc",
    max_query_items: int | None = None,
    logger: RichLogger | None = None,
) -> dict:
    """
    Query CDSE (Copernicus Data Space Ecosystem) using the OData API for data and return
    the response as a JSON object.

    Parameters
    ----------

    collection : str
        The name of the data collection, which as listed in the CDSE documentation
        (https://documentation.dataspace.copernicus.eu/APIs/OData.html#query-collection-of-products)
        may take the values:
        - `"SENTINEL-1"`
        - `"SENTINEL-2"`
        - `"SENTINEL-3"`
        - `"SENTINEL-5P"`
        - `"SENTINEL-6"`
        - `"SENTINEL-1-RTC"`
        - `"GLOBAL-MOSAICS"`
        - `"SMOS"`
        - `"ENVISAT"`
        - `"LANDSAT-5"`
        - `"LANDSAT-7"`
        - `"LANDSAT-8"`
        - `"COP-DEM"`
        - `"TERRAAQUA"`
        - `"S2GLC"`
        - `"CCM"`

    includes : str or list[str] or tuple[str], optional
        Including file name substring(s) to consider.

    excludes : str or list[str] or tuple[str], optional
        Excluding file name substring(s) to consider.

    start_sensing_time : str, optional
        Oldest start sensing datetime to consider.

    end_sensing_time : str, optional
        Newest end sensing datetime to consider.

    geometry : str, optional
        Geometry to intersect the data with (note that an intersection includes its
        boundary). This parameter needs to be a WKT string describing a polygon with
        same start and end vertices. Furthermore, its coordinates need to be expressed
        in EPSG:4326.

    orderby_start_sensing_time : {"desc", "asc"}, default="asc"
        Order of the returned querying items with respect to start sensing time. It may
        take the values:
        - `"desc"`: newest to oldest
        - `"asc"`: oldest to newest

    max_query_items : int, default=20
        Maximum number of items that may be returned by the query. It must be an integer
        between 0 and 1000 (inclusive). If not defined, 20 is considered. Check CDSE
        documentation for more details
        (https://documentation.dataspace.copernicus.eu/APIs/OData.html#top-option).

    logger : RichLogger or None, default=None
        A rich logger for stating progress.

    Returns
    -------
    response_json : dict
        The response of the query as a JSON object.

    """

    # ---> Check validity of the querying parameters
    # Check if collection pertains to the list of available collection names
    allowed_collection_values = [
        "SENTINEL-1",
        "SENTINEL-2",
        "SENTINEL-3",
        "SENTINEL-5P",
        "SENTINEL-6",
        "SENTINEL-1-RTC",
        "GLOBAL-MOSAICS",
        "SMOS",
        "ENVISAT",
        "LANDSAT-5",
        "LANDSAT-7",
        "LANDSAT-8",
        "COP-DEM",
        "TERRAAQUA",
        "S2GLC",
        "CCM",
    ]
    if collection is not None and collection not in allowed_collection_values:
        if logger is not None:
            logger.error(
                f"[bold red]Invalid collection: {collection!r}. Allowed values are:\n"
                + "\n".join(f"- {v!r}" for v in allowed_collection_values)
                + "\nRun will stop.[/bold red]",
            )
        raise ValueError(
            f"Invalid collection: {collection!r}. Allowed values are:\n"
            + "\n".join(f"- {v!r}" for v in allowed_collection_values)
        )

    # Check if geometry is of WKT format (if not None)
    if geometry is not None:
        try:
            wkt.loads(geometry)
        except ShapelyError as e:
            if logger is not None:
                logger.error(
                    "[bold red]Invalid geometry. It must be in WKT format."
                    + "\nRun will stop.[/bold red]",
                )
            raise GeometryError(
                "Invalid geometry. It must be in WKT format." + f"\nError message: {e}"
            )

    # Check if orderby_start_sensing_time is valid (that is either "desc", "asc" if not
    # None)
    if orderby_start_sensing_time is not None and (
        orderby_start_sensing_time not in ["desc", "asc"]
    ):
        raise ValueError(
            f"Invalid orderby_start_sensing_time: {orderby_start_sensing_time!r}."
            + " Allowed values are 'desc' or 'asc'."
        )

    # Check max_query_items is valid (that is, an integer between 0 and 1000, inclusive,
    # if not None)
    if max_query_items is not None and (
        not isinstance(max_query_items, int) or not (0 <= max_query_items <= 1000)
    ):
        raise ValueError(
            f"Invalid max_query_items: {max_query_items}."
            + " It must be an integer between 0 and 1000 (inclusive)."
        )

    # ---> Define query body
    # Define query URL
    query_url = (
        # URL for the OData products search endpoint
        "https://catalogue.dataspace.copernicus.eu/odata/v1/Products?"
        # Filter parameter
        + "$filter="
        # Filter for collection name
        #  NOTE: check
        #  https://documentation.dataspace.copernicus.eu/APIs/OData.html#query-collection-of-products
        + f"Collection/Name eq '{collection}'"
        # Filter for including file name substrings
        # NOTE: check
        # https://learn.microsoft.com/en-us/dynamics365/business-central/dev-itpro/webservices/use-filter-expressions-in-odata-uris
        + (
            " and "
            + " and ".join(
                f"contains(Name, '{includes_i}')"
                for includes_i in (
                    includes if isinstance(includes, (list, tuple)) else [includes]
                )
            )
            if includes is not None
            else ""
        )
        # Filter for excluding file name substrings
        # NOTE: check
        # https://learn.microsoft.com/en-us/dynamics365/business-central/dev-itpro/webservices/use-filter-expressions-in-odata-uris
        + (
            " and "
            + " and ".join(
                f"not contains(Name, '{excludes_i}')"
                for excludes_i in (
                    excludes if isinstance(excludes, (list, tuple)) else [excludes]
                )
            )
            if excludes is not None
            else ""
        )
        # Filter for newest start sensing date
        # NOTE: check
        # https://documentation.dataspace.copernicus.eu/APIs/OData.html#query-by-sensing-date
        + (
            f" and ContentDate/Start ge {start_sensing_time}"
            if start_sensing_time is not None
            else ""
        )
        # Filter for oldest end sensing date
        + (
            f" and ContentDate/End le {end_sensing_time}"
            if end_sensing_time is not None
            else ""
        )
        # Filter for intersection with geometry
        # NOTE: check
        # https://documentation.dataspace.copernicus.eu/APIs/OData.html#query-by-geographic-criteria
        + (
            f" and OData.CSC.Intersects(area=geography'SRID=4326;{geometry}')"
            if geometry is not None
            else ""
        )
        # Order of the returned query items as according to start sensing time
        # NOTE: check
        # https://documentation.dataspace.copernicus.eu/APIs/OData.html#orderby-option
        + (
            f"&$orderby=ContentDate/Start {orderby_start_sensing_time}"
            if orderby_start_sensing_time is not None
            else ""
        )
        # Maximum number of return query items
        # NOTE: check
        # https://documentation.dataspace.copernicus.eu/APIs/OData.html#top-option
        + (f"&$top={max_query_items}" if max_query_items is not None else "")
        # Consider "attributes expansion" to get the full metadata of each returned
        # query item (it will be associated with the key "Attributes" of each item)
        # NOTE: check
        # https://documentation.dataspace.copernicus.eu/APIs/OData.html#expand-attributes
        + ("&$expand=Attributes")
    )

    # ---> Perform query and get response as JSON object
    r = requests.get(query_url)

    # Raise an exception in case of request error, stopping the code run
    r.raise_for_status()

    # If no error try decoding response as Json object
    try:
        r_json = r.json()
        return r_json
    except json.JSONDecodeError as e:
        if logger is not None:
            logger.error(
                "[bold red]Error occurred while decoding response as JSON object."
                + f"\nError message: {e}"
                + "\nRun will stop.[/bold red]",
            )
        raise JSONDecodeError(
            "Error occurred while decoding response as JSON object."
            + f"\nError message: {e}"
        )


# --------------------------- CDSE data downloader --------------------------- #
def download_cdse(
    products_info: pd.DataFrame,
    path_out_dir: Path,
    access_token: str,
    max_workers: int = 4,
    show_progress: bool = True,
    logger: RichLogger | None = None,
) -> None:
    """
    Download products from CDSE (Copernicus Data Space Ecosystem) using the OData API,
    as described in the documentation
    (https://documentation.dataspace.copernicus.eu/APIs/OData.html#product-download).

    Parallel requesting is done for efficiency. Note that downloading files may by slow
    due to waiting for the external system, not because of CPU processing. Parallel
    request would increase the number of files downloaded per time.

    Parameters
    ----------

    products_info : pd.DataFrame
        DataFrame with products info previously obtained in a query to CDSE search
        endpoint.

    path_out_dir : Path
        Path to directory where to save downloaded products. The respective files would
        have the same name as the one issued in the `Name` field of `products_info`.

    access_token : str
        The CDSE access token for downloading the data.

    max_workers : int, default=4
        Maximum number of processes to run in parallel. This value cannot be higher than
        4 or error `429 Client Error: Too Many Requests for url` would occur. According
        CDSE's "Quotas and Limitations" page, the "Number of concurrent connections
        limit" is 4 (for more details, read
        https://documentation.dataspace.copernicus.eu/Quotas.html). If `1`, no
        parallelization is done.

    show_progress : bool, default=True
        `True` to display the download progress in real-time.

    logger : RichLogger or None, default=None
        A rich logger for stating progress.

    Returns
    -------

    None
    """

    # Define a request session (to reuse the same connection for each request, speeding
    # the process)
    session = requests.Session()

    # Create output folder if it does not exist already
    path_out_dir.mkdir(parents=True, exist_ok=True)

    # Define worker function for downloading a single file (to be used in threading)
    def worker(
        prod_id: str,
        prod_name: str,
    ) -> None:

        # Define file path where to save downloaded file content
        path_out = path_out_dir / prod_name

        # Define request header for the token and use it in the request session
        session.headers.update({"Authorization": f"Bearer {access_token}"})

        # Define download request body
        # NOTE: check
        # https://documentation.dataspace.copernicus.eu/APIs/OData.html#product-download
        download_url = (
            # URL for the OData products download endpoint
            "https://download.dataspace.copernicus.eu/odata/v1/Products"
            # Product Id
            + "("
            + prod_id
            + ")"
            + "/$value"
        )

        # Perform download request
        # NOTE: with stream set to True, the resultant response is of a stream kind.
        # In that case, the file is not fully loaded into memory and downloaded. The
        # streaming allows one to do the loading and download in chunks, which further
        # avoids memory problems
        r = session.get(download_url, stream=True)

        # Raise an exception in case of request error, stopping the code run
        r.raise_for_status()
        # If no error, try opening file in write-binary mode and writing the content in
        # chunks of 8 KB
        # NOTE: this works with any file format since it writes binary data
        try:
            with open(path_out, "wb") as file:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)
        except Exception as e:  # noqa: BLE001
            if logger is not None:
                logger.error(
                    "[bold red]Error occurred while downloading and writing file"
                    + f" {prod_name}."
                    + f"\nError message: {e}"
                    + "\nRun will stop.[/bold red]",
                )
            raise WritingError(
                "Error occurred while downloading and writing file"
                + f" {prod_name}."
                + f"\nError message: {e}"
            )

    # Define a progress bar to show download progress, if wanted
    pbar = (
        tqdm(
            # Prefix for the progressbar
            bar_format=f"{'':9}" + "{l_bar}{bar}{r_bar}",
            desc=f"{'':8}",
            total=len(products_info["Id"]),
            unit="file",
            options={"console": logger.console},
        )
        if show_progress is True and logger is not None
        else None
    )

    # Perform download using multithreading
    if max_workers != 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # List of placeholders for the eventual result of a computation
            futures = [
                executor.submit(worker, prod_id, prod_name)
                for prod_id, prod_name in zip(
                    products_info["Id"], products_info["Name"]
                )
            ]

            for future in as_completed(futures):
                # Call result of the completed process to raise exception if there is any
                # (if not done, exceptions may occur but not be raised)
                _ = future.result()
                # Update progress bar with one more count per completed process
                if pbar is not None:
                    pbar.update(1)

            # At the end close progress bar
            if pbar is not None:
                pbar.close()

    else:
        for prod_id, prod_name in zip(products_info["Id"], products_info["Name"]):
            worker(prod_id, prod_name)

            # Update progress bar with one more count per completed process
            if pbar is not None:
                pbar.update(1)
