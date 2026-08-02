from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# Sentinel-3 relative orbit numbers whose respective orbits fully cover CLIM4cities AOI
# in a descending approach
# NOTE: More information on the relative orbits may be obtained from Sentinel-3 Mission
# Wiki page
# (https://sentiwiki.copernicus.eu/web/s3-mission#S3Mission-OrbitS3-Mission-Orbittrue)
CLIM4CITIES_REL_ORBIT = [
    8,
    22,
    36,
    65,
    79,
    93,
    122,
    136,
    150,
    179,
    193,
    207,
    236,
    250,
    264,
    293,
    307,
    350,
    364,
]


@dataclass
class Sentinel3Config:
    """
    Configurations for querying, downloading and filtering Sentinel-3 companion LST and
    SYN products (to be used in download_sentinel3.py).

    Querying is done through OData API
    (https://documentation.dataspace.copernicus.eu/APIs/OData.html). A query on the LST
    products is firstly done so that they satisfy the given criteria (`timeliness`,
    `start_sensing_dates`, `end_sensing_dates` and `geometry_query`). The resultant
    query items are filtered for extra criteria (`cloud_cover_lims`, `relative_orbit`
    and `orbit_dir`). Only after that, a query for the companion SYN products is done
    for each filtered LST product. If a companion SYN product exists, the LST product is
    downloaded and subject to even more filtering criteria (`cloud_cover_max_aoi` as
    according to `geometry_aoi`). If the LST product satisfies such criteria, the
    companion SYN product is then downloaded.

    For more details on LST and SYN products access
    https://sentiwiki.copernicus.eu/web/slstr-products#S3-SLSTR-Products-L2-LST-Products
    and
    https://sentiwiki.copernicus.eu/web/synergy-products#SYNERGYProducts-L2SYNProductsS3-Synergy-Products-L2-SYN-Products,
    respectively.

    Note that the script download_sentinel3.py is such that the order of the returned
    querying items is with respect to start sensing time in ascending order (oldest to
    newest).

    Parameters
    ----------

    out_dir : Path
        The directory path to save the LST and SYN downloaded data.

    timeliness: {"NR", "NT"}, optional
        Timeliness of the LST and SYN products to consider in the query. This may be
        either:
            - `"NR"` (Near Real Time);
            - `"NT"` (Non-Time Critical).

        If not issued, both timeliness products are considered.

        Note that the revisit times of Sentinel-3 OLCI (SYN) and SLSTR (LST) are less
        than 2 days and less than 1 day, respectively
        (https://sentiwiki.copernicus.eu/web/s3-mission#S3Mission-OrbitS3-Mission-Orbittrue),
        and the uploading time after acquisition for NR (Near Real Time), and NT
        (Non-Time Critical) products less than 3 hours, and after 48 hours (but possibly
        up to 1 month), respectively, for OLCI
        (https://sentiwiki.copernicus.eu/__attachments/1672112/OMPC.ACR.HBK.001%20-%20Sentinel%203%20OLCI%20Land%20Handbook%20-%201.3.pdf#page=24)
        and within 3 hours, and within 24-48 hours, respectively, for SLSTR
        (https://sentiwiki.copernicus.eu/__attachments/1672112/OMPC.ACR.HBK.002%20-%20Sentinel%203%20SLSTR%20Land%20Handbook%202024%20-%201.4.pdf#page=26).


    start_sensing_dates : str or list[str], optional
        Oldest start sensing date of the LST and SYN products to consider in the query,
        with format "%Y-%m-%d". If not issued, it is set as 31 days before the newest
        end sensing date (`end_sensing_dates`). Note that the respective time is set to
        millisecond 0 of the date (in UTC) and the filtering is such that the limit is
        included in the admissible datetime range. If a list is issued,
        `end_sensing_dates` must also correspond to a list - with each entry of one
        associated with the respective entry of the other - and a query is done for each
        resultant datetime range defined by the paired start and end sensing dates.

    end_sensing_dates : str or list[str], optional
        Newest end sensing date of the LST and SYN products to consider in the query,
        with format "%Y-%m-%d". If not issued, it is set as current date. Note that the
        respective time is set to last millisecond of the date (in UTC), and the
        filtering is such that the limit is included in the admissible datetime range.
        If a list is issued, `start_sensing_dates` must also correspond to a list. If a
        list is issued, `start_sensing_dates` must also correspond to a list - with each
        entry of one associated with the respective entry of the other - and a query is
        done for each resultant datetime range defined by the paired start and end
        sensing dates.

    geometry_aoi : str, Path, optional
        WKT string or path to AOI geometry file to filter the LST downloaded data with
        respect to cloud cover fraction. Data would be considered of interest if the
        cloud cover fraction in the AOI does not exceed parameter `cloud_cover_max_aoi`.
        Data with higher cloud cover fraction in the AOI would be deleted. Note that if
        a path to a file is issued, the file format must be supported by
        `geopandas.read_file()` (e.g. `.shp`, `.geojson`, `.json`, `.gpkg`). Also note
        that this filtering is solely considered if `process` is set to `True`.

    geometry_query : str, Path, optional
        WKT string or path to WKT geometry file to intersect the querying LST and SYN
        data with (note that an intersection includes its boundary). Data would be
        considered of interest if its geometry has at least one point in common with
        this one. Note that file's geometry should correspond to a polygon with same
        start and end vertices. Furthermore, its coordinates must be expressed in
        EPSG:4326. The parameter is not only used for querying but also for subsetting
        the downloaded Sentinel-3 products (if `mask_clouds` is `True`). If
        `geometry_aoi` is issued, but `geometry_query` is not, `geometry_query` would be
        set as the convex hull of `geometry_aoi`.

    cloud_cover_lims : tuple[float, float], optional
        Minimum and maximum cloud cover fraction (in percentage) that the queried LST
        data must have. Data with a cloud cover fraction out of the issued range would
        not be considered. The parameter is solely used for filtering the queried
        products before downloading them.

    relative_orbit : list[int], optional
        Relative orbit numbers that the queried LST data must have. Data with numbers
        out of the issued list would not be considered. The parameter is solely used for
        filtering the queried products before downloading them.

        More information on relative orbit numbers may be obtained from Sentinel-3
        Mission Wiki page
        (https://sentiwiki.copernicus.eu/web/s3-mission#S3Mission-OrbitS3-Mission-Orbittrue)

    orbit_dir : {"ASCENDING", "DESCENDING"}, optional
        Orbit direction that the LST queried data must have. Data with directions out of
        the issued list would not be considered. The parameter is solely used for
        filtering the queried products before downloading them.

    cloud_cover_max_aoi : float, default=100
        Maximum cloud cover fraction (in percentage) that the LST data must have in the
        AOI (associated with parameter `geometry_aoi`). Data with a higher value would
        be deleted. The parameter is solely used for filtering after downloading and if
        parameter `geometry_aoi` is issued and `process` is set to `True`.

    process : bool, default=True
        Whether to process the downloaded products by georeferencing, subsetting them to
        bands and domain of interest and in the case of the LST product also masking out
        the clouded pixels (if `mask_clouds` is `True`).

    process_engine : {"rioxarray", "snappy"}, default="rioxarray"
        The alias of the backend Python package to use in the processing of the
        Sentinel-3 products. This is either
            - [`"rioxarray"`](https://corteva.github.io/rioxarray/stable/);
            - [`"snappy"`](https://github.com/senbox-org/esa-snappy) (that is, the
            aliases for the respective backend Python packages).

    filter_max_footprint_aoi_overlap : bool, default=False
        If `True` and more than one LST product is queried for a given date and AOI,
        only the one with the maximum footprint overlap with the AOI is considered.
        Furthermore, if there are multiple products with maximum footprint/AOI overlap,
        the one with the lowest cloud cover fraction in the footprint is considered.

    log_mode : {None, "console", "file", "both"}, default="both"
        The logging mode for the querying, downloading and filtering processes:
            - `None`: No logging is done;
            - `"console"`: Logging is done to console only;
            - `"file"`: Logging is done to a log file only;
            - `"both"`: Logging is done to both console and a log file.
        Note the log file would be defined as `sentinel3_download.log` at `out_dir`.
    """

    out_dir: Path
    timeliness: Literal["NR", "NT"] | None = None
    start_sensing_dates: str | list[str] | None = None
    end_sensing_dates: str | list[str] | None = None
    geometry_aoi: str | Path | None = None
    geometry_query: str | Path | None = None
    cloud_cover_lims: tuple[float, float] | None = None
    relative_orbit: list[int] | None = None
    orbit_dir: list[Literal["ASCENDING", "DESCENDING"]] | None = None
    cloud_cover_max_aoi: float = 100
    process: bool = True
    process_engine: Literal["rioxarray", "snappy"] = "rioxarray"
    mask_clouds: bool = True
    filter_max_footprint_aoi_overlap: bool = False
    log_mode: Literal["console", "file", "both"] | None = "both"


# Configurations for for querying, downloading and filtering Sentinel-3 companion LST
# and SYN products
# (https://sentiwiki.copernicus.eu/web/slstr-products#S3-SLSTR-Products-L2-LST-Products)
config = Sentinel3Config(
    out_dir=Path(__file__).resolve().parents[3] / "assets/data/raw/sentinel3_trial",
    timeliness="NT",
    start_sensing_dates=[
        "20200530T093807",
        "20200615T092312",
        "20220422T093637",
        "20220423T091026",
        "20221019T101010",
        "20230508T091950",
        "20230608T091609",
        "20230904T093454",
    ],
    end_sensing_dates=[
        "20200530T093807",
        "20200615T092312",
        "20220422T093637",
        "20220423T091026",
        "20221019T101010",
        "20230508T091950",
        "20230608T091609",
        "20230904T093454",
    ],
    geometry_aoi="POLYGON ((10 55, 11 55, 11 56, 10 56, 10 55))",
    geometry_query=None,
    cloud_cover_lims=(0, 100),
    relative_orbit=None,
    orbit_dir=["DESCENDING"],
    cloud_cover_max_aoi=25,
    process=True,
    process_engine="snappy",
    mask_clouds=True,
    filter_max_footprint_aoi_overlap=False,
    log_mode="both",
)
