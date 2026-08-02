import re
from pathlib import Path

import geopandas as gpd
from shapely import wkt


def load_aoi_to_gdf(aoi: Path | str | None = None) -> gpd.GeoDataFrame | None:
    """
    Convert a variety of AOI formats (WKT string, file path, vector file)
    into a GeoDataFrame. Returns None if aoi is None.
    """

    if aoi is None:
        return None

    # Case 1: AOI is a string
    if isinstance(aoi, str):
        # Interpret as WKT
        try:
            geom = wkt.loads(aoi)
            return gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")
        except Exception:  # noqa: BLE001, S110
            pass  # Not WKT but maybe it's a file path

        # Interpret as a file path
        path = Path(aoi)
        if path.exists():
            return _load_from_path(path)

        raise ValueError(
            "String provided is neither valid WKT nor an existing file path: " + aoi
        )

    # Case 2: AOI is a Path
    if isinstance(aoi, Path):
        if not aoi.exists():
            raise FileNotFoundError(f"Path does not exist: {aoi}")
        return _load_from_path(aoi)

    raise TypeError(f"Unsupported AOI type: {type(aoi)}")


def _load_from_path(path: Path) -> gpd.GeoDataFrame:
    """Internal helper to read AOI from path."""
    ext = path.suffix.lower()

    # Vector-file case (GeoJSON, Shp, GPKG, etc.)
    if ext in {".shp", ".gpkg", ".geojson", ".json"}:
        return gpd.read_file(path)

    # Otherwise assume text-based file containing WKT
    try:
        text = path.read_text().strip()
        geom = wkt.loads(text)
        return gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"Failed to parse WKT from file '{path}': {e}")


def to_convex_hull_wkt(aoi: gpd.GeoDataFrame | None, buffer: float = 0) -> str | None:
    """
    Get the convex hull of the geometry in `aoi` as WKT string. If `aoi` is None,
    return None.
    """

    if aoi is None:
        return None

    # Get the simplest convex polygon that encloses the whole detailed AOI geometry,
    # considering a small buffer margin
    aoi_simplified = aoi.to_crs(epsg=4326).buffer(buffer).union_all().convex_hull
    return aoi_simplified.wkt


def footprint_aoi_overlap(
    footprint_wkt: str,
    aoi_gdf: gpd.GeoDataFrame | None,
) -> float:
    """
    Extract valid WKT geometry from a Sentinel product's footprint string, and compute
    overlap fraction (in percentage) with AOI. If no AOI is provided, return 0 %.
    """

    if not footprint_wkt:
        return 0.0

    # Extract re's Match object of WKT string in from footprint string
    match = re.search(
        r"(MULTIPOLYGON\s*\(\(.*\)\)|POLYGON\s*\(\(.*\)\))",
        footprint_wkt,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # If no WKT string is found, return zero
    if not match:
        return 0.0

    # Get WKT string from Match object
    fp_wkt = match.group(1)

    # Convert WKT string to GeoDataFrame
    try:
        fp_gdf = load_aoi_to_gdf(fp_wkt)
    except Exception:  # noqa: BLE001
        return 0.0

    # If the footprint is empty or the AOI is not issued or it is empty, return 0
    if fp_gdf is None or fp_gdf.empty or aoi_gdf is None or aoi_gdf.empty:
        return 0.0

    # Convert CRSs of footprint and AOI GeoDataFrames to Universal Transverse Mercator
    # to compute area in squared metres
    fp_gdf = fp_gdf.to_crs(fp_gdf.estimate_utm_crs())
    aoi_gdf = aoi_gdf.to_crs(aoi_gdf.estimate_utm_crs())

    # Get single geometries of footprint and AOI GeoDataFrames
    footprint_geom = fp_gdf.union_all()
    aoi_geom = aoi_gdf.union_all()

    # Intersect footprint and AOI geometries
    inter = footprint_geom.intersection(aoi_geom)

    # If no intersection, return zero
    if inter.is_empty:
        return 0.0

    # Compute overlap fraction (in percentage) between footprint and AOI
    overlap = inter.area / footprint_geom.area * 100.0

    return overlap
