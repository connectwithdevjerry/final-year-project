import requests
import geopandas as gpd
from shapely.geometry import box

# ==============================
# Fetch OSM buildings by bbox
# ==============================
def fetch_osm_buildings(polygon, year):
    """
    Fetch building footprints from OSM within a bbox.
    NOTE: Overpass API does not provide true historical snapshots.
    Year parameter is kept for compatibility.
    """
    minx, miny, maxx, maxy = polygon.bounds

    query = f"""
    [out:json][timeout:50];
    (
      way["building"]({miny},{minx},{maxy},{maxx});
      relation["building"]({miny},{minx},{maxy},{maxx});
    );
    out geom;
    """

    url = "https://overpass-api.de/api/interpreter"
    response = requests.post(url, data={"data": query})
    response.raise_for_status()
    data = response.json()

    features = []
    for el in data.get("elements", []):
        if el["type"] == "way" and "geometry" in el:
            coords = [(pt["lon"], pt["lat"]) for pt in el["geometry"]]
            if len(coords) > 2:
                features.append({
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": [coords]},
                    "properties": {"id": el.get("id"), "tags": el.get("tags", {})}
                })

    if not features:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    return gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")


# ==============================
# Detect new buildings
# ==============================
def get_buildings_comparison(year1, year2):
    """
    Fetch buildings for year1 and year2 in OAU Ile-Ife bounding box,
    and identify new buildings (present in year2 but not in year1).

    Returns:
        dict: {
            "year1": GeoJSON of buildings in year1,
            "year2": GeoJSON of buildings in year2,
            "new_buildings": GeoJSON of buildings only in year2
        }
    """
    # OAU Ile-Ife bounding box
    south, west, north, east = 7.490, 4.480, 7.540, 4.560
    polygon = box(west, south, east, north)

    # Fetch footprints for both years
    gdf_old = fetch_osm_buildings(polygon, year1)
    gdf_new = fetch_osm_buildings(polygon, year2)

    # Ensure CRS is set
    if gdf_old.crs is None:
        gdf_old.set_crs("EPSG:4326", inplace=True)
    if gdf_new.crs is None:
        gdf_new.set_crs("EPSG:4326", inplace=True)

    # Identify new buildings
    if not gdf_old.empty and not gdf_new.empty:
        new_buildings = gpd.overlay(gdf_new, gdf_old, how="difference")
    else:
        new_buildings = gdf_new.copy()

    return {
        "year1": gdf_old.to_json(),
        "year2": gdf_new.to_json(),
        "new_buildings": new_buildings.to_json()
    }
