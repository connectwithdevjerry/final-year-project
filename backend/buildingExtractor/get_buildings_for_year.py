import requests
import geojson
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, box
from retrying import retry

# Overpass API endpoint
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Hardcoded OAU Ile-Ife bounding box [south, west, north, east]
OAU_BBOX = [7.490, 4.480, 7.540, 4.560]


def build_overpass_query(bbox, year):
    """Construct Overpass QL query for buildings in a bounding box."""
    south, west, north, east = bbox
    query = f"""
    [out:json][timeout:90][date:"{year}-01-01T00:00:00Z"];
    (
      way["building"]({south},{west},{north},{east});
      relation["building"]({south},{west},{north},{east});
    );
    out geom;
    """
    return query


@retry(stop_max_attempt_number=3, wait_fixed=5000)
def fetch_osm_data(query: str):
    """Fetch data from Overpass API with retry."""
    response = requests.post(OVERPASS_URL, data={"data": query}, timeout=180)
    response.raise_for_status()
    return response.json()


def osm_to_geojson(osm_data, osm_year: int):
    """Convert OSM data to GeoJSON features and add osm_year field."""
    features = []

    for element in osm_data.get("elements", []):
        if element["type"] == "way" and "geometry" in element:
            coords = [(node["lon"], node["lat"]) for node in element["geometry"]]
            if len(coords) > 3 and coords[0] == coords[-1]:
                try:
                    polygon = Polygon(coords)
                    if polygon.is_valid and not polygon.is_empty:
                        properties = element.get("tags", {})
                        properties.update({
                            "osm_id": element["id"],
                            "type": "way",
                            "osm_year": osm_year
                        })
                        feature = geojson.Feature(geometry=polygon, properties=properties)
                        features.append(feature)
                except Exception:
                    pass

        elif element["type"] == "relation" and "members" in element:
            polygons = []
            for member in element["members"]:
                if member["type"] == "way" and "geometry" in member:
                    coords = [(node["lon"], node["lat"]) for node in member["geometry"]]
                    if len(coords) > 3 and coords[0] == coords[-1]:
                        try:
                            poly = Polygon(coords)
                            if poly.is_valid and not poly.is_empty:
                                polygons.append(poly)
                        except Exception:
                            pass
            if len(polygons) > 1:
                try:
                    multipolygon = MultiPolygon(polygons)
                    if multipolygon.is_valid and not multipolygon.is_empty:
                        properties = element.get("tags", {})
                        properties.update({
                            "osm_id": element["id"],
                            "type": "relation",
                            "osm_year": osm_year
                        })
                        feature = geojson.Feature(geometry=multipolygon, properties=properties)
                        features.append(feature)
                except Exception:
                    pass

    return geojson.FeatureCollection(features)


def get_buildings_for_year(year: int):
    """
    Get OSM buildings inside OAU Ile-Ife bounding box for a specific year.
    Returns a GeoJSON FeatureCollection.
    """
    bbox = OAU_BBOX
    query = build_overpass_query(bbox, year)
    osm_data = fetch_osm_data(query)

    if osm_data:
        return osm_to_geojson(osm_data, osm_year=year)

    return geojson.FeatureCollection([])  # empty result


def get_new_buildings(year1: int, year2: int):
    """
    Compare buildings between two years and return only the new buildings.
    Buildings existing in year1 are removed from year2.
    Returns a GeoJSON FeatureCollection.
    """
    gdf_old = gpd.GeoDataFrame.from_features(get_buildings_for_year(year1)["features"], crs="EPSG:4326")
    gdf_new = gpd.GeoDataFrame.from_features(get_buildings_for_year(year2)["features"], crs="EPSG:4326")

    if not gdf_old.empty and not gdf_new.empty:
        new_buildings = gdf_new.overlay(gdf_old, how="difference")
    else:
        new_buildings = gdf_new

    return new_buildings.to_json()
