# ileife_building_features.py
# Requirements:
# pip install geopandas shapely pystac-client planetary-computer rasterio rioxarray xarray numpy pandas geopy scikit-image tqdm pyproj

import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape
from shapely.geometry import mapping
from shapely.affinity import rotate
from shapely.ops import unary_union
from geopy.geocoders import Nominatim
from pystac_client import Client
import planetary_computer as pc
import json
import rasterio
import rioxarray as rxr
import xarray as xr
from rasterio.features import geometry_mask
from tqdm import tqdm
from math import pi, atan2, degrees

# -------------------------
# 1. Get bbox for Ile-Ife
# -------------------------
def get_bbox_for_place(place_name="Ile-Ife, Nigeria", user_agent="my_geocoder"):
    geolocator = Nominatim(user_agent=user_agent)
    loc = geolocator.geocode(place_name, exactly_one=True, timeout=30)
    if loc is None:
        raise ValueError(f"Could not geocode {place_name}")
    # Nominatim returns lat/lon; build a small bbox around the point if not area
    # Prefer bounding box if provided:
    # Some locations return bounding box string; attempt to access via raw
    try:
        raw = loc.raw
        bbox = raw.get("boundingbox")
        if bbox:
            # boundingbox is [south, north, west, east]
            south, north = float(bbox[0]), float(bbox[1])
            west, east = float(bbox[2]), float(bbox[3])
            return [west, south, east, north]
    except Exception:
        pass
    # fallback: make a 0.03deg buffer square ~ ~3km
    lat, lon = loc.latitude, loc.longitude
    d = 0.03
    return [lon - d, lat - d, lon + d, lat + d]

# -------------------------
# 2. Query Microsoft ms-buildings
# -------------------------
def fetch_ms_buildings(bbox, out_geojson="ileife_buildings.geojson"):
    from pystac_client import Client
    import planetary_computer as pc
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import shape

    # connect to planetary computer STAC
    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")

    # no need to set limit, pagination will be handled by get_all_items()
    search = catalog.search(collections=["ms-buildings"], bbox=bbox)
    items = list(search.get_all_items())
    if len(items) == 0:
        raise RuntimeError("No ms-buildings items found for bbox; try increasing bbox or check dataset coverage.")

    geoms = []
    for it in items:
        for asset_key, asset in it.assets.items():
            if asset.media_type and "application/geo+json" in asset.media_type:
                url = pc.sign(asset.href)
                try:
                    tmp = gpd.read_file(url)
                    geoms.append(tmp)
                except Exception as e:
                    print(f"Warning reading asset {asset_key}: {e}")

    if len(geoms) == 0:
        # fallback: use item.geometry if asset not found
        for it in items:
            geom = it.geometry
            if geom:
                g = gpd.GeoDataFrame([it.properties], geometry=[shape(geom)], crs="EPSG:4326")
                geoms.append(g)

    if len(geoms) == 0:
        raise RuntimeError("Could not load any building geometry assets for this bbox.")

    gdf = gpd.GeoDataFrame(pd.concat(geoms, ignore_index=True))
    gdf = gdf.set_crs("EPSG:4326", allow_override=True)
    gdf.to_file(out_geojson, driver="GeoJSON")
    print(f"✅ Saved {len(gdf)} footprints to {out_geojson}")
    return gdf

# -------------------------
# 3. Compute geometric features
# -------------------------
def compute_geometric_features(gdf):
    # Project to an appropriate local projection for metric measures (UTM zone)
    # Use centroid to pick UTM zone
    centroid = gdf.unary_union.centroid
    lon, lat = centroid.x, centroid.y

    # helper: get UTM zone EPSG
    def lonlat_to_utm_epsg(lon, lat):
        zone = int((lon + 180) / 6) + 1
        epsg = 32600 + zone if lat >= 0 else 32700 + zone
        return int(epsg)

    epsg = lonlat_to_utm_epsg(lon, lat)
    gdf_m = gdf.to_crs(epsg)

    # area (m2), perimeter (m), compactness, centroid_x/y, orientation (deg), bbox dims
    rows = []
    for idx, row in gdf_m.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        area = geom.area
        perimeter = geom.length
        compactness = (4 * pi * area / (perimeter ** 2)) if perimeter > 0 else np.nan
        centroid = geom.centroid
        # orientation: calculate angle of minimum rotated rectangle long axis
        minrect = geom.minimum_rotated_rectangle
        # measure major axis by checking rectangle edges lengths
        try:
            coords = list(minrect.exterior.coords)[:-1]
            edges = []
            for i in range(len(coords)):
                x1, y1 = coords[i]
                x2, y2 = coords[(i + 1) % len(coords)]
                edges.append(((x2 - x1), (y2 - y1)))
            lens = [np.hypot(dx, dy) for (dx, dy) in edges]
            # pick longer edge as major axis
            max_idx = int(np.argmax(lens))
            dx, dy = edges[max_idx]
            angle_rad = atan2(dy, dx)
            angle_deg = (degrees(angle_rad) + 360) % 180  # orientation 0-180
        except Exception:
            angle_deg = np.nan

        rows.append({
            "orig_index": idx,
            "area_m2": area,
            "perimeter_m": perimeter,
            "compactness": compactness,
            "centroid_x": centroid.x,
            "centroid_y": centroid.y,
            "orientation_deg": angle_deg
        })

    feats = pd.DataFrame(rows).set_index("orig_index")
    # join back to original gdf
    gdf_features = gdf.join(feats, how="left")
    return gdf_features, epsg

# -------------------------
# 4. Download Sentinel-2 median composite (Planetary Computer)
# -------------------------
def get_sentinel2_median(bbox, time_range=("2020-01-01","2024-12-31"), bands=["B02","B03","B04","B08"], out_tif="s2_median.tif"):
    # connect STAC and search sentinel-2-l2a
    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    search = catalog.search(collections=["sentinel-2-l2a"], bbox=bbox, datetime="/".join(time_range), query={"eo:cloud_cover":{"lt":30}}, limit=500)
    items = list(search.get_all_items())
    if len(items) == 0:
        raise RuntimeError("No Sentinel-2 items found for bbox/time.")
    # build VRT-like stack by opening each band's asset and computing median via xarray
    band_arrays = []
    for band in bands:
        arrays = []
        for it in tqdm(items, desc=f"loading {band} assets"):
            # assets names tend to be like"B02" etc; sign url
            if band in it.assets:
                href = it.assets[band].href
                signed = pc.sign(href)
                try:
                    arr = rxr.open_rasterio(signed, masked=True).squeeze()
                    arrays.append(arr)
                except Exception:
                    pass
        if len(arrays) == 0:
            raise RuntimeError(f"No assets found for band {band}")
        # stack along time and compute median (note: memory heavy)
        stack = xr.concat(arrays, dim="time")
        median = stack.median(dim="time", skipna=True)
        band_arrays.append(median)

    # merge bands into a single xarray dataset
    ds = xr.concat(band_arrays, dim="band")
    ds = ds.assign_coords(band=bands)
    # Save as COG via rioxarray (if dataset has geospatial info)
    ds.rio.to_raster(out_tif)
    print(f"Saved composite to {out_tif}")
    return out_tif

# -------------------------
# 5. Sample spectral features per building
# -------------------------
def sample_raster_by_shapes(raster_path, gdf, epsg):
    # open raster and reproject buildings to raster CRS
    with rasterio.open(raster_path) as src:
        rast_crs = src.crs
    gdf_r = gdf.to_crs(rast_crs)
    samples = []
    with rasterio.open(raster_path) as src:
        bands = src.count
        for idx, row in gdf_r.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            # create mask (True for pixels inside shape)
            mask = geometry_mask([mapping(geom)], transform=src.transform, invert=True, out_shape=(src.height, src.width))
            if mask.sum() == 0:
                # no overlap with raster
                continue
            vals = []
            for b in range(1, src.count+1):
                data = src.read(b, masked=True)
                band_vals = data[mask]
                if band_vals.size == 0:
                    vals.append(np.nan)
                else:
                    vals.append(float(band_vals.mean()))
            samples.append({
                "orig_index": idx,
                **{f"band_{i+1}": vals[i] for i in range(len(vals))}
            })

    samp_df = pd.DataFrame(samples).set_index("orig_index")
    return samp_df

# -------------------------
# main
# -------------------------
def main():
    place = "Ile-Ife, Nigeria"
    print("Geocoding place...")
    bbox = get_bbox_for_place(place)
    print("BBox:", bbox)

    print("Fetching building footprints from Microsoft Planetary Computer (ms-buildings)...")
    gdf = fetch_ms_buildings(bbox, out_geojson="ileife_buildings.geojson")

    print("Computing geometric features...")
    gdf_features, epsg = compute_geometric_features(gdf)
    print("Feature example:\n", gdf_features.head())

    # Optionally produce the sentinel composite and sample spectral features
    print("Creating Sentinel-2 median composite (this can take time and memory)...")
    try:
        s2_file = get_sentinel2_median(bbox, time_range=("2021-01-01","2024-12-31"), bands=["B02","B03","B04","B08"], out_tif="ileife_s2_median.tif")
        print("Sampling spectral bands for each building...")
        samples = sample_raster_by_shapes(s2_file, gdf_features, epsg)
        # combine
        combined = gdf_features.join(samples, how="left")
        # compute NDVI approximation using B08 (band_4) and B04 (band_3) depending on your band ordering:
        # our get_sentinel2_median used bands ["B02","B03","B04","B08"] => band indices 1..4
        combined["ndvi"] = (combined["band_4"] - combined["band_3"]) / (combined["band_4"] + combined["band_3"])
        combined.to_file("ileife_buildings_with_features.geojson", driver="GeoJSON")
        print("Saved results to ileife_buildings_with_features.geojson")
    except Exception as e:
        print("Sentinel-2 processing skipped/failed:", e)
        # still save geom features
        gdf_features.to_file("ileife_building_geometry_features.geojson", driver="GeoJSON")
        print("Saved geometric features to ileife_building_geometry_features.geojson")

if __name__ == "__main__":
    main()
