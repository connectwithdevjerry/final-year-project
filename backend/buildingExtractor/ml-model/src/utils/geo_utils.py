"""
Geospatial utilities for coordinate transformation and metadata handling.
"""

import rasterio
import numpy as np
from typing import Tuple, Dict, Any, Optional
from rasterio.transform import from_bounds
from rasterio.crs import CRS
import json


class GeospatialHandler:
    """Handle geospatial operations for aerial imagery."""
    
    def __init__(self, tif_path: str):
        """Initialize with a TIF file path."""
        self.tif_path = tif_path
        self.metadata = self._load_metadata()
        
    def _load_metadata(self) -> Dict[str, Any]:
        """Load geospatial metadata from TIF file."""
        with rasterio.open(self.tif_path) as src:
            return {
                'crs': src.crs.to_string(),
                'transform': src.transform,
                'bounds': src.bounds,
                'width': src.width,
                'height': src.height,
                'nodata': src.nodata,
                'dtype': str(src.dtypes[0])
            }
    
    def pixel_to_coords(self, pixel_x: int, pixel_y: int) -> Tuple[float, float]:
        """Convert pixel coordinates to geographic coordinates."""
        row, col = pixel_y, pixel_x
        lon, lat = rasterio.transform.xy(self.metadata['transform'], row, col)
        return lon, lat
    
    def coords_to_pixel(self, lon: float, lat: float) -> Tuple[int, int]:
        """Convert geographic coordinates to pixel coordinates."""
        row, col = rasterio.transform.rowcol(self.metadata['transform'], lon, lat)
        return col, row
    
    def get_bounds_in_crs(self, target_crs: str = 'EPSG:4326') -> Tuple[float, float, float, float]:
        """Get image bounds in specified coordinate reference system."""
        if self.metadata['crs'] == target_crs:
            return self.metadata['bounds']
        
        # Transform bounds to target CRS
        from pyproj import Transformer
        transformer = Transformer.from_crs(self.metadata['crs'], target_crs, always_xy=True)
        
        bounds = self.metadata['bounds']
        left, bottom = transformer.transform(bounds.left, bounds.bottom)
        right, top = transformer.transform(bounds.right, bounds.top)
        
        return left, bottom, right, top
    
    def calculate_pixel_size(self) -> Tuple[float, float]:
        """Calculate pixel size in meters."""
        transform = self.metadata['transform']
        return abs(transform[0]), abs(transform[4])
    
    def get_resolution_info(self) -> Dict[str, float]:
        """Get detailed resolution information."""
        pixel_x, pixel_y = self.calculate_pixel_size()
        
        # Convert to different units if possible
        info = {
            'pixel_size_x_meters': pixel_x,
            'pixel_size_y_meters': pixel_y,
            'pixel_size_x_cm': pixel_x * 100,
            'pixel_size_y_cm': pixel_y * 100,
            'ground_sample_distance': max(pixel_x, pixel_y)
        }
        
        return info
    
    def export_annotation_with_coords(self, annotations: list, output_path: str):
        """Export annotations with preserved coordinate information."""
        annotation_data = {
            'image_path': self.tif_path,
            'crs': self.metadata['crs'],
            'transform': list(self.metadata['transform']),
            'bounds': list(self.metadata['bounds']),
            'resolution_info': self.get_resolution_info(),
            'annotations': []
        }
        
        for annotation in annotations:
            # Convert pixel coordinates to geographic coordinates
            pixel_coords = annotation.get('bbox', [])
            if len(pixel_coords) == 4:
                x1, y1, x2, y2 = pixel_coords
                
                # Get corner coordinates
                lon1, lat1 = self.pixel_to_coords(x1, y1)
                lon2, lat2 = self.pixel_to_coords(x2, y2)
                
                annotation_with_coords = {
                    **annotation,
                    'pixel_coordinates': {
                        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2
                    },
                    'geographic_coordinates': {
                        'lon1': lon1, 'lat1': lat1,
                        'lon2': lon2, 'lat2': lat2,
                        'center_lon': (lon1 + lon2) / 2,
                        'center_lat': (lat1 + lat2) / 2
                    }
                }
                
                annotation_data['annotations'].append(annotation_with_coords)
        
        with open(output_path, 'w') as f:
            json.dump(annotation_data, f, indent=2)
    
    def create_geojson(self, annotations: list, output_path: str):
        """Create GeoJSON file from annotations."""
        geojson = {
            "type": "FeatureCollection",
            "crs": {
                "type": "name",
                "properties": {"name": self.metadata['crs']}
            },
            "features": []
        }
        
        for i, annotation in enumerate(annotations):
            pixel_coords = annotation.get('bbox', [])
            if len(pixel_coords) == 4:
                x1, y1, x2, y2 = pixel_coords
                
                # Get corner coordinates
                lon1, lat1 = self.pixel_to_coords(x1, y1)
                lon2, lat2 = self.pixel_to_coords(x2, y2)
                lon3, lat3 = self.pixel_to_coords(x2, y1)
                lon4, lat4 = self.pixel_to_coords(x1, y2)
                
                feature = {
                    "type": "Feature",
                    "properties": {
                        "id": i,
                        "building_type": annotation.get('class', 'unknown'),
                        "confidence": annotation.get('confidence', 1.0),
                        "area_sqm": annotation.get('area_sqm', 0)
                    },
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [lon1, lat1],
                            [lon3, lat3],
                            [lon2, lat2],
                            [lon4, lat4],
                            [lon1, lat1]
                        ]]
                    }
                }
                
                geojson["features"].append(feature)
        
        with open(output_path, 'w') as f:
            json.dump(geojson, f, indent=2)


def validate_coordinate_system(crs_string: str) -> bool:
    """Validate if coordinate system string is valid."""
    try:
        CRS.from_string(crs_string)
        return True
    except:
        return False


def get_common_crs_info() -> Dict[str, Dict[str, Any]]:
    """Get information about common coordinate reference systems."""
    return {
        'EPSG:4326': {
            'name': 'WGS84',
            'description': 'World Geodetic System 1984',
            'units': 'degrees',
            'use_case': 'Global, web mapping'
        },
        'EPSG:3857': {
            'name': 'Web Mercator',
            'description': 'Web Mercator Projection',
            'units': 'meters',
            'use_case': 'Web mapping, Google Maps'
        },
        'EPSG:32633': {
            'name': 'WGS 84 / UTM zone 33N',
            'description': 'Universal Transverse Mercator Zone 33N',
            'units': 'meters',
            'use_case': 'Europe, Africa'
        }
    }
