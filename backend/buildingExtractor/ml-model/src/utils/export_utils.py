"""
Export utilities for different annotation and prediction formats.
"""

import json
import csv
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from datetime import datetime


class AnnotationExporter:
    """Export annotations and predictions in various formats."""
    
    def __init__(self, output_dir: str = "data/exports"):
        """Initialize exporter."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export_to_csv(self, 
                     annotations: List[Dict], 
                     filename: str,
                     include_coordinates: bool = True) -> str:
        """
        Export annotations to CSV format.
        
        Args:
            annotations: List of annotation dictionaries
            filename: Output filename
            include_coordinates: Whether to include coordinate information
            
        Returns:
            Path to exported CSV file
        """
        output_path = self.output_dir / f"{filename}.csv"
        
        rows = []
        for i, ann in enumerate(annotations):
            row = {
                'id': i,
                'class': ann.get('class', 'unknown'),
                'confidence': ann.get('confidence', 1.0)
            }
            
            if include_coordinates:
                pixel_coords = ann.get('pixel_coordinates', {})
                geo_coords = ann.get('geographic_coordinates', {})
                
                row.update({
                    'pixel_x1': pixel_coords.get('x1', 0),
                    'pixel_y1': pixel_coords.get('y1', 0),
                    'pixel_x2': pixel_coords.get('x2', 0),
                    'pixel_y2': pixel_coords.get('y2', 0),
                    'geo_lon1': geo_coords.get('lon1', 0),
                    'geo_lat1': geo_coords.get('lat1', 0),
                    'geo_lon2': geo_coords.get('lon2', 0),
                    'geo_lat2': geo_coords.get('lat2', 0),
                    'center_lon': geo_coords.get('center_lon', 0),
                    'center_lat': geo_coords.get('center_lat', 0)
                })
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        
        return str(output_path)
    
    def export_to_excel(self, 
                       annotations: List[Dict], 
                       filename: str,
                       include_coordinates: bool = True) -> str:
        """
        Export annotations to Excel format.
        
        Args:
            annotations: List of annotation dictionaries
            filename: Output filename
            include_coordinates: Whether to include coordinate information
            
        Returns:
            Path to exported Excel file
        """
        output_path = self.output_dir / f"{filename}.xlsx"
        
        # Create DataFrame
        rows = []
        for i, ann in enumerate(annotations):
            row = {
                'id': i,
                'class': ann.get('class', 'unknown'),
                'confidence': ann.get('confidence', 1.0)
            }
            
            if include_coordinates:
                pixel_coords = ann.get('pixel_coordinates', {})
                geo_coords = ann.get('geographic_coordinates', {})
                
                row.update({
                    'pixel_x1': pixel_coords.get('x1', 0),
                    'pixel_y1': pixel_coords.get('y1', 0),
                    'pixel_x2': pixel_coords.get('x2', 0),
                    'pixel_y2': pixel_coords.get('y2', 0),
                    'geo_lon1': geo_coords.get('lon1', 0),
                    'geo_lat1': geo_coords.get('lat1', 0),
                    'geo_lon2': geo_coords.get('lon2', 0),
                    'geo_lat2': geo_coords.get('lat2', 0),
                    'center_lon': geo_coords.get('center_lon', 0),
                    'center_lat': geo_coords.get('center_lat', 0)
                })
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Create Excel file with multiple sheets
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Main data sheet
            df.to_excel(writer, sheet_name='Annotations', index=False)
            
            # Summary sheet
            summary_data = self._create_summary_data(annotations)
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Statistics sheet
            stats_data = self._create_statistics_data(annotations)
            stats_df = pd.DataFrame(stats_data)
            stats_df.to_excel(writer, sheet_name='Statistics', index=False)
        
        return str(output_path)
    
    def export_to_shapefile(self, 
                           annotations: List[Dict], 
                           filename: str,
                           crs: str = "EPSG:4326") -> str:
        """
        Export annotations to Shapefile format.
        
        Args:
            annotations: List of annotation dictionaries
            filename: Output filename
            crs: Coordinate reference system
            
        Returns:
            Path to exported Shapefile directory
        """
        try:
            import geopandas as gpd
            from shapely.geometry import Polygon
        except ImportError:
            raise ImportError("geopandas and shapely are required for Shapefile export")
        
        output_dir = self.output_dir / filename
        output_dir.mkdir(exist_ok=True)
        
        geometries = []
        properties = []
        
        for i, ann in enumerate(annotations):
            geo_coords = ann.get('geographic_coordinates', {})
            
            if geo_coords:
                # Create polygon from coordinates
                lon1 = geo_coords.get('lon1', 0)
                lat1 = geo_coords.get('lat1', 0)
                lon2 = geo_coords.get('lon2', 0)
                lat2 = geo_coords.get('lat2', 0)
                
                polygon = Polygon([
                    (lon1, lat1),
                    (lon2, lat1),
                    (lon2, lat2),
                    (lon1, lat2),
                    (lon1, lat1)
                ])
                
                geometries.append(polygon)
                
                properties.append({
                    'id': i,
                    'class': ann.get('class', 'unknown'),
                    'confidence': ann.get('confidence', 1.0),
                    'area_sqm': self._calculate_area(polygon, crs)
                })
        
        if geometries:
            gdf = gpd.GeoDataFrame(properties, geometry=geometries, crs=crs)
            gdf.to_file(output_dir / f"{filename}.shp")
        
        return str(output_dir)
    
    def export_to_kml(self, 
                     annotations: List[Dict], 
                     filename: str) -> str:
        """
        Export annotations to KML format for Google Earth.
        
        Args:
            annotations: List of annotation dictionaries
            filename: Output filename
            
        Returns:
            Path to exported KML file
        """
        output_path = self.output_dir / f"{filename}.kml"
        
        kml_content = '''<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
    <name>Building Annotations</name>
    <description>Building annotations from aerial imagery</description>
'''
        
        for i, ann in enumerate(annotations):
            geo_coords = ann.get('geographic_coordinates', {})
            class_name = ann.get('class', 'unknown')
            confidence = ann.get('confidence', 1.0)
            
            if geo_coords:
                lon1 = geo_coords.get('lon1', 0)
                lat1 = geo_coords.get('lat1', 0)
                lon2 = geo_coords.get('lon2', 0)
                lat2 = geo_coords.get('lat2', 0)
                
                kml_content += f'''
    <Placemark>
        <name>Building {i+1}: {class_name}</name>
        <description>Confidence: {confidence:.2f}</description>
        <Polygon>
            <outerBoundaryIs>
                <LinearRing>
                    <coordinates>
                        {lon1},{lat1},0
                        {lon2},{lat1},0
                        {lon2},{lat2},0
                        {lon1},{lat2},0
                        {lon1},{lat1},0
                    </coordinates>
                </LinearRing>
            </outerBoundaryIs>
        </Polygon>
    </Placemark>
'''
        
        kml_content += '''
</Document>
</kml>'''
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(kml_content)
        
        return str(output_path)
    
    def export_to_yolo_format(self, 
                             annotations: List[Dict], 
                             filename: str,
                             image_width: int = 1000,
                             image_height: int = 1000) -> str:
        """
        Export annotations to YOLO format.
        
        Args:
            annotations: List of annotation dictionaries
            filename: Output filename
            image_width: Image width for normalization
            image_height: Image height for normalization
            
        Returns:
            Path to exported YOLO file
        """
        output_path = self.output_dir / f"{filename}.txt"
        
        class_mapping = {
            "residential_single": 0,
            "residential_multi": 1,
            "commercial": 2,
            "industrial": 3,
            "institutional": 4,
            "other": 5
        }
        
        with open(output_path, 'w') as f:
            for ann in annotations:
                pixel_coords = ann.get('pixel_coordinates', {})
                class_name = ann.get('class', 'other')
                class_id = class_mapping.get(class_name, 5)
                
                x1 = pixel_coords.get('x1', 0)
                y1 = pixel_coords.get('y1', 0)
                x2 = pixel_coords.get('x2', 0)
                y2 = pixel_coords.get('y2', 0)
                
                # Convert to YOLO format (normalized center coordinates and dimensions)
                x_center = (x1 + x2) / 2 / image_width
                y_center = (y1 + y2) / 2 / image_height
                width = (x2 - x1) / image_width
                height = (y2 - y1) / image_height
                
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        return str(output_path)
    
    def _create_summary_data(self, annotations: List[Dict]) -> List[Dict]:
        """Create summary data for Excel export."""
        summary = []
        
        # Count by class
        class_counts = {}
        for ann in annotations:
            class_name = ann.get('class', 'unknown')
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        for class_name, count in class_counts.items():
            summary.append({
                'Building Type': class_name,
                'Count': count,
                'Percentage': (count / len(annotations)) * 100 if annotations else 0
            })
        
        return summary
    
    def _create_statistics_data(self, annotations: List[Dict]) -> List[Dict]:
        """Create statistics data for Excel export."""
        stats = []
        
        if not annotations:
            return stats
        
        # Overall statistics
        stats.append({
            'Metric': 'Total Buildings',
            'Value': len(annotations)
        })
        
        # Confidence statistics
        confidences = [ann.get('confidence', 1.0) for ann in annotations]
        stats.extend([
            {'Metric': 'Average Confidence', 'Value': sum(confidences) / len(confidences)},
            {'Metric': 'Min Confidence', 'Value': min(confidences)},
            {'Metric': 'Max Confidence', 'Value': max(confidences)}
        ])
        
        # Area statistics (if available)
        areas = []
        for ann in annotations:
            if 'area_sqm' in ann:
                areas.append(ann['area_sqm'])
        
        if areas:
            stats.extend([
                {'Metric': 'Total Area (sqm)', 'Value': sum(areas)},
                {'Metric': 'Average Area (sqm)', 'Value': sum(areas) / len(areas)},
                {'Metric': 'Min Area (sqm)', 'Value': min(areas)},
                {'Metric': 'Max Area (sqm)', 'Value': max(areas)}
            ])
        
        return stats
    
    def _calculate_area(self, polygon, crs: str) -> float:
        """Calculate area of polygon in square meters."""
        try:
            import geopandas as gpd
            # Convert to UTM for accurate area calculation
            gdf = gpd.GeoDataFrame([1], geometry=[polygon], crs=crs)
            utm_crs = gdf.estimate_utm_crs()
            gdf_utm = gdf.to_crs(utm_crs)
            return gdf_utm.geometry.iloc[0].area
        except:
            return 0.0


class BatchExporter:
    """Export multiple annotation files in batch."""
    
    def __init__(self, input_dir: str, output_dir: str = "data/exports"):
        """Initialize batch exporter."""
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.exporter = AnnotationExporter(output_dir)
    
    def export_all_formats(self, 
                          formats: List[str] = None,
                          include_coordinates: bool = True) -> Dict[str, List[str]]:
        """
        Export all annotation files in multiple formats.
        
        Args:
            formats: List of formats to export ['csv', 'excel', 'geojson', 'kml', 'yolo']
            include_coordinates: Whether to include coordinate information
            
        Returns:
            Dictionary mapping format to list of exported files
        """
        if formats is None:
            formats = ['csv', 'excel', 'geojson', 'kml', 'yolo']
        
        results = {format: [] for format in formats}
        
        # Find all annotation files
        annotation_files = list(self.input_dir.glob("*_annotations.json"))
        
        for ann_file in annotation_files:
            filename = ann_file.stem.replace('_annotations', '')
            
            # Load annotations
            with open(ann_file, 'r') as f:
                data = json.load(f)
                annotations = data.get('annotations', [])
            
            if not annotations:
                continue
            
            # Export in each format
            for format_type in formats:
                try:
                    if format_type == 'csv':
                        output_path = self.exporter.export_to_csv(
                            annotations, filename, include_coordinates)
                    elif format_type == 'excel':
                        output_path = self.exporter.export_to_excel(
                            annotations, filename, include_coordinates)
                    elif format_type == 'geojson':
                        # Use existing GeoJSON export from geo_utils
                        from utils.geo_utils import GeospatialHandler
                        geo_handler = GeospatialHandler(data.get('image_path', ''))
                        output_path = f"{self.output_dir}/{filename}_annotations.geojson"
                        geo_handler.create_geojson(annotations, output_path)
                    elif format_type == 'kml':
                        output_path = self.exporter.export_to_kml(annotations, filename)
                    elif format_type == 'yolo':
                        output_path = self.exporter.export_to_yolo_format(annotations, filename)
                    
                    results[format_type].append(output_path)
                    
                except Exception as e:
                    print(f"Error exporting {filename} to {format_type}: {e}")
        
        return results
    
    def create_summary_report(self) -> str:
        """Create a summary report of all annotations."""
        output_path = self.output_dir / "summary_report.csv"
        
        all_annotations = []
        
        # Load all annotation files
        annotation_files = list(self.input_dir.glob("*_annotations.json"))
        
        for ann_file in annotation_files:
            with open(ann_file, 'r') as f:
                data = json.load(f)
                annotations = data.get('annotations', [])
                
                for ann in annotations:
                    ann['source_file'] = ann_file.name
                    all_annotations.append(ann)
        
        if all_annotations:
            # Create summary DataFrame
            rows = []
            for ann in all_annotations:
                row = {
                    'source_file': ann.get('source_file', ''),
                    'class': ann.get('class', 'unknown'),
                    'confidence': ann.get('confidence', 1.0)
                }
                
                geo_coords = ann.get('geographic_coordinates', {})
                if geo_coords:
                    row.update({
                        'center_lon': geo_coords.get('center_lon', 0),
                        'center_lat': geo_coords.get('center_lat', 0)
                    })
                
                rows.append(row)
            
            df = pd.DataFrame(rows)
            df.to_csv(output_path, index=False)
            
            print(f"Summary report created: {output_path}")
            print(f"Total annotations: {len(all_annotations)}")
            print(f"Unique building types: {df['class'].nunique()}")
            print(f"Files processed: {len(annotation_files)}")
        
        return str(output_path)
