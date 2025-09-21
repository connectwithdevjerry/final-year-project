"""
Interactive annotation tool for aerial imagery with coordinate preservation.
"""

import os
import sys
import cv2
import numpy as np
import json
from typing import List, Dict, Tuple, Optional
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.geo_utils import GeospatialHandler


class AnnotationTool:
    """Interactive annotation tool for aerial imagery."""
    
    def __init__(self, image_path: str, output_dir: str = "data/annotations"):
        """Initialize the annotation tool."""
        self.image_path = image_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load image and geospatial handler
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        self.geo_handler = GeospatialHandler(image_path)
        self.display_image = self.image.copy()
        
        # Annotation state
        self.annotations = []
        self.current_annotation = None
        self.drawing = False
        self.start_point = None
        self.end_point = None
        
        # Building types
        self.building_types = [
            "residential_single",
            "residential_multi", 
            "commercial",
            "industrial",
            "institutional",
            "other"
        ]
        self.current_building_type = 0
        
        # Display settings
        self.window_name = "Aerial Imagery Annotation Tool"
        self.scale_factor = 1.0
        self.pan_x, self.pan_y = 0, 0
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for annotation."""
        # Adjust coordinates for pan and scale
        x = int((x - self.pan_x) / self.scale_factor)
        y = int((y - self.pan_y) / self.scale_factor)
        
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.current_annotation = {
                'bbox': [x, y, x, y],
                'class': self.building_types[self.current_building_type],
                'confidence': 1.0
            }
            
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.end_point = (x, y)
            if self.current_annotation:
                self.current_annotation['bbox'] = [
                    min(self.start_point[0], x),
                    min(self.start_point[1], y),
                    max(self.start_point[0], x),
                    max(self.start_point[1], y)
                ]
            self.update_display()
            
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing and self.current_annotation:
                # Only save if box is large enough
                bbox = self.current_annotation['bbox']
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                
                if width > 10 and height > 10:  # Minimum size threshold
                    self.annotations.append(self.current_annotation.copy())
                    print(f"Added annotation: {self.current_annotation}")
                
                self.drawing = False
                self.current_annotation = None
                self.start_point = None
                self.end_point = None
                self.update_display()
    
    def key_callback(self, key):
        """Handle keyboard input."""
        if key == ord('n') or key == ord('N'):
            # Next building type
            self.current_building_type = (self.current_building_type + 1) % len(self.building_types)
            print(f"Building type: {self.building_types[self.current_building_type]}")
            
        elif key == ord('p') or key == ord('P'):
            # Previous building type
            self.current_building_type = (self.current_building_type - 1) % len(self.building_types)
            print(f"Building type: {self.building_types[self.current_building_type]}")
            
        elif key == ord('s') or key == ord('S'):
            # Save annotations
            self.save_annotations()
            
        elif key == ord('l') or key == ord('L'):
            # Load annotations
            self.load_annotations()
            
        elif key == ord('d') or key == ord('D'):
            # Delete last annotation
            if self.annotations:
                removed = self.annotations.pop()
                print(f"Removed annotation: {removed}")
                self.update_display()
                
        elif key == ord('c') or key == ord('C'):
            # Clear all annotations
            self.annotations = []
            print("Cleared all annotations")
            self.update_display()
            
        elif key == ord('h') or key == ord('H'):
            # Show help
            self.show_help()
            
        elif key == 27:  # ESC
            return False
            
        return True
    
    def update_display(self):
        """Update the display with current annotations."""
        self.display_image = self.image.copy()
        
        # Draw existing annotations
        for i, annotation in enumerate(self.annotations):
            bbox = annotation['bbox']
            color = self.get_color_for_class(annotation['class'])
            
            cv2.rectangle(self.display_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # Add label
            label = f"{i+1}: {annotation['class']}"
            cv2.putText(self.display_image, label, (bbox[0], bbox[1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw current annotation being created
        if self.current_annotation:
            bbox = self.current_annotation['bbox']
            color = self.get_color_for_class(self.current_annotation['class'])
            cv2.rectangle(self.display_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        
        # Add info overlay
        self.add_info_overlay()
        
        # Apply scale and pan
        if self.scale_factor != 1.0:
            h, w = self.display_image.shape[:2]
            new_h, new_w = int(h * self.scale_factor), int(w * self.scale_factor)
            self.display_image = cv2.resize(self.display_image, (new_w, new_h))
        
        cv2.imshow(self.window_name, self.display_image)
    
    def get_color_for_class(self, class_name: str) -> Tuple[int, int, int]:
        """Get color for building class."""
        colors = {
            "residential_single": (0, 255, 0),    # Green
            "residential_multi": (0, 200, 0),     # Dark Green
            "commercial": (255, 0, 0),            # Blue
            "industrial": (0, 0, 255),            # Red
            "institutional": (255, 255, 0),       # Cyan
            "other": (128, 128, 128)              # Gray
        }
        return colors.get(class_name, (255, 255, 255))
    
    def add_info_overlay(self):
        """Add information overlay to the display."""
        info_text = [
            f"Building Type: {self.building_types[self.current_building_type]}",
            f"Annotations: {len(self.annotations)}",
            f"Scale: {self.scale_factor:.2f}",
            "",
            "Controls:",
            "N/P - Next/Previous building type",
            "S - Save annotations",
            "L - Load annotations", 
            "D - Delete last annotation",
            "C - Clear all annotations",
            "H - Show help",
            "ESC - Exit"
        ]
        
        y_offset = 30
        for text in info_text:
            cv2.putText(self.display_image, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(self.display_image, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            y_offset += 20
    
    def show_help(self):
        """Show detailed help information."""
        help_text = """
AERIAL IMAGERY ANNOTATION TOOL HELP

BUILDING TYPES:
- Residential Single: Single-family homes
- Residential Multi: Multi-family buildings, apartments
- Commercial: Stores, offices, shopping centers
- Industrial: Factories, warehouses, manufacturing
- Institutional: Schools, hospitals, government buildings
- Other: Any other building type

MOUSE CONTROLS:
- Left Click + Drag: Draw bounding box around building
- Release: Complete annotation

KEYBOARD CONTROLS:
- N/P: Cycle through building types
- S: Save annotations to file
- L: Load annotations from file
- D: Delete last annotation
- C: Clear all annotations
- H: Show this help
- ESC: Exit tool

ANNOTATION FILE FORMAT:
Annotations are saved with both pixel and geographic coordinates
for maximum compatibility with GIS systems and research tools.

RESOLUTION INFORMATION:
This tool preserves the original image resolution and coordinate
system for accurate spatial analysis.
        """
        print(help_text)
    
    def save_annotations(self):
        """Save annotations with coordinate information."""
        if not self.annotations:
            print("No annotations to save")
            return
        
        # Create output filename
        image_name = Path(self.image_path).stem
        output_path = self.output_dir / f"{image_name}_annotations.json"
        
        # Save with coordinate preservation
        self.geo_handler.export_annotation_with_coords(self.annotations, str(output_path))
        
        # Also create GeoJSON
        geojson_path = self.output_dir / f"{image_name}_annotations.geojson"
        self.geo_handler.create_geojson(self.annotations, str(geojson_path))
        
        print(f"Saved {len(self.annotations)} annotations to:")
        print(f"  - {output_path}")
        print(f"  - {geojson_path}")
        
        # Print coordinate information
        resolution_info = self.geo_handler.get_resolution_info()
        print(f"\nImage Resolution Information:")
        print(f"  - Pixel size: {resolution_info['pixel_size_x_meters']:.2f}m x {resolution_info['pixel_size_y_meters']:.2f}m")
        print(f"  - Ground Sample Distance: {resolution_info['ground_sample_distance']:.2f}m")
        print(f"  - Coordinate System: {self.geo_handler.metadata['crs']}")
    
    def load_annotations(self):
        """Load annotations from file."""
        image_name = Path(self.image_path).stem
        annotation_file = self.output_dir / f"{image_name}_annotations.json"
        
        if annotation_file.exists():
            with open(annotation_file, 'r') as f:
                data = json.load(f)
                self.annotations = data.get('annotations', [])
                print(f"Loaded {len(self.annotations)} annotations")
                self.update_display()
        else:
            print(f"No annotation file found: {annotation_file}")
    
    def run(self):
        """Run the annotation tool."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        # Try to load existing annotations
        self.load_annotations()
        
        # Initial display
        self.update_display()
        self.show_help()
        
        print(f"\nStarting annotation for: {self.image_path}")
        print("Press 'H' for help, 'ESC' to exit")
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key != 255:  # Key was pressed
                if not self.key_callback(key):
                    break
        
        cv2.destroyAllWindows()
        
        # Auto-save on exit
        if self.annotations:
            save = input("Save annotations before exiting? (y/n): ").lower().strip()
            if save in ['y', 'yes']:
                self.save_annotations()


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description="Aerial Imagery Annotation Tool")
    parser.add_argument("image_path", help="Path to TIF image file")
    parser.add_argument("--output_dir", default="data/annotations", 
                       help="Output directory for annotations")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image_path):
        print(f"Error: Image file not found: {args.image_path}")
        return
    
    try:
        tool = AnnotationTool(args.image_path, args.output_dir)
        tool.run()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
