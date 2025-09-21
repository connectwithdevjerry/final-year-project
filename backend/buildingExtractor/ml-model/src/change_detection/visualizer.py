#!/usr/bin/env python3
"""
Visual Change Detection System
Creates visual overlays showing detected changes on aerial images.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon, Rectangle, Circle
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass
from enum import Enum

# Import our change detection classes
from .change_detector import ChangeType, BuildingChange, ChangeDetectionResult

logger = logging.getLogger(__name__)


class Visualizer:
    """Creates visual representations of detected changes."""
    
    def __init__(self):
        self.colors = {
            ChangeType.BUILDING_ADDED: (0, 255, 0),      # Green
            ChangeType.BUILDING_REMOVED: (0, 0, 255),    # Red
            ChangeType.BUILDING_MODIFIED: (0, 255, 255), # Yellow
            ChangeType.BUILDING_UNCHANGED: (128, 128, 128) # Gray
        }
        
        self.color_names = {
            ChangeType.BUILDING_ADDED: "Added",
            ChangeType.BUILDING_REMOVED: "Removed", 
            ChangeType.BUILDING_MODIFIED: "Modified",
            ChangeType.BUILDING_UNCHANGED: "Unchanged"
        }
    
    def create_change_overlay(self, image: np.ndarray, changes: List[BuildingChange], 
                            alpha: float = 0.6) -> np.ndarray:
        """
        Create a visual overlay showing detected changes on the image.
        
        Args:
            image: Original image as numpy array
            changes: List of detected building changes
            alpha: Transparency of the overlay
            
        Returns:
            Image with change overlays
        """
        overlay = image.copy()
        
        for change in changes:
            color = self.colors[change.change_type]
            
            # Get polygon if available, otherwise use bounding box
            if hasattr(change, 'polygon') and change.polygon and len(change.polygon) >= 3:
                # Draw polygon outline for the building (matches your annotation style)
                polygon_points = np.array(change.polygon, dtype=np.int32)
                
                if change.change_type == ChangeType.BUILDING_ADDED:
                    # Green polygon outline for added buildings
                    cv2.polylines(overlay, [polygon_points], True, color, 3)
                    # Add semi-transparent fill
                    overlay_fill = overlay.copy()
                    cv2.fillPoly(overlay_fill, [polygon_points], color)
                    overlay = cv2.addWeighted(overlay, 1-alpha, overlay_fill, alpha, 0)
                    
                elif change.change_type == ChangeType.BUILDING_REMOVED:
                    # Red polygon outline for removed buildings
                    cv2.polylines(overlay, [polygon_points], True, color, 3)
                    # Add diagonal lines to show removal
                    center = np.mean(polygon_points, axis=0).astype(int)
                    for point in polygon_points:
                        cv2.line(overlay, tuple(center), tuple(point), color, 2)
                    
                elif change.change_type == ChangeType.BUILDING_MODIFIED:
                    # Yellow polygon outline for modified buildings
                    cv2.polylines(overlay, [polygon_points], True, color, 3)
                    # Add corner marks
                    for i, point in enumerate(polygon_points):
                        if i % 2 == 0:  # Mark every other corner
                            cv2.circle(overlay, tuple(point), 8, color, -1)
                
                # Add confidence text near the polygon
                bbox = cv2.boundingRect(polygon_points)
                x1, y1 = bbox[0], bbox[1]
                confidence_text = f"{change.confidence:.2f}"
                cv2.putText(overlay, confidence_text, (x1, max(y1-10, 15)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Add building type if available
                if hasattr(change, 'building_type') and change.building_type:
                    type_text = change.building_type.replace('_', ' ').title()
                    cv2.putText(overlay, type_text, (x1, y1-35), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
            elif hasattr(change, 'bbox') and change.bbox:
                # Fallback to bounding box if polygon not available
                x1, y1, x2, y2 = change.bbox
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                if change.change_type == ChangeType.BUILDING_ADDED:
                    # Green rectangle outline for added buildings
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 3)
                    # Add semi-transparent fill
                    overlay_fill = overlay.copy()
                    cv2.rectangle(overlay_fill, (x1, y1), (x2, y2), color, -1)
                    overlay = cv2.addWeighted(overlay, 1-alpha, overlay_fill, alpha, 0)
                    
                elif change.change_type == ChangeType.BUILDING_REMOVED:
                    # Red rectangle outline for removed buildings
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 3)
                    # Add diagonal lines to show removal
                    cv2.line(overlay, (x1, y1), (x2, y2), color, 2)
                    cv2.line(overlay, (x2, y1), (x1, y2), color, 2)
                    
                elif change.change_type == ChangeType.BUILDING_MODIFIED:
                    # Yellow rectangle outline for modified buildings
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 3)
                    # Add corner marks
                    corner_size = min(20, (x2-x1)//4, (y2-y1)//4)
                    cv2.rectangle(overlay, (x1, y1), (x1+corner_size, y1+corner_size), color, -1)
                    cv2.rectangle(overlay, (x2-corner_size, y1), (x2, y1+corner_size), color, -1)
                    cv2.rectangle(overlay, (x1, y2-corner_size), (x1+corner_size, y2), color, -1)
                    cv2.rectangle(overlay, (x2-corner_size, y2-corner_size), (x2, y2), color, -1)
                
                # Add confidence text at top-left of bounding box
                confidence_text = f"{change.confidence:.2f}"
                cv2.putText(overlay, confidence_text, (x1, max(y1-10, 15)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Add building type if available
                if hasattr(change, 'building_type') and change.building_type:
                    type_text = change.building_type.replace('_', ' ').title()
                    cv2.putText(overlay, type_text, (x1, y1-35), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
            elif change.pixel_coordinates:
                # Fallback to point-based visualization for older data
                x, y = change.pixel_coordinates[0]
                
                if change.change_type == ChangeType.BUILDING_ADDED:
                    # Green circle for added buildings
                    cv2.circle(overlay, (int(x), int(y)), 15, color, -1)
                    cv2.circle(overlay, (int(x), int(y)), 15, (255, 255, 255), 2)
                elif change.change_type == ChangeType.BUILDING_REMOVED:
                    # Red X for removed buildings
                    size = 20
                    cv2.line(overlay, (int(x-size), int(y-size)), (int(x+size), int(y+size)), color, 3)
                    cv2.line(overlay, (int(x+size), int(y-size)), (int(x-size), int(y+size)), color, 3)
                elif change.change_type == ChangeType.BUILDING_MODIFIED:
                    # Yellow rectangle for modified buildings
                    cv2.rectangle(overlay, (int(x-15), int(y-15)), (int(x+15), int(y+15)), color, -1)
                    cv2.rectangle(overlay, (int(x-15), int(y-15)), (int(x+15), int(y+15)), (255, 255, 255), 2)
                
                # Add confidence text
                confidence_text = f"{change.confidence:.2f}"
                cv2.putText(overlay, confidence_text, (int(x+20), int(y-10)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return overlay
    
    def create_comparison_view(self, image1: np.ndarray, image2: np.ndarray,
                             changes1: List[BuildingChange], changes2: List[BuildingChange],
                             title1: str = "Reference Image", title2: str = "Comparison Image") -> np.ndarray:
        """
        Create a side-by-side comparison view with change overlays.
        
        Args:
            image1: First image with overlays
            image2: Second image with overlays
            changes1: Changes for first image
            changes2: Changes for second image
            title1: Title for first image
            title2: Title for second image
            
        Returns:
            Combined comparison image
        """
        # Create overlays
        overlay1 = self.create_change_overlay(image1, changes1)
        overlay2 = self.create_change_overlay(image2, changes2)
        
        # Resize images to same height
        target_height = min(image1.shape[0], image2.shape[0])
        scale1 = target_height / image1.shape[0]
        scale2 = target_height / image2.shape[0]
        
        new_width1 = int(image1.shape[1] * scale1)
        new_width2 = int(image2.shape[1] * scale2)
        
        resized1 = cv2.resize(overlay1, (new_width1, target_height))
        resized2 = cv2.resize(overlay2, (new_width2, target_height))
        
        # Create combined image
        combined_width = new_width1 + new_width2 + 50  # 50px gap
        combined_height = target_height + 100  # Space for titles and legend
        
        combined = np.ones((combined_height, combined_width, 3), dtype=np.uint8) * 255
        
        # Place images
        combined[50:target_height+50, 0:new_width1] = resized1
        combined[50:target_height+50, new_width1+50:new_width1+new_width2+50] = resized2
        
        # Add titles
        cv2.putText(combined, title1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(combined, title2, (new_width1+60, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Add legend
        legend_y = target_height + 70
        legend_items = [
            (ChangeType.BUILDING_ADDED, "Added"),
            (ChangeType.BUILDING_REMOVED, "Removed"),
            (ChangeType.BUILDING_MODIFIED, "Modified")
        ]
        
        for i, (change_type, label) in enumerate(legend_items):
            color = self.colors[change_type]
            x_pos = 10 + i * 150
            
            # Draw color indicator
            cv2.circle(combined, (x_pos, legend_y), 8, color, -1)
            cv2.circle(combined, (x_pos, legend_y), 8, (0, 0, 0), 1)
            
            # Draw label
            cv2.putText(combined, label, (x_pos + 15, legend_y + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        return combined
    
    def create_detailed_change_map(self, image: np.ndarray, changes: List[BuildingChange],
                                 zoom_factor: float = 2.0) -> np.ndarray:
        """
        Create a detailed change map with zoomed-in view of changes.
        
        Args:
            image: Original image
            changes: List of detected changes
            zoom_factor: Zoom level for detailed view
            
        Returns:
            Detailed change map image
        """
        if not changes:
            return image
        
        # Create overlay
        overlay = self.create_change_overlay(image, changes)
        
        # If we have changes, create a detailed view around the first few changes
        detailed_views = []
        max_views = min(4, len(changes))  # Show up to 4 detailed views
        
        for i in range(max_views):
            change = changes[i]
            if change.pixel_coordinates:
                x, y = change.pixel_coordinates[0]
                
                # Extract region around change
                region_size = 100
                x1 = max(0, int(x - region_size))
                y1 = max(0, int(y - region_size))
                x2 = min(image.shape[1], int(x + region_size))
                y2 = min(image.shape[0], int(y + region_size))
                
                region = overlay[y1:y2, x1:x2]
                
                # Resize for better visibility
                region_resized = cv2.resize(region, (200, 200))
                
                # Add border and label
                region_with_border = cv2.copyMakeBorder(region_resized, 30, 10, 10, 10, 
                                                       cv2.BORDER_CONSTANT, value=(255, 255, 255))
                
                # Add change type label
                change_type_text = self.color_names[change.change_type]
                cv2.putText(region_with_border, change_type_text, (10, 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                
                # Add confidence
                conf_text = f"Conf: {change.confidence:.2f}"
                cv2.putText(region_with_border, conf_text, (10, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
                detailed_views.append(region_with_border)
        
        # Combine detailed views
        if detailed_views:
            # Calculate grid layout
            cols = 2
            rows = (len(detailed_views) + 1) // 2
            
            view_height = detailed_views[0].shape[0]
            view_width = detailed_views[0].shape[1]
            
            grid_height = rows * view_height
            grid_width = cols * view_width
            
            grid = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255
            
            for i, view in enumerate(detailed_views):
                row = i // cols
                col = i % cols
                
                y1 = row * view_height
                y2 = y1 + view_height
                x1 = col * view_width
                x2 = x1 + view_width
                
                grid[y1:y2, x1:x2] = view
            
            # Combine with main image
            main_resized = cv2.resize(overlay, (400, 300))
            main_with_border = cv2.copyMakeBorder(main_resized, 10, 10, 10, 10, 
                                                cv2.BORDER_CONSTANT, value=(255, 255, 255))
            
            # Add title to main image
            cv2.putText(main_with_border, "Full Image View", (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            # Combine main image and detailed views
            combined_height = max(main_with_border.shape[0], grid_height)
            combined_width = main_with_border.shape[1] + grid_width + 20
            
            combined = np.ones((combined_height, combined_width, 3), dtype=np.uint8) * 255
            combined[0:main_with_border.shape[0], 0:main_with_border.shape[1]] = main_with_border
            combined[0:grid_height, main_with_border.shape[1]+20:] = grid
            
            return combined
        
        return overlay
    
    def save_visualization(self, image: np.ndarray, output_path: str, 
                          title: str = "Change Detection Results"):
        """Save visualization image to file."""
        # Add title
        result_image = cv2.copyMakeBorder(image, 40, 10, 10, 10, 
                                        cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(result_image, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        cv2.imwrite(output_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
        logger.info(f"Visualization saved to {output_path}")


def create_change_visualization(result: ChangeDetectionResult, output_dir: str = "results"):
    """
    Create comprehensive visualizations for change detection results.
    
    Args:
        result: ChangeDetectionResult object
        output_dir: Directory to save visualization files
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Load original images
    import rasterio
    
    with rasterio.open(result.image1_path) as src1:
        image1 = src1.read()
        if image1.shape[0] == 1:
            image1 = np.repeat(image1, 3, axis=0)
        elif image1.shape[0] == 4:
            image1 = image1[:3]
        image1 = np.transpose(image1, (1, 2, 0))
        if image1.dtype == np.uint16:
            image1 = (image1 / 256).astype(np.uint8)
    
    with rasterio.open(result.image2_path) as src2:
        image2 = src2.read()
        if image2.shape[0] == 1:
            image2 = np.repeat(image2, 3, axis=0)
        elif image2.shape[0] == 4:
            image2 = image2[:3]
        image2 = np.transpose(image2, (1, 2, 0))
        if image2.dtype == np.uint16:
            image2 = (image2 / 256).astype(np.uint8)
    
    # Initialize visualizer
    visualizer = Visualizer()
    
    # Separate changes by type
    removed_changes = [c for c in result.changes if c.change_type == ChangeType.BUILDING_REMOVED]
    added_changes = [c for c in result.changes if c.change_type == ChangeType.BUILDING_ADDED]
    modified_changes = [c for c in result.changes if c.change_type == ChangeType.BUILDING_MODIFIED]
    
    # Create individual overlays
    overlay1 = visualizer.create_change_overlay(image1, removed_changes)
    overlay2 = visualizer.create_change_overlay(image2, added_changes)
    
    # Save individual overlays
    visualizer.save_visualization(overlay1, str(output_path / "image1_with_removed.png"), 
                                 "Reference Image - Buildings Removed")
    visualizer.save_visualization(overlay2, str(output_path / "image2_with_added.png"), 
                                 "Comparison Image - Buildings Added")
    
    # Create comparison view
    comparison = visualizer.create_comparison_view(
        image1, image2, removed_changes, added_changes,
        "Reference Image", "Comparison Image"
    )
    visualizer.save_visualization(comparison, str(output_path / "comparison_view.png"), 
                                 "Side-by-Side Comparison")
    
    # Create detailed change map
    all_changes = result.changes[:20]  # Limit to first 20 changes for clarity
    if all_changes:
        detailed_map = visualizer.create_detailed_change_map(image2, all_changes)
        visualizer.save_visualization(detailed_map, str(output_path / "detailed_changes.png"), 
                                     "Detailed Change Analysis")
    
    # Create summary visualization
    create_summary_visualization(result, str(output_path / "summary.png"))
    
    logger.info(f"All visualizations saved to {output_path}")


def create_summary_visualization(result: ChangeDetectionResult, output_path: str):
    """Create a summary visualization with statistics."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Change type distribution
    change_types = ['Added', 'Removed', 'Modified', 'Total']
    counts = [result.buildings_added, result.buildings_removed, 
              result.buildings_modified, result.total_changes]
    colors = ['green', 'red', 'orange', 'blue']
    
    ax1.bar(change_types, counts, color=colors)
    ax1.set_title('Change Type Distribution')
    ax1.set_ylabel('Number of Changes')
    
    # Confidence distribution
    confidences = [c.confidence for c in result.changes]
    if confidences:
        ax2.hist(confidences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.set_title('Confidence Score Distribution')
        ax2.set_xlabel('Confidence')
        ax2.set_ylabel('Frequency')
    
    # Building type distribution
    building_types = {}
    for change in result.changes:
        btype = change.building_type
        building_types[btype] = building_types.get(btype, 0) + 1
    
    if building_types:
        ax3.pie(building_types.values(), labels=building_types.keys(), autopct='%1.1f%%')
        ax3.set_title('Building Type Distribution')
    
    # Processing statistics
    stats_text = f"""
    Processing Time: {result.processing_time:.2f}s
    Coordinate System: {result.coordinate_system}
    Pixel Size: {result.pixel_size[0]:.3f}m x {result.pixel_size[1]:.3f}m
    Total Changes: {result.total_changes}
    """
    ax4.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center')
    ax4.set_title('Processing Statistics')
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Summary visualization saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    import json
    
    # Load a result file
    with open("test_change_detection_results.json", "r") as f:
        result_data = json.load(f)
    
    # Create visualizations
    create_change_visualization(result_data, "visualizations")

