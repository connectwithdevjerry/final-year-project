#!/usr/bin/env python3
"""
Smart Change Detection System for Aerial Imagery
Compares two aerial images and detects building changes while preserving coordinates.
"""

import numpy as np
import cv2
import rasterio
import torch
import torch.nn as nn
from pathlib import Path
import json
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChangeType(Enum):
    """Types of changes that can be detected."""
    BUILDING_ADDED = "building_added"
    BUILDING_REMOVED = "building_removed"
    BUILDING_MODIFIED = "building_modified"
    BUILDING_UNCHANGED = "building_unchanged"


@dataclass
class BuildingChange:
    """Represents a detected building change."""
    change_type: ChangeType
    coordinates: List[Tuple[float, float]]  # Geographic coordinates
    pixel_coordinates: List[Tuple[int, int]]  # Pixel coordinates
    confidence: float
    building_type: str
    area_pixels: int
    area_meters: Optional[float] = None
    metadata: Dict = None


@dataclass
class ChangeDetectionResult:
    """Result of change detection analysis."""
    image1_path: str
    image2_path: str
    total_changes: int
    buildings_added: int
    buildings_removed: int
    buildings_modified: int
    changes: List[BuildingChange]
    processing_time: float
    coordinate_system: str
    pixel_size: Tuple[float, float]  # meters per pixel


class ImageAligner:
    """Handles alignment and registration of two aerial images."""
    
    def __init__(self):
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    def align_images(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Align two images using feature matching.
        Returns aligned images and transformation matrix.
        """
        logger.info("Aligning images using ORB feature matching...")
        
        # Convert to grayscale for feature detection
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        
        # Detect keypoints and descriptors
        kp1, des1 = self.orb.detectAndCompute(gray1, None)
        kp2, des2 = self.orb.detectAndCompute(gray2, None)
        
        if des1 is None or des2 is None:
            logger.warning("Could not detect enough features for alignment")
            return img1, img2, np.eye(3)
        
        # Match features
        matches = self.matcher.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Keep only good matches
        good_matches = matches[:min(100, len(matches))]
        
        if len(good_matches) < 10:
            logger.warning("Not enough good matches for alignment")
            return img1, img2, np.eye(3)
        
        # Extract matched points
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Find homography
        homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if homography is None:
            logger.warning("Could not compute homography")
            return img1, img2, np.eye(3)
        
        # Apply transformation to align img1 to img2
        h, w = img2.shape[:2]
        aligned_img1 = cv2.warpPerspective(img1, homography, (w, h))
        
        logger.info(f"Image alignment completed with {len(good_matches)} matches")
        return aligned_img1, img2, homography


class BuildingDetector:
    """Detects buildings in aerial images using the trained model."""
    
    def __init__(self, model_path: str = "data/models/building_classifier.pth"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        self.class_names = ["residential_single", "residential_multi", "commercial", 
                           "industrial", "institutional", "other"]
        
    def _load_model(self, model_path: str):
        """Load the trained building detection model."""
        from robust_train import SimpleBuildingClassifier
        
        model = SimpleBuildingClassifier(num_classes=6)
        if Path(model_path).exists():
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.eval()
            logger.info(f"Model loaded from {model_path}")
        else:
            logger.warning(f"Model not found at {model_path}, using untrained model")
        
        return model.to(self.device)
    
    def detect_buildings(self, image: np.ndarray, geotransform: Tuple,
                        confidence_threshold: float = 0.5, image_path: str = None) -> List[Dict]:
        """
        Detect buildings by using your existing annotations and validating with your trained model.
        This approach uses the annotated building locations and validates them with your model.
        """
        logger.info("Detecting buildings in image...")
        
        buildings = []
        
        # Step 1: Try to load existing annotations for this image
        existing_annotations = self._load_existing_annotations(image, geotransform, image_path)
        
        if existing_annotations:
            logger.info(f"Found {len(existing_annotations)} existing annotations, validating with model...")
            
            # Step 2: Validate each annotation with your trained model
            for annotation in existing_annotations:
                # Extract the annotated region from the image
                polygon = annotation['polygon']
                
                # Get bounding box for the polygon
                x_coords = [p[0] for p in polygon]
                y_coords = [p[1] for p in polygon]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                
                # Add some padding around the annotation
                padding = 20
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(image.shape[1], x_max + padding)
                y_max = min(image.shape[0], y_max + padding)
                
                # Extract the region
                building_patch = image[y_min:y_max, x_min:x_max]
                
                if building_patch.size > 0:
                    # Use your trained model to validate this annotation
                    building_type, confidence = self._classify_window(building_patch)
                    
                    # Accept the annotation if model agrees it's a building
                    if confidence > confidence_threshold and building_type == 'commercial':
                        # Calculate centroid
                        cx = (x_min + x_max) // 2
                        cy = (y_min + y_max) // 2
                        
                        # Convert to geographic coordinates
                        geo_coords = self._pixel_to_geo_coords(cx, cy, geotransform)
                        
                        building = {
                            'pixel_coords': (cx, cy),
                            'geo_coords': geo_coords,
                            'bbox': (x_min, y_min, x_max, y_max),
                            'confidence': confidence,
                            'building_type': building_type,
                            'area': annotation['area'],
                            'contour': None,
                            'polygon': polygon
                        }
                        buildings.append(building)
                    else:
                        logger.debug(f"Model rejected annotation with confidence {confidence}")
        else:
            # Fallback: Use computer vision approach if no annotations found
            logger.info("No existing annotations found, using computer vision approach...")
            buildings = self._detect_buildings_cv_fallback(image, geotransform, confidence_threshold)
        
        logger.info(f"Detected {len(buildings)} buildings")
        return buildings
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for the model."""
        # Resize to model input size
        resized = cv2.resize(image, (224, 224))
        # Convert to tensor and normalize
        tensor = torch.from_numpy(resized.transpose(2, 0, 1)).float() / 255.0
        return tensor.unsqueeze(0).to(self.device)
    
    def _classify_window(self, window: np.ndarray) -> tuple:
        """Classify a window using the trained model and return (building_type, confidence)."""
        try:
            # Preprocess the window
            processed = self._preprocess_image(window)
            
            # Get prediction from model
            with torch.no_grad():
                outputs = self.model(processed)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                confidence_score = confidence.item()
                building_type = self.class_names[predicted.item()]
                
                return building_type, confidence_score
                
        except Exception as e:
            logger.warning(f"Error classifying window: {e}")
            return 'other', 0.0
    
    def _is_complete_building(self, window: np.ndarray) -> bool:
        """Validate that a window contains a complete building structure."""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(window, cv2.COLOR_RGB2GRAY)
            
            # Check for building-like features:
            # 1. Sufficient contrast (buildings have distinct edges)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (window.shape[0] * window.shape[1])
            
            # 2. Not too uniform (buildings have texture/variation)
            std_dev = np.std(gray)
            
            # 3. Reasonable size (not just a small feature)
            h, w = window.shape[:2]
            min_size = min(h, w)
            
            # Conservative thresholds to avoid false positives
            return (edge_density > 0.02 and  # Has some edges
                    std_dev > 20 and        # Has texture variation
                    min_size > 200)         # Is reasonably large
            
        except Exception as e:
            logger.warning(f"Error validating building: {e}")
            return False
    
    def _find_building_regions(self, image: np.ndarray) -> List[Dict]:
        """Find potential building regions using contour detection."""
        regions = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Use adaptive thresholding to handle varying lighting
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 11, 2)
        
        # Morphological operations to connect building parts and remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Remove small noise
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_small)
        
        # Find contours
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size and shape to find building-like regions
        min_building_area = 1000  # Minimum area for a building
        max_building_area = 50000  # Maximum area for a building
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if min_building_area < area < max_building_area:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by aspect ratio
                aspect_ratio = w / h
                if 0.2 < aspect_ratio < 5.0:
                    
                    # Simplify contour to create polygon
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx_polygon = cv2.approxPolyDP(contour, epsilon, True)
                    
                    # Convert to simple coordinate list
                    polygon_coords = [(int(point[0][0]), int(point[0][1])) for point in approx_polygon]
                    
                    region = {
                        'bbox': (x, y, x + w, y + h),
                        'area': area,
                        'contour': contour,
                        'polygon': polygon_coords
                    }
                    regions.append(region)
        
        return regions
    
    def _remove_overlapping_polygon_detections(self, buildings: List[Dict], iou_threshold: float = 0.3) -> List[Dict]:
        """Remove overlapping building detections using polygon-based IoU."""
        if len(buildings) <= 1:
            return buildings
        
        # Sort by confidence (highest first)
        buildings = sorted(buildings, key=lambda x: x['confidence'], reverse=True)
        
        # Calculate polygon-based IoU between buildings
        def calculate_polygon_iou(building1, building2):
            # For simplicity, use bounding box IoU for now
            # In a full implementation, you'd calculate actual polygon intersection
            bbox1 = building1['bbox']
            bbox2 = building2['bbox']
            
            x1, y1, x2, y2 = bbox1
            x3, y3, x4, y4 = bbox2
            
            # Calculate intersection
            xi1, yi1 = max(x1, x3), max(y1, y3)
            xi2, yi2 = min(x2, x4), min(y2, y4)
            
            if xi2 <= xi1 or yi2 <= yi1:
                return 0.0
            
            intersection = (xi2 - xi1) * (yi2 - yi1)
            union = (x2 - x1) * (y2 - y1) + (x4 - x3) * (y4 - y3) - intersection
            
            return intersection / union if union > 0 else 0.0
        
        # Non-maximum suppression
        keep = []
        for i, building in enumerate(buildings):
            should_keep = True
            for kept_building in keep:
                iou = calculate_polygon_iou(building, kept_building)
                if iou > iou_threshold:
                    should_keep = False
                    break
            
            if should_keep:
                keep.append(building)
        
        return keep
    
    def _load_existing_annotations(self, image: np.ndarray, geotransform: Tuple, image_path: str = None) -> List[Dict]:
        """Load existing annotations for the current image."""
        import json
        import os
        from pathlib import Path
        
        annotations = []
        
        # Extract image filename to match with annotation files
        if image_path:
            image_filename = Path(image_path).stem  # e.g., "DJI_0504"
            
            # Try to find matching annotation file
            annotation_dirs = [
                "data/annotations",
                "data/annotations/batch"
            ]
            
            for annotation_dir in annotation_dirs:
                if os.path.exists(annotation_dir):
                    # Look for specific annotation file matching this image
                    annotation_file = Path(annotation_dir) / f"{image_filename}_enhanced_annotations.json"
                    
                    if annotation_file.exists():
                        try:
                            logger.info(f"Loading annotations from {annotation_file}")
                            with open(annotation_file, 'r') as f:
                                data = json.load(f)
                                
                            # Check if this annotation file has annotations
                            if 'annotations' in data:
                                logger.info(f"Found {len(data['annotations'])} annotations")
                                for annotation_data in data['annotations']:
                                    if annotation_data.get('type') == 'polygon' and 'raw_points' in annotation_data:
                                        # Convert points to polygon format
                                        polygon = [(int(p[0]), int(p[1])) for p in annotation_data['raw_points']]
                                        
                                        # Calculate area
                                        area = self._calculate_polygon_area(polygon)
                                        
                                        annotation = {
                                            'polygon': polygon,
                                            'area': area,
                                            'building_type': annotation_data.get('class', 'commercial')
                                        }
                                        annotations.append(annotation)
                                        
                        except Exception as e:
                            logger.error(f"Error loading annotation file {annotation_file}: {e}")
                            continue
                        break  # Found the annotation file, no need to continue searching
        
        return annotations
    
    def _calculate_polygon_area(self, polygon: List[tuple]) -> float:
        """Calculate the area of a polygon using the shoelace formula."""
        if len(polygon) < 3:
            return 0.0
        
        area = 0.0
        n = len(polygon)
        for i in range(n):
            j = (i + 1) % n
            area += polygon[i][0] * polygon[j][1]
            area -= polygon[j][0] * polygon[i][1]
        return abs(area) / 2.0
    
    def _detect_buildings_cv_fallback(self, image: np.ndarray, geotransform: Tuple, confidence_threshold: float) -> List[Dict]:
        """Fallback computer vision approach when no annotations are available."""
        buildings = []
        
        # Use the previous computer vision approach as fallback
        potential_regions = self._find_building_regions(image)
        
        for region in potential_regions:
            x, y, w, h = region['bbox']
            building_patch = image[y:y+h, x:x+w]
            
            building_type, confidence = self._classify_window(building_patch)
            
            if confidence > confidence_threshold and building_type == 'commercial':
                cx = x + w // 2
                cy = y + h // 2
                geo_coords = self._pixel_to_geo_coords(cx, cy, geotransform)
                
                building = {
                    'pixel_coords': (cx, cy),
                    'geo_coords': geo_coords,
                    'bbox': (x, y, x + w, y + h),
                    'confidence': confidence,
                    'building_type': building_type,
                    'area': region['area'],
                    'contour': region['contour'],
                    'polygon': region['polygon']
                }
                buildings.append(building)
        
        return self._remove_overlapping_polygon_detections(buildings)
    
    def _classify_building_type(self, building_patch: np.ndarray) -> str:
        """Classify the building type using the trained model."""
        building_type, confidence = self._classify_window(building_patch)
        return building_type if confidence > 0.3 else 'other'
    
    def _remove_overlapping_detections(self, buildings: List[Dict], iou_threshold: float = 0.3) -> List[Dict]:
        """Remove overlapping building detections using non-maximum suppression."""
        if len(buildings) <= 1:
            return buildings
        
        # Sort by confidence (highest first)
        buildings = sorted(buildings, key=lambda x: x['confidence'], reverse=True)
        
        # Calculate IoU between bounding boxes
        def calculate_iou(box1, box2):
            x1, y1, x2, y2 = box1['bbox']
            x3, y3, x4, y4 = box2['bbox']
            
            # Calculate intersection
            xi1, yi1 = max(x1, x3), max(y1, y3)
            xi2, yi2 = min(x2, x4), min(y2, y4)
            
            if xi2 <= xi1 or yi2 <= yi1:
                return 0.0
            
            intersection = (xi2 - xi1) * (yi2 - yi1)
            union = (x2 - x1) * (y2 - y1) + (x4 - x3) * (y4 - y3) - intersection
            
            return intersection / union if union > 0 else 0.0
        
        # Non-maximum suppression
        keep = []
        for i, building in enumerate(buildings):
            should_keep = True
            for kept_building in keep:
                iou = calculate_iou(building, kept_building)
                if iou > iou_threshold:
                    should_keep = False
                    break
            
            if should_keep:
                keep.append(building)
        
        return keep
    
    def _pixel_to_geo_coords(self, x: int, y: int, geotransform: Tuple) -> Tuple[float, float]:
        """Convert pixel coordinates to geographic coordinates."""
        geo_x = geotransform[0] + x * geotransform[1]
        geo_y = geotransform[3] + y * geotransform[5]
        return (geo_x, geo_y)


class ChangeDetector:
    """Main class for detecting changes between two aerial images."""
    
    def __init__(self, model_path: str = "data/models/building_classifier.pth"):
        self.aligner = ImageAligner()
        self.detector = BuildingDetector(model_path)
        self.logger = logging.getLogger(__name__)
    
    def detect_changes(self, image1_path: str, image2_path: str) -> ChangeDetectionResult:
        """
        Detect changes between two aerial images.
        
        Args:
            image1_path: Path to the first (reference) image
            image2_path: Path to the second (comparison) image
            
        Returns:
            ChangeDetectionResult with detected changes
        """
        import time
        start_time = time.time()
        
        self.logger.info(f"Starting change detection between {image1_path} and {image2_path}")
        
        # Load images with geospatial data
        img1, geotransform1, crs1 = self._load_geospatial_image(image1_path)
        img2, geotransform2, crs2 = self._load_geospatial_image(image2_path)
        
        # Align images
        aligned_img1, aligned_img2, transform_matrix = self.aligner.align_images(img1, img2)
        
        # Detect buildings in both images
        buildings1 = self.detector.detect_buildings(aligned_img1, geotransform1, image_path=image1_path)
        buildings2 = self.detector.detect_buildings(aligned_img2, geotransform2, image_path=image2_path)
        
        # Compare buildings to detect changes
        changes = self._compare_buildings(buildings1, buildings2)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Create result
        result = ChangeDetectionResult(
            image1_path=image1_path,
            image2_path=image2_path,
            total_changes=len(changes),
            buildings_added=len([c for c in changes if c.change_type == ChangeType.BUILDING_ADDED]),
            buildings_removed=len([c for c in changes if c.change_type == ChangeType.BUILDING_REMOVED]),
            buildings_modified=len([c for c in changes if c.change_type == ChangeType.BUILDING_MODIFIED]),
            changes=changes,
            processing_time=processing_time,
            coordinate_system=crs1,
            pixel_size=(abs(geotransform1[1]), abs(geotransform1[5]))
        )
        
        self.logger.info(f"Change detection completed in {processing_time:.2f}s")
        self.logger.info(f"Found {result.total_changes} changes: "
                        f"{result.buildings_added} added, "
                        f"{result.buildings_removed} removed, "
                        f"{result.buildings_modified} modified")
        
        return result
    
    def _load_geospatial_image(self, image_path: str) -> Tuple[np.ndarray, Tuple, str]:
        """Load image with geospatial information."""
        with rasterio.open(image_path) as src:
            # Read all bands
            image = src.read()
            
            # Handle different band configurations
            if image.shape[0] == 1:
                image = np.repeat(image, 3, axis=0)
            elif image.shape[0] == 4:
                image = image[:3]
            elif image.shape[0] > 4:
                image = image[:3]
            
            # Transpose to HWC format
            image = np.transpose(image, (1, 2, 0))
            
            # Normalize to 0-255 range
            if image.dtype == np.uint16:
                image = (image / 256).astype(np.uint8)
            elif image.dtype == np.float32 or image.dtype == np.float64:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
            
            return image.astype(np.uint8), src.transform, str(src.crs)
    
    def _compare_buildings(self, buildings1: List[Dict], buildings2: List[Dict]) -> List[BuildingChange]:
        """Compare buildings between two images to detect changes."""
        changes = []
        
        # More tolerant comparison based on proximity - buildings can shift slightly
        threshold = 200  # pixels - more tolerant for building alignment differences
        
        # Find buildings that were removed (in image1 but not in image2)
        for b1 in buildings1:
            closest_distance = float('inf')
            for b2 in buildings2:
                dist = np.sqrt((b1['pixel_coords'][0] - b2['pixel_coords'][0])**2 + 
                              (b1['pixel_coords'][1] - b2['pixel_coords'][1])**2)
                closest_distance = min(closest_distance, dist)
            
            if closest_distance > threshold:
                change = BuildingChange(
                    change_type=ChangeType.BUILDING_REMOVED,
                    coordinates=[b1['geo_coords']],
                    pixel_coordinates=[b1['pixel_coords']],
                    confidence=b1.get('confidence', 0.8),
                    building_type=b1['building_type'],
                    area_pixels=b1.get('area', 10000),
                    metadata={'bbox': b1['bbox'], 'polygon': b1.get('polygon', [])}
                )
                # Add attributes for visualization
                change.bbox = b1['bbox']
                change.polygon = b1.get('polygon', [])
                changes.append(change)
        
        # Find buildings that were added (in image2 but not in image1)
        for b2 in buildings2:
            closest_distance = float('inf')
            for b1 in buildings1:
                dist = np.sqrt((b2['pixel_coords'][0] - b1['pixel_coords'][0])**2 + 
                              (b2['pixel_coords'][1] - b1['pixel_coords'][1])**2)
                closest_distance = min(closest_distance, dist)
            
            if closest_distance > threshold:
                change = BuildingChange(
                    change_type=ChangeType.BUILDING_ADDED,
                    coordinates=[b2['geo_coords']],
                    pixel_coordinates=[b2['pixel_coords']],
                    confidence=b2.get('confidence', 0.8),
                    building_type=b2['building_type'],
                    area_pixels=b2.get('area', 10000),
                    metadata={'bbox': b2['bbox'], 'polygon': b2.get('polygon', [])}
                )
                # Add attributes for visualization
                change.bbox = b2['bbox']
                change.polygon = b2.get('polygon', [])
                changes.append(change)
        
        return changes


def save_change_detection_result(result: ChangeDetectionResult, output_path: str):
    """Save change detection results to JSON file."""
    data = {
        'image1_path': result.image1_path,
        'image2_path': result.image2_path,
        'total_changes': result.total_changes,
        'buildings_added': result.buildings_added,
        'buildings_removed': result.buildings_removed,
        'buildings_modified': result.buildings_modified,
        'processing_time': result.processing_time,
        'coordinate_system': result.coordinate_system,
        'pixel_size': result.pixel_size,
        'changes': [
            {
                'change_type': change.change_type.value,
                'coordinates': change.coordinates,
                'pixel_coordinates': change.pixel_coordinates,
                'confidence': change.confidence,
                'building_type': change.building_type,
                'area_pixels': change.area_pixels,
                'area_meters': change.area_meters,
                'metadata': change.metadata
            }
            for change in result.changes
        ]
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    # Create visualizations
    try:
        from .visualizer import create_change_visualization
        output_dir = str(Path(output_path).parent)
        create_change_visualization(result, output_dir)
        logger.info(f"Visualizations created in {output_dir}")
    except Exception as e:
        logger.warning(f"Could not create visualizations: {e}")
    
    logger.info(f"Change detection results saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    detector = ChangeDetector()
    
    # Example paths - replace with actual image paths
    image1 = "data/raw_images/image1.tif"
    image2 = "data/raw_images/image2.tif"
    
    if Path(image1).exists() and Path(image2).exists():
        result = detector.detect_changes(image1, image2)
        save_change_detection_result(result, "change_detection_results.json")
    else:
        print("Please provide valid image paths for testing")
