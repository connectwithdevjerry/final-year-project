#!/usr/bin/env python3
"""
Advanced building detection system that identifies ALL buildings in a TIF file
and provides detailed annotations with coordinates.
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
import rasterio
from torchvision import models
from pathlib import Path
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.patches as patches
from skimage import measure, morphology
from scipy import ndimage
import tempfile
import os

class EnhancedBuildingClassifier(nn.Module):
    """Enhanced building classifier using ResNet."""
    
    def __init__(self, num_classes=6):
        super().__init__()
        self.backbone = models.resnet18(pretrained=False)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)

class AdvancedBuildingDetector:
    """Advanced building detection system."""
    
    def __init__(self, model_path='data/models/building_classifier.pth'):
        self.model_path = model_path
        self.model = None
        self.class_names = [
            "residential_single",
            "residential_multi", 
            "commercial",
            "industrial",
            "institutional",
            "other"
        ]
        self.load_model()
    
    def load_model(self):
        """Load the trained model."""
        print("🤖 Loading enhanced model for building detection...")
        try:
            self.model = EnhancedBuildingClassifier(num_classes=6)
            self.model.load_state_dict(torch.load(self.model_path, map_location='cpu'))
            self.model.eval()
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def load_and_preprocess_image(self, image_path, target_size=(224, 224)):
        """Load and preprocess image for building detection."""
        print(f"📷 Loading image: {image_path}")
        
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        with rasterio.open(image_path) as src:
            # Get original image dimensions and geotransform
            self.original_shape = src.shape
            self.geotransform = src.get_transform()
            self.crs = src.crs
            
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
            
            # Store original image for processing
            self.original_image = image.copy()
            
            # Resize for display
            self.display_image = cv2.resize(image, target_size)
            
            # Normalize to 0-255 range
            if image.dtype == np.uint16:
                image = (image / 256).astype(np.uint8)
            elif image.dtype == np.float32 or image.dtype == np.float64:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
            
            self.processed_image = image.astype(np.uint8)
            print(f"✅ Image loaded: {self.original_shape} -> {self.processed_image.shape}")
            
            return self.processed_image
    
    def detect_building_regions(self, image, min_area=100, max_area=50000):
        """Detect potential building regions using computer vision."""
        print("🔍 Detecting building regions...")
        
        # Convert to grayscale for processing
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Morphological operations to clean up
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area and aspect ratio
        building_regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                # Filter by aspect ratio (buildings are typically rectangular)
                if 0.3 < aspect_ratio < 3.0:
                    # Get polygon approximation
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx_poly = cv2.approxPolyDP(contour, epsilon, True)
                    
                    building_regions.append({
                        'contour': contour,
                        'polygon': approx_poly,
                        'bbox': (x, y, w, h),
                        'area': area,
                        'aspect_ratio': aspect_ratio
                    })
        
        print(f"✅ Found {len(building_regions)} potential building regions")
        return building_regions
    
    def classify_building_region(self, image, region):
        """Classify a building region using the trained model."""
        x, y, w, h = region['bbox']
        
        # Extract region with padding
        padding = 20
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(image.shape[1], x + w + padding)
        y2 = min(image.shape[0], y + h + padding)
        
        # Crop region
        region_crop = image[y1:y2, x1:x2]
        
        if region_crop.size == 0:
            return {'class': 'other', 'confidence': 0.0}
        
        # Resize to model input size
        region_resized = cv2.resize(region_crop, (224, 224))
        
        # Convert to tensor
        region_tensor = torch.from_numpy(region_resized.transpose(2, 0, 1)).float() / 255.0
        region_tensor = region_tensor.unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(region_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        return {
            'class': self.class_names[predicted_class],
            'confidence': confidence,
            'all_probabilities': {name: prob.item() for name, prob in zip(self.class_names, probabilities[0])}
        }
    
    def pixel_to_geographic(self, x, y):
        """Convert pixel coordinates to geographic coordinates."""
        if hasattr(self, 'geotransform'):
            geo_x = self.geotransform[0] + x * self.geotransform[1]
            geo_y = self.geotransform[3] + y * self.geotransform[5]
            return geo_x, geo_y
        return x, y
    
    def detect_all_buildings(self, image_path, min_confidence=0.3):
        """Detect and classify all buildings in the image."""
        print("🏗️ ADVANCED BUILDING DETECTION")
        print("=" * 50)
        
        # Load image
        image = self.load_and_preprocess_image(image_path)
        
        # Detect building regions
        building_regions = self.detect_building_regions(image)
        
        # Classify each region
        buildings = []
        for i, region in enumerate(building_regions):
            print(f"🔍 Analyzing region {i+1}/{len(building_regions)}...")
            
            # Classify the region
            classification = self.classify_building_region(image, region)
            
            if classification['confidence'] >= min_confidence:
                # Get geographic coordinates
                x, y, w, h = region['bbox']
                center_x, center_y = x + w//2, y + h//2
                geo_x, geo_y = self.pixel_to_geographic(center_x, center_y)
                
                # Get polygon coordinates
                polygon_coords = []
                for point in region['polygon']:
                    px, py = point[0]
                    geo_px, geo_py = self.pixel_to_geographic(px, py)
                    polygon_coords.append([geo_px, geo_py])
                
                building = {
                    'id': i + 1,
                    'class': classification['class'],
                    'confidence': classification['confidence'],
                    'bbox': region['bbox'],
                    'area_pixels': region['area'],
                    'center_pixel': (center_x, center_y),
                    'center_geographic': (geo_x, geo_y),
                    'polygon_coordinates': polygon_coords,
                    'aspect_ratio': region['aspect_ratio'],
                    'all_probabilities': classification['all_probabilities']
                }
                
                buildings.append(building)
                print(f"  ✅ {classification['class']} ({classification['confidence']*100:.1f}%)")
            else:
                print(f"  ❌ Low confidence ({classification['confidence']*100:.1f}%) - skipping")
        
        print(f"\n📊 DETECTION SUMMARY:")
        print(f"  Total regions found: {len(building_regions)}")
        print(f"  Buildings detected: {len(buildings)}")
        print(f"  Average confidence: {np.mean([b['confidence'] for b in buildings]):.3f}")
        
        # Count by class
        class_counts = {}
        for building in buildings:
            cls = building['class']
            class_counts[cls] = class_counts.get(cls, 0) + 1
        
        print(f"\n📋 BUILDING TYPES:")
        for cls, count in sorted(class_counts.items()):
            print(f"  {cls}: {count}")
        
        return {
            'image_path': image_path,
            'total_buildings': len(buildings),
            'buildings': buildings,
            'class_distribution': class_counts,
            'image_shape': self.original_shape,
            'geotransform': self.geotransform.tolist() if hasattr(self, 'geotransform') else None
        }
    
    def create_visualization(self, result, output_path=None):
        """Create a visualization of detected buildings."""
        print("🎨 Creating visualization...")
        
        if output_path is None:
            output_path = 'building_detection_result.png'
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Original image
        ax1.imshow(self.processed_image)
        ax1.set_title('Original Image', fontsize=16, fontweight='bold')
        ax1.axis('off')
        
        # Image with building annotations
        ax2.imshow(self.processed_image)
        ax2.set_title(f'Detected Buildings ({len(result["buildings"])} found)', fontsize=16, fontweight='bold')
        
        # Color map for different building types
        colors = {
            'commercial': 'red',
            'residential_single': 'blue',
            'residential_multi': 'lightblue',
            'industrial': 'orange',
            'institutional': 'green',
            'other': 'purple'
        }
        
        # Draw buildings
        for building in result['buildings']:
            color = colors.get(building['class'], 'gray')
            
            # Draw bounding box
            x, y, w, h = building['bbox']
            rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                                   edgecolor=color, facecolor='none')
            ax2.add_patch(rect)
            
            # Draw center point
            center_x, center_y = building['center_pixel']
            ax2.plot(center_x, center_y, 'o', color=color, markersize=8)
            
            # Add label
            ax2.text(x, y-5, f"{building['class'][:3]} {building['confidence']:.2f}", 
                    color=color, fontweight='bold', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        ax2.axis('off')
        
        # Add legend
        legend_elements = [patches.Patch(color=color, label=cls.replace('_', ' ').title()) 
                          for cls, color in colors.items()]
        ax2.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Visualization saved to: {output_path}")
        return output_path
    
    def save_detection_results(self, result, output_path=None):
        """Save detection results to JSON file."""
        if output_path is None:
            output_path = 'building_detection_results.json'
        
        # Convert numpy arrays to lists for JSON serialization
        json_result = result.copy()
        for building in json_result['buildings']:
            building['bbox'] = list(building['bbox'])
            building['center_pixel'] = list(building['center_pixel'])
            building['center_geographic'] = list(building['center_geographic'])
            building['polygon_coordinates'] = building['polygon_coordinates']
        
        with open(output_path, 'w') as f:
            json.dump(json_result, f, indent=2)
        
        print(f"✅ Results saved to: {output_path}")
        return output_path

def main():
    """Main function for testing building detection."""
    import sys
    
    print("🚀 ADVANCED BUILDING DETECTION SYSTEM")
    print("=" * 60)
    
    # Initialize detector
    detector = AdvancedBuildingDetector()
    
    # Get image path
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Default to orthomosaic
        image_path = "data/raw_images/orthomosaic.tif"
    
    print(f"🔍 Analyzing image: {image_path}")
    
    try:
        # Detect buildings
        result = detector.detect_all_buildings(image_path)
        
        # Create visualization
        viz_path = detector.create_visualization(result)
        
        # Save results
        json_path = detector.save_detection_results(result)
        
        print(f"\n🎉 BUILDING DETECTION COMPLETE!")
        print(f"📊 Results:")
        print(f"  - Buildings found: {result['total_buildings']}")
        print(f"  - Visualization: {viz_path}")
        print(f"  - JSON results: {json_path}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()

