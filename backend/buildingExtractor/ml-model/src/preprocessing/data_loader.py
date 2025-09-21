"""
Data loading and preprocessing utilities for aerial imagery.
"""

import os
import numpy as np
import cv2
import rasterio
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import json
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class AerialImageDataset(Dataset):
    """Dataset class for aerial imagery with annotations."""
    
    def __init__(self, 
                 image_dir: str,
                 annotation_dir: str,
                 transform=None,
                 target_size: Tuple[int, int] = (640, 640),
                 augmentation: bool = True):
        """
        Initialize the dataset.
        
        Args:
            image_dir: Directory containing TIF images
            annotation_dir: Directory containing annotation files
            transform: Optional transform to be applied
            target_size: Target size for images (width, height)
            augmentation: Whether to apply data augmentation
        """
        self.image_dir = Path(image_dir)
        self.annotation_dir = Path(annotation_dir)
        self.target_size = target_size
        self.transform = transform
        
        # Get all image files
        self.image_files = list(self.image_dir.glob("*.tif")) + list(self.image_dir.glob("*.tiff"))
        self.image_files = [f for f in self.image_files if f.is_file()]
        
        # Filter images that have annotations
        self.valid_files = []
        for img_file in self.image_files:
            annotation_file = self.annotation_dir / f"{img_file.stem}_annotations.json"
            if annotation_file.exists():
                self.valid_files.append((img_file, annotation_file))
        
        print(f"Found {len(self.valid_files)} image-annotation pairs")
        
        # Define augmentation pipeline
        if augmentation:
            self.augmentation = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.HueSaturationValue(p=0.2),
                A.GaussNoise(p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
        else:
            self.augmentation = None
    
    def __len__(self):
        return len(self.valid_files)
    
    def __getitem__(self, idx):
        """Get item from dataset."""
        img_file, annotation_file = self.valid_files[idx]
        
        # Load image
        image = self._load_image(img_file)
        
        # Load annotations
        annotations = self._load_annotations(annotation_file)
        
        # Extract bounding boxes and labels
        bboxes = []
        class_labels = []
        class_ids = []
        
        for ann in annotations:
            bbox = ann['pixel_coordinates']
            bboxes.append([bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']])
            
            class_name = ann['class']
            class_labels.append(class_name)
            class_ids.append(self._class_to_id(class_name))
        
        # Apply augmentation
        if self.augmentation and len(bboxes) > 0:
            augmented = self.augmentation(image=image, bboxes=bboxes, class_labels=class_labels)
            image = augmented['image']
            bboxes = augmented['bboxes']
            class_labels = augmented['class_labels']
        
        # Convert to tensor
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        
        return {
            'image': image,
            'bboxes': torch.tensor(bboxes, dtype=torch.float32) if bboxes else torch.zeros((0, 4)),
            'class_ids': torch.tensor(class_ids, dtype=torch.long) if class_ids else torch.zeros((0,), dtype=torch.long),
            'class_labels': class_labels,
            'image_path': str(img_file)
        }
    
    def _load_image(self, image_path: Path) -> np.ndarray:
        """Load image from file."""
        with rasterio.open(image_path) as src:
            # Read all bands
            image = src.read()
            
            # Handle different band configurations
            if image.shape[0] == 1:
                # Grayscale
                image = np.repeat(image, 3, axis=0)
            elif image.shape[0] == 4:
                # RGBA, remove alpha
                image = image[:3]
            elif image.shape[0] > 4:
                # Multi-spectral, take first 3 bands
                image = image[:3]
            
            # Transpose to HWC format
            image = np.transpose(image, (1, 2, 0))
            
            # Normalize to 0-255 range
            image = self._normalize_image(image)
            
            return image.astype(np.uint8)
    
    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to 0-255 range."""
        # Handle different data types
        if image.dtype == np.uint16:
            # 16-bit to 8-bit conversion
            image = (image / 256).astype(np.uint8)
        elif image.dtype == np.float32 or image.dtype == np.float64:
            # Float to 8-bit conversion
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
        
        return image
    
    def _load_annotations(self, annotation_file: Path) -> List[Dict]:
        """Load annotations from JSON file."""
        with open(annotation_file, 'r') as f:
            data = json.load(f)
            return data.get('annotations', [])
    
    def _class_to_id(self, class_name: str) -> int:
        """Convert class name to ID."""
        class_mapping = {
            "residential_single": 0,
            "residential_multi": 1,
            "commercial": 2,
            "industrial": 3,
            "institutional": 4,
            "other": 5
        }
        return class_mapping.get(class_name, 5)
    
    @staticmethod
    def get_class_names() -> List[str]:
        """Get list of class names."""
        return [
            "residential_single",
            "residential_multi", 
            "commercial",
            "industrial",
            "institutional",
            "other"
        ]


class DataPreprocessor:
    """Preprocessing utilities for aerial imagery."""
    
    def __init__(self, target_size: Tuple[int, int] = (640, 640)):
        self.target_size = target_size
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocess a single image."""
        with rasterio.open(image_path) as src:
            image = src.read()
            
            # Handle different band configurations
            if image.shape[0] == 1:
                image = np.repeat(image, 3, axis=0)
            elif image.shape[0] == 4:
                image = image[:3]
            elif image.shape[0] > 4:
                image = image[:3]
            
            # Transpose to HWC
            image = np.transpose(image, (1, 2, 0))
            
            # Resize if needed
            if image.shape[:2] != self.target_size:
                image = cv2.resize(image, self.target_size)
            
            # Normalize
            image = self._normalize_image(image)
            
            return image
    
    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to 0-255 range."""
        if image.dtype == np.uint16:
            image = (image / 256).astype(np.uint8)
        elif image.dtype == np.float32 or image.dtype == np.float64:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
        
        return image
    
    def create_yolo_format(self, annotation_file: str, output_dir: str):
        """Convert annotations to YOLO format."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(annotation_file, 'r') as f:
            data = json.load(f)
        
        image_name = Path(annotation_file).stem.replace('_annotations', '')
        yolo_file = output_dir / f"{image_name}.txt"
        
        with open(yolo_file, 'w') as f:
            for ann in data.get('annotations', []):
                pixel_coords = ann['pixel_coordinates']
                class_name = ann['class']
                class_id = self._class_to_id(class_name)
                
                # Convert to YOLO format (normalized coordinates)
                # Assuming image dimensions from metadata
                image_width = data.get('width', 1000)
                image_height = data.get('height', 1000)
                
                x_center = (pixel_coords['x1'] + pixel_coords['x2']) / 2 / image_width
                y_center = (pixel_coords['y1'] + pixel_coords['y2']) / 2 / image_height
                width = (pixel_coords['x2'] - pixel_coords['x1']) / image_width
                height = (pixel_coords['y2'] - pixel_coords['y1']) / image_height
                
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    def _class_to_id(self, class_name: str) -> int:
        """Convert class name to ID."""
        class_mapping = {
            "residential_single": 0,
            "residential_multi": 1,
            "commercial": 2,
            "industrial": 3,
            "institutional": 4,
            "other": 5
        }
        return class_mapping.get(class_name, 5)


def create_data_loaders(image_dir: str,
                       annotation_dir: str,
                       batch_size: int = 8,
                       train_split: float = 0.8,
                       target_size: Tuple[int, int] = (640, 640)) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders."""
    
    # Create full dataset
    full_dataset = AerialImageDataset(
        image_dir=image_dir,
        annotation_dir=annotation_dir,
        target_size=target_size
    )
    
    # Split dataset
    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader


def collate_fn(batch):
    """Custom collate function for handling variable number of objects."""
    images = torch.stack([item['image'] for item in batch])
    bboxes = [item['bboxes'] for item in batch]
    class_ids = [item['class_ids'] for item in batch]
    class_labels = [item['class_labels'] for item in batch]
    image_paths = [item['image_path'] for item in batch]
    
    return {
        'images': images,
        'bboxes': bboxes,
        'class_ids': class_ids,
        'class_labels': class_labels,
        'image_paths': image_paths
    }
