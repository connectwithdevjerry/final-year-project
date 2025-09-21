"""
Building detection models for aerial imagery.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
from typing import Dict, List, Tuple, Optional
import cv2
from ultralytics import YOLO
import os


class BuildingDetector(nn.Module):
    """Custom CNN for building detection in aerial imagery."""
    
    def __init__(self, num_classes: int = 6, pretrained: bool = True):
        """
        Initialize building detector.
        
        Args:
            num_classes: Number of building classes
            pretrained: Whether to use pretrained weights
        """
        super(BuildingDetector, self).__init__()
        
        # Use ResNet50 as backbone
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Remove the original classifier
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # Add custom head for building detection
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        # Bounding box regression head
        self.bbox_head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 4)  # x, y, w, h
        )
    
    def forward(self, x):
        """Forward pass."""
        features = self.backbone(x)
        
        # Global pooling
        pooled = self.global_pool(features)
        pooled = pooled.view(pooled.size(0), -1)
        
        # Classification and bbox regression
        class_logits = self.classifier(pooled)
        bbox_coords = self.bbox_head(pooled)
        
        return class_logits, bbox_coords


class YOLOBuildingDetector:
    """YOLO-based building detector with coordinate preservation."""
    
    def __init__(self, model_path: Optional[str] = None, num_classes: int = 6):
        """
        Initialize YOLO building detector.
        
        Args:
            model_path: Path to trained YOLO model
            num_classes: Number of building classes
        """
        self.num_classes = num_classes
        self.class_names = [
            "residential_single",
            "residential_multi", 
            "commercial",
            "industrial",
            "institutional",
            "other"
        ]
        
        if model_path and os.path.exists(model_path):
            self.model = YOLO(model_path)
        else:
            # Initialize with pretrained YOLOv8
            self.model = YOLO('yolov8n.pt')
            # Modify for building detection
            self._setup_custom_model()
    
    def _setup_custom_model(self):
        """Setup custom model architecture for building detection."""
        # This would involve modifying the YOLO model for our specific classes
        # For now, we'll use the pretrained model and retrain it
        pass
    
    def train(self, 
              data_config: str,
              epochs: int = 100,
              batch_size: int = 16,
              img_size: int = 640,
              device: str = 'auto'):
        """
        Train the YOLO model.
        
        Args:
            data_config: Path to YAML config file
            epochs: Number of training epochs
            batch_size: Batch size for training
            img_size: Input image size
            device: Device to use for training
        """
        results = self.model.train(
            data=data_config,
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            device=device,
            project='data/models',
            name='building_detection',
            save_period=10,
            patience=20,
            lr0=0.01,
            weight_decay=0.0005,
            warmup_epochs=3,
            box=7.5,
            cls=0.5,
            dfl=1.5
        )
        
        return results
    
    def predict(self, 
                image_path: str,
                conf_threshold: float = 0.25,
                iou_threshold: float = 0.45,
                save_results: bool = True) -> Dict:
        """
        Predict buildings in an image.
        
        Args:
            image_path: Path to input image
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS
            save_results: Whether to save prediction results
            
        Returns:
            Dictionary containing predictions
        """
        results = self.model.predict(
            source=image_path,
            conf=conf_threshold,
            iou=iou_threshold,
            save=save_results,
            save_txt=True,
            save_conf=True
        )
        
        # Process results
        predictions = self._process_results(results[0])
        
        return predictions
    
    def _process_results(self, result) -> Dict:
        """Process YOLO results into standardized format."""
        predictions = {
            'boxes': [],
            'scores': [],
            'class_ids': [],
            'class_names': []
        }
        
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
            scores = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            
            for i in range(len(boxes)):
                predictions['boxes'].append(boxes[i].tolist())
                predictions['scores'].append(float(scores[i]))
                predictions['class_ids'].append(int(class_ids[i]))
                predictions['class_names'].append(self.class_names[class_ids[i]])
        
        return predictions
    
    def predict_with_coordinates(self, 
                               image_path: str,
                               geo_handler,
                               conf_threshold: float = 0.25) -> Dict:
        """
        Predict buildings with coordinate preservation.
        
        Args:
            image_path: Path to input image
            geo_handler: GeospatialHandler instance
            conf_threshold: Confidence threshold
            
        Returns:
            Dictionary with predictions and coordinates
        """
        # Get predictions
        predictions = self.predict(image_path, conf_threshold, save_results=False)
        
        # Add coordinate information
        predictions_with_coords = {
            'image_path': image_path,
            'crs': geo_handler.metadata['crs'],
            'transform': list(geo_handler.metadata['transform']),
            'bounds': list(geo_handler.metadata['bounds']),
            'resolution_info': geo_handler.get_resolution_info(),
            'predictions': []
        }
        
        for i, (box, score, class_id, class_name) in enumerate(
            zip(predictions['boxes'], predictions['scores'], 
                predictions['class_ids'], predictions['class_names'])):
            
            x1, y1, x2, y2 = box
            
            # Convert pixel coordinates to geographic coordinates
            lon1, lat1 = geo_handler.pixel_to_coords(int(x1), int(y1))
            lon2, lat2 = geo_handler.pixel_to_coords(int(x2), int(y2))
            
            prediction_with_coords = {
                'id': i,
                'class': class_name,
                'class_id': class_id,
                'confidence': score,
                'pixel_coordinates': {
                    'x1': float(x1), 'y1': float(y1),
                    'x2': float(x2), 'y2': float(y2)
                },
                'geographic_coordinates': {
                    'lon1': lon1, 'lat1': lat1,
                    'lon2': lon2, 'lat2': lat2,
                    'center_lon': (lon1 + lon2) / 2,
                    'center_lat': (lat1 + lat2) / 2
                }
            }
            
            predictions_with_coords['predictions'].append(prediction_with_coords)
        
        return predictions_with_coords


class BuildingSegmentationModel(nn.Module):
    """U-Net based model for building segmentation."""
    
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        """
        Initialize segmentation model.
        
        Args:
            num_classes: Number of classes (background + buildings)
            pretrained: Whether to use pretrained weights
        """
        super(BuildingSegmentationModel, self).__init__()
        
        # Use ResNet50 as encoder
        self.encoder = models.resnet50(pretrained=pretrained)
        
        # Remove classifier and average pooling
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-2])
        
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(1024, 512, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, num_classes, 1)
        )
    
    def forward(self, x):
        """Forward pass."""
        features = self.encoder(x)
        output = self.decoder(features)
        return output


class ModelTrainer:
    """Training utilities for building detection models."""
    
    def __init__(self, model, device: str = 'auto'):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            device: Device to use for training
        """
        self.model = model
        self.device = torch.device(device if device != 'auto' else 
                                 ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model.to(self.device)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
    
    def train_epoch(self, dataloader, optimizer, criterion):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch in dataloader:
            images = batch['images'].to(self.device)
            class_ids = batch['class_ids'].to(self.device)
            bboxes = batch['bboxes'].to(self.device)
            
            optimizer.zero_grad()
            
            if hasattr(self.model, 'forward'):
                # Custom model
                class_logits, bbox_pred = self.model(images)
                
                # Classification loss
                class_loss = criterion(class_logits, class_ids)
                
                # Bbox regression loss (simplified)
                bbox_loss = F.mse_loss(bbox_pred, bboxes.float())
                
                total_loss_batch = class_loss + 0.1 * bbox_loss
            else:
                # YOLO model (handled differently)
                total_loss_batch = 0  # YOLO handles its own loss
            
            total_loss_batch.backward()
            optimizer.step()
            
            total_loss += total_loss_batch.item()
            
            if hasattr(self.model, 'forward'):
                _, predicted = torch.max(class_logits.data, 1)
                total += class_ids.size(0)
                correct += (predicted == class_ids).sum().item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100 * correct / total if total > 0 else 0
        
        return avg_loss, accuracy
    
    def validate_epoch(self, dataloader, criterion):
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                images = batch['images'].to(self.device)
                class_ids = batch['class_ids'].to(self.device)
                bboxes = batch['bboxes'].to(self.device)
                
                if hasattr(self.model, 'forward'):
                    class_logits, bbox_pred = self.model(images)
                    
                    class_loss = criterion(class_logits, class_ids)
                    bbox_loss = F.mse_loss(bbox_pred, bboxes.float())
                    total_loss_batch = class_loss + 0.1 * bbox_loss
                    
                    _, predicted = torch.max(class_logits.data, 1)
                    total += class_ids.size(0)
                    correct += (predicted == class_ids).sum().item()
                else:
                    total_loss_batch = 0
                
                total_loss += total_loss_batch.item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100 * correct / total if total > 0 else 0
        
        return avg_loss, accuracy
    
    def train(self, 
              train_loader,
              val_loader,
              epochs: int = 100,
              learning_rate: float = 0.001,
              save_path: str = 'data/models/building_detector.pth'):
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            learning_rate: Learning rate
            save_path: Path to save the trained model
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=10, factor=0.5
        )
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            
            # Validate
            val_loss, val_acc = self.validate_epoch(val_loader, criterion)
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Save history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Print progress
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print(f'  LR: {optimizer.param_groups[0]["lr"]:.6f}')
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), save_path)
                print(f'  New best model saved to {save_path}')
            
            print('-' * 50)
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }
