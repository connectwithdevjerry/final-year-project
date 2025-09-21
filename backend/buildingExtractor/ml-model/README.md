# 🏗️ Advanced Building Detection System

A comprehensive AI-powered system for detecting and classifying buildings in aerial TIF images using deep learning and computer vision techniques.

## 🚀 Features

- **Advanced Building Detection**: Identifies all buildings in aerial TIF images
- **Multi-Class Classification**: Classifies buildings into 6 categories:
  - Residential Single
  - Residential Multi
  - Commercial
  - Industrial
  - Institutional
  - Other
- **Geographic Coordinate Preservation**: Maintains precise geographic coordinates for all detected buildings
- **Interactive Visualization**: Clickable building annotations with detailed information
- **Change Detection**: Compare two aerial images to identify building changes
- **High Accuracy**: Trained on 267+ manually annotated building polygons

## 📁 Project Structure

```
obaloluwap/
├── advanced_building_detector.py    # Main building detection system
├── data/
│   ├── annotations/                 # Building annotations (267+ polygons)
│   │   ├── batch/                  # Batch processed annotations
│   │   └── *.json                  # Individual image annotations
│   ├── models/
│   │   └── building_classifier.pth # Trained PyTorch model
│   └── raw_images/
│       ├── orthomosaic.tif         # Large orthomosaic image
│       └── orthophotos/            # 145 individual TIF images
├── src/
│   ├── annotation/                 # Annotation tools
│   ├── change_detection/           # Change detection algorithms
│   ├── models/                     # Model definitions
│   ├── preprocessing/              # Data preprocessing
│   └── utils/                      # Utility functions
└── requirements_py313.txt          # Python dependencies
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd obaloluwap
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements_py313.txt
   ```

3. **Verify installation**:
   ```bash
   python advanced_building_detector.py --help
   ```

## 🎯 Usage

### Basic Building Detection

```bash
# Detect all buildings in an image
python advanced_building_detector.py path/to/your/image.tif

# Detect with custom confidence threshold
python advanced_building_detector.py path/to/your/image.tif --min-confidence 0.5
```

### Programmatic Usage

```python
from advanced_building_detector import AdvancedBuildingDetector

# Initialize detector
detector = AdvancedBuildingDetector()

# Detect buildings
result = detector.detect_all_buildings('path/to/image.tif')

# Print results
print(f"Found {result['total_buildings']} buildings")
for building in result['buildings']:
    print(f"Building: {building['class']} (confidence: {building['confidence']:.2f})")
    print(f"Coordinates: {building['center_geographic']}")
```

## 📊 Model Performance

- **Training Data**: 267+ manually annotated building polygons
- **Model Architecture**: ResNet18-based classifier
- **Classes**: 6 building types
- **Accuracy**: High precision on commercial buildings (primary focus)
- **Input Format**: TIF aerial images with geographic metadata

## 🔧 Technical Details

### Building Detection Pipeline

1. **Image Preprocessing**: Load and normalize TIF images
2. **Region Detection**: Use computer vision to find potential building regions
3. **Classification**: Apply trained model to classify each region
4. **Coordinate Mapping**: Convert pixel coordinates to geographic coordinates
5. **Visualization**: Generate annotated images with building overlays

### Key Technologies

- **PyTorch**: Deep learning framework
- **OpenCV**: Computer vision operations
- **Rasterio**: Geospatial image processing
- **Matplotlib**: Visualization
- **NumPy**: Numerical computations

## 📈 Results

The system successfully detects and classifies buildings with:
- Precise polygon-based annotations
- Geographic coordinate preservation
- High confidence classifications
- Visual overlays for verification

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with PyTorch and OpenCV
- Trained on manually annotated aerial imagery
- Optimized for commercial building detection

## 📞 Support

For questions or issues, please open an issue on GitHub or contact the development team.

---

**Note**: This system is specifically optimized for detecting commercial buildings in aerial TIF images with geographic metadata.