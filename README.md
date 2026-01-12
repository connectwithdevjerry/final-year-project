# 📊 Spatio-Temporal Urban Monitoring Dashboard (OAU, Ile-Ife)

This dashboard is a **web-based geospatial visualization and analysis system** developed to monitor **urban growth dynamics** within **Obafemi Awolowo University (OAU) campus and its environs, Ile-Ife, Nigeria**, from **2019 to October 2025**.

The system integrates **deep learning–based building extraction**, satellite imagery, drone data, and volunteered geographic information to support **evidence-based urban planning and management**.

---

## 🎯 Purpose of the Dashboard

The dashboard was designed to:
- Visualize **spatio-temporal urban expansion**
- Support **real-time monitoring** of building growth
- Enable **spatial pattern analysis** for planners and researchers
- Serve as a scalable prototype for urban monitoring in **resource-constrained environments**

---

## 🧩 Core Dashboard Capabilities

### 1. Spatio-Temporal Building Growth Visualization
- Interactive map displaying extracted buildings across multiple years (2019–2025)
- Temporal comparison of urban expansion patterns
- Visual differentiation of building footprints by time period

---

### 2. Automated Building Extraction Outputs
- Displays results from **DeepLabV3 semantic segmentation**
- Supports raster-based building masks derived from:
  - Sentinel-2 imagery
  - High-resolution drone imagery (5 cm/pixel)
- Noise reduction applied using a **guided filter** for improved footprint clarity

---

### 3. Spatial Pattern Analysis
- Building density mapping
- Spatial autocorrelation analysis using **Moran’s I (0.45)** to identify clustering
- Hotspot visualization for high-growth zones within and around OAU

---

### 4. OSM-Based Reference and Validation Layer
- Integration of **OpenStreetMap (OSM)** building footprints
- Temporal comparison of OSM building counts:
  - 2019: 2,194 buildings
  - 2025: 18,043 buildings
- Used as a fallback and reference dataset where model outputs were uncertain

---

### 5. Metrics & Accuracy Dashboard
The dashboard presents model validation statistics, including:
- Intersection over Union (IoU):
  - 50% (pre-processing)
  - 40% (post-processing)
- Precision: 70%
- Recall: 68%

These metrics provide transparency on model performance and reliability.

---

## 🗺 Data Sources

### Satellite & Aerial Data
- Sentinel-2 multispectral imagery
- Drone imagery:
  - 145 images
  - 5 cm spatial resolution
  - Acquired: August 22, 2024

### Ancillary Data
- OpenStreetMap (OSM) building footprints
- Administrative and campus boundary data

---

## 🏗 System Architecture

1. **Image Processing & Analysis**
   - DeepLabV3 for building extraction
   - Guided filtering for noise reduction
   - Spatial metrics computation

2. **Backend**
   - Django framework
   - Handles spatial data ingestion, processing, and storage

3. **Frontend Visualization**
   - OpenLayers for interactive mapping
   - Temporal layer toggling and spatial querying
   - Attribute tables and summary statistics

---

## 📈 Key Findings Visualized in the Dashboard

- **722% increase in building count** between 2019 and 2025
- Clear clustering of new developments around campus peripheries
- Predictive indicators suggest **continued urban expansion**, increasing pressure on infrastructure and land management systems

---

## ⚠ Limitations Reflected in the Dashboard

- **Small training dataset** limited DeepLabV3 generalization, especially in dense urban areas with overlapping rooftops
- OSM data quality varied due to **inconsistent volunteer mapping**, affecting early-year reliability
- Hardware constraints slowed processing of large GeoTIFF files
- Trade-off observed between:
  - DeepLabV3 (efficient, less accurate in dense zones)
  - Mask R-CNN (more accurate, computationally expensive)

These limitations are transparently documented to guide interpretation of dashboard outputs.

---

## 🔮 Planned Improvements

- Expansion of training datasets for better model generalization
- Integration of higher-resolution imagery (e.g., Landsat, commercial data)
- GPU-enabled processing for large-scale uploads
- Improved temporal prediction models for urban growth forecasting
- Enhanced dashboard analytics for planners and policymakers

---

## 👤 Author

**Adedeji Jeremiah**  
Surveying & Geoinformatics  
Geospatial & Web Systems Developer  

GitHub: https://github.com/connectwithdevjerry
