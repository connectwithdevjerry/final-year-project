import torch
import numpy as np
import rasterio
import cv2
import geopandas as gpd
from rasterio.features import shapes
import segmentation_models_pytorch as smp
import json
import os
from django.conf import settings

def run_building_extraction(img_path=None):

    # -----------------------------
    # Load Orthophoto
    # -----------------------------
    # img_path = r"C:\Users\USER\Documents\obaloluwa\orthomosaic.tif"  # Change this path
    with rasterio.open(img_path) as src:
        img = src.read([1, 2, 3])  # use RGB bands only
        transform = src.transform
        crs = src.crs

    # Reshape to HWC format
    img = np.moveaxis(img, 0, -1)

    # -----------------------------
    # Preprocess for Model
    # -----------------------------
    inp = cv2.resize(img, (256, 256))   # resize for model input
    inp = inp.astype(np.float32) / 255.0
    inp = np.transpose(inp, (2, 0, 1))  # CHW format
    inp = torch.from_numpy(inp).unsqueeze(0)  # add batch

    # -----------------------------
    # Load Pretrained U-Net
    # -----------------------------
    model = smp.Unet(
        encoder_name="resnet34",        # backbone
        encoder_weights="imagenet",          # no ImageNet weights
        in_channels=3,
        classes=1,                      # 1 class = building
        activation=None
    )

    weights_path = os.path.join(settings.BASE_DIR, "buildingExtractor", "ml-model", "data", "models", "building_classifier.pth")

    model.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=False)
    model.eval()

    # -----------------------------
    # Run Prediction
    # -----------------------------
    with torch.no_grad():
        pr_mask = model(inp)
        pr_mask = torch.sigmoid(pr_mask).squeeze().cpu().numpy()

    # Threshold → binary mask
    mask = (pr_mask > 0.5).astype(np.uint8)

    # Resize back to original image size
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    # -----------------------------
    # Convert Mask → Polygons
    # -----------------------------
    results = (
        {"properties": {"raster_val": int(v)}, "geometry": s}
        for i, (s, v) in enumerate(shapes(mask, mask > 0, transform=transform))
    )

    # Make GeoDataFrame
    gdf = gpd.GeoDataFrame.from_features(list(results), crs=crs)

    # Save polygons to GeoJSON
    # out_file = r"C:\Users\USER\Documents\obaloluwa\buildings.geojson"
    # gdf.to_file(out_file, driver="GeoJSON")

    # print("Buildings extracted and saved to:", out_file)
    # print("Number of buildings detected:", len(gdf))

    gdf = gdf.to_crs(epsg=4326)  # convert to WGS84 for web maps

    return json.loads(gdf.to_json())  # GeoJSON dict
