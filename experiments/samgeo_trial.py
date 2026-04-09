from samgeo import SamGeo

# Heavy model: vit_h (~2.4GB). For lighter: vit_l or vit_b
sam = SamGeo(
    model_type="vit_h",
    automatic=True,
    device="cuda",
)

img_path = "/home/anas/datasets/exp1_photos/snow1.png"   # GeoTIFF or JPG/PNG
output_mask = "segmentation_mask.tif"

# Auto-generate all masks (no prompt needed)
sam.generate(img_path, output_mask)

# Optional: convert to vector polygons
sam.tiff_to_vector(output_mask, "segmentation.gpkg")

# --- Alternatively: prompt with a bounding box ---
sam_prompted = SamGeo(model_type="vit_h", automatic=False, device="cuda")
sam_prompted.set_image(img_path)

# box = [x_min, y_min, x_max, y_max] in pixel coords
masks, scores, _ = sam_prompted.predict(
    point_coords=None,
    box=[100, 200, 400, 500],
    multimask_output=True
)


