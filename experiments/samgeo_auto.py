"""
samgeo_automatic.py
Segments everything in an image automatically.
Install: pip install "segment-geospatial[samgeo]" matplotlib pillow
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from samgeo import SamGeo

# ── Config ───────────────────────────────────────────────────────────────────
IMAGE_PATH = "/home/anas/datasets/exp1/field1.png"       # swap in your own image
MASKS_OUT  = "masks_auto.png"
ANNS_OUT   = "annotations.png"

MODEL_TYPE = "vit_b"           # vit_b (fast) | vit_l | vit_h (best quality)
# ─────────────────────────────────────────────────────────────────────────────


def visualize(image_path, masks_path):
    img  = np.array(Image.open(image_path).convert("RGB"))
    mask = np.array(Image.open(masks_path))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img);                      axes[0].set_title("Original");   axes[0].axis("off")
    axes[1].imshow(mask, cmap="tab20");       axes[1].set_title("Masks");      axes[1].axis("off")
    axes[2].imshow(img)
    axes[2].imshow(mask, cmap="tab20", alpha=0.5); axes[2].set_title("Overlay"); axes[2].axis("off")

    plt.tight_layout()
    plt.savefig("result_automatic.png", dpi=150)
    print("Saved → result_automatic.png")
    plt.show()


if __name__ == "__main__":
    sam = SamGeo(
        model_type=MODEL_TYPE,
        automatic=True,
        sam_kwargs={
            "points_per_side": 32,           # increase for denser segmentation
            "pred_iou_thresh": 0.86,
            "stability_score_thresh": 0.92,
            "min_mask_region_area": 100,     # drop blobs smaller than 100 px²
        },
    )

    sam.generate(
        source=IMAGE_PATH,
        output=MASKS_OUT,
        foreground=True,   # filter background regions
        unique=True,       # each object gets a unique integer ID
    )

    # save a coloured annotation image
    sam.show_anns(axis="off", alpha=0.6, output=ANNS_OUT)
    print(f"Masks       → {MASKS_OUT}")
    print(f"Annotations → {ANNS_OUT}")

    visualize(IMAGE_PATH, MASKS_OUT)