"""
samgeo_point_prompt.py
Segments a specific object by pointing at it with pixel coordinates.
Install: pip install "segment-geospatial[samgeo]" matplotlib pillow
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from samgeo import SamGeo

# ── Config ───────────────────────────────────────────────────────────────────
IMAGE_PATH = "/home/anas/from_cluster/the_newer_version_models/snow1_reconstructed_24k.png/snow1_bahdanau-hyperprior-v2_checkpoint_best_loss_bahdanau_v2_k24_cluster_64_96_24.pth_recon.png"       # swap in your own image
MASK_OUT   = "mask_point.tif"

MODEL_TYPE = "vit_b"

# Each entry in POINT_COORDS is [col, row] (x, y in pixel space).
# Label 1 = foreground (include), 0 = background (exclude).
POINT_COORDS = [
    [200, 200],   # foreground: object you want
    # [350, 200],   # background: region to exclude (remove if not needed)
]
POINT_LABELS = [1, 0]
# ─────────────────────────────────────────────────────────────────────────────


def visualize(image_path, mask_path, point_coords, point_labels):
    img  = np.array(Image.open(image_path).convert("RGB"))
    mask = np.array(Image.open(mask_path))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img)
    axes[0].set_title("Original + prompts")
    axes[0].axis("off")
    # draw the prompt points
    for (col, row), label in zip(point_coords, point_labels):
        color = "lime" if label == 1 else "red"
        marker = "*" if label == 1 else "x"
        axes[0].plot(col, row, marker=marker, color=color, markersize=12, markeredgewidth=2)

    axes[1].imshow(mask, cmap="Blues");  axes[1].set_title("Mask");   axes[1].axis("off")
    axes[2].imshow(img)
    axes[2].imshow(mask, cmap="Blues", alpha=0.5)
    axes[2].set_title("Overlay");        axes[2].axis("off")

    plt.tight_layout()
    plt.savefig("result_point_prompt.png", dpi=150)
    print("Saved → result_point_prompt.png")
    plt.show()


if __name__ == "__main__":
    sam = SamGeo(model_type=MODEL_TYPE, automatic=False)
    sam.set_image(IMAGE_PATH)

    sam.predict(
        point_coords=POINT_COORDS,
        point_labels=POINT_LABELS,
        output=MASK_OUT,
    )

    print(f"Mask → {MASK_OUT}")
    visualize(IMAGE_PATH, MASK_OUT, POINT_COORDS, POINT_LABELS)