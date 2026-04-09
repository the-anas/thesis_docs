# seg_snow.py
# pip install ultralytics opencv-python matplotlib numpy

from ultralytics import SAM
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import random

# ── config ────────────────────────────────────────────────────────────────────
IMAGE_PATH   = "snow1.png"          # your local image
OUTPUT_PATH  = "snow1_segmented.png"

# Model options (auto-downloaded on first run):
#   "sam2.1_t.pt"  ~38  MB  ← lightest, plenty for 264px
#   "sam2.1_s.pt"  ~91  MB
#   "sam2.1_b+.pt" ~163 MB
#   "sam2.1_l.pt"  ~898 MB  ← heaviest / most accurate
MODEL        = "sam2.1_b+.pt"      # good balance for your RTX 3060
# ──────────────────────────────────────────────────────────────────────────────

def overlay_masks(image_bgr, result, alpha=0.45):
    """Draw semi-transparent colored masks + contours over the original image."""
    h, w = image_bgr.shape[:2]
    overlay = image_bgr.copy().astype(np.float32)
    legend_items = []

    if result.masks is None:
        print("No masks returned.")
        return image_bgr, []

    masks = result.masks.data.cpu().numpy()   # [N, H, W]  boolean
    N = len(masks)
    print(f"  → {N} segments found")

    # generate N distinct colours
    rng = random.Random(42)
    colours = [
        tuple(rng.randint(60, 255) for _ in range(3))
        for _ in range(N)
    ]

    for i, (mask, colour) in enumerate(zip(masks, colours)):
        mask_bool = mask.astype(bool)

        # coloured fill
        coloured = np.zeros_like(image_bgr, dtype=np.float32)
        coloured[mask_bool] = colour
        overlay[mask_bool] = (
            overlay[mask_bool] * (1 - alpha) + coloured[mask_bool] * alpha
        )

        # contour
        mask_u8 = mask_bool.astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay.astype(np.uint8), contours, -1,
                         colour, thickness=1)

        area_pct = mask_bool.sum() / (h * w) * 100
        legend_items.append((colour, f"Segment {i+1}  ({area_pct:.1f}%)"))

    return np.clip(overlay, 0, 255).astype(np.uint8), legend_items


def build_figure(image_bgr, overlay_bgr, legend_items, out_path):
    img_rgb     = cv2.cvtColor(image_bgr,  cv2.COLOR_BGR2RGB)
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].imshow(img_rgb);     axes[0].set_title("Original");           axes[0].axis("off")
    axes[1].imshow(overlay_rgb); axes[1].set_title("SAM 2.1 Segments");   axes[1].axis("off")

    patches = [
        mpatches.Patch(
            facecolor=tuple(c / 255 for c in colour),
            edgecolor="white", linewidth=0.5,
            label=label
        )
        for colour, label in legend_items
    ]
    if patches:
        axes[1].legend(
            handles=patches, loc="lower left",
            fontsize=6, framealpha=0.7,
            ncol=max(1, len(patches) // 10)
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out_path}")


def main():
    image_bgr = cv2.imread(IMAGE_PATH)
    if image_bgr is None:
        raise FileNotFoundError(f"Cannot open: {IMAGE_PATH}")
    print(f"Image: {image_bgr.shape[1]}×{image_bgr.shape[0]} px")

    model = SAM(MODEL)
    print(f"Model loaded: {MODEL}")

    # ── automatic everything segmentation (no prompts) ──────────────────────
    print("Running automatic mask generation …")
    results = model(IMAGE_PATH)
    result  = results[0]

    overlay, legend = overlay_masks(image_bgr, result)
    build_figure(image_bgr, overlay, legend, OUTPUT_PATH)

    # ── optional: prompt with a bounding box ────────────────────────────────
    # If auto-seg misses something, you can prompt explicitly:
    #   results = model(IMAGE_PATH, bboxes=[[x1, y1, x2, y2]])
    # Or with a point (foreground label=1 / background label=0):
    #   results = model(IMAGE_PATH, points=[[132, 132]], labels=[1])


if __name__ == "__main__":
    main()