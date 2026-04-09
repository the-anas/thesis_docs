# HERE IS SOLUTION 
from ultralytics import FastSAM
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random

IMAGE_PATH = "/home/anas/datasets/exp1_photos/sandy2.png"
OUTPUT_PATH = "/home/anas/datasets/exp1_photos/sandy1_fastsam.png"

model = FastSAM("FastSAM-s.pt")  # FastSAM-x.pt for better quality

# "everything" mode — segments all regions automatically, no prompts
results = model(
    IMAGE_PATH,
    device="cpu",
    retina_masks=True,   # full-res masks instead of downscaled
    imgsz=264,
    conf=0.3,
    iou=0.85,
)

result = results[0]

# ── extract masks ─────────────────────────────────────────────────────────────
image_bgr = cv2.imread(IMAGE_PATH)
h, w = image_bgr.shape[:2]
overlay = image_bgr.copy().astype(np.float32)
legend_items = []

if result.masks is not None:
    masks = result.masks.data.cpu().numpy()   # [N, H, W]
    N = len(masks)
    print(f"{N} segments found")

    rng = random.Random(42)
    colours = [tuple(rng.randint(60, 255) for _ in range(3)) for _ in range(N)]

    for mask, colour in zip(masks, colours):
        mask_bool = mask.astype(bool)
        coloured = np.zeros_like(image_bgr, dtype=np.float32)
        coloured[mask_bool] = colour
        overlay[mask_bool] = overlay[mask_bool] * 0.45 + coloured[mask_bool] * 0.55

        area_pct = mask_bool.sum() / (h * w) * 100
        legend_items.append((colour, f"{area_pct:.1f}%"))
else:
    print("No masks found — try lowering conf threshold")

overlay = np.clip(overlay, 0, 255).astype(np.uint8)

# ── plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
axes[0].set_title("Original"); axes[0].axis("off")

axes[1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
axes[1].set_title("FastSAM Segments"); axes[1].axis("off")

patches = [
    mpatches.Patch(facecolor=tuple(c/255 for c in col), label=lbl)
    for col, lbl in legend_items
]
if patches:
    axes[1].legend(handles=patches, loc="lower left", fontsize=6,
                   framealpha=0.7, ncol=max(1, len(patches)//8))

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=180, bbox_inches="tight")
print(f"Saved → {OUTPUT_PATH}")




# ##### for edges only, no predictions
# from skimage.segmentation import felzenszwalb, slic, mark_boundaries
# from skimage.io import imread
# import matplotlib.pyplot as plt

# img = imread("snow1.png")

# # Felzenszwalb: good for images with clear region boundaries (your ice cracks)
# segments = felzenszwalb(img, scale=150, sigma=0.5, min_size=200)

# # OR SLIC (superpixels): more uniform segments
# # segments = slic(img, n_segments=20, compactness=10, sigma=1)

# print(f"{segments.max() + 1} segments")

# fig, axes = plt.subplots(1, 2, figsize=(10, 5))
# axes[0].imshow(img);                         axes[0].set_title("Original")
# axes[1].imshow(mark_boundaries(img, segments)); axes[1].set_title("Segments")
# plt.savefig("segmented.png", dpi=150)


# => I think these two models are good enough right here, just use these