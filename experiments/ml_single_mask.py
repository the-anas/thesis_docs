import random
from ultralytics import FastSAM
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ORIGINAL_PATH   = "/home/anas/datasets/exp1/256/field1_256.png"
COMPRESSED_PATH = "/home/anas/datasets/exp1/recon_K192/field1_recon.png"
OUTPUT_PATH     = "/home/anas/datasets/exp1/field1_global_iou.png"

model = FastSAM("FastSAM-s.pt")

def run_fastsam(image_path):
    results = model(
        image_path,
        device="cpu",
        retina_masks=True,
        imgsz=264,
        conf=0.3,
        iou=0.85,
    )
    result = results[0]
    if result.masks is not None:
        return result.masks.data.cpu().numpy()
    return np.array([])

def collapse_masks(masks):
    if len(masks) == 0:
        return None
    return masks.any(axis=0).astype(bool)

def compute_iou(a, b):
    intersection = (a & b).sum()
    union        = (a | b).sum()
    return intersection / union if union > 0 else 0.0

def build_overlay(image_bgr, masks):
    overlay = image_bgr.copy().astype(np.float32)
    rng = random.Random(42)
    for mask in masks:
        colour = tuple(rng.randint(60, 255) for _ in range(3))
        mask_bool = mask.astype(bool)
        coloured = np.zeros_like(image_bgr, dtype=np.float32)
        coloured[mask_bool] = colour
        overlay[mask_bool] = overlay[mask_bool] * 0.45 + coloured[mask_bool] * 0.55
    return np.clip(overlay, 0, 255).astype(np.uint8)

# ── run model ─────────────────────────────────────────────────────────────────
masks_orig = run_fastsam(ORIGINAL_PATH)
masks_comp = run_fastsam(COMPRESSED_PATH)

image_orig = cv2.imread(ORIGINAL_PATH)
image_comp = cv2.imread(COMPRESSED_PATH)
h, w = image_orig.shape[:2]

# ── overlays ──────────────────────────────────────────────────────────────────
overlay_orig = build_overlay(image_orig, masks_orig)
overlay_comp = build_overlay(image_comp, masks_comp)

# ── coverage iou ──────────────────────────────────────────────────────────────
cov_orig = collapse_masks(masks_orig)
cov_comp = collapse_masks(masks_comp)
coverage_iou = compute_iou(cov_orig, cov_comp)

# ── diff ──────────────────────────────────────────────────────────────────────
diff = np.zeros((h, w, 3), dtype=np.uint8)
diff[cov_orig & ~cov_comp] = (0,   0, 255)
diff[~cov_orig & cov_comp] = (0, 255,   0)

print(f"Original segments:   {len(masks_orig)}")
print(f"Compressed segments: {len(masks_comp)}")
print(f"Coverage IoU:        {coverage_iou:.4f}")

# ── plot ──────────────────────────────────────────────────────────────────────
base = OUTPUT_PATH.replace(".png", "")

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.imshow(cv2.cvtColor(overlay_orig, cv2.COLOR_BGR2RGB))
# ax.set_title(f"Original ({len(masks_orig)} segments)")
ax.axis("off")
plt.tight_layout()
plt.savefig(f"{base}_original.png", dpi=180, bbox_inches="tight")
plt.close()

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.imshow(cv2.cvtColor(overlay_comp, cv2.COLOR_BGR2RGB))
# ax.set_title(f"Reconstructed ({len(masks_comp)} segments)")
ax.axis("off")
plt.tight_layout()
plt.savefig(f"{base}_reconstructed.png", dpi=180, bbox_inches="tight")
plt.close()

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.imshow(cv2.cvtColor(diff, cv2.COLOR_BGR2RGB))
ax.set_title("")
ax.axis("off")
red_patch   = mpatches.Patch(color=(1, 0, 0), label="Lost after compression")
green_patch = mpatches.Patch(color=(0, 1, 0), label="Gained after compression")
ax.legend(handles=[red_patch, green_patch], loc="lower left", fontsize=7, framealpha=0.7)
plt.tight_layout()
plt.savefig(f"{base}_diff.png", dpi=180, bbox_inches="tight")
plt.close()

print(f"Saved → {base}_original.png")
print(f"Saved → {base}_reconstructed.png")
print(f"Saved → {base}_diff.png")