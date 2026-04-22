from ultralytics import FastSAM
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random

ORIGINAL_PATH = "/home/anas/datasets/exp1/256/green2_256.png"
COMPRESSED_PATH = "/home/anas/datasets/exp1/recon_K192/green2_recon.png"
OUTPUT_PATH = "/home/anas/datasets/exp1/green2_iou.png"

IOU_MATCH_THRESHOLD = 0.5

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
        return result.masks.data.cpu().numpy()  # [N, H, W]
    return np.array([])

def build_overlay(image_bgr, masks, colours):
    overlay = image_bgr.copy().astype(np.float32)
    for mask, colour in zip(masks, colours):
        mask_bool = mask.astype(bool)
        coloured = np.zeros_like(image_bgr, dtype=np.float32)
        coloured[mask_bool] = colour
        overlay[mask_bool] = overlay[mask_bool] * 0.45 + coloured[mask_bool] * 0.55
    return np.clip(overlay, 0, 255).astype(np.uint8)

def compute_iou(mask_a, mask_b):
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    intersection = (a & b).sum()
    union = (a | b).sum()
    return intersection / union if union > 0 else 0.0

def greedy_match(masks_orig, masks_comp, threshold=IOU_MATCH_THRESHOLD):
    """
    For each mask in orig, find the best IoU partner in comp.
    Returns list of (orig_idx, comp_idx, iou) for matched pairs,
    and sets of unmatched indices on each side.
    """
    matched = []
    used_comp = set()

    for i, mo in enumerate(masks_orig):
        best_iou = 0.0
        best_j = -1
        for j, mc in enumerate(masks_comp):
            if j in used_comp:
                continue
            iou = compute_iou(mo, mc)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_iou >= threshold and best_j != -1:
            matched.append((i, best_j, best_iou))
            used_comp.add(best_j)

    unmatched_orig = set(range(len(masks_orig))) - {m[0] for m in matched}
    unmatched_comp = set(range(len(masks_comp))) - {m[1] for m in matched}
    return matched, unmatched_orig, unmatched_comp

# ── run model on both images ───────────────────────────────────────────────────
masks_orig = run_fastsam(ORIGINAL_PATH)
masks_comp = run_fastsam(COMPRESSED_PATH)

print(f"Original:   {len(masks_orig)} segments")
print(f"Compressed: {len(masks_comp)} segments")
print(f"Segment count delta: {abs(len(masks_orig) - len(masks_comp))}")

# ── match and compute IoU ──────────────────────────────────────────────────────
matched, unmatched_orig, unmatched_comp = greedy_match(masks_orig, masks_comp)

ious = [iou for _, _, iou in matched]
mean_iou = np.mean(ious) if ious else 0.0
unmatched_rate = len(unmatched_orig) / len(masks_orig) if len(masks_orig) > 0 else 0.0

print(f"\nMatched pairs:    {len(matched)}")
print(f"Mean IoU:         {mean_iou:.4f}")
print(f"Unmatched (orig): {len(unmatched_orig)} ({unmatched_rate*100:.1f}% lost)")
print(f"Unmatched (comp): {len(unmatched_comp)} (new/split regions)")

# ── build overlays ────────────────────────────────────────────────────────────
image_orig = cv2.imread(ORIGINAL_PATH)
image_comp = cv2.imread(COMPRESSED_PATH)
h, w = image_orig.shape[:2]

rng = random.Random(42)

# shared colours for matched pairs so visually matched regions look the same
n_orig = len(masks_orig)
n_comp = len(masks_comp)
colours_orig = [tuple(rng.randint(60, 255) for _ in range(3)) for _ in range(n_orig)]
colours_comp = [None] * n_comp

for i, j, _ in matched:
    colours_comp[j] = colours_orig[i]  # same colour = same matched region

# unmatched comp masks get a distinct red tint to flag new/split regions
for j in unmatched_comp:
    colours_comp[j] = (60, 60, 220)

overlay_orig = build_overlay(image_orig, masks_orig, colours_orig)
overlay_comp = build_overlay(image_comp, masks_comp, colours_comp)

# ── diff mask: pixels where matched masks disagree ────────────────────────────
diff = np.zeros((h, w, 3), dtype=np.uint8)
for i, j, iou in matched:
    a = masks_orig[i].astype(bool)
    b = masks_comp[j].astype(bool)
    lost = a & ~b        # in orig but not comp
    gained = ~a & b      # in comp but not orig
    diff[lost] = (0, 0, 255)    # red  = lost after compression
    diff[gained] = (0, 255, 0)  # green = gained after compression

# ── plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(cv2.cvtColor(overlay_orig, cv2.COLOR_BGR2RGB))
axes[0].set_title(f"Original ({len(masks_orig)} segments)")
axes[0].axis("off")

axes[1].imshow(cv2.cvtColor(overlay_comp, cv2.COLOR_BGR2RGB))
axes[1].set_title(f"Compressed ({len(masks_comp)} segments)")
axes[1].axis("off")

axes[2].imshow(cv2.cvtColor(diff, cv2.COLOR_BGR2RGB))
axes[2].set_title(f"Diff  |  Mean IoU: {mean_iou:.3f}  |  Lost: {len(unmatched_orig)}")
axes[2].axis("off")

red_patch = mpatches.Patch(color=(1, 0, 0), label="Lost after compression")
green_patch = mpatches.Patch(color=(0, 1, 0), label="Gained after compression")
axes[2].legend(handles=[red_patch, green_patch], loc="lower left", fontsize=7, framealpha=0.7)

plt.suptitle(
    f"FastSAM Segmentation Consistency  |  "
    f"Matched: {len(matched)}  Unmatched rate: {unmatched_rate*100:.1f}%",
    fontsize=10
)
plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=180, bbox_inches="tight")
print(f"\nSaved → {OUTPUT_PATH}")