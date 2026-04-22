"""
patch_cosine_similarity.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────

IMAGE_PATH        = "/home/anas/datasets/exp1_clean/mix1_256.png"
PATCH_SIZE        = 32
IMAGE_SIZE        = 256
QUERY_PATCH_NUM   =  15       # 1-indexed
CMAP              = "RdYlBu_r"

# ─────────────────────────────────────────────────────────────────────────────


def load_and_resize(image_path: str, image_size: int) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    img = img.resize((image_size, image_size), Image.BILINEAR)
    return np.array(img)


def extract_patches(img: np.ndarray, patch_size: int) -> np.ndarray:
    H, W, C = img.shape
    n = H // patch_size
    patches = []
    for row in range(n):
        for col in range(n):
            patch = img[
                row * patch_size : (row + 1) * patch_size,
                col * patch_size : (col + 1) * patch_size,
                :
            ]
            patches.append(patch.flatten().astype(np.float32))
    return np.array(patches)  # (N, D)


def cosine_similarity(query: np.ndarray, all_patches: np.ndarray) -> np.ndarray:
    query_norm = query / (np.linalg.norm(query) + 1e-8)
    norms      = np.linalg.norm(all_patches, axis=1, keepdims=True) + 1e-8
    normed     = all_patches / norms
    return normed @ query_norm  # (N,)


def save_figure(fig, base_path: str, suffix: str):
    out = base_path.rsplit(".", 1)[0] + suffix + ".png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    print(f"Saved to {out}")
    plt.close(fig)


def visualize_similarity_heatmap(
    image_path: str,
    patch_size: int,
    image_size: int,
    query_patch_num: int,
    cmap: str = "RdYlBu_r",
):
    img = load_and_resize(image_path, image_size)

    n_per_side    = image_size // patch_size
    total_patches = n_per_side ** 2

    if query_patch_num < 1 or query_patch_num > total_patches:
        raise ValueError(
            f"Query patch {query_patch_num} out of range. "
            f"Valid range: 1–{total_patches} for a {n_per_side}×{n_per_side} grid."
        )

    patches   = extract_patches(img, patch_size)
    query_vec = patches[query_patch_num - 1]
    sims      = cosine_similarity(query_vec, patches)
    sim_grid  = sims.reshape(n_per_side, n_per_side)
    sim_full  = np.repeat(np.repeat(sim_grid, patch_size, axis=0), patch_size, axis=1)

    q_idx = query_patch_num - 1
    q_row = q_idx // n_per_side
    q_col = q_idx  % n_per_side
    q_x   = q_col * patch_size
    q_y   = q_row * patch_size

    def query_rect():
        return mpatches.Rectangle(
            (q_x - 0.5, q_y - 0.5), patch_size, patch_size,
            linewidth=2, edgecolor="cyan", facecolor="none",
        )

    stats = (
        f"{image_size}×{image_size}px  |  patch {patch_size}×{patch_size}  |  "
        f"{n_per_side}×{n_per_side} grid  |  query patch {query_patch_num}\n"
        f"min={sims.min():.3f}  max={sims.max():.3f}  "
        f"mean={sims.mean():.3f}  std={sims.std():.3f}"
    )

    # ── 1. original with query patch marked ──────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    ax1.imshow(img)
    ax1.add_patch(query_rect())
    ax1.text(
        q_x + patch_size / 2, q_y + patch_size / 2, str(query_patch_num),
        color="cyan", fontsize=8, ha="center", va="center", fontweight="bold",
    )
    # ax1.set_title(f"Query patch: {query_patch_num}", fontsize=10)
    ax1.axis("off")
    # fig1.suptitle(stats, fontsize=8)
    plt.tight_layout()
    save_figure(fig1, image_path, f"_patch{query_patch_num}_original")

    # ── 2. heatmap only ───────────────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    im = ax2.imshow(sim_full, cmap=cmap, vmin=0, vmax=1)
    ax2.add_patch(query_rect())
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04, )#label="Cosine similarity"
    # ax2.set_title("Cosine similarity heatmap", fontsize=10)
    ax2.axis("off")
    # fig2.suptitle(stats, fontsize=8)
    plt.tight_layout()
    save_figure(fig2, image_path, f"_patch{query_patch_num}_heatmap")

    # ── 3. overlay ────────────────────────────────────────────────────────────
    fig3, ax3 = plt.subplots(figsize=(6, 6))
    ax3.imshow(img)
    ax3.imshow(sim_full, cmap=cmap, vmin=0, vmax=1, alpha=0.55)
    ax3.add_patch(query_rect())
    # ax3.set_title("Overlay", fontsize=10)
    ax3.axis("off")
    # fig3.suptitle(stats, fontsize=8)
    plt.tight_layout()
    save_figure(fig3, image_path, f"_patch{query_patch_num}_overlay")


if __name__ == "__main__":
    visualize_similarity_heatmap(
        image_path=IMAGE_PATH,
        patch_size=PATCH_SIZE,
        image_size=IMAGE_SIZE,
        query_patch_num=QUERY_PATCH_NUM,
        cmap=CMAP,
    )