"""
visualize_patches.py

Usage:
    python visualize_patches.py <image_path> <patch_size> <image_size> <patch_numbers>

    image_path    : path to input image
    patch_size    : size of each square patch (e.g. 32)
    image_size    : resize image to this square size before patching (e.g. 224)
    patch_numbers : comma-separated 1-indexed patch numbers to highlight (e.g. 1,5,12)

Example:
    python visualize_patches.py /home/anas/datasets/tile.png 32 224 1,5,12
"""

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image


def load_and_resize(image_path: str, image_size: int) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    img = img.resize((image_size, image_size), Image.BILINEAR)
    return np.array(img)


def visualize_patches(
    image_path: str,
    patch_size: int = 32,
    image_size: int = 256,
    highlight: list[int] = None,  # 1-indexed
    border_color: str = "red",
    border_linewidth: float = 1.5,
    label_patches: bool = True,
):
    """
    Displays the resized image with patch grid, highlighting only the
    patches whose 1-indexed numbers are in `highlight`.

    Patch numbering: row-major, upper-left is patch 1.
    """
    img = load_and_resize(image_path, image_size)

    n_patches_per_side = image_size // patch_size  # assumes image_size divisible by patch_size
    total_patches = n_patches_per_side ** 2

    if highlight is None:
        highlight = []

    # Validate
    invalid = [p for p in highlight if p < 1 or p > total_patches]
    if invalid:
        raise ValueError(
            f"Patch numbers out of range {invalid}. "
            f"Valid range: 1–{total_patches} for a {n_patches_per_side}×{n_patches_per_side} grid."
        )

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.imshow(img)
    ax.set_title(
        f"{image_size}×{image_size}px  |  patch size {patch_size}×{patch_size}  |  "
        f"{n_patches_per_side}×{n_patches_per_side} = {total_patches} patches\n"
        f"Highlighted: {sorted(highlight)}",
        fontsize=9,
    )
    ax.axis("off")

    highlight_set = set(highlight)

    for idx in range(total_patches):
        patch_num = idx + 1  # 1-indexed
        row = idx // n_patches_per_side
        col = idx % n_patches_per_side

        x = col * patch_size   # left edge in pixels
        y = row * patch_size   # top edge in pixels

        if patch_num in highlight_set:
            rect = mpatches.Rectangle(
                (x - 0.5, y - 0.5),          # -0.5 for pixel-center alignment
                patch_size,
                patch_size,
                linewidth=border_linewidth,
                edgecolor=border_color,
                facecolor="none",
            )
            ax.add_patch(rect)

            if label_patches:
                ax.text(
                    x + patch_size / 2,
                    y + patch_size / 2,
                    str(patch_num),
                    color=border_color,
                    fontsize=7,
                    ha="center",
                    va="center",
                    fontweight="bold",
                )

    plt.tight_layout()
    output_path = image_path.rsplit(".", 1)[0] + "_patches_highlighted.png"
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    print(f"Saved to {output_path}")
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize selected patches on an image.")
    parser.add_argument("image_path", type=str, help="Path to input image")
    parser.add_argument("patch_size", type=int, help="Patch size in pixels (e.g. 32)")
    parser.add_argument("image_size", type=int, help="Resize image to this square size (e.g. 224)")
    parser.add_argument(
        "patch_numbers",
        type=str,
        help="Comma-separated 1-indexed patch numbers to highlight (e.g. 1,5,12)",
    )
    parser.add_argument(
        "--color", type=str, default="red", help="Border color (default: red)"
    )
    parser.add_argument(
        "--lw", type=float, default=1.5, help="Border line width (default: 1.5)"
    )
    parser.add_argument(
        "--no-labels", action="store_true", help="Suppress patch number labels"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    patch_nums = [int(p.strip()) for p in args.patch_numbers.split(",")]

    visualize_patches(
        image_path=args.image_path,
        patch_size=args.patch_size,
        image_size=args.image_size,
        highlight=patch_nums,
        border_color=args.color,
        border_linewidth=args.lw,
        label_patches=not args.no_labels,
    )
