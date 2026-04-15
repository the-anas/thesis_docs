"""
reconstruct.py

Reconstruct an image using a trained compression model and save the result.

Usage:
    python reconstruct.py \
        --image /path/to/image.png \
        --arch bmshj2018-hyperprior-bahdanau \
        --checkpoint /path/to/checkpoint_128_192_32.pth.tar \
        --output /path/to/output_dir \
        --image-size 256 \
        --patch-size 16

Supported archs:
    bmshj2018-hyperprior
    bmshj2018-hyperprior-bahdanau
    bmshj2018-hyperprior-crossattention
"""

import argparse
import os
import re

import torch
import torch.nn.functional as F
from pathlib import Path

from new_utils import load_image, save_tensor_as_image
from loader import models_dict   # your existing loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(arch: str, checkpoint_path: str) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)

    if all(k.startswith("module.") for k in state_dict):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}

    # parse N, M, K from filename  e.g. _128_192_32.pth.tar
    match = re.search(r'_(\d+)_(\d+)_(\d+)\.pth\.tar$', checkpoint_path)
    if match:
        N, M, K = map(int, match.groups())
        print(f"Parsed N={N}, M={M}, K={K} from filename")
    else:
        # fallback for bmshj2018-hyperprior which only needs N, M
        match2 = re.search(r'_(\d+)_(\d+)\.pth\.tar$', checkpoint_path)
        if match2:
            N, M = map(int, match2.groups())
            K = None
            print(f"Parsed N={N}, M={M} from filename")
        else:
            raise ValueError(
                "Could not parse N, M[, K] from checkpoint filename. "
                "Expected format: ..._N_M_K.pth.tar or ..._N_M.pth.tar"
            )

    if arch == "bmshj2018-hyperprior":
        net = models_dict[arch](N, M)
    else:
        net = models_dict[arch](N, M, K, embedding_type="downsample_cnn")

    net.load_state_dict(state_dict)
    net.eval()
    return net


@torch.no_grad()
def reconstruct(
    image_path: str,
    arch: str,
    checkpoint_path: str,
    output_dir: str,
    image_size: int = 256,
    patch_size: int = 16,
):
    os.makedirs(output_dir, exist_ok=True)

    # ── load image ────────────────────────────────────────────────────────────
    x = load_image(image_path, image_size=image_size).to(device)  # (1, C, H, W)

    # ── load model ────────────────────────────────────────────────────────────
    model = load_checkpoint(arch, checkpoint_path).to(device)

    # ── forward pass ─────────────────────────────────────────────────────────
    out = model(x)
    x_hat = out["x_hat"].clamp(0, 1).squeeze(0)  # (C, H, W)

    # ── save original and reconstruction side by side ─────────────────────────
    stem = Path(image_path).stem
    ckpt_stem = Path(checkpoint_path).stem

    original_path = os.path.join(output_dir, f"{stem}_original.png")
    recon_path    = os.path.join(output_dir, f"{stem}_{arch}_{ckpt_stem}_recon.png")

    save_tensor_as_image(x.squeeze(0), original_path)
    save_tensor_as_image(x_hat,        recon_path)

    print(f"Original   : {original_path}")
    print(f"Reconstruct: {recon_path}")

    # ── also save a side-by-side comparison ──────────────────────────────────
    try:
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(mpimg.imread(original_path))
        axes[0].set_title("Original")
        axes[0].axis("off")
        axes[1].imshow(mpimg.imread(recon_path))
        axes[1].set_title(f"Reconstruction\n{arch}")
        axes[1].axis("off")
        plt.tight_layout()

        comparison_path = os.path.join(output_dir, f"{stem}_{arch}_{ckpt_stem}_comparison.png")
        fig.savefig(comparison_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"Comparison : {comparison_path}")
    except ImportError:
        print("matplotlib not available, skipping comparison plot")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image",       required=True,  help="Path to input image")
    p.add_argument("--arch",        required=True,  choices=list(models_dict.keys()))
    p.add_argument("--checkpoint",  required=True,  help="Path to .pth.tar checkpoint")
    p.add_argument("--output",      required=True,  help="Output directory")
    p.add_argument("--image-size",  type=int, default=256)
    p.add_argument("--patch-size",  type=int, default=16)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    reconstruct(
        image_path=args.image,
        arch=args.arch,
        checkpoint_path=args.checkpoint,
        output_dir=args.output,
        image_size=args.image_size,
        patch_size=args.patch_size,
    )