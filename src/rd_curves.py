"""
rd_curves.py

Evaluates checkpoints across multiple models and lambdas on an eval dataset
and plots rate-distortion curves (PSNR and MS-SSIM vs bpp).

Usage:
    python rd_curves.py --output-dir ./rd_results
"""

import os
import re
import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from torchmetrics.image import PeakSignalNoiseRatio, MultiScaleStructuralSimilarityIndexMeasure


from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class PNGFolderDataset(Dataset):
    def __init__(self, folder: str):
        self.paths = sorted(Path(folder).glob("*.png"))
        if not self.paths:
            raise ValueError(f"No PNG files found in {folder}")
        
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return self.transform(Image.open(self.paths[idx]).convert("RGB"))


def get_eval_loader(folder: str, batch_size: int = 22):
    dataset = PNGFolderDataset(folder)
    print(f"Found {len(dataset)} images in {folder}")
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)


from models import ScaleHyperpriorBahdanau, ScaleHyperprior

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Model registry ────────────────────────────────────────────────────────────
models_dict = {
    "basic-hyperprior": ScaleHyperprior,
    "bahdanau-hyperprior":         ScaleHyperpriorBahdanau,
}

# ── Checkpoint config ─────────────────────────────────────────────────────────
# Each entry: (arch, lambda, checkpoint_path)
# arch must be a key in models_dict.
# N, M, K are parsed automatically from the checkpoint filename (_N_M_K.pth.tar).
# Remove or comment out any entries whose checkpoints are missing.
MODELS = {
    # "small-bahdanau": {
    #     "arch": "bahdanau-hyperprior",
    #     "checkpoints": [
    #         # (lambda, path)
    #         #missing 0.001 here
    #         (0.003, "/home/anas/from_cluster/downloaded_14_04_2026/checkpoints/checkpoint_best_loss_low_lambda_bahdanau_small_cluster_128_192_128.pth.tar"),
    #         (0.01,  "/home/anas/from_cluster/20_04_2026/checkpoint_best_loss_bahdanau_small_cluster_128_192_128.pth.tar"),
    #         #missing 0.1 here
    #         (0.03,  "/home/anas/from_cluster/20_04_2026/checkpoint_l_003_bahdanau_small_cluster_128_192_128.pth.tar"),
    #         (0.3,   "/home/anas/from_cluster/downloaded_14_04_2026/checkpoints/checkpoint_best_loss_high_lambda_bahdanau_small_cluster_128_192_128.pth.tar"),
    #         # missing 1 here
    #         (3.0,   "/home/anas/from_cluster/20_04_2026/checkpoint_best_loss_l_3_bahdanau_small_cluster_128_192_128.pth.tar"),
    #     ],
    # },
    "large-bahdanau": {
        "arch": "bahdanau-hyperprior",
        "checkpoints": [
            (0.003, "/home/anas/from_cluster/downloaded_14_04_2026/checkpoints/checkpoint_best_loss_low_lambda_bahdanau_big_cluster_192_320_192.pth.tar"),
            (0.001, "/home/anas/from_cluster/20_04_2026/checkpoint_lambda_0001_bahdanau_big_cluster_192_320_192.pth.tar"),
            (0.01, "/home/anas/from_cluster/20_04_2026/checkpoint_best_loss_bahdanau_big_cluster_192_320_192.pth.tar"),
            (0.03, "/home/anas/from_cluster/20_04_2026/checkpoint_lambda_003_bahdanau_big_cluster_192_320_192.pth.tar"),
            (0.1,  "/home/anas/from_cluster/20_04_2026/checkpoint_lambda_01_bahdanau_big_cluster_192_320_192.pth.tar"),
            (0.3,  "/home/anas/from_cluster/downloaded_14_04_2026/checkpoints/checkpoint_high_lambda_bahdanau_big_cluster_192_320_192.pth.tar"),
            (1.0,  "/home/anas/from_cluster/20_04_2026/checkpoint_lambda_1_bahdanau_big_cluster_192_320_192.pth.tar"),
            (3.0,  "/home/anas/from_cluster/20_04_2026/checkpoint_lambda_3_bahdanau_big_cluster_192_320_192.pth.tar"),
            # add remaining checkpoints here
        ],
    },
    "small-basic": {
        "arch": "basic-hyperprior",
        "checkpoints": [
            (0.003, "/home/anas/from_cluster/downloaded_14_04_2026/checkpoints/checkpoint_best_loss_low_lambda_vanilla_small_cluster_128_192_128.pth.tar"),
            (0.001, "/home/anas/from_cluster/20_04_2026/checkpoint_lambda_0001_vanilla_small_cluster_128_192_128.pth.tar"),
            (0.01, "/home/anas/from_cluster/downloaded_14_04_2026/checkpoints/checkpoint_best_loss_basic-hyperprior_128_192_128.pth.tar"),
            (0.03, "/home/anas/from_cluster/20_04_2026/checkpoint_lambda_003_vanilla_small_cluster_128_192_128.pth.tar"),
            (0.1,  "/home/anas/from_cluster/20_04_2026/checkpoint_lambda_01_vanilla_small_cluster_128_192_128.pth.tar"),
            (0.3,  "/home/anas/from_cluster/downloaded_14_04_2026/checkpoints/checkpoint_high_lambda_vanilla_small_cluster_128_192_128.pth.tar"),
            (1.0,  "/home/anas/from_cluster/20_04_2026/checkpoint_lambda_1_vanilla_small_cluster_128_192_128.pth.tar"),
            (3.0,  "/home/anas/from_cluster/20_04_2026/checkpoint_lambda_3_vanilla_small_cluster_128_192_128.pth.tar"),
            # add remaining checkpoints here
        ],
    },
    "large-basic": {
        "arch": "basic-hyperprior",
        "checkpoints": [
            (0.003, "/home/anas/from_cluster/downloaded_14_04_2026/checkpoints/checkpoint_best_loss_low_lambda_vanilla_big_cluster_192_320_192.pth.tar"),
            (0.001, "/home/anas/from_cluster/20_04_2026/checkpoint_best_loss_lambda_0001_vanilla_big_cluster_192_320_192.pth.tar"),
            (0.01, "/home/anas/from_cluster/downloaded_14_04_2026/checkpoints/checkpoint_best_loss_basic-hyperprior_192_320_192.pth.tar"),
            (0.03, "/home/anas/from_cluster/20_04_2026/checkpoint_best_loss_lambda_003_vanilla_big_cluster_192_320_192.pth.tar"),
            (0.1,  "/home/anas/from_cluster/20_04_2026/checkpoint_best_loss_lambda_01_vanilla_big_cluster_192_320_192.pth.tar"),
            (0.3,  "/home/anas/from_cluster/downloaded_14_04_2026/checkpoints/checkpoint_best_loss_high_lambda_vanilla_big_cluster_192_320_192.pth.tar"),
            (1.0,  "/home/anas/from_cluster/20_04_2026/checkpoint_best_loss_lambda_1_vanilla_big_cluster_192_320_192.pth.tar"),
            (3.0,  "/home/anas/from_cluster/20_04_2026/checkpoint_best_loss_lambda_3_vanilla_big_cluster_192_320_192.pth.tar"),
            # add remaining checkpoints here
        ],
    },
}


# ── Metrics ───────────────────────────────────────────────────────────────────
psnr_fn   = PeakSignalNoiseRatio(data_range=1.0).to(device)
msssim_fn = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to(device)

# ── Checkpoint loading ────────────────────────────────────────────────────────
def load_checkpoint(arch: str, checkpoint_path: str) -> nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["state_dict"]

    # strip DataParallel 'module.' prefix if present
    if all(k.startswith("module.") for k in state_dict):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}

    # parse N, M, K from filename — expects _N_M_K.pth.tar suffix
    match = re.search(r'_(\d+)_(\d+)_(\d+)\.pth\.tar$', checkpoint_path)
    if not match:
        raise ValueError(
            f"Could not extract N, M, K from checkpoint filename: {checkpoint_path}\n"
            f"Expected filename ending in _N_M_K.pth.tar"
        )
    N, M, K = tuple(map(int, match.groups()))
    print(f"Loading {arch}  N={N}  M={M}  K={K}\nfrom {checkpoint_path}")

    if arch == "basic-hyperprior":
        net = models_dict[arch](N, M)
    else:
        net = models_dict[arch](N, M, K, embedding_type="downsample_cnn")

    net.load_state_dict(state_dict)
    net.update()    # build entropy coding tables before compress/decompress
    print("Model loaded successfully.\n")
    return net.eval()


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate_checkpoint(model: nn.Module, loader) -> dict:
    model.to(device)
    total_bpp = total_psnr = total_msssim = 0.0
    n_batches = 0

    with torch.no_grad():
        for x in loader:
            x = x.to(device)
            B, C, H, W = x.shape
            num_pixels = H * W

            out   = model.compress(x)
            x_hat = model.decompress(out["strings"], out["shape"])["x_hat"]
            x_hat = x_hat.clamp(0.0, 1.0)

            bpp = sum(
                len(s) * 8 for strings in out["strings"] for s in strings
            ) / (B * num_pixels)

            total_bpp    += bpp
            total_psnr   += psnr_fn(x_hat, x).item()
            total_msssim += msssim_fn(x_hat, x).item()
            n_batches    += 1

    return {
        "bpp":    total_bpp    / n_batches,
        "psnr":   total_psnr   / n_batches,
        "msssim": total_msssim / n_batches,
    }


def run_evaluation() -> dict:
    loader  = get_eval_loader("/home/anas/datasets/exp1_clean", batch_size=1)
    results = {}

    for model_name, cfg in MODELS.items():
        arch = cfg["arch"]
        print(f"\n{'='*60}")
        print(f"Model: {model_name}  (arch: {arch})")
        print(f"{'='*60}")
        results[model_name] = []

        for lam, ckpt_path in cfg["checkpoints"]:
            print(f"  lambda={lam}")
            model   = load_checkpoint(arch, ckpt_path)
            metrics = evaluate_checkpoint(model, loader)
            metrics["lambda"] = lam
            results[model_name].append(metrics)
            print(f"  bpp={metrics['bpp']:.4f}  "
                  f"PSNR={metrics['psnr']:.2f} dB  "
                  f"MS-SSIM={metrics['msssim']:.4f}\n")

            # free VRAM between checkpoints
            del model
            torch.cuda.empty_cache()

    return results


# ── Plotting ──────────────────────────────────────────────────────────────────
MARKERS = ["o", "s", "^", "D"]
COLORS  = ["steelblue", "coral", "mediumseagreen", "mediumpurple"]


def plot_rd(results: dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    fig_psnr,   ax_psnr   = plt.subplots(figsize=(6, 4.5))
    fig_msssim, ax_msssim = plt.subplots(figsize=(6, 4.5))

    for idx, (model_name, points) in enumerate(results.items()):
        points  = sorted(points, key=lambda p: p["bpp"])
        bpps    = [p["bpp"]    for p in points]
        psnrs   = [p["psnr"]   for p in points]
        msssims = [p["msssim"] for p in points]

        color  = COLORS[idx % len(COLORS)]
        marker = MARKERS[idx % len(MARKERS)]

        ax_psnr.plot(bpps, psnrs,     marker=marker, color=color,
                     linewidth=1.5, markersize=5, label=model_name)
        ax_msssim.plot(bpps, msssims, marker=marker, color=color,
                       linewidth=1.5, markersize=5, label=model_name)

    for ax, ylabel in [(ax_psnr, "PSNR (dB)"), (ax_msssim, "MS-SSIM")]:
        ax.set_xlabel("Rate (bpp)")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    fig_psnr.tight_layout()
    fig_msssim.tight_layout()

    psnr_path   = os.path.join(output_dir, "rd_psnr.png")
    msssim_path = os.path.join(output_dir, "rd_msssim.png")
    fig_psnr.savefig(psnr_path,   bbox_inches="tight", dpi=150)
    fig_msssim.savefig(msssim_path, bbox_inches="tight", dpi=150)
    plt.close("all")
    print(f"Saved: {psnr_path}")
    print(f"Saved: {msssim_path}")


# ── Entry point ───────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="/home/anas/datasets/r-d_curves")
    return p.parse_args()


if __name__ == "__main__":
    args    = parse_args()
    results = run_evaluation()
    plot_rd(results, args.output_dir)