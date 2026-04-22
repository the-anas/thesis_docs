"""
rd_curves.py

Evaluates 6 checkpoints (2 models x 3 lambdas) on an eval dataset and plots
rate-distortion curves (PSNR and MS-SSIM vs bpp).

Usage:
    python rd_curves.py --output-dir ./rd_results
"""

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from torchmetrics.image import PeakSignalNoiseRatio, MultiScaleStructuralSimilarityIndexMeasure
from models import ScaleHyperpriorBahdanau, ScaleHyperprior

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────────────────
# FILL IN: define your models and checkpoints here
#
# Structure:
#   MODELS = {
#       "ModelName": {
#           "lambdas": [lam1, lam2, lam3],          # must match checkpoint order
#           "checkpoints": ["/path/a.pth", ...],     # one per lambda
#           "model_fn": lambda: YourModelClass(...), # callable that returns an uninitialised model
#       },
#       ...
#   }
# ─────────────────────────────────────────────────────────────────────────────
MODELS = {
    "small-bahdanau": {
        "lambdas":     [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3],
        "checkpoints": [
            "/home/anas/from_cluster/20_04_2026/   modelA_lam0018.pth", # missing
            "/home/anas/from_cluster/downloaded_14_04_2026/checkpoints/checkpoint_best_loss_low_lambda_bahdanau_small_cluster_128_192_128.pth.tar", #0.003
            "/home/anas/from_cluster/20_04_2026/checkpoint_best_loss_bahdanau_small_cluster_128_192_128.pth.tar",
            "/home/anas/from_cluster/20_04_2026/checkpoint_l_003_bahdanau_small_cluster_128_192_128.pth.tar", 
            "/home/anas/from_cluster/20_04_2026/", # missing
            "/home/anas/from_cluster/downloaded_14_04_2026/checkpoints/checkpoint_best_loss_high_lambda_bahdanau_small_cluster_128_192_128.pth.tar",
            "/home/anas/from_cluster/20_04_2026/", # missing
            "/home/anas/from_cluster/20_04_2026/checkpoint_best_loss_l_3_bahdanau_small_cluster_128_192_128.pth.tar",
        ],
        "model_fn": lambda: None,  # replace with e.g. ScaleHyperpriorBahdanau(...)
    },

    "large-bahdanau": {
        "lambdas":     [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3],
        "checkpoints": [
            "/path/to/modelB_lam0018.pth",
            "/home/anas/from_cluster/downloaded_14_04_2026/checkpoints/checkpoint_best_loss_low_lambda_bahdanau_big_cluster_192_320_192.pth.tar",
            "/path/to/modelB_lam025.pth",
        ],
        "model_fn": lambda: None,  # replace with e.g. ScaleHyperpriorConcat(...)
    },
    "small-basic": {
        "lambdas":     [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3],
        "checkpoints": [
            "/path/to/modelA_lam0018.pth",
            "/home/anas/from_cluster/downloaded_14_04_2026/checkpoints/checkpoint_best_loss_low_lambda_bahdanau_small_cluster_128_192_128.pth.tar",
            "/path/to/modelA_lam025.pth",
        ],
        "model_fn": lambda: None,  # replace with e.g. ScaleHyperpriorBahdanau(...)
    },
    "big-basic": {
        "lambdas":     [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3],
        "checkpoints": [
            "/path/to/modelA_lam0018.pth",
            "/home/anas/from_cluster/downloaded_14_04_2026/checkpoints/checkpoint_best_loss_low_lambda_vanilla_big_cluster_192_320_192.pth.tar",
            "/path/to/modelA_lam025.pth",
        ],
        "model_fn": lambda: None,  # replace with e.g. ScaleHyperpriorBahdanau(...)
    }
}

# ─────────────────────────────────────────────────────────────────────────────
# FILL IN: build your eval dataloader here
# Should yield batches of (B, C, H, W) tensors normalised to [0, 1]
# ─────────────────────────────────────────────────────────────────────────────
def get_eval_loader():
    raise NotImplementedError("Fill in your eval dataloader here.")


# ── Metrics ───────────────────────────────────────────────────────────────────
psnr_fn  = PeakSignalNoiseRatio(data_range=1.0).to(device)
msssim_fn = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to(device)


def evaluate_checkpoint(model, loader) -> dict:
    """
    Returns {"bpp": float, "psnr": float, "msssim": float}
    averaged over the entire eval set.
    """
    model.eval()
    total_bpp = total_psnr = total_msssim = 0.0
    n_batches = 0

    with torch.no_grad():
        for x in loader:
            x = x.to(device)
            B, C, H, W = x.shape
            num_pixels = H * W

            out = model.compress(x)
            x_hat = model.decompress(out["strings"], out["shape"])["x_hat"]
            x_hat = x_hat.clamp(0.0, 1.0)

            # bpp: total bits across all strings / (batch * pixels)
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


# ── Evaluation loop ───────────────────────────────────────────────────────────
def run_evaluation() -> dict:
    """
    Returns:
        {
            "Model A": [{"lambda": .., "bpp": .., "psnr": .., "msssim": ..}, ...],
            "Model B": [...],
        }
    """
    loader = get_eval_loader()
    results = {}

    for model_name, cfg in MODELS.items():
        print(f"\n{'='*50}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*50}")
        results[model_name] = []

        for lam, ckpt_path in zip(cfg["lambdas"], cfg["checkpoints"]):
            print(f"  lambda={lam}  checkpoint={ckpt_path}")

            model = cfg["model_fn"]()
            state = torch.load(ckpt_path, map_location=device)
            # handles both raw state dicts and checkpoint dicts
            state_dict = state.get("state_dict", state.get("model", state))
            model.load_state_dict(state_dict)
            model = model.to(device)

            metrics = evaluate_checkpoint(model, loader)
            metrics["lambda"] = lam
            results[model_name].append(metrics)

            print(f"    bpp={metrics['bpp']:.4f}  "
                  f"PSNR={metrics['psnr']:.2f} dB  "
                  f"MS-SSIM={metrics['msssim']:.4f}")

    return results


# ── Plotting ──────────────────────────────────────────────────────────────────
MARKERS = ["o", "s", "^", "D", "v", "P"]
COLORS  = ["steelblue", "coral", "mediumseagreen", "mediumpurple"]


def plot_rd(results: dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    fig_psnr,   ax_psnr   = plt.subplots(figsize=(6, 4.5))
    fig_msssim, ax_msssim = plt.subplots(figsize=(6, 4.5))

    for idx, (model_name, points) in enumerate(results.items()):
        # sort by bpp so the line goes left to right
        points = sorted(points, key=lambda p: p["bpp"])

        bpps    = [p["bpp"]    for p in points]
        psnrs   = [p["psnr"]   for p in points]
        msssims = [p["msssim"] for p in points]

        color  = COLORS[idx % len(COLORS)]
        marker = MARKERS[idx % len(MARKERS)]

        ax_psnr.plot(bpps, psnrs,   marker=marker, color=color,
                     linewidth=1.5, markersize=5, label=model_name)
        ax_msssim.plot(bpps, msssims, marker=marker, color=color,
                       linewidth=1.5, markersize=5, label=model_name)

    for ax, ylabel in [
        (ax_psnr,   "PSNR (dB)"),
        (ax_msssim, "MS-SSIM"),
    ]:
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
    print(f"\nSaved: {psnr_path}")
    print(f"Saved: {msssim_path}")


# ── Entry point ───────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="./rd_results")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results = run_evaluation()
    plot_rd(results, args.output_dir)
