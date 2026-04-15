"""
experiment_yg_ablation.py
=========================
Ablation study over the y_g latent space in ScaleHyperpriorBahdanau.
Covers two experimental workflows:

  Workflow A — probe (Experiments 2, 3, 5)
      Single-image manipulation probes on a trained checkpoint.
      Exp 2: zero-out y_g
      Exp 3: Gaussian noise injection, sigma in {0.01, 0.05, 0.1, 0.25, 0.5, 1.0}
      Exp 5: cross-image y_g mixing

  Workflow B — info (Experiment 4)
      Information-content probes over a small evaluation set (~20 images).
      Metrics: empirical entropy, effective rank, activation sparsity,
               y_g / y energy correlation.

Usage
-----
# Workflow A – single-image probes (Exps 2, 3, 5)
python experiment_yg_ablation.py probe \\
    --checkpoint checkpoints/model_128_192_32.pth.tar \\
    --arch bahdanau-hyperprior \\
    --image /path/to/image.png \\
    --other-image /path/to/other_image.png \\
    --results-dir probe_results \\
    --cuda

# Workflow B – information probes over eval set (Exp 4)
python experiment_yg_ablation.py info \\
    --checkpoint checkpoints/model_128_192_32.pth.tar \\
    --arch bahdanau-hyperprior \\
    --dataset /path/to/eval/images \\
    --n-images 20 \\
    --results-dir info_results \\
    --cuda
"""

import argparse
import os
import random
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from models import ScaleHyperprior, ScaleHyperpriorBahdanau
from loader import SSL4EOS12RGBDataset
from new_utils import patchify, unpatchify, save_tensor_as_image, load_image

import re

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PATCH_SIZE   = 16
TIMESTAMP    = datetime.now().strftime("%Y-%m-%d_%H-%M")
NOISE_SIGMAS = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]

models_dict = {
    "basic-hyperprior":    ScaleHyperprior,
    "bahdanau-hyperprior": ScaleHyperpriorBahdanau,
}


# ===========================================================================
# Shared utilities
# ===========================================================================

def set_seed(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_psnr(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """PSNR in dB, inputs in [0, 1]."""
    mse = torch.mean((original - reconstructed.clamp(0, 1)) ** 2).item()
    if mse < 1e-10:
        return 100.0
    return -10.0 * np.log10(mse)


def compute_bpp(out_net: dict, num_pixels: int) -> float:
    """Bits-per-pixel from likelihood tensors."""
    total_bits = sum(
        -torch.log2(lkl).sum()
        for lkl in out_net["likelihoods"].values()
    )
    return (total_bits / num_pixels).item()



def load_checkpoint(checkpoint_path: str, arch: str, device: str):
    """
    Load model from checkpoint.
    Extracts N, M, K from filename suffix: expects _N_M_K.pth.tar.
    Returns (net, N, M, K).
    """
    match = re.search(r'_(\d+)_(\d+)_(\d+)\.pth\.tar$', checkpoint_path)
    if not match:
        raise ValueError(
            f"Cannot extract N, M, K from '{checkpoint_path}'. "
            "Filename must end with _N_M_K.pth.tar"
        )
    N, M, K = tuple(map(int, match.groups()))

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["state_dict"]

    # Strip DataParallel 'module.' prefix if present
    if all(k.startswith("module.") for k in state_dict):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}

    if arch == "basic-hyperprior":
        net = models_dict[arch](N, M)
    else:
        net = models_dict[arch](N, M, K, embedding_type="downsample_cnn")

    net.load_state_dict(state_dict)
    net.eval()
    net.to(device)

    print(f"Loaded {arch}  N={N}  M={M}  K={K}  from {checkpoint_path}")
    return net, N, M, K


# ===========================================================================
# Experiment 4 — information-content helpers
# ===========================================================================

def empirical_entropy_bits(tensor: torch.Tensor, bins: int = 64) -> float:
    """
    Marginal empirical entropy of a tensor (treated as i.i.d. samples).
    Returns bits per element.
    Flattens all dims and builds a histogram over the scalar values.
    """
    vals = tensor.detach().cpu().float().numpy().flatten()
    counts, _ = np.histogram(vals, bins=bins, density=False)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def effective_rank(tensor: torch.Tensor) -> float:
    """
    Roy & Vetterli effective rank: exp(H(sigma)) where sigma are
    normalised singular values. Measures how many dimensions are
    actually used in y_g across the batch.
    Input shape: (B*P, K, h, w) — spatial dims are collapsed first.
    """
    flat = tensor.detach().cpu().float()
    flat = flat.reshape(flat.shape[0], -1)         # (B*P, K*h*w)
    _, s, _ = torch.linalg.svd(flat, full_matrices=False)
    s = s / (s.sum() + 1e-12)
    s = s[s > 1e-10]
    return float(torch.exp(-torch.sum(s * torch.log(s))))


def activation_sparsity(tensor: torch.Tensor, threshold: float = 0.01) -> float:
    """Fraction of near-zero elements — dead channels indicator."""
    return float((tensor.abs() < threshold).float().mean().item())


def yg_y_energy_corr(y_g: torch.Tensor, y: torch.Tensor) -> float:
    """
    Pearson correlation between per-patch mean absolute energy of y_g
    and y (the analysis transform output).
    Proxy for whether y_g energy tracks the main latent's energy.
    Both inputs: (B*P, C, h, w).
    """
    yg_mean = y_g.detach().abs().mean(dim=(1, 2, 3)).cpu()   # (B*P,)
    y_mean  = y.detach().abs().mean(dim=(1, 2, 3)).cpu()     # (B*P,)
    return float(torch.corrcoef(torch.stack([yg_mean, y_mean]))[0, 1].item())


# ===========================================================================
# Experiments 2, 3, 5 — y_g manipulation probes
# ===========================================================================

@torch.no_grad()
def probe_zero_yg(model, x: torch.Tensor) -> dict:
    """
    Experiment 2 – zero out y_g at decode time.
    Measures how much the decoder relies on y_g vs y_hat alone.
    """
    model.eval()
    B, C, H, W = x.shape
    x_p   = patchify(x, patch_size=model.patch_size)
    _, P, _, Hp, Wp = x_p.shape
    Gh, Gw = H // Hp, W // Wp
    x_flat = x_p.reshape(B * P, C, Hp, Wp)

    y_g      = model._embed_patches(x_flat)
    y_g_zero = torch.zeros_like(y_g)

    y        = model.g_a(x_p, y_g)
    z        = model.h_a(torch.abs(y))
    z_hat, _ = model.entropy_bottleneck(z)
    scales   = model.h_s(z_hat)
    y_hat, _ = model.gaussian_conditional(y, scales)
    y_g_hat_zero, _ = model.y_ent_bot(y_g_zero)

    x_hat_flat = model.g_s(y_hat, y_g_hat_zero).clamp(0, 1)
    x_hat_p    = x_hat_flat.reshape(B, P, 3, x_hat_flat.shape[-2], x_hat_flat.shape[-1])
    x_hat      = unpatchify(x_hat_p, (Gh, Gw))

    compute_bpp_result = compute_bpp(model.forward(x), num_pixels=H*W)

    return {"psnr_zero_yg": compute_psnr(x, x_hat), "x_hat": x_hat, "bpp": compute_bpp_result}


@torch.no_grad()
def probe_noisy_yg(model, x: torch.Tensor, sigma: float) -> dict:
    """
    Experiment 3 – inject Gaussian noise N(0, sigma^2) into y_g.
    Traces a PSNR-vs-noise-level sensitivity curve.
    """
    model.eval()
    B, C, H, W = x.shape
    x_p   = patchify(x, patch_size=model.patch_size)
    _, P, _, Hp, Wp = x_p.shape
    Gh, Gw = H // Hp, W // Wp
    x_flat = x_p.reshape(B * P, C, Hp, Wp)

    y_g       = model._embed_patches(x_flat)
    y_g_noisy = y_g + sigma * torch.randn_like(y_g)

    y        = model.g_a(x_p, y_g)
    z        = model.h_a(y.abs())
    z_hat, _ = model.entropy_bottleneck(z)
    scales   = model.h_s(z_hat)
    y_hat, _ = model.gaussian_conditional(y, scales)
    y_g_hat_noisy, _ = model.y_ent_bot(y_g_noisy)

    x_hat_flat = model.g_s(y_hat, y_g_hat_noisy).clamp(0, 1)
    x_hat_p    = x_hat_flat.reshape(B, P, 3, x_hat_flat.shape[-2], x_hat_flat.shape[-1])
    x_hat      = unpatchify(x_hat_p, (Gh, Gw))

    compute_bpp_result = compute_bpp(model.forward(x), num_pixels=H*W)

    return {"sigma": sigma, "psnr_noisy_yg": compute_psnr(x, x_hat), "x_hat": x_hat, "bpp": compute_bpp_result}


@torch.no_grad()
def probe_mixed_yg(model, x: torch.Tensor, x_other: torch.Tensor) -> dict:
    """
    Experiment 5 – decode x using y_g from x_other.
    Quantifies how much image-specific semantic content is baked into y_g.
    Also runs a baseline decode with x's own y_g for direct comparison.
    """
    model.eval()
    B, C, H, W = x.shape
    x_p       = patchify(x,       patch_size=model.patch_size)
    x_other_p = patchify(x_other, patch_size=model.patch_size)
    _, P, _, Hp, Wp = x_p.shape
    Gh, Gw = H // Hp, W // Wp

    x_flat       = x_p.reshape(B * P, C, Hp, Wp)
    x_other_flat = x_other_p.reshape(B * P, C, Hp, Wp)

    y_g       = model._embed_patches(x_flat)
    y_g_other = model._embed_patches(x_other_flat)

    # Encode x normally
    y        = model.g_a(x_p, y_g)
    z        = model.h_a(y.abs())
    z_hat, _ = model.entropy_bottleneck(z)
    scales   = model.h_s(z_hat)
    y_hat, _ = model.gaussian_conditional(y, scales)

    # Decode with foreign y_g
    y_g_other_hat, _ = model.y_ent_bot(y_g_other)
    x_hat_flat  = model.g_s(y_hat, y_g_other_hat).clamp(0, 1)
    x_hat_p     = x_hat_flat.reshape(B, P, 3, x_hat_flat.shape[-2], x_hat_flat.shape[-1])
    x_hat_mixed = unpatchify(x_hat_p, (Gh, Gw))

    # Baseline: decode with own y_g
    y_g_hat, _ = model.y_ent_bot(y_g)
    x_hat_base_flat = model.g_s(y_hat, y_g_hat).clamp(0, 1)
    x_hat_base_p    = x_hat_base_flat.reshape(B, P, 3, x_hat_flat.shape[-2], x_hat_flat.shape[-1])
    x_hat_base      = unpatchify(x_hat_base_p, (Gh, Gw))

    psnr_base  = compute_psnr(x, x_hat_base)
    psnr_mixed = compute_psnr(x, x_hat_mixed)

    compute_bpp_result_x = compute_bpp(model.forward(x), num_pixels=H*W)
    compute_bpp_result_other = compute_bpp(model.forward(x_other), num_pixels=H*W)

    return {
        "psnr_own_yg":     psnr_base,
        "psnr_foreign_yg": psnr_mixed,
        "psnr_delta":      psnr_base - psnr_mixed,   # positive = y_g carries real info
        "x_hat_mixed":     x_hat_mixed,
        "x_hat_base":      x_hat_base,
        "bpp":             compute_bpp_result_x,
        "bpp_other":       compute_bpp_result_other
    }


# ===========================================================================
# Workflow A — Experiments 2, 3, 5 on a single chosen image pair
# ===========================================================================

def run_probe_workflow(model, image_path: str, other_image_path: str,
                       K: int, device: str,
                       results_dir: str = "probe_results",
                       output_file: str = "results.json"):
    """
    Runs Experiments 2, 3, and 5 on a single image (+ a second image for Exp 5).
    Saves all reconstructions as PNGs and all metrics to a JSON file.
    """
    # results_dir = Path(results_dir)
    # results_dir.mkdir(parents=True, exist_ok=True)

    x       = load_image(image_path).to(device)
    x_other = load_image(other_image_path).to(device)

    data = {}

    # ---- Exp 2: zero y_g ------------------------------------------------
    print("\n[Exp 2] Zero y_g ...")
    res2 = probe_zero_yg(model, x)
    save_tensor_as_image(res2["x_hat"], Path(results_dir) / f"K{K}_zero_yg.png")
    data["psnr_zero_yg"] = res2["psnr_zero_yg"]
    data["bpp_zero_yg"]  = res2["bpp"]
    print(f"  PSNR = {res2['psnr_zero_yg']:.4f} dB  →  K{K}_zero_yg.png")

    # ---- Exp 3: noisy y_g -----------------------------------------------
    print("\n[Exp 3] Noisy y_g ...")
    for sigma in NOISE_SIGMAS:
        res3  = probe_noisy_yg(model, x, sigma=sigma)
        fname = f"K{K}_noisy_yg_sigma_{str(sigma).replace('.', 'p')}.png"
        save_tensor_as_image(res3["x_hat"], Path(results_dir) / fname)
        key = f"psnr_noise_sigma_{sigma}"
        data[key] = res3["psnr_noisy_yg"]
        data[f"bpp_noise_sigma_{sigma}"] = res3["bpp"]
        print(f"  sigma={sigma:<6}  PSNR = {res3['psnr_noisy_yg']:.4f} dB  →  {fname}")

    # ---- Exp 5: cross-image mixing --------------------------------------
    print("\n[Exp 5] Cross-image mixing ...")
    res5 = probe_mixed_yg(model, x, x_other)
    save_tensor_as_image(res5["x_hat_mixed"], Path(results_dir) / f"K{K}_mixed_yg.png")
    save_tensor_as_image(res5["x_hat_base"],  Path(results_dir) / f"K{K}_base_yg.png")
    data["psnr_own_yg"]     = res5["psnr_own_yg"]
    data["psnr_foreign_yg"] = res5["psnr_foreign_yg"]
    data["psnr_delta"]      = res5["psnr_delta"]
    data["bpp_mixed_yg"]    = res5["bpp"]
    data["bpp_base_yg"]     = res5["bpp_other"]
    print(f"  PSNR (own y_g)     = {res5['psnr_own_yg']:.4f} dB  →  K{K}_base_yg.png")
    print(f"  PSNR (foreign y_g) = {res5['psnr_foreign_yg']:.4f} dB  →  K{K}_mixed_yg.png")
    print(f"  PSNR delta         = {res5['psnr_delta']:.4f} dB")

    # ---- Save metrics ---------------------------------------------------
    out_path = Path(results_dir) / output_file
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nMetrics → {out_path}")
    return data


# ===========================================================================
# Workflow B — Experiment 4 information-content probes over an eval set
# ===========================================================================

@torch.no_grad()
def run_info_workflow(model, dataset_path: str, K: int, device: str,
                      n_images: int = 20,
                      results_dir: str = "info_results",
                      output_file: str = "info_results.json"):
    """
    Runs Experiment 4 (information-content probes on y_g) over a small
    evaluation set of n_images images sampled randomly from dataset_path.

    For each image computes:
      - empirical entropy of y_g activations
      - effective rank of y_g across patches (Roy-Vetterli)
      - activation sparsity of y_g
      - Pearson correlation between y_g energy and y energy

    Aggregates mean ± std across all images and saves to JSON.
    """
    # results_dir = Path(results_dir)
    # results_dir.mkdir(parents=True, exist_ok=True)

    dataset = SSL4EOS12RGBDataset(root=dataset_path)
    # indices = random.sample(range(len(dataset)), min(n_images, len(dataset)))
    # subset  = Subset(dataset, indices)
    loader  = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    model.eval()

    # One scalar per image
    records = {
        "entropy_bits":        [],
        "effective_rank":      [],
        "activation_sparsity": [],
        "yg_y_energy_corr":    [],
    }

    print(f"\nRunning Exp 4 on {len(dataset)} images ...")
    for i, x_batch in enumerate(loader):
        x      = x_batch.to(device)
        x_p    = patchify(x, patch_size=model.patch_size)
        B, P, C, Hp, Wp = x_p.shape
        x_flat = x_p.reshape(B * P, C, Hp, Wp)

        y_g = model._embed_patches(x_flat)    # (B*P, K, h, w)
        y   = model.g_a(x_p, y_g)            # (B*P, M, h', w')

        records["entropy_bits"].append(empirical_entropy_bits(y_g))
        records["effective_rank"].append(effective_rank(y_g))
        records["activation_sparsity"].append(activation_sparsity(y_g))
        records["yg_y_energy_corr"].append(yg_y_energy_corr(y_g, y))

        print(f"  [{i+1:>3}/{len(loader)}] "
              f"entropy={records['entropy_bits'][-1]:.3f}  "
              f"eff_rank={records['effective_rank'][-1]:.2f}  "
              f"sparsity={records['activation_sparsity'][-1]:.3f}  "
              f"corr={records['yg_y_energy_corr'][-1]:.3f}")

    # Aggregate: mean ± std across all images
    summary = {"K": K, "n_images": len(records["entropy_bits"])}
    for key, vals in records.items():
        arr = np.array(vals, dtype=np.float32)
        summary[f"{key}_mean"] = float(arr.mean())
        summary[f"{key}_std"]  = float(arr.std())

    out_path = Path(results_dir) / output_file
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    _print_info_summary(summary)
    print(f"Info metrics → {out_path}")
    return summary


def _print_info_summary(s: dict):
    print(f"\n{'='*60}")
    print(f"  EXP 4 SUMMARY  K={s['K']}  n={s['n_images']}")
    print(f"{'='*60}")
    for key in ["entropy_bits", "effective_rank", "activation_sparsity", "yg_y_energy_corr"]:
        mean = s.get(f"{key}_mean", float("nan"))
        std  = s.get(f"{key}_std",  float("nan"))
        print(f"  {key:<30s}  {mean:.4f} ± {std:.4f}")
    print(f"{'='*60}\n")


# ===========================================================================
# CLI
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(description="y_g ablation experiments")
    sub = p.add_subparsers(dest="mode", required=True)

    def add_shared_args(sp):
        sp.add_argument("--checkpoint",  type=str, required=True,
                        help="Path to checkpoint, must end with _N_M_K.pth.tar")
        sp.add_argument("--arch",        type=str, required=True,
                        choices=["basic-hyperprior", "bahdanau-hyperprior"])
        sp.add_argument("--results-dir", dest="results_dir", type=str, default="probe_results")
        sp.add_argument("--output-file", dest="output_file", type=str, default="results.json")
        sp.add_argument("--cuda",        action="store_true")
        sp.add_argument("--seed",        type=int, default=42)

    # ---- probe (Exps 2, 3, 5) -------------------------------------------
    pr = sub.add_parser("probe", help="Run Exps 2, 3, 5 on a single image pair")
    add_shared_args(pr)
    pr.add_argument("--image",       type=str, required=True,
                    help="Path to primary image (Exps 2, 3, and x in Exp 5)")
    pr.add_argument("--other-image", dest="other_image", type=str, required=True,
                    help="Path to second image (Exp 5 foreign y_g source)")

    # ---- info (Exp 4) ---------------------------------------------------
    inf = sub.add_parser("info", help="Run Exp 4 information probes over an eval set")
    add_shared_args(inf)
    inf.add_argument("--dataset",  type=str, required=True,
                     help="Root directory of the evaluation dataset")
    inf.add_argument("--n-images", dest="n_images", type=int, default=20,
                     help="Number of images to randomly sample from the dataset")

    return p.parse_args()


def main():
    args   = parse_args()
    set_seed(args.seed)
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    net, N, M, K = load_checkpoint(args.checkpoint, args.arch, device)

    # ---------------------------------------------------------------- PROBE
    if args.mode == "probe":
        print(f"\n{'='*60}")
        print(f"  PROBE  arch={args.arch}  N={N}  M={M}  K={K}")
        print(f"{'='*60}")
        run_probe_workflow(
            model=net,
            image_path=args.image,
            other_image_path=args.other_image,
            K=K,
            device=device,
            results_dir=args.results_dir,
            output_file=args.output_file,
        )

    # ----------------------------------------------------------------- INFO
    elif args.mode == "info":
        print(f"\n{'='*60}")
        print(f"  INFO  arch={args.arch}  N={N}  M={M}  K={K}  n={args.n_images}")
        print(f"{'='*60}")
        run_info_workflow(
            model=net,
            dataset_path=args.dataset,
            K=K,
            device=device,
            n_images=args.n_images,
            results_dir=args.results_dir,
            output_file=args.output_file,
        )


if __name__ == "__main__":
    main()
