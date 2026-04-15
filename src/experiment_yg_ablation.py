"""
experiment_yg_ablation.py
=========================
Ablation study over the K dimension of the y_g latent space in
ScaleHyperpriorBahdanau.  Covers five experimental axes:

  2. y_g manipulation    — ablations applied at inference to a trained checkpoint
  3. Noise injection     — Gaussian noise σ ∈ {0.01,0.05,0.1,0.25,0.5,1.0}
  4. Information content — empirical entropy + effective rank of y_g
  5. Cross-image mixing  — replace y_g with embedding from a different image

Usage
-----

# Phase 2 – run inference probes on a trained checkpoint
python experiment_yg_ablation.py probe \
    --checkpoint checkpoints/K32_best.pth.tar \
    --dataset /path/to/data \
    --K 32 --cuda
"""

import argparse
import os
import random
import json
from pathlib import Path
from datetime import datetime
from xml.parsers.expat import model

import numpy as np
from skimage import data
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from compressai.losses import RateDistortionLoss
from compressai.optimizers import net_aux_optimizer

from models import ScaleHyperprior, ScaleHyperpriorBahdanau
from loader import SSL4EOS12RGBDataset
from new_utils import save_tensor_as_image
import wandb
from new_utils import patchify, unpatchify, save_tensor_as_image

from torchvision import transforms
from PIL import Image

import re

from new_utils import load_image

transform = transforms.Compose([
    transforms.CenterCrop((256, 256)),  # target size (H, W)
    transforms.ToTensor(),              # converts to [0,1] automatically
])

# ---------------------------------------------------------------------------
# Hyper-parameter grid for the K sweep (Experiment 1)
# ---------------------------------------------------------------------------
K_SWEEP = [8, 24, 48, 72, 96, 128]

# Fixed architectural params – only K varies
# N_FIXED = 30
# M_FIXED = 24
PATCH_SIZE = 16

TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M")

models_dict = {"basic-hyperprior": ScaleHyperprior, 
               "bahdanau-hyperprior":ScaleHyperpriorBahdanau}



# ===========================================================================
# Utility helpers
# ===========================================================================

def set_seed(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# [] maybe just use metrics you already have??
# ===========================================================================
# Metric helpers
# ===========================================================================

def compute_psnr(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """PSNR in dB, inputs in [0,1]."""
    mse = torch.mean((original - reconstructed.clamp(0, 1)) ** 2).item()
    if mse < 1e-10:
        return 100.0
    return -10.0 * np.log10(mse)


# [] make sure shape and format out of this function is compatible
# [] do i need to compute bpp??

def compute_bpp(out_net: dict, num_pixels: int) -> float:
    """Bits-per-pixel from likelihood tensors."""
    total_bits = sum(
        -torch.log2(lkl).sum()
        for lkl in out_net["likelihoods"].values()
    )
    return (total_bits / num_pixels).item()

# ===========================================================================
# Information-content probes for y_g (Experiment 4)
# ===========================================================================

def empirical_entropy_bits(tensor: torch.Tensor, bins: int = 64) -> float:
    """
    Marginal empirical entropy of a tensor (treated as i.i.d. samples).
    Returns bits per element.
    """
    vals = tensor.detach().cpu().float().numpy().flatten()
    counts, _ = np.histogram(vals, bins=bins, density=False)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def effective_rank(tensor: torch.Tensor) -> float:
    """
    Roy & Vetterli effective rank: exp(H(sigma)) where sigma are
    normalised singular values.  Measures how many dimensions are
    actually used in y_g across the batch.
    tensor: (B*P, K, h, w) — flatten spatial dims first.
    """
    flat = tensor.detach().cpu().float()
    # collapse spatial: (B*P, K*h*w)
    flat = flat.reshape(flat.shape[0], -1)
    # SVD on the (samples x features) matrix
    _, s, _ = torch.linalg.svd(flat, full_matrices=False)
    s = s / (s.sum() + 1e-12)
    s = s[s > 1e-10]
    eff_rank = float(torch.exp(-torch.sum(s * torch.log(s))))
    return eff_rank


def activation_sparsity(tensor: torch.Tensor, threshold: float = 0.01) -> float:
    """Fraction of near-zero elements (dead channels indicator)."""
    return float((tensor.abs() < threshold).float().mean().item())


def mutual_info_proxy(y_g, x, x_hat, y_hat):  
    if y_hat is None:
        return {"yg_yhat_energy_corr": float("nan")}
    yg_mean   = y_g.detach().abs().mean(dim=(1,2,3)).cpu()
    yhat_energy = y_hat.detach().abs().mean(dim=(1,2,3)).cpu()
    corr = float(torch.corrcoef(torch.stack([yg_mean, yhat_energy]))[0,1].item())
    return {"yg_yhat_energy_corr": corr}



# ===========================================================================
# y_g MANIPULATION PROBES  (Experiments 2, 3, 5)
# ===========================================================================

@torch.no_grad()
def probe_zero_yg(model, x: torch.Tensor) -> dict:
    """Experiment 2 – blank y_g (zero vector): measures how much the
    decoder relies on y_g vs y_hat alone."""

    model.eval()
    B, C, H, W = x.shape
    x_p   = patchify(x, patch_size=model.patch_size)
    _, P, _, Hp, Wp = x_p.shape
    Gh, Gw = H // Hp, W // Wp
    x_flat = x_p.reshape(B * P, C, Hp, Wp)

    y_g    = model._embed_patches(x_flat)          # real embedding
    y_g_zero = torch.zeros_like(y_g)              # ← ZEROED

    y      = model.g_a(x_p, y_g)                  # encode normally
    z      = model.h_a(torch.abs(y)) # change to torch.abs
    z_hat, _ = model.entropy_bottleneck(z)
    scales = model.h_s(z_hat)
    y_hat, _ = model.gaussian_conditional(y, scales)
    y_g_hat_zero, _ = model.y_ent_bot(y_g_zero)   # encode zeros

    x_hat_flat = model.g_s(y_hat, y_g_hat_zero).clamp(0, 1)
    x_hat_p    = x_hat_flat.reshape(B, P, 3, x_hat_flat.shape[-2], x_hat_flat.shape[-1])
    x_hat      = unpatchify(x_hat_p, (Gh, Gw))

    psnr = compute_psnr(x, x_hat)
    return {"psnr_zero_yg": psnr, "x_hat": x_hat}


# [] maybe replace with gaussian noise
@torch.no_grad()
def probe_noisy_yg(model, x: torch.Tensor, sigma: float) -> dict:
    """Experiment 3 – add Gaussian noise σ to y_g and measure PSNR degradation."""

    model.eval()
    B, C, H, W = x.shape
    x_p   = patchify(x, patch_size=model.patch_size)
    _, P, _, Hp, Wp = x_p.shape
    Gh, Gw = H // Hp, W // Wp
    x_flat = x_p.reshape(B * P, C, Hp, Wp)

    y_g       = model._embed_patches(x_flat)
    y_g_noisy = y_g + sigma * torch.randn_like(y_g) # vary sigma here to meaure reliance on y_g

    y      = model.g_a(x_p, y_g)
    z      = model.h_a(y.abs())
    z_hat, _ = model.entropy_bottleneck(z)
    scales = model.h_s(z_hat)
    y_hat, _ = model.gaussian_conditional(y, scales)
    y_g_hat_noisy, _ = model.y_ent_bot(y_g_noisy)

    x_hat_flat = model.g_s(y_hat, y_g_hat_noisy).clamp(0, 1)
    x_hat_p    = x_hat_flat.reshape(B, P, 3, x_hat_flat.shape[-2], x_hat_flat.shape[-1])
    x_hat      = unpatchify(x_hat_p, (Gh, Gw))

    psnr = compute_psnr(x, x_hat)
    return {"sigma": sigma, "psnr_noisy_yg": psnr, "x_hat": x_hat}


@torch.no_grad()
def probe_mixed_yg(model, x: torch.Tensor, x_other: torch.Tensor) -> dict:
    """Experiment 5 – use the y_g from x_other during decoding of x.
    Quantifies how much semantic content is baked into y_g."""

    model.eval()
    B, C, H, W = x.shape
    x_p     = patchify(x,       patch_size=model.patch_size)
    x_other_p = patchify(x_other, patch_size=model.patch_size)
    _, P, _, Hp, Wp = x_p.shape
    Gh, Gw = H // Hp, W // Wp

    x_flat       = x_p.reshape(B * P, C, Hp, Wp)
    x_other_flat = x_other_p.reshape(B * P, C, Hp, Wp)

    y_g       = model._embed_patches(x_flat)         # own embedding
    y_g_other = model._embed_patches(x_other_flat)   # foreign embedding

    # encode x normally
    y      = model.g_a(x_p, y_g)
    z      = model.h_a(y.abs())
    z_hat, _ = model.entropy_bottleneck(z)
    scales = model.h_s(z_hat)
    y_hat, _ = model.gaussian_conditional(y, scales)

    # decode with foreign y_g
    y_g_other_hat, _ = model.y_ent_bot(y_g_other)
    x_hat_flat = model.g_s(y_hat, y_g_other_hat).clamp(0, 1)
    x_hat_p    = x_hat_flat.reshape(B, P, 3, x_hat_flat.shape[-2], x_hat_flat.shape[-1])
    x_hat      = unpatchify(x_hat_p, (Gh, Gw))

    psnr_mixed = compute_psnr(x, x_hat)

    # baseline: decode with own y_g
    y_g_hat, _ = model.y_ent_bot(y_g)
    x_hat_base_flat = model.g_s(y_hat, y_g_hat).clamp(0, 1)
    x_hat_base_p    = x_hat_base_flat.reshape(B, P, 3, x_hat_flat.shape[-2], x_hat_flat.shape[-1])
    x_hat_base      = unpatchify(x_hat_base_p, (Gh, Gw))
    psnr_base = compute_psnr(x, x_hat_base)

    return {
        "psnr_own_yg":    psnr_base,
        "psnr_foreign_yg": psnr_mixed,
        "psnr_delta":     psnr_base - psnr_mixed,  # positive = y_g carries real info
        "x_hat_mixed": x_hat,
        "x_hat_base": x_hat_base,
    }

# ===========================================================================
# PROBE PHASE – run all manipulations on a trained checkpoint
# ===========================================================================

@torch.no_grad()
def run_probes(model, image_path, other_image_path: str, K: int, device: str,
               results_dir: str = "probe_results", output_file: str = "results.json"):
    """
    Runs Experiments 2-5 on `n_images` validation images.
    Saves a JSON summary to results_dir/K{K}_probes.json.
    """
    data = {}
    x = load_image(image_path)  
    x_other = load_image(other_image_path)
    x.to(device)
    x_other.to(device)

    # img = Image.open(image_path).convert("RGB")
    # img_other = Image.open(other_image_path).convert("RGB") 
    # x = transform(img)
    # x_other = transform(img_other)
    # # [] check if batch dim is needed 
    # x = x.unsqueeze(0).to(device)  # add batch dim
    # x_other = x_other.unsqueeze(0).to(device)
    # # delete below and change  with single image
    

    # [] edit these later
    noise_sigmas = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]



    # ---- Exp 2: zero y_g ----
    zero_out_result = probe_zero_yg(model, x) # other output is psnr with key psnr_zero_yg
    # [] import function below
    save_tensor_as_image(zero_out_result["x_hat"], Path(results_dir) / f"K{K}_zero_yg.png")
    data["psnr_zero_yg"] = zero_out_result["psnr_zero_yg"]

    # ---- Exp 3: noise ----
    for sigma in noise_sigmas:
        res_noise = probe_noisy_yg(model, x, sigma=sigma)
        save_tensor_as_image(res_noise["x_hat"], Path(results_dir) / f"K{K}_noisy_yg_sigma_{sigma}.png")
        data[f"psnr_noise_sigma_{sigma}"] = res_noise["psnr_noisy_yg"]
    # other output is "sigma": sigma, "psnr_noisy_yg": psnr

    
    # ---- Exp 5: cross-image mixing ----
    res_mix = probe_mixed_yg(model, x, x_other)
    save_tensor_as_image(res_mix["x_hat_mixed"], Path(results_dir) / f"K{K}_mixed_yg.png")
    save_tensor_as_image(res_mix["x_hat_base"], Path(results_dir) / f"K{K}_base_yg.png")
    data["psnr_own_yg"] = res_mix["psnr_own_yg"]
    data["psnr_foreign_yg"] = res_mix["psnr_foreign_yg"]
    data["psnr_delta"] = res_mix["psnr_delta"]

    with open(Path(results_dir)/ Path(output_file), "w") as f:
        json.dump(data, f, indent=2)

           

    # [] figure out what is going below with exp 4 adn what oyu need out of it
    # [] prbly not good enough
    agg = {
        "K": K,
        "n_images": n_images,
        # Exp 2: zero
        "psnr_zero_yg": [],
        # Exp 3: noise
        **{f"psnr_noise_sigma_{s}": [] for s in noise_sigmas},
        # Exp 4: information probes
        "entropy_bits": [],
        "effective_rank": [],
        "activation_sparsity": [],
        "yg_yhat_energy_corr": [],
        # Exp 5: cross-image mix
        "psnr_own_yg": [],
        "psnr_foreign_yg": [],
        "psnr_delta": [],
    }


    for (x_batch, x_other_batch) in zip(loader, other_loader):
        x       = x_batch.to(device)
        x_other = x_other_batch.to(device)

        # ---- Exp 4: information content of y_g ----
        x_p    = patchify(x, patch_size=model.patch_size)
        B, P, C, Hp, Wp = x_p.shape
        x_flat = x_p.reshape(B * P, C, Hp, Wp)
        y_g    = model._embed_patches(x_flat)

        agg["entropy_bits"].append(empirical_entropy_bits(y_g))
        agg["effective_rank"].append(effective_rank(y_g))
        agg["activation_sparsity"].append(activation_sparsity(y_g))

        # energy correlation proxy
        z      = model.h_a(y_g.abs().mean(1, keepdim=True).expand_as(y_g[:, :M_FIXED]))  # rough proxy, skip if shape mismatch
        # safe MI proxy: use y and y_g directly
        y      = model.g_a(x_p, y_g)
        mi     = mutual_info_proxy(y_g, x=None, x_hat=None, y_hat=y)
        # redefine to match signature
        corr = float(torch.corrcoef(
            torch.stack([
                y_g.abs().mean(dim=(1,2,3)).cpu(),
                y.abs().mean(dim=(1,2,3)).cpu()
            ])
        )[0,1].item())
        agg["yg_yhat_energy_corr"].append(corr)


    # Summarise (mean ± std)
    summary = {"K": K}
    for key, vals in agg.items():
        if key == "K" or not isinstance(vals, list) or len(vals) == 0:
            continue
        arr = np.array(vals, dtype=np.float32)
        summary[key + "_mean"] = float(arr.mean())
        summary[key + "_std"]  = float(arr.std())

    out_path = results_dir / f"K{K}_probes.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[Probes K={K}] Results saved to {out_path}")
    _print_probe_summary(summary)
    return summary


def _print_probe_summary(s: dict):
    print(f"\n{'='*60}")
    print(f"  PROBE SUMMARY  K={s['K']}")
    print(f"{'='*60}")
    for k, v in s.items():
        if k == "K": continue
        print(f"  {k:<40s}  {v:.4f}")
    print(f"{'='*60}\n")


# ===========================================================================
# Aggregate results across all K variants
# ===========================================================================

def aggregate_results(results_dir: str = "probe_results"):
    """Read all K*_probes.json and print a comparison table."""
    results_dir = Path(results_dir)
    files = sorted(results_dir.glob("K*_probes.json"),
                   key=lambda p: int(p.stem[1:].split("_")[0]))

    if not files:
        print("No probe result files found.")
        return

    print(f"\n{'K':>6} | {'entropy':>10} | {'eff_rank':>10} | "
          f"{'psnr_zero':>10} | {'psnr_noise0.1':>14} | "
          f"{'psnr_own':>10} | {'psnr_foreign':>12} | {'psnr_delta':>10}")
    print("-" * 100)

    for f in files:
        with open(f) as fp:
            s = json.load(fp)
        K  = s["K"]
        en = s.get("entropy_bits_mean", float("nan"))
        er = s.get("effective_rank_mean", float("nan"))
        pz = s.get("psnr_zero_yg_mean", float("nan"))
        pn = s.get("psnr_noise_sigma_0.1_mean", float("nan"))
        po = s.get("psnr_own_yg_mean", float("nan"))
        pf = s.get("psnr_foreign_yg_mean", float("nan"))
        pd = s.get("psnr_delta_mean", float("nan"))
        print(f"{K:>6} | {en:>10.3f} | {er:>10.2f} | "
              f"{pz:>10.2f} | {pn:>14.2f} | "
              f"{po:>10.2f} | {pf:>12.2f} | {pd:>10.2f}")


# ===========================================================================
# CLI
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(description="y_g ablation experiments")
    sub = p.add_subparsers(dest="mode", required=True)

    # ---- probe ----
    pr = sub.add_parser("probe", help="Run manipulation probes on a checkpoint")
    pr.add_argument("--checkpoint", type=str,   required=True)
    pr.add_argument("--image",    type=str,   required=True, help="Path to image (Experiments 2,3,5)")
    pr.add_argument("--other-image",dest="other_image", type=str, required=True, help="Path to other image (Experiment 5)")
    pr.add_argument("--n-images",   type=int,   default=64)
    pr.add_argument("--cuda",       action="store_true")
    pr.add_argument("--results-dir", dest="results_dir", type=str,   default="probe_results")
    pr.add_argument("--output-file", dest="output_file", type=str,   default="results.json")
    pr.add_argument("--seed",       type=int,   default=42)
    pr.add_argument("--arch", choices=["basic-hyperprior", "bahdanau-hyperprior"], help="Model architecture to load")

    # ---- aggregate ----
    ag = sub.add_parser("aggregate", help="Print table aggregating all probe JSON files")
    ag.add_argument("--results-dir", type=str, default="probe_results")

    return p.parse_args()


# [] check how different the train mode actually is and what you are going to do about it
def main():
    args = parse_args()
    set_seed(getattr(args, "seed", 42))
    device = "cuda" if getattr(args, "cuda", False) and torch.cuda.is_available() else "cpu"


    # load model, uses single model per run
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint["state_dict"]

    # Strip the DataParallel 'module.' prefix if present
    if all(k.startswith("module.") for k in state_dict):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}



    match = re.search(r'_(\d+)_(\d+)_(\d+)\.pth\.tar$', args.checkpoint)
 
    if match:
        values = tuple(map(int, match.groups()))
    else:
        print("Could not extract N, M, K from checkpoint filename. Make sure it ends with _N_M_K.pth.tar")
        raise ValueError("Invalid checkpoint filename format")

    print(f"Loading model {args.arch} with N={values[0]}, M={values[1]}, K={values[2]}\nfrom {args.checkpoint}\n\n")
  

    if args.arch=="basic-hyperprior":
        net = models_dict[args.arch](values[0], values[1])
    else:
        net = models_dict[args.arch](values[0], values[1], values[2], embedding_type="downsample_cnn")
    
    net.load_state_dict(state_dict)
    net.eval()
    net.to(device)
    print("MODEL LOADED FINE")

    # [] if running through multiple K, edit code
    run_probes(net, args.image, args.other_image, values[2], device,
               args.results_dir, args.output_file)
  




    # --------------------------------------------------------------- AGGREGATE
    if args.mode == "aggregate":
        aggregate_results(args.results_dir)



if __name__ == "__main__":
    main()
