"""
boxplots.py

Usage:
    python boxplots.py <results_json> [--output-dir <dir>]

Produces one PNG per metric, pooling all images into a single box per metric.
No per-image separation, no title.
"""

import argparse
import json
import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)

PAIRWISE_METRICS = ["cosine", "mi", "kl_ij", "kl_ji"]
METRIC_LABELS = {
    "entropy": "Entropy (bits)",
    "cosine":  "Cosine Similarity",
    "mi":      "Mutual Information (bits)",
    "kl_ij":   "KL(i || j)  (bits)",
    "kl_ji":   "KL(j || i)  (bits)",
}


def build_dataframes(results: dict):
    entropies = []
    pairwise_rows = []

    for data in results.values():
        entropies.extend(data["entropy"])
        for pair in data["pairwise"]:
            pairwise_rows.append({
                "cosine": pair["cosine"],
                "mi":     pair["mi"],
                "kl_ij":  pair["kl_ij"],
                "kl_ji":  pair["kl_ji"],
            })

    return entropies, pd.DataFrame(pairwise_rows)


def plot(results_path: str, output_dir: str | None):
    with open(results_path) as f:
        results = json.load(f)

    entropies, df_pair = build_dataframes(results)

    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(results_path))
    os.makedirs(output_dir, exist_ok=True)

    # entropy is a plain list, pairwise metrics are columns in df_pair
    all_data = {"entropy": pd.DataFrame({"entropy": entropies})}
    for m in PAIRWISE_METRICS:
        all_data[m] = df_pair[[m]].rename(columns={m: m})

    for metric, df in all_data.items():
        fig, ax = plt.subplots(figsize=(4, 5))
        sns.boxplot(
            data=df, y=metric, ax=ax,
            width=0.4,
            flierprops=dict(marker="o", markersize=2, alpha=0.4),
        )
        ax.set_xlabel("")
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.set_xticks([])
        plt.tight_layout()

        out = os.path.join(output_dir, f"boxplot_{metric}.png")
        fig.savefig(out, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"Saved: {out}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("results_json")
    p.add_argument("--output-dir", default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot(args.results_json, args.output_dir)