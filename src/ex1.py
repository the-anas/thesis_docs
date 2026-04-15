import os
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from itertools import combinations
from new_utils import patchify, patch_entropy, cosine_similarity, mutual_information, kl_divergence
import pickle 
import json

from new_utils import load_image

def run_pipeline(folder_path: str, patch_size: int = 32, image_size: int = 224):
    """
    For every image in folder:
      - Extract patches using patchify
      - Compute entropy for every patch
      - Compute pairwise cosine similarity, MI, KL divergence for every patch pair

    Args:
        folder_path: path to folder containing images
        patch_size:  size of each square patch
        image_size:  images are resized to (image_size x image_size) before patching

    Returns:
        results: dict structured as:
            {
                "image_name.jpg": {
                    "entropy":  [e0, e1, e2, ...],          # one per patch
                    "pairwise": [                            # one per pair
                        {
                            "patch_i":   int,
                            "patch_j":   int,
                            "cosine":    float,
                            "mi":        float,
                            "kl_ij":     float,   # KL(i || j)
                            "kl_ji":     float,   # KL(j || i)
                        },
                        ...
                    ]
                },
                ...
            }
    """
    valid_extensions = {".jpg", ".jpeg", ".png", ".tiff", ".bmp"}
    image_files = [
        f for f in os.listdir(folder_path)
        if os.path.splitext(f)[1].lower() in valid_extensions
    ]

    if not image_files:
        print(f"No images found in {folder_path}")
        return {}

    results = {}

    for filename in image_files:
        print(f"\nProcessing {filename}...")
        path = os.path.join(folder_path, filename)

        # --- Load and patchify ---
        image_tensor = load_image(path, image_size=image_size)  # (1, C, H, W)
        patches = patchify(image_tensor, patch_size=patch_size)  # (1, P, C, ph, pw)
        patches = patches[0]                                     # (P, C, ph, pw) drop batch dim

        num_patches = patches.shape[0]
        print(f"  {num_patches} patches of size {patch_size}x{patch_size}")

        # --- Entropy: one value per patch ---
        entropies = []
        for i in range(num_patches):
            e = patch_entropy(patches[i])
            entropies.append(e)
            print(f"  Patch {i:03d} entropy: {e:.4f} bits")

        # --- Pairwise metrics ---
        pairwise_results = []
        patch_pairs = list(combinations(range(num_patches), 2))
        print(f"  Computing {len(patch_pairs)} patch pairs...")

        for i, j in patch_pairs:
            cos  = cosine_similarity(patches[i], patches[j])
            mi   = mutual_information(patches[i], patches[j])
            kl_ij = kl_divergence(patches[i], patches[j])   # KL(i || j)
            kl_ji = kl_divergence(patches[j], patches[i])   # KL(j || i)

            pairwise_results.append({
                "patch_i": i,
                "patch_j": j,
                "cosine":  cos,
                "mi":      mi,
                "kl_ij":   kl_ij,
                "kl_ji":   kl_ji,
            })

        results[filename] = {
            "entropy":  entropies,
            "pairwise": pairwise_results,
        }

    return results


if __name__ == "__main__":
    results = run_pipeline(
        folder_path="/home/anas/datasets/exp1",
        patch_size=32,
        image_size=224
    )

    # Example: print most similar patch pair per image by cosine similarity
    for image_name, data in results.items():
        pairs = data["pairwise"]
        most_similar = max(pairs, key=lambda x: x["cosine"])
        print(f"\n{image_name} — most similar pair:")
        print(f"  Patches {most_similar['patch_i']} & {most_similar['patch_j']}")
        print(f"  Cosine: {most_similar['cosine']:.4f}")
        print(f"  MI:     {most_similar['mi']:.4f}")
        print(f"  KL_ij:  {most_similar['kl_ij']:.4f}")
        print(f"  KL_ji:  {most_similar['kl_ji']:.4f}")

        print(type(results))


    with open('results_total.json', 'w') as fp:
        json.dump(results, fp)