import torch
from rshf.satmae import SatMAE
import torchvision.io as io
import torch.nn.functional as F
import torch.nn as nn
from compressai.models.utils import conv
from compressai.layers import GDN
from PIL import Image
from torchvision import transforms
from pathlib import Path

# function to patch within model 
def patchify(images: torch.Tensor, patch_size: int = 16):
    """
    Patchify a batch of images without mixing samples.

    Args:
        images: Tensor of shape (B, 3, H, W)
        patch_size: size of each square patch (ph = pw)

    Returns:
        patches: Tensor of shape (B, P, 3, patch_size, patch_size)
                 where P = (H // patch_size) * (W // patch_size)
    """
    B, C, H, W = images.shape
    assert H % patch_size == 0 and W % patch_size == 0, \
        "H and W must be divisible by patch_size"

    ph = pw = patch_size
    h_patches = H // ph
    w_patches = W // pw

    # Step 1: reshape
    x = images.reshape(
        B, C,
        h_patches, ph,
        w_patches, pw
    )

    # Step 2: move patch dimensions together
    x = x.permute(0, 2, 4, 1, 3, 5)

    # Step 3: flatten patch grid
    patches = x.reshape(
        B,
        h_patches * w_patches,
        C,
        ph,
        pw
    )

    return patches



# fucntion to call and use embedding model within compression model 


# A lighter SatMAE checkpoint (ViT-Base). You can swap for others.
# few other embeddings to try out
    # MVRL/satmaepp_ViT-L_pretrain_fmow_rgb -> loads
    # MVRL/remote-clip-resnet-50 -> IDK how to properl load this, autoload acts sus with it

def embed_image(embedding_model, 
                image, 
                device="cude" if torch.cuda.is_available() else "cpu"):
    """
    Let the model take in the actual embedding model as an arg
    Load the embedding model in the trainnning loop
    Image assumed to already have been normalized
    """

    # [] needs to handle batch dimension
    # => supposedly happened automatically here, just make sure it actually works fine 
    # with shapes before hand



    with torch.no_grad():
        tokens, _, _ = embedding_model.forward_encoder(image, mask_ratio=0.0)  # (B, num_tokens, dim)
    
    return tokens



def unpatchify(patches: torch.Tensor, grid_hw: tuple[int, int]) -> torch.Tensor:
    """
    Reconstruct images from non-overlapping patches.

    Args:
        patches: Tensor of shape (B, P, C, Hp, Wp)
        grid_hw: (Gh, Gw) where P = Gh * Gw (patch grid height/width)

    Returns:
        images: Tensor of shape (B, C, Gh*Hp, Gw*Wp)
    """
    if patches.ndim != 5:
        raise ValueError(f"patches must be (B,P,C,Hp,Wp), got {tuple(patches.shape)}")

    B, P, C, Hp, Wp = patches.shape
    Gh, Gw = grid_hw

    if P != Gh * Gw:
        raise ValueError(f"P={P} must equal Gh*Gw={Gh*Gw}")

    x = patches.reshape(B, Gh, Gw, C, Hp, Wp)   # (B, Gh, Gw, C, Hp, Wp)
    x = x.permute(0, 3, 1, 4, 2, 5)             # (B, C, Gh, Hp, Gw, Wp)
    x = x.reshape(B, C, Gh * Hp, Gw * Wp)       # (B, C, H, W)

    return x



class LowResMask(nn.Module):
    def __init__(self, kernel_size=16, stride=16):
        super(LowResMask, self).__init__()
        
        self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        mask = self.pool(x)
        return mask
    
    # [] produce a downsampling

class DownsampleCNN(nn.Module):
    def __init__(self, N, K):
        super(DownsampleCNN, self).__init__()
    # good enough downsampling cnn
        self.conv_global_y = nn.Sequential(
            conv(3, N), # 128
            GDN(N),
            conv(N, N), #64
            GDN(N),
            conv(N, N), #32
            GDN(N),
            conv(N, N), #16
            GDN(N),
            conv(N, N), #8
            GDN(N),
            conv(N, K), #4
            # GDN(N),
            # conv(N, N), #
            # GDN(N),
            # conv(N, K),  # -> (B*P, K, h', w')
        )
    
    def forward(self, x):
        embedding = self.conv_global_y(x)
        return embedding
    
def save_tensor_as_image(tensor, path):

    tensor = tensor.clamp(0, 1)
    # 2. Rescale back to [0, 255]
    tensor = (tensor * 255).byte()   # or .to(torch.uint8)
    # 3. Convert to (H, W, C) for PIL
    img_array = tensor.permute(1, 2, 0).cpu().numpy()
    # 4. Save
    # print(img_array)
    # print(tensor.shape)
    img = Image.fromarray(img_array)
    # print("PATH is")
    # print(path)
    img.save(path)

# new_utils_v2.py

class PatchEmbedCNN(nn.Module):
    """
    Lightweight CNN to embed a single patch.
    For a 16x16 patch: 2 stride-2 convs -> 4x4 spatial output.
    For a 32x32 patch: 3 stride-2 convs -> 4x4 spatial output.
    """
    def __init__(self, K: int, patch_size: int = 16):
        super().__init__()
        # choose depth based on patch size so output is always ~4x4
        n_stride = max(1, (patch_size - 1).bit_length() - 2)   # 16->2, 32->3, 64->4
        layers = [conv(3, K), GDN(K)]
        for _ in range(n_stride - 1):
            layers += [conv(K, K), GDN(K)]
        layers += [conv(K, K, stride=1, kernel_size=3)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):  # (B*P, 3, Hp, Wp) -> (B*P, K, ~4, ~4)
        return self.net(x)
    


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────

def patch_entropy(patch: torch.Tensor, num_bins: int = 64) -> float:
    vals = patch.flatten().float()
    if vals.max() > 1.0:
        vals = vals / 255.0
    hist = torch.histc(vals, bins=num_bins, min=0.0, max=1.0)
    prob = hist / hist.sum()
    prob = prob[prob > 0]
    return -torch.sum(prob * torch.log2(prob)).item()


def cosine_similarity(patch_a: torch.Tensor, patch_b: torch.Tensor) -> float:
    vec_a = patch_a.flatten().float()
    vec_b = patch_b.flatten().float()
    return F.cosine_similarity(vec_a.unsqueeze(0), vec_b.unsqueeze(0)).item()


def mutual_information(patch_a: torch.Tensor, patch_b: torch.Tensor, num_bins: int = 64) -> float:
    def to_prob(patch):
        vals = patch.flatten().float()
        if vals.max() > 1.0:
            vals = vals / 255.0
        hist = torch.histc(vals, bins=num_bins, min=0.0, max=1.0)
        return hist / hist.sum()

    def entropy_from_prob(p):
        p = p[p > 0]
        return -torch.sum(p * torch.log2(p))

    prob_a = to_prob(patch_a)
    prob_b = to_prob(patch_b)

    vals_a = patch_a.flatten().float()
    vals_b = patch_b.flatten().float()
    if vals_a.max() > 1.0:
        vals_a = vals_a / 255.0
    if vals_b.max() > 1.0:
        vals_b = vals_b / 255.0

    num_bins_j = num_bins
    idx_a = (vals_a * (num_bins_j - 1)).long().clamp(0, num_bins_j - 1)
    idx_b = (vals_b * (num_bins_j - 1)).long().clamp(0, num_bins_j - 1)
    joint_hist = torch.zeros(num_bins_j, num_bins_j)
    joint_hist.index_put_((idx_a, idx_b), torch.ones_like(vals_a), accumulate=True)
    prob_joint = joint_hist / joint_hist.sum()

    H_a  = entropy_from_prob(prob_a)
    H_b  = entropy_from_prob(prob_b)
    H_ab = entropy_from_prob(prob_joint.flatten())

    return max((H_a + H_b - H_ab).item(), 0.0)


def kl_divergence(patch_a: torch.Tensor, patch_b: torch.Tensor, num_bins: int = 64, epsilon: float = 1e-10) -> float:
    def to_prob(patch):
        vals = patch.flatten().float()
        if vals.max() > 1.0:
            vals = vals / 255.0
        hist = torch.histc(vals, bins=num_bins, min=0.0, max=1.0)
        return hist / hist.sum()

    p = to_prob(patch_a)
    q = to_prob(patch_b) + epsilon
    q = q / q.sum()

    return max(torch.sum(p * torch.log2((p + epsilon) / q)).item(), 0.0)


def average_entropy(likelihood):
    return (-torch.log2(likelihood + 1e-9)).mean() 


def load_image(path: str, image_size: int = 256) -> torch.Tensor:
    """
    Load an image and return a (1, C, H, W) tensor, normalized to [0, 1].
    Resizes to image_size x image_size so patchify divides evenly.
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),   # converts to (C, H, W) and normalizes to [0, 1]
    ])
    image = Image.open(path).convert("RGB")
    tensor = transform(image)        # (C, H, W)
    return tensor.unsqueeze(0)       # type: ignore # (1, C, H, W)  ← batch dim for patchify


class DownsampleCNN_v2(nn.Module):
    """
    Shallow downsampler that preserves 4x4 spatial output for L_k=16 tokens.
    For a 16x16 patch: 2 stride-2 convs -> 4x4.
    """
    def __init__(self, N, K):
        super().__init__()
        self.conv_global_y = nn.Sequential(
            conv(3, N),   # 16 -> 8
            GDN(N),
            conv(N, K),   # 8  -> 4
        )

    def forward(self, x):
        return self.conv_global_y(x)
    
    