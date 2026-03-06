import torch
from rshf.satmae import SatMAE
import torchvision.io as io
import torch.nn.functional as F
import torch.nn as nn
from compressai.models.utils import conv
from compressai.layers import GDN
from PIL import Image


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