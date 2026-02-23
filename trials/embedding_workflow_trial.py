import torch
from rshf.satmae import SatMAE
import torchvision.io as io
device = "cpu"  # consumer CPU is fine

# A lighter SatMAE checkpoint (ViT-Base). You can swap for others.
# few other embeddings to try out
    # MVRL/satmaepp_ViT-L_pretrain_fmow_rgb -> loads
    # MVRL/remote-clip-resnet-50 -> IDK how to properl load this, autoload acts sus with it


model_id = "MVRL/satmaepp_ViT-L_pretrain_fmow_rgb"
model = SatMAE.from_pretrained(model_id).to(device).eval()


x = io.read_image("/home/anas/datasets/ssl42eo-small-torun/0000006_20201109T110301_20201109T110451_T31UDQ.png")  # uint8, (C,H,W)
x_norm = x.float() / 255.0  


x = x.unsqueeze(0)

import torch.nn.functional as F

x = F.interpolate(
    x,
    size=(224, 224),
    mode="bilinear",
    align_corners=False
)

x.shape
with torch.no_grad():
    tokens, _, _ = model.forward_encoder(x, mask_ratio=0.0)  # (B, num_tokens, dim)

tokens[0].shape


# Option B: CLS token if present (some ViT configs)
# emb = tokens[:, 0, :]

# Normalize (often helpful)
emb = emb / emb.norm(dim=-1, keepdim=True)

print("embedding shape:", emb.shape)
