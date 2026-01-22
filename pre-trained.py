import rasterio
import torch
import compressai
import torch.nn.functional as F
from pympler import asizeof

model = compressai.zoo.bmshj2018_hyperprior(
    quality=8, metric="ms-ssim", pretrained=True, progress=True
).eval()

# pass single rgb image through
with rasterio.open("/home/anasnamouchi/thesis/src/hrsc_image.bmp") as src:
    img = src.read()  # shape: (bands, height, width)


img = torch.from_numpy(img).float()

# [] do i need to normalize? why?
# [] if this is truly a standard why donest model normalize by itself.
# x = torch.from_numpy(img).float() / 255.0  # -> [0,1]

factor = model.downsampling_factor  # 64 for this model
print(f"model factor is {factor}")
img = img.unsqueeze(0)
print(f"image shape is {img.shape}")
print(f"size of unpadded un compressed: {(img.numel() * img.element_size())}")
# getting values to pad
h, w = img.shape[-2:]
pad_h = (factor - h % factor) % factor
pad_w = (factor - w % factor) % factor

x_pad = F.pad(img, (0, pad_w, 0, pad_h), mode="replicate")

print(f"shape of padded image: {x_pad.shape}")
# print(F"size of padded uncompressed: {(x_pad.numel()*x_pad.element_size())}")

compressed = model.compress(x_pad)
print(type(compressed))
print(compressed.keys())
print(compressed["shape"])
print(f"size of compressed: {asizeof.asizeof(compressed)}")
# we observe that there is indeed compression (smaller file size)

# =====================


# (optional but safe) ensure entropy-model tables are ready
# model.update(force=True)

# with rasterio.open("/home/anasnamouchi/thesis/src/hrsc_image.bmp") as src:
#     img = src.read()  # (C,H,W), typically uint8


# out = model.decompress(compressed["strings"], compressed["shape"])

# x_hat = out["x_hat"][..., :h, :w]  # crop back to original size

# few notes about hrsc2016
# - images have arbitrary shapes and need padding to work on this model
# [] how does it work on other models
# [] are the images suppose to always have fix lengths that work with the models in a real scenarion or what exactly
#
#
