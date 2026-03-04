# import sys
# from pathlib import Path

# ROOT = Path("/home/anas/thesis")

# sys.path.insert(0, str(ROOT))


import rasterio
import torch
import compressai
import torch.nn.functional as F
from pympler import asizeof
from src.models import ScaleHyperpriorCrossAttention

from src.new_utils import LowResMask
import torch

#####
# Some reusable functions
def simple_parameters_number(model):
    num_params = sum(p.numel() for p in model.parameters())
    return num_params

def detailed_parameter_count(model):

    # for name, p in model_pretrained.named_parameters():
    #     print(f"{name:40} {p.numel():,}")

    for name, module in model.named_modules():
        params = sum(p.numel() for p in module.parameters(recurse=False))
        if params > 0:
            print(f"{name:30} {params:,}")
# ###########


checkpoint = torch.load("checkpoint_best_loss.pth.tar", map_location="cpu")


model_pretrained = compressai.zoo.bmshj2018_hyperprior(
    quality=8, metric="ms-ssim", pretrained=True, progress=True
).eval()

embedding_model = LowResMask()

my_model = ScaleHyperpriorCrossAttention(30, 24, 30, embedding_model=embedding_model, embedding_type="downsample_cnn")

my_model.load_state_dict(checkpoint["state_dict"])


# why the difference 
print("Pre trained model:")
# print(f"pretrained model parameters: {simple_parameters_number(model_pretrained)}") # -> 11816323
detailed_parameter_count(model_pretrained)
print("\n\n")
print("My model:")
# print(f"my model parameters: {simple_parameters_number(my_model)}")# -> 424725
detailed_parameter_count(my_model)

