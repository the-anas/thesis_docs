from models import ScaleHyperprior
import torch

checkpoint_path = "/home/anas/downloaded_08_04_2026/checkpoints/checkpoint_best_loss_basic-hyperprior_192_320_192.pth.tar"
model = ScaleHyperprior(192, 320)
checkpoint = torch.load(checkpoint_path, map_location="cpu")

state_dict = checkpoint["state_dict"]

# Strip the DataParallel 'module.' prefix if present
if all(k.startswith("module.") for k in state_dict):
    state_dict = {k[len("module."):]: v for k, v in state_dict.items()}

model.load_state_dict(state_dict)
model.eval()