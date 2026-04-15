from models import ScaleHyperprior, ScaleHyperpriorBahdanau
import torch
from train import images_every_10_epochs


checkpoint_path = "/home/anas/from_cluster/downloaded_14_04_2026/checkpoints/checkpoint_bahdanau-hyperprior_128_192_128.pth.tar"

# "/home/anas/downloaded_08_04_2026/checkpoints/checkpoint_best_loss_basic-hyperprior_192_320_192.pth.tar"
model = ScaleHyperpriorBahdanau(128,192,128)
checkpoint = torch.load(checkpoint_path, map_location="cpu")

state_dict = checkpoint["state_dict"]

# Strip the DataParallel 'module.' prefix if present
if all(k.startswith("module.") for k in state_dict):
    state_dict = {k[len("module."):]: v for k, v in state_dict.items()}

model.load_state_dict(state_dict)
model.eval()

print("Model loaded successfully. Now saving example reconstructions every 10 epochs...")

images_every_10_epochs(
    image_dir="/home/anas/datasets/reconstruction_dir",
    model=model,
    epoch=0,
    reconstruction_path="/home/anas/datasets/reconstruction_dir/reconstructions",
    cropped_path="/home/anas/datasets/reconstruction_dir/cropped",
)

print("Done saving example reconstructions.")