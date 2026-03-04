from pathlib import Path
import torch 
from src.models import ScaleHyperpriorCrossAttention
from src.new_utils import LowResMask

folder = Path("/home/anas/thesis/images/to_compress/originals")

from PIL import Image
from torchvision import transforms


transform = transforms.ToTensor()

patch_size = 16

test_transforms = transforms.Compose(
    [transforms.CenterCrop(patch_size), transforms.ToTensor()]
)


files = [f for f in folder.iterdir() if f.is_file()]
embedding_model = LowResMask()
checkpoint = torch.load("checkpoint_best_loss.pth.tar", map_location="cpu")
my_model = ScaleHyperpriorCrossAttention(30, 24, 30, embedding_model=embedding_model, embedding_type="downsample_cnn")
sd = checkpoint["state_dict"]
print("Model keys sample:", list(my_model.state_dict().keys())[:30])
print("Checkpoint keys sample:", list(sd.keys())[:30])

# And most importantly:
print("Current g_a.local:", my_model.g_a.local)

my_model.load_state_dict(checkpoint["state_dict"])

my_model.eval()
my_model.update(force=True)   # important

for file in files:
    image = image = Image.open(file).convert("RGB")
    tensor = test_transforms(image)
    tensor = tensor.unsqueeze(0)
    strings,shape = my_model.compress(tensor)
    x_hat = my_model.decompress(strings, shape)
    print(type(x_hat))
    print(x_hat)
    print("\n")