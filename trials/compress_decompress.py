from pathlib import Path
import torch 
from src.models import ScaleHyperpriorCrossAttention
from src.new_utils import LowResMask
import os

cropped_path = "/home/anas/thesis/results/cropped/"
reconstructed_path = "/home/anas/thesis/results/reconstructed/"

def save_tensor_as_image(tensor, path):

    tensor = tensor.clamp(0, 1)
    # 2. Rescale back to [0, 255]
    tensor = (tensor * 255).byte()   # or .to(torch.uint8)
    # 3. Convert to (H, W, C) for PIL
    img_array = tensor.permute(1, 2, 0).numpy()
    # 4. Save
    print(img_array)
    print(tensor.shape)
    img = Image.fromarray(img_array)
    print(path)
    img.save(path)

folder = Path("/home/anas/thesis/images/to_compress/originals")

output_dir = "/home/anas/thesis/results"
os.makedirs(output_dir, exist_ok=True)

from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

transform = transforms.ToTensor()

patch_size = 16

test_transforms = transforms.Compose(
    [transforms.CenterCrop(256), transforms.ToTensor()]
)


files = [f for f in folder.iterdir() if f.is_file()]
embedding_model = LowResMask()
checkpoint = torch.load("checkpoint_best_loss.pth.tar", map_location="cpu")
my_model = ScaleHyperpriorCrossAttention(30, 24, 30, embedding_model=embedding_model, embedding_type="downsample_cnn")
sd = checkpoint["state_dict"]

# print("Model keys sample:", list(my_model.state_dict().keys())[:30])
# print("Checkpoint keys sample:", list(sd.keys())[:30])

# # And most importantly:
# print("Current g_a.local:", my_model.g_a.local)

my_model.load_state_dict(checkpoint["state_dict"])

my_model.eval()
my_model.update(force=True)   # important

for file in files:
    image = image = Image.open(file).convert("RGB")
    tensor = test_transforms(image)
    # add 1 at batch dimension
    save_tensor_as_image(tensor, cropped_path + file.name)
    tensor = tensor.unsqueeze(0)
    # print(tensor*255)
    # save_image((tensor*255), f"{output_dir}/cropped/{file.name}")
    
    out = my_model.compress(tensor)
    x_hat = my_model.decompress(out["strings"], out["shape"])
    # print(type(x_hat))
    # print(type(x_hat["x_hat"]))
    # print(x_hat["x_hat"].shape)
    save_tensor_as_image(x_hat["x_hat"].squeeze(0), reconstructed_path+file.name)
    # print("\n")
     