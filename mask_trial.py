import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt



# ---------------------------
# 1. Load image
# ---------------------------
img = Image.open("hrsc_image.bmp").convert("RGB")

# ############
type(img)
img.size


transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),  # (3, H, W), values in [0,1]
])

x = transform(img).unsqueeze(0)  # (1, 3, 256, 256)
x.shape

# ---------------------------
# 2. CNN that produces a low-res mask
# ---------------------------
class LowResMaskCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # 256 → 128
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, stride=2, padding=1), # 128 → 64
            nn.Sigmoid(),  # mank in [0,1]
        )

    def forward(self, x):
        return self.net(x)

model = LowResMaskCNN()

# ---------------------------
# 3. Forward pass
# ---------------------------
with torch.no_grad():
    # mask = model(x)  # (1, 1, 64, 64)

    mask = F.avg_pool2d(x, kernel_size=8, stride=8)
# (1, 3, 256, 256) → (1, 3, 32, 32)


print("Low-res mask tensor shape:", mask.shape)

# ---------------------------
# 4. Save mask as image
# ---------------------------
mask_img = mask.squeeze().cpu().numpy()  # (64, 64)

img = mask_img.transpose(1, 2, 0)  # (H, W, 3)
plt.imshow(img)
plt.axis("off")

# plt.imshow(mask_img, cmap="gray")
# plt.axis("off")
plt.savefig("low_res_mask.png", bbox_inches="tight")
plt.show()

