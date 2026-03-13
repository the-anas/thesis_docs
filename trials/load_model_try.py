from models import ScaleHyperpriorBahdanau, ScaleHyperpriorCrossAttention
import torch 
from new_utils import LowResMask, save_tensor_as_image

embedding_model = LowResMask()


checkpoint = torch.load(
    "/home/anas/thesis/checkpoints/most_basic_model_checkpoints/checkpoint_big_model_small_dataset.pth.tar",
    map_location="cpu")
my_model = ScaleHyperpriorCrossAttention(192, 320, 192, embedding_model=embedding_model, embedding_type="downsample_cnn")
sd = checkpoint["state_dict"]

my_model.load_state_dict(checkpoint["state_dict"])


# print(checkpoint.keys())
# # typically: dict_keys(['epoch', 'state_dict', 'optimizer', ...])

# # look at the state dict
# state_dict = checkpoint['state_dict']  # or just ckpt if it was saved as state_dict directly

# for name, tensor in state_dict.items():
#     print(f"{name:60s} {tuple(tensor.shape)}")

#     # N = first conv output channels
# N = state_dict['g_a.local.0.weight'].shape[0]

# # M = encoder output channels (last conv in g_a)
# M = state_dict['g_a.local.4.weight'].shape[0]

# # K = embedding dim (k_proj input dim)
# K = state_dict['g_a.k_proj.weight'].shape[1]

# print(f"N={N}, M={M}, K={K}")