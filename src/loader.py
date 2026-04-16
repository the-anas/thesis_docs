import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import zarr.storage
import xarray as xr
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
# from models import ScaleHyperprior, ScaleHyperpriorBahdanau
from models import ScaleHyperprior, ScaleHyperpriorBahdanau, ScaleHyperpriorBahdanau_v2

# models_dict = {"basic-hyperprior": ScaleHyperprior, 
#                "bahdanau-hyperprior":ScaleHyperpriorBahdanau}


models_dict = {
    "basic-hyperprior":      ScaleHyperprior,
    "bahdanau-hyperprior":   ScaleHyperpriorBahdanau,    # old — loads existing checkpoints
    "bahdanau-hyperprior-v2": ScaleHyperpriorBahdanau_v2, # new — train fresh
}


class SSL4EOS12RGBDataset(Dataset):
    def __init__(self, data_dir, crop_size=256, is_train=True):
        # if is_train:
        #     data_dir = os.path.join(data_dir, "train/S2RGB/")
        # else:
        #     data_dir = os.path.join(data_dir, "val/S2RGB/")

        self.files                 = sorted(glob.glob(os.path.join(data_dir, '*.zarr.zip')))
        self.crop_size             = crop_size
        self.is_train              = is_train
        self.samples_per_file      = 64
        self.timestamps_per_sample = 4
        self.images_per_file       = self.samples_per_file * self.timestamps_per_sample  # 256
        self.total_images          = len(self.files) * self.images_per_file

        assert len(self.files) > 0, f"No .zarr.zip files found in {data_dir}"

        # simple file cache
        self._cache_path = None
        # self._cache_data = None   # ← defined here in __init__, that's where it comes from

    def __len__(self):
        return self.total_images

    def __getitem__(self, idx):
        # ---- map flat idx to (file, sample, time) ----
        file_idx   = idx // self.images_per_file
        remainder  = idx  % self.images_per_file
        sample_idx = remainder // self.timestamps_per_sample
        time_idx   = remainder  % self.timestamps_per_sample

        filepath = self.files[file_idx]

        # ---- open file only if not already cached ----
        if filepath != self._cache_path:
            store            = zarr.storage.ZipStore(filepath, mode='r')
            ds               = xr.open_zarr(store, consolidated=False)
            self._cache_data = ds.bands.values   # (64, 4, 3, 264, 264)
            self._cache_path = filepath

        # ---- extract single image ----
        img = self._cache_data[sample_idx, time_idx]             # (3, 264, 264) uint8

        # ---- numpy -> float tensor [0, 1] ----
        img = torch.from_numpy(img.astype(np.float32)/ 255.0)   # (3, 264, 264)

        # ---- crop ----
        if self.is_train:
            i, j, h, w = transforms.RandomCrop.get_params(img, (self.crop_size, self.crop_size))
            img = TF.crop(img, i, j, h, w)
            # img = transforms.RandomCrop(img, self.crop_size)
        else:
            img = TF.center_crop(img, [self.crop_size])            # (3, 256, 256)

        return img


# ---- datasets ----
# train_dataset = SSL4EOS12RGBDataset('/home/anas/thesis/checkpoints/train/', is_train=True)
# val_dataset   = SSL4EOS12RGBDataset('/home/anas/thesis/checkpoints/val/',   is_train=False)

# print(f"Train images: {len(train_dataset):,}")
# print(f"Val images:   {len(val_dataset):,}")

# # ---- loaders ----
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,  num_workers=4, pin_memory=True)
# val_loader   = DataLoader(val_dataset,   batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

# # ---- sanity check ----
# batch = next(iter(train_loader))
# print(f"Batch shape:  {batch.shape}")                            # (8, 3, 256, 256)
# print(f"Value range:  [{batch.min():.3f}, {batch.max():.3f}]")  # [0.0, 1.0]