import zarr.storage
import xarray as xr

store = zarr.storage.ZipStore(
    "/home/anas/thesis/checkpoints/ssl4eos12_train_seasonal_data_003791.zarr.zip"
    , mode='r')
ds = xr.open_zarr(store, consolidated=False)
print(ds)


# /home/anas/thesis/checkpoints/ssl4eos12_train_seasonal_data_003791.zarr.zip

############
# trying to save single image, make sure  all is well with these

import zarr.storage
import xarray as xr
from PIL import Image
import numpy as np

# load
store = zarr.storage.ZipStore(
    "/home/anas/thesis/checkpoints/ssl4eos12_train_seasonal_data_003791.zarr.zip",
    mode='r')
ds = xr.open_zarr(store, consolidated=False)
data = ds.bands.values  # (64, 4, 3, 264, 264)

# pick any sample and timestamp
img = data[30, 2]  # (3, 264, 264)

# convert to HWC uint8 for PIL
img_hwc = img.transpose(1, 2, 0).astype(np.uint8)  # (264, 264, 3)

# save
Image.fromarray(img_hwc).save('sample.png')
print("saved to sample.png")

# => IMAGE SAVED ACTUALLY