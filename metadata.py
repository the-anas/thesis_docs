import rasterio
from pathlib import Path
import torch


def metadata_reader(path):
    with rasterio.open(path) as src:
        img = src.read()
        img = torch.from_numpy(img).float()
        profile = src.profile  # metadata (dtype, CRS, transform, etc.)

        return profile, img


# ESA DATASET ===============================
#  tif images are too big to open on local computer i think
# for ESA dataset they defe are, have gotten the following error
# PIL.Image.DecompressionBombError: Image size (1296000000 pixels) exceeds limit of 178956970 pixels, could be decompression bomb DOS attack.


# esa image metadata exploreation
# ==================
"""
metadata:
{'driver': 'GTiff', -> image is georeferenced 
'dtype': 'uint8', -> enncoding 
'nodata': 0.0, -> 0 is a non value (valid values 1-255)
'width': 36000, 'height': 36000, -> size of the image, would be about 1.3gb uncompressed 
'count': 1, -> single band -> This is a classification map and not really a proper image
'crs': CRS.from_wkt('GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AXIS["Latitude",NORTH],AXIS["Longitude",EAST],AUTHORITY["EPSG","4326"]]'), 
'transform': Affine(8.333333333333333e-05, 0.0, 0.0, 0.0, -8.333333333333333e-05, 33.0), -> maps position to earth  
'blockxsize': 1024, 'blockysize': 1024, 'tiled': True, -> this is the size of a single tile, image is tiled for compression on disk and easier time reading -> this is for phyisical storage on disk
'compress': 'deflate', -> lossless compression -> It gets decopmressed as you read it
'interleave': 'band'}
"""
# -> (1, 36000, 36000) why is it a single band??
# => ESA world cover does not contain satellite images but land cover maps, which i think is useless

# ==================================================================

# HRSC2016 dataset
# file format is bmp -> standard image (not special satellite image)
# => can be visualzed, these are basic areial images, don't even look like satellite ones
# => do not appear to be high resolution either in this format (maybe there is more in uncompressed files)

from PIL import Image
import numpy as np

img = Image.open("/home/anasnamouchi/thesis/src/hrsc_image.bmp")
arr = np.array(img)

print(img.mode)  # 'RGB'
print(arr.shape)  # (753, 1166, 3)
print(arr.dtype)  # uint8

# => regular image, all look extremely basic and not high resolution to me
# why did it say hrsc on kaggle?? idk

# =====================================
# SSL4EO
# => dataset is also divided into the patches seperately, not images all togther
# ==============================================
# the FLAIR dataset has the patches seperate from each other, im not touching that for now
# ===================================================

# EUROSat
# metadata:
"""
{'driver': 'GTiff', -> georeferenced
'dtype': 'uint16', -> encoding, two bits per pixel (1-65535)
Note by chatgpt to be reviewed later:
reflectance is often stored as: reflectance × 10 000 → uint16 So a value like: 4231 → 0.4231 reflectance

'nodata': None, -> no special character for invalid pixel
'width': 64, 'height':n64, 
=> this is likely not a full image
=> from the paper, we see that this is a dataset of patches, not full images, 
=> this is by design, it was made for classification
=> i dont think this is a big deal since it contains all the bands, can be accounted for as just images
'count': 13, -> 13 bands, most likely bands from sent-2
'crs': CRS.from_wkt('PROJCS["WGS 84 / UTM zone 35N",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",27],PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["EPSG","32635"]]'), 
'transform': Affine(10.00570688714736, 0.0, 624602.2348443292, 0.0, -9.994088099999352, 4877286.033637), 
=> 2 lines above, coordinate system, pixel resolution 10 meters
'blockxsize': 64, 'blockysize': 4, 
'tiled': False, 'interleave': 'pixel'}
"""
img = torch.from_numpy(img).float()
img.shape  # -> ([13,64,64])

# ================
# ROIs (this is the first version form sent1, containing c-band imagery with two bands)
with rasterio.open("/home/anasnamouchi/thesis/src/weird1_image.tif") as src:
    img = src.read()  # shape: (bands, height, width)
    profile = src.profile  # metadata (dtype, CRS, transform, etc.)

profile
img = torch.from_numpy(img).float()
img.shape  # => has two bands (sent 1 i assume)

# ========================
# bigearth1 (this is the first version form sent1, containing c-band imagery with two bands)
with rasterio.open("/home/anasnamouchi/thesis/src/bigearthnet1_image.tif") as src:
    img = src.read()  # shape: (bands, height, width)
    profile = src.profile  # metadata (dtype, CRS, transform, etc.)

profile
img = torch.from_numpy(img).float()

img.shape  # => has 1 band, wtff
# => this actually makes sense, the two bands are completely seperated, there are two different files for them
# ====================================
#
# ROI sent-2
img_path = "/home/anasnamouchi/thesis/src/from_remote/ROI_image.tif"
data, img = metadata_reader(img_path)
data
img.shape # torch.Size([13, 256, 256])
# => normally does not need padding
# notes from metadata
"""
- contains 13 bands, just like sent2 is supposed to
- location on earth can is preserved
""" 


# ======================
# Bigearethnet2

repo_root = Path(
    "/home/anasnamouchi/thesis/src/from_remote/S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_90_90/"
)

all_metadata = []
all_img_shape = []
for path in repo_root.rglob("*"):
    if path.is_file():
        prf, img = metadata_reader(path)
        all_metadata.append(prf)
        all_img_shape.append(img.shape)

# contains different shapes, most likely due to different spectral resolutions, 
# [] how is this normally handeled
for x in all_img_shape:
    print(x)

# nothing special in metadata besides the different spectral resolutions between channels
for x in all_metadata:
    print(x)
    print("\n")
# padding might not be enough for this
# can't be used for experiments until we handle a 