# YOU HAVE TO INPUT PASSWORD TO OS DRIVE BEFORE RUNNING FOR THIS TO WORK 
# file coming from m1454690 dataset

import h5py

file_path = "/run/media/anasnamouchi/OS/datasets/m1454690/training.h5"

# keys
with h5py.File(file_path, "r") as f:
    print(list(f.keys()))
# type of object stored under each key

with h5py.File(file_path, "r") as f:
    for key in f.keys():
        print(key, type(f[key]))


with h5py.File("/run/media/anasnamouchi/OS/datasets/m1454690/training.h5", "r") as f:
    for key in f.keys():
        print(key)
        dset = f[key]
        print(dset.shape)
        print(dset.dtype)
        print(dset.attrs)
        print("\n")

# shapes of sen1 and sen2:
# sen1: (352366, 32, 32, 8)
# sen2: (352366, 32, 32, 10)
# => I am guessing these are patches, 8 bands and 10 bands

# lab are classification labels, doesn't matter for what
# => I cannot find any information to reconstruct full images, but it seems to me that the patches will do just fine for now, maybe?? (maybe issue with trying to learn a global similarity??)