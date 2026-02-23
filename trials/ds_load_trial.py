import torchvision


eurosat_ds= torchvision.datasets.EuroSAT(root="/home/anas/datasets/eurosat_1", download=True)
type(eurosat_ds)
len(eurosat_ds)
eurosat_ds.samples
