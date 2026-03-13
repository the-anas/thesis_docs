import argparse
import random
import shutil
import sys

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms

from compressai.datasets import ImageFolder
from compressai.losses import RateDistortionLoss
from compressai.optimizers import net_aux_optimizer
from compressai.zoo import image_models

from models import ScaleHyperpriorCrossAttention, ScaleHyperpriorBahdanau

from rshf.satmae import SatMAE

from new_utils import LowResMask, save_tensor_as_image

import wandb

from datetime import datetime
from pathlib import Path
from torch.utils.data import Subset, DataLoader
import os
import random
from loader import SSL4EOS12RGBDataset

from torchmetrics.functional import peak_signal_noise_ratio as psnr_metric
from torchmetrics.functional import structural_similarity_index_measure as ssim_metric
# [] not that lpips can only be used with rgb images
# [] also omitting lpips for now, answer the following questions beforehand
    # - are we supposed to run lpips on this specific model or on other alternate pre-trained models?? (different symantics)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity



timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

# [] LPIPS commented out for now
# Initialised once and reused — LPIPS has learnable weights so we keep it as a module
# _lpips_metric = None

# def get_lpips_metric(device):
#     global _lpips_metric
#     if _lpips_metric is None:
#         _lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device)
#         _lpips_metric.eval()
#     return _lpips_metric

# local path
cropped_path = Path("/home/anas/thesis/results/cropped/")
reconstruction_path = Path("/home/anas/thesis/results/reconstructed/")

# mcml cluster paths
# reconstruction_path = Path("/dss/dsshome1/0E/ra42tif2/thesis_docs/images/results/reconstructed/")
# cropped_path = Path("/dss/dsshome1/0E/ra42tif2/thesis_docs/images/results/cropped/")

# cip pool gpu path
# cropped_path = Path("/home/ra42tif/thesis_docs/results/cropped")
# reconstruction_path = Path("/home/ra42tif/thesis_docs/results/reconstructed")


os.makedirs(reconstruction_path, exist_ok=True)
os.makedirs(cropped_path, exist_ok=True)


# save example images from test suite every 10 epochs
def images_every_10_epochs(test_dataset, model,epoch ): 
    device = next(model.parameters()).device   
    model.eval()
    model.update(force=True)

    random_indices = random.sample(range(len(test_dataset)), 10)
    os.makedirs(Path(reconstruction_path/f"epoch_{epoch}"), exist_ok=True) 
    os.makedirs(Path(cropped_path/f"epoch_{epoch}"), exist_ok=True) 

    # Create a subset and a new dataloader
    subset = Subset(test_dataset, random_indices)
    random_loader = DataLoader(subset, batch_size=10, shuffle=False)
    # print("init saving images")
    for ind, tens in enumerate(random_loader):
        # print(f"Currently in image number {ind}")
        counter=0
        # print("shape tens", tens.shape)
        for sec_ind, image in enumerate(tens):
            # print("image type and shape", image.shape, type(image))
            # print(f"counter is {counter}")
            image = image.to(device)
            
            save_tensor_as_image(image, Path(cropped_path / f"epoch_{epoch}"/f"image{sec_ind}_epoch{epoch}.png"))
            # print("cropped saved")
            tensor = image.unsqueeze(0)
            # print("relevant")
            # print(type(tensor))
            # print(tensor.shape)
            out = model.compress(tensor)
            x_hat = model.decompress(out["strings"], out["shape"])
            # print(type(x_hat))
            # print(type(x_hat["x_hat"]))
            # print(x_hat["x_hat"].shape)
            save_tensor_as_image(x_hat["x_hat"].squeeze(0), Path(reconstruction_path / f"epoch_{epoch}"/f"image{sec_ind}_epoch{epoch}.png"))
            # print("reconstructed saved")
            counter+=1
    
    model.train()


def compute_metrics(original, reconstructed):
    """Compute PSNR, SSIM, and LPIPS between original and reconstructed tensors.
    Expects tensors of shape (B, C, H, W) in [0, 1] range.
    Returns a dict with scalar float values.
    """
    device = original.device

    # PSNR & SSIM — torchmetrics handles batches natively
    psnr_val  = psnr_metric(reconstructed, original, data_range=1.0).item()
    ssim_val  = ssim_metric(reconstructed, original, data_range=1.0).item()

    # Lpip commented out for now
    # LPIPS expects inputs in [-1, 1] when normalize=False, but we set normalize=True
    # so [0, 1] inputs are fine.  Use no_grad to avoid storing the graph.
    # lpips_fn  = get_lpips_metric(device)
    # # LPIPS only supports 3-channel images; if satellite data has more channels,
    # # fall back to the first 3 bands.
    # orig_3ch  = original[:, :3].clamp(0, 1)
    # recon_3ch = reconstructed[:, :3].clamp(0, 1)
    # with torch.no_grad():
    #     lpips_val = lpips_fn(recon_3ch, orig_3ch).item()

    return {"psnr": psnr_val, "ssim": ssim_val} # , "lpips": lpips_val

class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""
    conf = {
        "net": {"type": "Adam", "lr": args.learning_rate},
        "aux": {"type": "Adam", "lr": args.aux_learning_rate},
    }
    optimizer = net_aux_optimizer(net, conf)
    return optimizer["net"], optimizer["aux"]


def train_one_epoch(
    model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm, wandb_obj
):
    model.train()
    device = next(model.parameters()).device

    for i, d in enumerate(train_dataloader):


        d = d.to(device)
        # print("type of d", type(d))
        # pritn("shape of d", d.shape)
        # print
        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out_net = model(d)

        out_criterion = criterion(out_net, d)
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        if i % 10 == 0:
            print(
                f"Train epoch {epoch}: ["
                f"{i*len(d)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tLoss: {out_criterion["loss"].item():.3f} |'
                f'\tMSE loss: {out_criterion["mse_loss"].item():.3f} |'
                f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
                f"\tAux loss: {aux_loss.item():.2f}"
            )

        # Compute perceptual metrics on the current batch (detached, no grad needed)
        with torch.no_grad():
            x_hat = out_net["x_hat"].clamp(0, 1)
            batch_metrics = compute_metrics(d, x_hat)

        wandb_obj.log( {
            "train/Rate Distortion Loss": out_criterion["loss"].item(),
            "train/MSE Loss Training" : out_criterion["mse_loss"].item(), 
            "train/Bpp Loss Training" : out_criterion["bpp_loss"].item(),
            "train/Aux Loss Training" : aux_loss.item(),
            "train/PSNR Training": batch_metrics["psnr"],
            "train/SSIM Training": batch_metrics["ssim"],
            # "train/LPIPS":                batch_metrics["lpips"]
        })


def test_epoch(epoch, test_dataloader, model, criterion):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()
    psnr_meter  = AverageMeter()
    ssim_meter  = AverageMeter()
    # lpips_meter = AverageMeter()

    with torch.no_grad():
        for d in test_dataloader:
            d = d.to(device)
            out_net = model(d)
            out_criterion = criterion(out_net, d)

            aux_loss.update(model.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            mse_loss.update(out_criterion["mse_loss"])

            x_hat = out_net["x_hat"].clamp(0, 1)
            metrics = compute_metrics(d, x_hat)
            psnr_meter.update(metrics["psnr"],  n=d.size(0))
            ssim_meter.update(metrics["ssim"],  n=d.size(0))
            # lpips_meter.update(metrics["lpips"], n=d.size(0))

    print(
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg:.3f} |"
        f"\tMSE loss: {mse_loss.avg:.3f} |"
        f"\tBpp loss: {bpp_loss.avg:.2f} |"
        f"\tAux loss: {aux_loss.avg:.2f}\n"
    )

    return {
        "Loss_ma" : loss.avg, # _ma stands for moving average, this comes from AverageMeter()
        "MSE_loss_ma" : mse_loss.avg, 
        "Bpp_loss_ma" : bpp_loss.avg,
        "Aux_loss_ma" : aux_loss.avg, 
        "PSNR_ma":      psnr_meter.avg,
        "SSIM_ma":      ssim_meter.avg,
        # "LPIPS_ma":     lpips_meter.avg,
    }


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "checkpoint_best_loss.pth.tar")


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-m",
        "--model",
        default="bmshj2018-factorized",
        choices=image_models.keys(),
        help="Model architecture (default: %(default)s)",
    )
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="Training dataset"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=100,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-nm",
        "--run_name",
        default=timestamp,
        type=str,
        help="Name of the run in Weights and biases (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=4,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=1e-2,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=64,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        type=float,
        default=1e-3,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument("--seed", type=int, help="Set random seed for reproducibility")
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    # init wandb
    run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="anasnamouchi",
    # Set the wandb project where this run will be logged.
    project="Thesis",
    # Track hyperparameters and run metadata.
    config={
        # "learning_rate": 0.02,
        "architecture": "Hyperprior + Downsample_CNN",
        "dataset": "ssl4eo-small",
        "epochs": args.epochs,
        "name": args.run_name
    },
)
    
    # Random crop below is not causing any problems, quite the opposite
    # it is needed for the proper image sizes 
    # the model will end up being trained on 256*256 images regardless of their size in the dataset
    # [] 256 is hardcoded and needs to change
    train_transforms = transforms.Compose(
        [transforms.RandomCrop(256), transforms.ToTensor()]
    )

    test_transforms = transforms.Compose(
        [transforms.CenterCrop(256), transforms.ToTensor()]
    )

    # train_dataset = ImageFolder("/home/anas/datasets/ssl42eo-small-torun", split="train", transform=train_transforms)
    # test_dataset = ImageFolder("/home/anas/datasets/ssl42eo-small-torun", split="test", transform=test_transforms)

    # train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms)
    # test_dataset = ImageFolder(args.dataset, split="test", transform=test_transforms)

    train_dataset = SSL4EOS12RGBDataset('/home/anas/thesis/checkpoints/train/', is_train=True)
    test_dataset   = SSL4EOS12RGBDataset('/home/anas/thesis/checkpoints/val/',   is_train=False)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )
    
    # LOAD EMBEDDING MODEL 
    # embedding_model_id = "MVRL/satmaepp_ViT-L_pretrain_fmow_rgb"
    # embedding_model = SatMAE.from_pretrained(embedding_model_id).to(device).eval()

    embedding_model = LowResMask()
    

    #net = image_models[args.model](quality=3)
    # net = ScaleHyperpriorCrossAttention(30, 24, 30, embedding_model=embedding_model, embedding_type="downsample_cnn")
    net = ScaleHyperpriorBahdanau(30, 24, 30, embedding_type="downsample_cnn")
    net = net.to(device)

    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)

    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    criterion = RateDistortionLoss(lmbda=args.lmbda)

    last_epoch = 0
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        net.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    best_loss = float("inf")
    for epoch in range(last_epoch, args.epochs):

        ####
        # if epoch %10 == 0:
        #     images_every_10_epochs(test_dataset,net,epoch)


        ###
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
            wandb_obj = run
        )
        losses = test_epoch(epoch, test_dataloader, net, criterion)
        lr_scheduler.step(losses["Loss_ma"])
        
        # [] put the saving images below
        if epoch %10 == 0:
            images_every_10_epochs(test_dataset,net,epoch)
        

            

        # [] above is the update to lr, make sure you tack it then
        is_best = losses["Loss_ma"] < best_loss
        best_loss = min(losses["Loss_ma"], best_loss)

        run.log({
            # [] just commeneted for now, need to find a way for tracking the scheduler properly
            #"lr_scheduler": lr_scheduler,
            "eval/Loss_ma" : losses["Loss_ma"], 
            "eval/MSE_loss_ma" : losses["MSE_loss_ma"], 
            "eval/Bpp_loss_ma" : losses["Bpp_loss_ma"],
            "eval/Aux_loss_ma" : losses["Aux_loss_ma"], 
            "eval/PSNR":        losses["PSNR_ma"],
            "eval/SSIM":        losses["SSIM_ma"],
            # "eval/LPIPS":       losses["LPIPS_ma"],

            }
        )
        if args.save:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "loss": losses["Loss_ma"],
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                is_best,
            )


if __name__ == "__main__":
    main(sys.argv[1:])
