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

from new_utils import LowResMask, save_tensor_as_image, average_entropy

import wandb
# THIS AND NEW_UTILS WERE EDITED


from datetime import datetime
from pathlib import Path
from torch.utils.data import Subset, DataLoader
import os
import random
from loader import SSL4EOS12RGBDataset

from torchmetrics.functional import peak_signal_noise_ratio as psnr_metric
from torchmetrics.functional import structural_similarity_index_measure as ssim_metric

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from loader import models_dict
from PIL import Image
from torchvision import transforms

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")


# save example images from test suite every 10 epochs
# def images_every_10_epochs(test_dataset, model,epoch, reconstruction_path, cropped_path ): 
    # device = next(model.parameters()).device   
    # model.eval()
    # model.update(force=True)

    # random_indices = random.sample(range(len(test_dataset)), 10)
    # os.makedirs(Path(reconstruction_path/f"epoch_{epoch}"), exist_ok=True) 
    # os.makedirs(Path(cropped_path/f"epoch_{epoch}"), exist_ok=True) 

    # # Create a subset and a new dataloader
    # subset = Subset(test_dataset, random_indices)
    # random_loader = DataLoader(subset, batch_size=10, shuffle=False)
    # for ind, tens in enumerate(random_loader):
    #     counter=0
    #     for sec_ind, image in enumerate(tens):
    #         image = image.to(device)            
    #         save_tensor_as_image(image, Path(cropped_path / f"epoch_{epoch}"/f"image{sec_ind}_epoch{epoch}.png"))
    #         tensor = image.unsqueeze(0)
    #         out = model.compress(tensor)
    #         x_hat = model.decompress(out["strings"], out["shape"])
    #         save_tensor_as_image(x_hat["x_hat"].squeeze(0), Path(reconstruction_path / f"epoch_{epoch}"/f"image{sec_ind}_epoch{epoch}.png"))
    #         counter+=1
    
    # model.train()


def images_every_10_epochs(image_dir, model, epoch, reconstruction_path, cropped_path):

    device = next(model.parameters()).device
    model.eval()
    model.update(force=True)

    all_images = sorted(Path(image_dir).glob("*.png"))
    assert len(all_images) > 0, f"No .png files found in {image_dir}"

    selected = random.sample(all_images, min(10, len(all_images)))

    os.makedirs(Path(reconstruction_path) / f"epoch_{epoch}", exist_ok=True)
    os.makedirs(Path(cropped_path) / f"epoch_{epoch}", exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    for sec_ind, img_path in enumerate(selected):
        image = transform(Image.open(img_path).convert("RGB")).to(device)  # (C, H, W)
        save_tensor_as_image(image, Path(cropped_path) / f"epoch_{epoch}" / f"image{sec_ind}_epoch{epoch}.png")

        tensor = image.unsqueeze(0)
        out = model.compress(tensor)
        x_hat = model.decompress(out["strings"], out["shape"])
        save_tensor_as_image(x_hat["x_hat"].squeeze(0), Path(reconstruction_path) / f"epoch_{epoch}" / f"image{sec_ind}_epoch{epoch}.png")

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

    return {"psnr": psnr_val, "ssim": ssim_val}

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
    model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm, wandb_obj,
    lr_scheduler
):
    model.train()
    device = next(model.parameters()).device

    for i, d in enumerate(train_dataloader):


        d = d.to(device)
        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out_net = model(d)

        out_criterion = criterion(out_net, d)
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()
        lr_scheduler.step(out_criterion["loss"].item()) 

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
            "train/Learning Rate":        optimizer.param_groups[0]["lr"], 
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
    y_entropy = AverageMeter()
    z_entropy = AverageMeter()
    y_g_present = False
    y_g_entropy = AverageMeter()

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

            y_entropy.update(average_entropy(out_net["likelihoods"]["y"]))
            z_entropy.update(average_entropy(out_net["likelihoods"]["z"]))

            y_g = out_net["likelihoods"].get("y_g")

            if y_g is not None:
                y_g_present = True
                y_g_entropy.update(average_entropy(y_g))
            

    print(
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg:.3f} |"
        f"\tMSE loss: {mse_loss.avg:.3f} |"
        f"\tBpp loss: {bpp_loss.avg:.2f} |"
        f"\tAux loss: {aux_loss.avg:.2f}\n"
    )

    # handeling whether we are working with bahdanau or vanilla model when meaasuring entropy of likelihoods

    if y_g_present:
        return {
        "Loss_ma" : loss.avg, 
        "MSE_loss_ma" : mse_loss.avg, 
        "Bpp_loss_ma" : bpp_loss.avg,
        "Aux_loss_ma" : aux_loss.avg, 
        "PSNR_ma":      psnr_meter.avg,
        "SSIM_ma":      ssim_meter.avg,
        "Y Entropy":    y_entropy.avg,
        "Z Entropy":    z_entropy.avg,
        "Y_G Entropy":  y_g_entropy.avg
    }


    return {
        "Loss_ma" : loss.avg, 
        "MSE_loss_ma" : mse_loss.avg, 
        "Bpp_loss_ma" : bpp_loss.avg,
        "Aux_loss_ma" : aux_loss.avg, 
        "PSNR_ma":      psnr_meter.avg,
        "SSIM_ma":      ssim_meter.avg,
        "Y Entropy":    y_entropy.avg,
        "Z Entropy":    z_entropy.avg,
    }


def save_checkpoint(state, is_best, filename=f"checkpoint.pth.tar", copy_name="checkpoint_best_loss.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, copy_name)


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-m",
        "--model",
        # default="bmshj2018-factorized",
        required=True,
        choices=models_dict.keys(),
        help="Model architecture (default: %(default)s)",
    )
    parser.add_argument(
        "-d", "--dataset", type=str,  help="Training dataset", required=True
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=20,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "--wandb_mode",
        default="online",
        type=str,
        choices=["offline", "online"],
        help="W&B experiment tracking default mode (default: %(default)s)",
    )
    parser.add_argument(
        "-nm",
        "--run_name",
        default=timestamp,
        required=True,
        type=str,
        help="Name of the run in Weights and biases (default: %(default)s)",
    )
    parser.add_argument(
        "-s",
        "--system",
        required=True,
        type=str,
        choices=["cip_pool", "mcml", "lightning", "coder-iabg"],
        help="Which system are you running on? mcml cluster or cip pool (default: %(default)s)",
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
        "--batch-size", type=int, default=32, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=32,
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
    parser.add_argument(
    "--tags",
    nargs="+",         
    default=[],
    type=str,
    help="Tags for the run"
    )
    parser.add_argument("-K", type=int, help="size of K latent space", required=True)
    parser.add_argument("-M", type=int, help="size of M latent space", required=True)
    parser.add_argument("-N", type=int, help="size of K latent space", required=True)

    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)

    if args.system == "cip_pool":    
        reconstruction_path = Path(f"/home/ra42tif/images_experiments/images/{args.model}_{args.N}_{args.M}_{args.K}/reconstructed/")
        cropped_path = Path(f"/home/ra42tif/images_experiments/images/{args.model}_{args.N}_{args.M}_{args.K}/cropped/")

        train_dataset = SSL4EOS12RGBDataset(
            "/home/ra42tif/datasets/train_10gb_version/subset_train_big_dataset"
            , is_train=True)
        
        test_dataset   = SSL4EOS12RGBDataset(
        "/home/ra42tif/datasets/eval_10gb_version/S2RGB"
       ,is_train=False)

    elif args.system == "mcml":
        reconstruction_path = Path(f"/dss/dsshome1/0E/ra42tif2/thesis_docs/images/{args.run_name}_{args.N}_{args.M}_{args.K}/reconstructed/")
        cropped_path = Path(f"/dss/dsshome1/0E/ra42tif2/thesis_docs/images/{args.run_name}_{args.N}_{args.M}_{args.K}/cropped/")

        train_dataset = SSL4EOS12RGBDataset(     
            args.dataset   
        # "/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra42tif2/subset_train_big_dataset"
        # "/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra42tif2/20gb_subset_ssl4eo"
        , is_train=True)

        test_dataset   = SSL4EOS12RGBDataset(
            "/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra42tif2/data/ssl4eo-s12/val/S2RGB"
            ,is_train=False)
        
    
    elif args.system == "lightning":
        reconstruction_path = Path(f"/teamspace/studios/this_studio/images/{args.model}_{args.N}_{args.M}_{args.K}/reconstructed/")
        cropped_path = Path(f"/teamspace/studios/this_studio/images/{args.model}_{args.N}_{args.M}_{args.K}/cropped/")

        train_dataset = SSL4EOS12RGBDataset(
            "/teamspace/studios/this_studio/train_split"
            , is_train=True)
        
        test_dataset   = SSL4EOS12RGBDataset(
            "/teamspace/studios/this_studio/val_split"
            ,is_train=False)

    elif args.system == "coder-iabg":
        reconstruction_path = Path(f"/home/ubuntu/images_experiments/{args.model}_{args.N}_{args.M}_{args.K}/reconstructed/")
        cropped_path = Path(f"/home/ubuntu/images_experiments/{args.model}_{args.N}_{args.M}_{args.K}/cropped/")

        train_dataset = SSL4EOS12RGBDataset(
            "/home/ubuntu/data/small_dataset_to_transfer/subset_train_big_dataset"
            , is_train=True)
        
        test_dataset   = SSL4EOS12RGBDataset(
            "/home/ubuntu/data/small_val_to_transfer/S2RGB"
           ,is_train=False)

    reconstruction_dir = "/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra42tif2/reconstruction_dir"
    
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    # init wandb
    print("here pre init of wandb")
    run = wandb.init(
    mode=args.wandb_mode,
    # dir=f"wandb/{args.run_name}",
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="anasnamouchi",
    notes=f"Ran using 20gb subset of dataset, for 20 epochs, with patch size of {args.patch_size}, and lambda of {args.lmbda}. This is a run for the {args.model} architecture with N={args.N}, M={args.M}, K={args.K}. The tags for this run are {args.tags}",
    # Set the wandb project where this run will be logged.
    project="Thesis",
    name= args.run_name,
    # use tags too
    tags=args.tags,

    # Track hyperparameters and run metadata.
    config={
        # "learning_rate": 0.02,
        "architecture": "Hyperprior + Downsample_CNN",
        "dataset": "ssl4eo-small",
        "epochs": args.epochs,
    },
)
    
    # [] 256 is hardcoded and needs to change
    # train_transforms = transforms.Compose(
    #     [transforms.RandomCrop(256), transforms.ToTensor()]
    # )

    # test_transforms = transforms.Compose(
    #     [transforms.CenterCrop(256), transforms.ToTensor()]
    # )


    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    print(f"lambda: {args.lmbda}")

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
    
    
    if args.model=="basic-hyperprior":
        net = models_dict[args.model](args.N, args.M)
    else:
        net = models_dict[args.model](args.N, args.M, args.K, embedding_type="downsample_cnn")


    net = net.to(device)

    print(f"Model initilaized is {args.model} with {args.N,args.M,args.K}")

    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)

    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=500) # maybe play with patience if you notice problems, or og back to updating every epoch
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

        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
            wandb_obj = run,
            lr_scheduler=lr_scheduler
        )

        # perform evaluation only every 3 epochs
        if epoch%3==0:
                
            losses = test_epoch(epoch, test_dataloader, net, criterion)
            # lr_scheduler updated to update every step using training loss instead of validation loss
            # lr_scheduler.step(losses["Loss_ma"])
            

            is_best = losses["Loss_ma"] < best_loss
            best_loss = min(losses["Loss_ma"], best_loss)

            # handeling whether we are working with bahdanau or vanilla model 

            log_dict = {
                "eval/Loss_ma" : losses["Loss_ma"], 
                "eval/MSE_loss_ma" : losses["MSE_loss_ma"], 
                "eval/Bpp_loss_ma" : losses["Bpp_loss_ma"],
                "eval/Aux_loss_ma" : losses["Aux_loss_ma"], 
                "eval/PSNR":        losses["PSNR_ma"],
                "eval/SSIM":        losses["SSIM_ma"],
                "eval/Y Entropy":    losses["Y Entropy"],
                "eval/Z Entropy":    losses["Z Entropy"],
            }

            if "Y_G Entropy" in losses:
                log_dict["eval/Y_G Entropy"] = losses["Y_G Entropy"]
            
            run.log(log_dict)
            

        if args.save:
        # trying to save only if there is an improvement, if there are problems with this go back to older version
            if is_best:
                        
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
		filename=f"checkpoint_{args.run_name}_{args.N}_{args.M}_{args.K}.pth.tar",
		copy_name= f"checkpoint_best_loss_{args.run_name}_{args.N}_{args.M}_{args.K}.pth.tar"
                )
        
        # save example images
        if epoch %5 == 0:
            images_every_10_epochs(reconstruction_dir,net,epoch, reconstruction_path, cropped_path)
        
        


if __name__ == "__main__":
    main(sys.argv[1:])
