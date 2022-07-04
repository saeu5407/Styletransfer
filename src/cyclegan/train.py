import argparse
import os
import torch
import sys
sys.path.append('..')
import GPUtil

from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from datasets import CycleGANDataset
from networks import CycleGANGenerator, CycleGANDiscriminator
from utils import init_weight, ImagePool, LossDisplayer

import glob
import time
from datetime import timedelta

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=1000)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--dataset_path", type=str,
                    default=os.path.join(os.getcwd().split("src")[0], "datasets", "horse2zebra"))
parser.add_argument("--size", type=int, default=256)
parser.add_argument("--lambda_ide", type=float, default=10)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--pool_size", type=int, default=50)
parser.add_argument("--identity", action="store_true")
parser.add_argument("--class_a", type=str, default='trainA')
parser.add_argument("--class_b", type=str, default='trainB')

args = parser.parse_args()

def train():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # Model
    num_blocks = 6 if args.size <= 256 else 8
    netG_A2B = CycleGANGenerator(num_blocks).to(device)
    netG_B2A = CycleGANGenerator(num_blocks).to(device)
    netD_A = CycleGANDiscriminator().to(device)
    netD_B = CycleGANDiscriminator().to(device)

    # Cuda
    if torch.cuda.device_count() > 1:
        print("Use Multi GPU : {} GPUs".format(torch.cuda.device_count()))
        netG_A2B = nn.DataParallel(netG_A2B)
        netG_B2A = nn.DataParallel(netG_B2A)
        netD_A = nn.DataParallel(netD_A)
        netD_B = nn.DataParallel(netD_B)

    netG_A2B.apply(init_weight)
    netG_B2A.apply(init_weight)
    netD_A.apply(init_weight)
    netD_B.apply(init_weight)
    epoch = 0

    netG_A2B.train()
    netG_B2A.train()
    netD_A.train()
    netD_B.train()

    # Dataset
    transform = transforms.Compose(
        [
            transforms.Resize((args.size, args.size)),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )

    dataloader = DataLoader(
        CycleGANDataset(args.dataset_path, [args.class_a, args.class_b], transform))

    dataset_name = os.path.basename(args.dataset_path)

    pool_fake_A = ImagePool(args.pool_size)
    pool_fake_B = ImagePool(args.pool_size)

    # Loss
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()
    criterion_GAN = nn.MSELoss()

    disp = LossDisplayer(["G_GAN", "G_recon", "D"])
    summary = SummaryWriter()

    # Optimizer, Schedular
    optim_G = optim.Adam(
        list(netG_A2B.parameters()) + list(netG_B2A.parameters()),
        lr=args.lr,
        betas=(0.5, 0.999),
    )
    optim_D_A = optim.Adam(netD_A.parameters(), lr=args.lr)
    optim_D_B = optim.Adam(netD_B.parameters(), lr=args.lr)

    lr_lambda = lambda epoch: 1 - ((epoch - 1) // 100) / (args.epoch / 100)
    scheduler_G = optim.lr_scheduler.LambdaLR(optimizer=optim_G, lr_lambda=lr_lambda)
    scheduler_D_A = optim.lr_scheduler.LambdaLR(
        optimizer=optim_D_A, lr_lambda=lr_lambda
    )
    scheduler_D_B = optim.lr_scheduler.LambdaLR(
        optimizer=optim_D_B, lr_lambda=lr_lambda
    )

    print("CHECKPOINT DIR")
    checkpoint_dir = os.path.join(os.getcwd().split("src")[0], "checkpoint", dataset_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(checkpoint_dir)

    # Checkpoint 있을 시 해당 체크포인트부터 다시 진행
    if len(glob.glob(os.path.join(checkpoint_dir, '*.pth'))) >= 1:
        checkpoint = torch.load(glob.glob(os.path.join(checkpoint_dir, '*.pth'))[-1], map_location=device)
        epoch = checkpoint['epoch']
        netG_A2B.load_state_dict(checkpoint["netG_A2B_state_dict"])
        netG_B2A.load_state_dict(checkpoint["netG_B2A_state_dict"])
        netD_A.load_state_dict(checkpoint["netD_A_state_dict"])
        netD_B.load_state_dict(checkpoint["netD_B_state_dict"])

        optim_G.load_state_dict(checkpoint["optim_G"])
        optim_D_A.load_state_dict(checkpoint["optim_D_A"])
        optim_D_B.load_state_dict(checkpoint["optim_D_B"])

        netG_A2B.train()
        netG_B2A.train()
        netD_A.train()
        netD_B.train()
        print("start checkpoint : {}".format(glob.glob(os.path.join(checkpoint_dir, '*.pth'))[-1]))

    # Training
    while epoch < args.epoch:
        epoch += 1
        # print(f"\nEpoch {epoch}")
        print("Epoch : {}".format(epoch))
        start = time.process_time()

        for idx, (real_A, real_B) in enumerate(dataloader):
            # print(f"{idx}/{len(dataloader)}", end="\r")
            # print("{}/{}".format(idx, len(dataloader)))
            real_A = real_A.to(device)
            real_B = real_B.to(device)

            # Forward gan_classifier
            fake_A = netG_B2A(real_B)
            fake_B = netG_A2B(real_A)

            cycle_A = netG_B2A(fake_B)
            cycle_B = netG_A2B(fake_A)

            pred_fake_A = netD_A(fake_A)
            pred_fake_B = netD_B(fake_B)

            # Calculate and backward generator gan_classifier losses
            loss_cycle_A = criterion_cycle(cycle_A, real_A)
            loss_cycle_B = criterion_cycle(cycle_B, real_B)
            loss_GAN_A = criterion_GAN(pred_fake_A, torch.ones_like(pred_fake_A))
            loss_GAN_B = criterion_GAN(pred_fake_B, torch.ones_like(pred_fake_B))

            loss_G = (
                args.lambda_ide * (loss_cycle_A + loss_cycle_B)
                + loss_GAN_A
                + loss_GAN_B
            )

            if args.identity:
                identity_A = netG_B2A(real_A)
                identity_B = netG_A2B(real_B)
                loss_identity_A = criterion_identity(identity_A, real_A)
                loss_identity_B = criterion_identity(identity_B, real_B)
                loss_G += 0.5 * args.lambda_ide * (loss_identity_A + loss_identity_B)

            optim_G.zero_grad()
            loss_G.backward()
            optim_G.step()

            # Calculate and backward discriminator gan_classifier losses
            pred_real_A = netD_A(real_A)
            pred_fake_A = netD_A(pool_fake_A.query(fake_A))

            loss_D_A = 0.5 * (
                criterion_GAN(pred_real_A, torch.ones_like(pred_real_A))
                + criterion_GAN(pred_fake_A, torch.zeros_like(pred_fake_A))
            )

            optim_D_A.zero_grad()
            loss_D_A.backward()
            optim_D_A.step()

            pred_real_B = netD_B(real_B)
            pred_fake_B = netD_B(pool_fake_B.query(fake_B))

            loss_D_B = 0.5 * (
                criterion_GAN(pred_real_B, torch.ones_like(pred_real_B))
                + criterion_GAN(pred_fake_B, torch.zeros_like(pred_fake_B))
            )

            optim_D_B.zero_grad()
            loss_D_B.backward()
            optim_D_B.step()

            # Record loss
            loss_G_GAN = loss_GAN_A + loss_GAN_B
            loss_G_recon = loss_G - loss_G_GAN
            loss_D = loss_D_A + loss_D_B
            disp.record([loss_G_GAN, loss_G_recon, loss_D])

        # Step scheduler
        scheduler_G.step()
        scheduler_D_A.step()
        scheduler_D_B.step()

        # Record and display loss
        avg_losses = disp.get_avg_losses()
        summary.add_scalar("loss_G_GAN", avg_losses[0], epoch)
        summary.add_scalar("loss_G_recon", avg_losses[1], epoch)
        summary.add_scalar("loss_D", avg_losses[2], epoch)

        disp.display()
        disp.reset()

        # Save checkpoint
        if epoch % 50 == 0 or epoch == 1:
            torch.save(
                {
                    "netG_A2B_state_dict": netG_A2B.state_dict(),
                    "netG_B2A_state_dict": netG_B2A.state_dict(),
                    "netD_A_state_dict": netD_A.state_dict(),
                    "netD_B_state_dict": netD_B.state_dict(),
                    "epoch": epoch,
                    "loss_G_GAN": loss_G_GAN,
                    "loss_G_recon": loss_G_recon,
                    "loss_D": loss_D,
                    "optim_G": optim_G.state_dict(),
                    "optim_D_A": optim_D_A.state_dict(),
                    "optim_D_B": optim_D_B.state_dict()
                },
                os.path.join(os.getcwd().split("src")[0], "checkpoint", dataset_name, str(epoch) + ".pth"),
            )

        # Calculate time required
        print("Time elapsed: ", timedelta(seconds=time.process_time()-start))
        torch.cuda.empty_cache()
        GPUtil.showUtilization()


if __name__ == "__main__":
    train()