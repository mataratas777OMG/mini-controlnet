import os
import argparse

import torch

from src.models.unet import UnetWithControl
from src.data.dataset import EdgeImageDataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--data-train", type=str, default="data/train")
    parser.add_argument("--data-val", type=str, default="data/val")
    parser.add_argument("--data-test", type=str, default="data/test")
    parser.add_argument("--out", type=str, default="models/", help="Output directory")
    parser.add_argument("--freeze_backbone", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = EdgeImageDataset(
        root_dir=args.data_train,
        size=args.size
    )
    val_dataset = EdgeImageDataset(
        root_dir=args.data_val,
        size=args.size
    )
    test_dataset = EdgeImageDataset(
        root_dir=args.data_test,
        size=args.size
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    model = UnetWithControl(freeze_backbone=args.freeze_backbone).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    loss = torch.nn.MSELoss()