"""ViT training script for CIFAR-10."""

import os
import math
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from vit import vit_tiny, vit_small, vit_base
from data import get_cifar10
from utils import AverageMeter, plot_history


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = (logits.argmax(dim=1) == y).float().mean().item() * 100.0
        loss_meter.update(loss.item(), x.size(0))
        acc_meter.update(acc, x.size(0))

    return loss_meter.avg, acc_meter.avg


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean().item() * 100.0
        loss_meter.update(loss.item(), x.size(0))
        acc_meter.update(acc, x.size(0))

    return loss_meter.avg, acc_meter.avg


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- config ----------------------------------------------------------
    batch_size    = 512
    epochs        = 100
    lr            = 3e-4
    weight_decay  = 0.05
    warmup_epochs = 5
    dropout       = 0.1

    model = vit_tiny(patch_size=4, num_classes=10, dropout=dropout).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_loader, _, test_loader = get_cifar10(batch_size=batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Cosine annealing with linear warmup
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
    scheduler = SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler],
                             milestones=[warmup_epochs])

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    best_acc = 0.0
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), os.path.join(save_dir, "best.pth"))

        print(f"Epoch {epoch:3d} | "
              f"lr {scheduler.get_last_lr()[0]:.2e} | "
              f"train loss {train_loss:.3f}  acc {train_acc:.2f}% | "
              f"test loss {test_loss:.3f}  acc {test_acc:.2f}%")

    print(f"\nBest test accuracy: {best_acc:.2f}%")
    plot_history(history, save_path="training.png")


if __name__ == "__main__":
    main()
